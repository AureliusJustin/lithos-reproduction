#define _GNU_SOURCE
/*
 * sched_stream.c — the stream registry: LithOS's LAUNCH QUEUES (Fig. 9, step 1).
 *
 * "A launch queue is created when an application creates a stream" (§5.2). Here
 * that queue is a StreamState slot: it carries the stream's compute quota, its
 * assigned TPC range, idle/outstanding bookkeeping (which TPC stealing and the
 * throttle read), and the identity the latency predictor keys off.
 *
 * This file owns the shared scheduler state; the other scheduler files
 * (tpc_alloc.c, dispatch.c, sched.c) read it through sched_internal.h.
 */
#include "sched_internal.h"
#include "lithos_sched.h"
#include "real.h"
#include "atomizer.h"
#include "qmd.h"
#include "coord.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- shared state (declared extern in sched_internal.h) ------------------- */
StreamState     g_streams[MAX_STREAMS];
pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
uint64_t        g_total_launches, g_total_atoms;
uint32_t        g_num_tpcs = 0;         /* 0 until detect_tpcs() runs           */
uint64_t        g_idle_ns  = 1000000;   /* 1 ms without a launch => idle        */
int             g_default_quota = -1;   /* LITHOS_QUOTA: quota for every stream */
int             g_tpc_base = 0;         /* LITHOS_TPC_BASE: this process's first
                                         * TPC. A central LithOS scheduler would
                                         * assign disjoint ranges per tenant; under
                                         * MPS this lets separate processes occupy
                                         * disjoint TPCs. */

/* Scheduling priority given to streams the application did not prioritise
 * itself (LITHOS_PRIO). Lower = more important, matching CUDA's convention, so a
 * stream created with cuStreamCreateWithPriority keeps the value the app chose
 * and needs no translation. */
static int      g_default_prio = 0;


/* ---- device topology ------------------------------------------------------ */

/* Discover how many TPCs this GPU has. Called lazily on the first launch because
 * at library-init time (first cuInit) no CUDA context exists yet. */
void detect_tpcs(void) {
    int sms = 0, major = 0;
    CUdevice dev;
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) cuDeviceGet(&dev, 0);

    /* Use the REAL attribute query rather than our own cuDeviceGetAttribute
     * wrapper: that wrapper reports the tenant's *allocated* SM count (the §6
     * spoof), and we need the true physical number here. */
    CUresult (*get_attr)(int*, CUdevice_attribute, CUdevice) =
        g_real.cuDeviceGetAttribute ? g_real.cuDeviceGetAttribute : cuDeviceGetAttribute;
    get_attr(&sms,   CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,      dev);
    get_attr(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,  dev);

    int sms_per_tpc = (major >= 6) ? 2 : 1;      /* Pascal and later: 2 SMs per TPC */
    g_num_tpcs = sms > 0 ? (uint32_t)(sms / sms_per_tpc) : 0;
    qmd_set_num_tpcs((int)g_num_tpcs);           /* for clean mask logging */
    SLOG("detected %d SMs -> %u TPCs", sms, g_num_tpcs);

    /* Now that the device is known, register with the system-wide tenant table
     * (§5.1). It hands back a range disjoint from every other live tenant, so
     * cross-application quotas need no hand-assigned LITHOS_TPC_BASE. */
    if (coord_join((int)g_num_tpcs, g_default_quota, g_lithos_cfg.priority)) {
        int lo, hi;
        if (coord_my_range(&lo, &hi) && g_default_quota > 0) {
            g_tpc_base = lo;
            SLOG("coordinator assigned TPC range [%d,%d) (%d tenants)", lo, hi, coord_tenant_count());
        }
    }
}

void lithos_sched_init(void) {
    memset(g_streams, 0, sizeof(g_streams));
    atomizer_init();
    const char* q = getenv("LITHOS_QUOTA");
    if (q) g_default_quota = atoi(q);
    const char* base = getenv("LITHOS_TPC_BASE");
    if (base) g_tpc_base = atoi(base);
    const char* prio = getenv("LITHOS_PRIO");
    if (prio) g_default_prio = atoi(prio);
    /* TPC count is detected lazily (see detect_tpcs). */
}

/* ---- stream registry ------------------------------------------------------ */

/* Look up an existing launch queue. Caller holds g_lock. */
StreamState* find_stream(CUstream s) {
    for (int i = 0; i < MAX_STREAMS; i++)
        if (g_streams[i].in_use && g_streams[i].s == s) return &g_streams[i];
    return NULL;
}

/* Look up, or lazily create, this stream's launch queue. Lazy creation matters
 * because the CUDA runtime (PyTorch/TF/JAX/TensorRT) launches on the default
 * stream and on streams it created before we were watching — without this a
 * global LITHOS_QUOTA would never attach to framework work.
 *
 * We always allocate a slot, even with no quota, because the predictor and the
 * per-stream bookkeeping key off it; masking simply no-ops when quota_tpcs <= 0,
 * so unquota'd streams stay unrestricted. Caller holds g_lock. */
/* How many quota'd streams have been handed a range, and whether any two of those
 * ranges actually differ. The paper's quota is per APPLICATION, so by default every
 * stream of this process shares one range — and then TPC stealing between them is a
 * provable no-op (lending a range identical to your own adds no TPCs). We track
 * that so tpc_alloc.c can skip the stealing scan entirely in that case. */
static int g_quota_streams   = 0;
int        g_ranges_disjoint = 0;   /* read by compute_disable_mask_ex */

/* Give a newly registered stream its TPC range.
 *
 *  default                    every stream shares [tpc_base, tpc_base+quota) —
 *                             one quota for the whole application (the paper's model)
 *  LITHOS_PERSTREAM_QUOTA=1   each stream gets its OWN disjoint quota-sized slice,
 *                             packed consecutively and wrapped within the device.
 *                             This is what makes intra-process TPC stealing
 *                             meaningful: an idle stream now owns TPCs that a busy
 *                             one can borrow. Caller holds g_lock. */
static void assign_tpc_range(StreamState* st) {
    st->priority   = g_default_prio;
    st->quota_tpcs = g_default_quota;
    if (g_default_quota <= 0) return;

    /* A stream is usually created BEFORE the first launch, so the device may not
     * have been probed yet — and probing is also what registers this process with
     * the system-wide tenant table and yields our assigned TPC range (§5.1).
     * Without this the stream would be pinned to the default base before the
     * coordinator ever spoke, and every tenant would land on the same TPCs. */
    if (g_num_tpcs == 0) detect_tpcs();

    if (g_lithos_cfg.perstream_quota) {
        int span = (g_num_tpcs > 0) ? (int)g_num_tpcs - g_tpc_base : g_default_quota;
        if (span < g_default_quota) span = g_default_quota;
        int slot = (g_quota_streams * g_default_quota) % span;
        st->tpc_lo = g_tpc_base + slot;
        st->tpc_hi = st->tpc_lo + g_default_quota;
        if (g_quota_streams > 0 && slot != 0) g_ranges_disjoint = 1;
    } else {
        st->tpc_lo = g_tpc_base;
        st->tpc_hi = g_tpc_base + g_default_quota;
    }
    g_quota_streams++;
}

StreamState* ensure_stream(CUstream s) {
    StreamState* st = find_stream(s);
    if (st) return st;
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (g_streams[i].in_use) continue;
        memset(&g_streams[i], 0, sizeof(StreamState));
        g_streams[i].in_use = 1;
        g_streams[i].s      = s;
        assign_tpc_range(&g_streams[i]);
        return &g_streams[i];
    }
    return NULL;    /* table full: run unrestricted rather than fail the launch */
}

/* Index of a slot in g_streams — the predictor's per-queue key. */
int stream_slot(StreamState* st) { return st ? (int)(st - g_streams) : -1; }

/* How many TPCs this stream's kernels currently run on (its quota, or all). */
int stream_tpcs(StreamState* st) {
    if (st && st->quota_tpcs > 0) return st->tpc_hi - st->tpc_lo;
    return g_num_tpcs > 0 ? (int)g_num_tpcs : 1;
}

/* ---- lifecycle hooks called by the interposer ----------------------------- */

void lithos_stream_created(CUstream s, int priority) {
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (g_streams[i].in_use) continue;
        memset(&g_streams[i], 0, sizeof(StreamState));
        g_streams[i].in_use   = 1;
        g_streams[i].s        = s;
        assign_tpc_range(&g_streams[i]);
        g_streams[i].priority = priority;   /* the app's own value wins over LITHOS_PRIO */
        break;
    }
    pthread_mutex_unlock(&g_lock);
}

/* Priority of a stream's launch queue, for the dispatcher's choice of what to
 * send next (§5.2). Streams we have not seen yet are not created here — a launch
 * on an unregistered stream is about to register it anyway — so they simply
 * report the process default. */
int lithos_stream_prio(CUstream s) {
    pthread_mutex_lock(&g_lock);
    StreamState* st = find_stream(s);
    int p = st ? st->priority : g_default_prio;
    pthread_mutex_unlock(&g_lock);
    return p;
}

void lithos_stream_destroyed(CUstream s) {
    pthread_mutex_lock(&g_lock);
    StreamState* st = find_stream(s);
    if (st) st->in_use = 0;
    pthread_mutex_unlock(&g_lock);
}

/* Set one stream's compute quota, then re-pack every quota'd stream into a
 * disjoint contiguous TPC range so no two streams overlap. */
void lithos_set_quota(CUstream s, int n_tpcs) {
    pthread_mutex_lock(&g_lock);
    StreamState* st = find_stream(s);
    if (st) st->quota_tpcs = n_tpcs;

    int cursor = 0;
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (!g_streams[i].in_use || g_streams[i].quota_tpcs <= 0) continue;
        g_streams[i].tpc_lo = cursor;
        g_streams[i].tpc_hi = cursor + g_streams[i].quota_tpcs;
        cursor += g_streams[i].quota_tpcs;
    }
    pthread_mutex_unlock(&g_lock);
}

/* SMs allocated to THIS tenant (§6): quota x 2 SMs-per-TPC, or 0 when there is no
 * quota (then the caller reports the physical count). interpose.c uses this to
 * spoof CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, so cross-block-sync and
 * persistent kernels size themselves to their partition instead of the device. */
int lithos_allocated_sms(void) {
    return g_default_quota > 0 ? g_default_quota * 2 : 0;
}

void lithos_sched_stats(uint64_t* launches, uint64_t* atoms) {
    pthread_mutex_lock(&g_lock);
    if (launches) *launches = g_total_launches;
    if (atoms)    *atoms    = g_total_atoms;
    pthread_mutex_unlock(&g_lock);
}
