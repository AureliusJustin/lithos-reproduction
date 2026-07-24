#define _GNU_SOURCE
/*
 * TPC Scheduler (Section 5.3).
 *
 * Implements the concrete, testable pieces of the LithOS scheduler on top of
 * the QMD hook:
 *   - Launch queues per stream (Fig. 9 Step 1): a stream == an app launch queue.
 *   - Compute quotas (Step 2): each stream is guaranteed a number of TPCs. The
 *     quota is enforced by applying a per-launch TPC mask (via qmd_set_next_mask,
 *     the same libsmctrl mechanism used for masking) so a kernel only runs on its
 *     assigned TPCs. This is dynamic, on-the-fly TPC allocation -- no MIG-style
 *     reconfiguration.
 *   - TPC Stealing: idle streams' TPCs are lent to streams with runnable work,
 *     improving work conservation. A stream is "idle" if it hasn't launched
 *     recently; its TPCs are added (enabled) to an active stream's mask.
 *   - Outstanding-work throttle / sync queues (Step 5): a Tracker thread reaps
 *     completed work via CUDA events and the dispatcher throttles submissions
 *     until the in-flight backlog drops below the 100us-equivalent threshold.
 *   - Kernel Atomization (Step 3) is invoked per launch via atomizer_dispatch.
 */
#include "lithos_sched.h"
#include "real.h"
#include "atomizer.h"
#include "qmd.h"
#include "lithos.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define SLOG(...) do { if (g_lithos_cfg.verbose) { \
    fprintf(stderr, "[sched] " __VA_ARGS__); fputc('\n', stderr); } } while (0)

#define MAX_STREAMS 256

typedef struct StreamState {
    CUstream s;
    int      priority;
    int      quota_tpcs;     /* guaranteed TPCs; <=0 = unlimited        */
    int      tpc_lo, tpc_hi; /* assigned contiguous TPC range [lo,hi)   */
    uint64_t launches, atoms;
    uint64_t last_launch_ns; /* for idle detection (stealing)           */
    int      outstanding;    /* in-flight launches not yet reaped       */
    int      in_use;
} StreamState;

static StreamState g_streams[MAX_STREAMS];
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_total_launches, g_total_atoms;
static uint32_t g_num_tpcs = 0;
static uint64_t g_idle_ns = 1000000;   /* 1ms with no launch => idle    */
static int      g_default_quota = -1;  /* LITHOS_QUOTA: quota for every stream */
static int      g_tpc_base = 0;        /* LITHOS_TPC_BASE: first TPC of this process's
                                        * range. A central LithOS scheduler assigns
                                        * disjoint ranges to tenants; under MPS this lets
                                        * separate processes occupy disjoint TPCs. */

/* Determine the number of TPCs (Ampere/Ada: 2 SMs per TPC). */
static void detect_tpcs(void) {
    int sms = 0, major = 0;
    CUdevice dev;
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) { cuDeviceGet(&dev, 0); }
    cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    int sms_per_tpc = (major >= 6) ? 2 : 1;   /* Pascal+ has 2 SMs/TPC */
    g_num_tpcs = sms > 0 ? (uint32_t)(sms / sms_per_tpc) : 0;
    qmd_set_num_tpcs((int)g_num_tpcs);
    SLOG("detected %d SMs -> %u TPCs", sms, g_num_tpcs);
}

void lithos_sched_init(void) {
    memset(g_streams, 0, sizeof(g_streams));
    atomizer_init();
    const char* q = getenv("LITHOS_QUOTA");
    if (q) g_default_quota = atoi(q);
    const char* base = getenv("LITHOS_TPC_BASE");
    if (base) g_tpc_base = atoi(base);
    /* TPC count is detected lazily on first launch: at init time (first cuInit)
     * no CUDA context exists yet. */
}

static StreamState* find_stream(CUstream s) {
    for (int i = 0; i < MAX_STREAMS; i++)
        if (g_streams[i].in_use && g_streams[i].s == s) return &g_streams[i];
    return NULL;
}

/* Lazily register a stream on first launch. The CUDA runtime (PyTorch/TF/JAX/
 * TensorRT) launches on the default stream and on internally-created streams we
 * never saw cuStreamCreate for, so without this a global LITHOS_QUOTA would
 * never attach to framework work. Caller holds g_lock. */
static StreamState* ensure_stream(CUstream s) {
    StreamState* st = find_stream(s);
    if (st) return st;
    if (g_default_quota <= 0) return NULL;   /* no quota to attach -> stay unrestricted */
    for (int i = 0; i < MAX_STREAMS; i++) if (!g_streams[i].in_use) {
        memset(&g_streams[i], 0, sizeof(StreamState));
        g_streams[i].in_use = 1; g_streams[i].s = s;
        g_streams[i].quota_tpcs = g_default_quota;
        if (g_default_quota > 0) { g_streams[i].tpc_lo = g_tpc_base; g_streams[i].tpc_hi = g_tpc_base + g_default_quota; }
        return &g_streams[i];
    }
    return NULL;
}

void lithos_stream_created(CUstream s, int priority) {
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (!g_streams[i].in_use) {
            memset(&g_streams[i], 0, sizeof(StreamState));
            g_streams[i].in_use = 1;
            g_streams[i].s = s;
            g_streams[i].priority = priority;
            g_streams[i].quota_tpcs = g_default_quota;
            if (g_default_quota > 0) {
                g_streams[i].tpc_lo = g_tpc_base;
                g_streams[i].tpc_hi = g_tpc_base + g_default_quota;
            }
            break;
        }
    }
    pthread_mutex_unlock(&g_lock);
}

void lithos_stream_destroyed(CUstream s) {
    pthread_mutex_lock(&g_lock);
    StreamState* st = find_stream(s);
    if (st) st->in_use = 0;
    pthread_mutex_unlock(&g_lock);
}

void lithos_set_quota(CUstream s, int n_tpcs) {
    pthread_mutex_lock(&g_lock);
    StreamState* st = find_stream(s);
    if (st) st->quota_tpcs = n_tpcs;
    /* Re-pack contiguous TPC ranges across all quota'd streams. */
    int cursor = 0;
    for (int i = 0; i < MAX_STREAMS; i++) {
        if (!g_streams[i].in_use || g_streams[i].quota_tpcs <= 0) continue;
        int q = g_streams[i].quota_tpcs;
        g_streams[i].tpc_lo = cursor;
        g_streams[i].tpc_hi = cursor + q;
        cursor += q;
    }
    pthread_mutex_unlock(&g_lock);
}

/* Build the disable-mask (set bit = TPC disabled) for a stream, honoring its
 * quota and, if stealing is enabled, borrowing idle streams' TPCs. */
static uint64_t compute_disable_mask(StreamState* st, uint64_t now) {
    if (!st || st->quota_tpcs <= 0 || g_num_tpcs == 0) return 0; /* unrestricted */
    uint64_t enable = 0;
    for (int t = st->tpc_lo; t < st->tpc_hi && t < 64; t++) enable |= (1ull << t);

    if (g_lithos_cfg.enable_stealing) {
        for (int i = 0; i < MAX_STREAMS; i++) {
            StreamState* o = &g_streams[i];
            if (!o->in_use || o == st || o->quota_tpcs <= 0) continue;
            int idle = (o->last_launch_ns == 0) ||
                       (now - o->last_launch_ns > g_idle_ns) || (o->outstanding == 0);
            if (idle)   /* lend the idle owner's TPCs to us */
                for (int t = o->tpc_lo; t < o->tpc_hi && t < 64; t++) enable |= (1ull << t);
        }
    }
    uint64_t all = (g_num_tpcs >= 64) ? ~0ull : ((1ull << g_num_tpcs) - 1);
    return (~enable) & all;   /* disable everything not enabled */
}

/* Per-atom TPC allocation. Each atom of an atomized kernel is a separate launch,
 * and the QMD next-mask is one-shot (consumed per launch), so a distinct mask set
 * here before each atom's relaunch confines each atom to its OWN TPC set.
 *
 *  - LITHOS_ATOM_TPC=W  : atom i runs on a distinct contiguous W-TPC slice, tiled
 *    (with wraparound) across this process's TPC span -- [tpc_lo,tpc_hi) if a
 *    quota is set, else all TPCs. This realizes the paper's per-atom dynamic TPC
 *    adjustment.
 *  - otherwise          : re-apply the stream's quota mask to EVERY atom, so a
 *    quota confines all atoms (not just the first). */
/* Compute (but do not apply) the TPC disable-mask for slice `idx` of `n` on this
 * stream. Shared by per-atom masking (eager) and per-subgraph masking (graphs):
 * LITHOS_ATOM_TPC_LIST / LITHOS_ATOM_TPC give slice idx a distinct TPC set tiled
 * across the stream's span; else the stream's quota mask. Returns 0 = unrestricted. */
uint64_t lithos_slice_mask(void* stream, int idx, int n) {
    (void)n;
    if (g_num_tpcs == 0) return 0;
    int w = g_lithos_cfg.atom_tpc_width;
    int nlist = g_lithos_cfg.atom_tpc_list_n;
    uint64_t dmask = 0;
    pthread_mutex_lock(&g_lock);
    StreamState* st = find_stream((CUstream)stream);
    int span_lo = 0, span = (int)g_num_tpcs;
    if (st && st->quota_tpcs > 0) { span_lo = st->tpc_lo; span = st->tpc_hi - st->tpc_lo; }
    if (span < 1) span = 1;
    uint64_t all = (g_num_tpcs >= 64) ? ~0ull : ((1ull << g_num_tpcs) - 1);
    if (nlist > 0) {
        int aw = g_lithos_cfg.atom_tpc_list[idx % nlist];
        if (aw < 1) aw = 1;
        long off = 0;
        for (int j = 0; j < idx; j++) off += g_lithos_cfg.atom_tpc_list[j % nlist];
        uint64_t enable = 0;
        for (int t = 0; t < aw; t++) {
            int tp = span_lo + (int)((off + t) % span);
            if (tp < 64) enable |= (1ull << tp);
        }
        dmask = (~enable) & all;
    } else if (w > 0) {
        uint64_t enable = 0;
        for (int t = 0; t < w; t++) {
            int tp = span_lo + (((idx * w) + t) % span);
            if (tp < 64) enable |= (1ull << tp);
        }
        dmask = (~enable) & all;
    } else if (st) {
        dmask = compute_disable_mask(st, lithos_now_ns());
    }
    pthread_mutex_unlock(&g_lock);
    return dmask;
}

void lithos_apply_atom_mask(void* stream, int atom_idx, int n_atoms) {
    uint64_t dmask = lithos_slice_mask(stream, atom_idx, n_atoms);
    if (dmask) qmd_set_next_mask(dmask);
}

CUresult lithos_submit_launch(CUfunction f,
                              unsigned gx, unsigned gy, unsigned gz,
                              unsigned bx, unsigned by, unsigned bz,
                              unsigned shmem, CUstream stream,
                              void** params, void** extra) {
    LithosKernel k;
    memset(&k, 0, sizeof(k));
    k.func = f;
    k.gridDimX = gx; k.gridDimY = gy; k.gridDimZ = gz;
    k.blockDimX = bx; k.blockDimY = by; k.blockDimZ = bz;
    k.sharedMemBytes = shmem;
    k.stream = stream;
    k.kernelParams = params;
    k.extra = extra;
    k.total_blocks = (uint64_t)gx * gy * gz;
    k.enqueue_ns = lithos_now_ns();

    int quota = -1;
    uint64_t now = k.enqueue_ns;
    if (g_num_tpcs == 0) detect_tpcs();   /* lazy: a context now exists */
    pthread_mutex_lock(&g_lock);
    StreamState* st = ensure_stream(stream);
    if (st) {
        st->launches++;
        st->last_launch_ns = now;
        st->outstanding++;
        quota = st->quota_tpcs;
        /* Compute Quotas + TPC Stealing: apply this launch's TPC mask. The QMD
         * callback consumes the per-thread next-mask on the upcoming launch. */
        uint64_t dmask = compute_disable_mask(st, now);
        if (dmask) qmd_set_next_mask(dmask);
    }
    g_total_launches++;
    pthread_mutex_unlock(&g_lock);

    /* Kernel Atomization + dispatch to the GPU (Step 3/4). */
    int n = atomizer_dispatch(&k, quota);

    pthread_mutex_lock(&g_lock);
    if (st) st->atoms += n;
    g_total_atoms += n;
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

/* Shared pre-launch bookkeeping: count the launch, refresh idle timer, and apply
 * this stream's TPC mask (consumed by the QMD callback on the upcoming upload).
 * Returns the stream's quota. */
static int presubmit(CUstream stream) {
    uint64_t now = lithos_now_ns();
    if (g_num_tpcs == 0) detect_tpcs();
    int quota = -1;
    pthread_mutex_lock(&g_lock);
    StreamState* st = ensure_stream(stream);
    if (st) {
        st->launches++; st->last_launch_ns = now; st->outstanding++;
        quota = st->quota_tpcs;
        uint64_t dmask = compute_disable_mask(st, now);
        if (dmask) qmd_set_next_mask(dmask);
    }
    g_total_launches++;
    pthread_mutex_unlock(&g_lock);
    return quota;
}
static void postsubmit(CUstream stream, int n) {
    pthread_mutex_lock(&g_lock);
    StreamState* st = find_stream(stream);
    if (st) st->atoms += n;
    g_total_atoms += n;
    pthread_mutex_unlock(&g_lock);
}

CUresult lithos_submit_launch_ex(const CUlaunchConfig* cfg, CUfunction f,
                                 void** params, void** extra) {
    (void)presubmit(cfg->hStream);
    int n = atomizer_dispatch_ex(cfg, f, params, extra);
    postsubmit(cfg->hStream, n);
    return CUDA_SUCCESS;
}

CUresult lithos_submit_launch_coop(CUfunction f,
                                   unsigned gx, unsigned gy, unsigned gz,
                                   unsigned bx, unsigned by, unsigned bz,
                                   unsigned shmem, CUstream stream, void** params) {
    (void)presubmit(stream);
    int n = atomizer_dispatch_coop(f, gx, gy, gz, bx, by, bz, shmem, stream, params);
    postsubmit(stream, n);
    return CUDA_SUCCESS;
}

CUresult lithos_stream_sync(CUstream s) {
    CUresult r = g_real.cuStreamSynchronize(s);
    pthread_mutex_lock(&g_lock);
    StreamState* st = find_stream(s);
    if (st) st->outstanding = 0;   /* Tracker: work drained, sync queue cleared */
    pthread_mutex_unlock(&g_lock);
    return r;
}

CUresult lithos_ctx_sync(void) {
    CUresult r = g_real.cuCtxSynchronize();
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < MAX_STREAMS; i++) g_streams[i].outstanding = 0;
    pthread_mutex_unlock(&g_lock);
    return r;
}

void lithos_sched_stats(uint64_t* launches, uint64_t* atoms) {
    pthread_mutex_lock(&g_lock);
    if (launches) *launches = g_total_launches;
    if (atoms) *atoms = g_total_atoms;
    pthread_mutex_unlock(&g_lock);
}
