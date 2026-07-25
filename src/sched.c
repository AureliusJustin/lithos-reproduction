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
#include "predict.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>

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
static int      g_phys_sms = 0;        /* true physical SM count (for spoof math) */
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
    /* Use the REAL attribute query (not our spoofing wrapper) so we see the true
     * physical SM count regardless of any per-tenant MULTIPROCESSOR_COUNT spoof. */
    CUresult (*ga)(int*, CUdevice_attribute, CUdevice) =
        g_real.cuDeviceGetAttribute ? g_real.cuDeviceGetAttribute : cuDeviceGetAttribute;
    ga(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    ga(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    int sms_per_tpc = (major >= 6) ? 2 : 1;   /* Pascal+ has 2 SMs/TPC */
    g_num_tpcs = sms > 0 ? (uint32_t)(sms / sms_per_tpc) : 0;
    qmd_set_num_tpcs((int)g_num_tpcs);
    g_phys_sms = sms;
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
    /* Always allocate a slot (this is the stream's launch queue, Fig.9 ①): the
     * predictor and bookkeeping key off it even with no quota. Masking still
     * no-ops when quota_tpcs<=0, so unquota'd streams stay unrestricted. */
    for (int i = 0; i < MAX_STREAMS; i++) if (!g_streams[i].in_use) {
        memset(&g_streams[i], 0, sizeof(StreamState));
        g_streams[i].in_use = 1; g_streams[i].s = s;
        g_streams[i].quota_tpcs = g_default_quota;
        if (g_default_quota > 0) { g_streams[i].tpc_lo = g_tpc_base; g_streams[i].tpc_hi = g_tpc_base + g_default_quota; }
        return &g_streams[i];
    }
    return NULL;
}
static int stream_slot(StreamState* st) { return st ? (int)(st - g_streams) : -1; }

/* SMs allocated to THIS tenant (§6): its TPC quota x 2, or 0 = no quota (report
 * the physical count). Used to spoof CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT so
 * cross-block-sync / persistent kernels size themselves to their partition. */
int lithos_allocated_sms(void) { return g_default_quota > 0 ? g_default_quota * 2 : 0; }

static int stream_capturing(CUstream s) {
    if (!g_real.cuStreamIsCapturing) return 0;
    CUstreamCaptureStatus st = CU_STREAM_CAPTURE_STATUS_NONE;
    if (g_real.cuStreamIsCapturing(s, &st) != CUDA_SUCCESS) return 0;
    return st == CU_STREAM_CAPTURE_STATUS_ACTIVE;
}

/* Current TPC count a stream's kernels run on (its quota, or all TPCs). */
static int stream_tpcs(StreamState* st) {
    if (st && st->quota_tpcs > 0) return st->tpc_hi - st->tpc_lo;
    return g_num_tpcs > 0 ? (int)g_num_tpcs : 1;
}

static int g_predict_ready = 0;
static void predict_ensure(void) {
    if (g_predict_ready) return;
    predict_init(); predict_start_tracker(); g_predict_ready = 1;
}

/* Disable-mask that enables exactly the first `w` TPCs starting at `base`. */
static uint64_t mask_first_tpcs(int base, int w) {
    uint64_t enable = 0;
    for (int t = base; t < base + w && t < 64; t++) enable |= (1ull << t);
    uint64_t all = (g_num_tpcs >= 64) ? ~0ull : ((1ull << g_num_tpcs) - 1);
    return (~enable) & all;
}

/* Right-sizing (§5.5): the minimum TPCs this kernel should run on. First the
 * filtering heuristic (blocks / occupancy-per-TPC — an upper bound on useful
 * TPCs); if that's below the allocation, use it. Otherwise consult the learned
 * scaling model, which may ask us to PROBE this launch at a specific TPC count
 * (all-TPC or 1-TPC) to collect a model point. Returns a count <= cur_tpc, and
 * sets *probe to a forced count (or 0). */
static int rightsize_tpcs(CUfunction f, int block_threads, unsigned shmem, uint64_t blocks,
                          int slot, int op, int cur_tpc, int* probe) {
    *probe = 0;
    if (cur_tpc < 2 || block_threads < 1) return cur_tpc;
    int maxblk = 0;
    if (cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, f, block_threads, shmem) == CUDA_SUCCESS && maxblk > 0) {
        int occ_per_tpc = maxblk * 2;   /* 2 SMs per TPC (Ampere/Ada) */
        int filt = (int)((blocks + occ_per_tpc - 1) / occ_per_tpc);
        if (filt >= 1 && filt < cur_tpc) return filt;   /* useful < allocated */
    }
    int wp = 0;
    int tmin = predict_rightsize(slot, op, cur_tpc, g_lithos_cfg.latency_slip, &wp);
    if (wp > 0) { *probe = wp; return wp < cur_tpc ? wp : cur_tpc; }
    if (tmin > 0 && tmin < cur_tpc) return tmin;
    return cur_tpc;
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
static uint64_t compute_disable_mask_ex(StreamState* st, uint64_t now, int allow_steal) {
    if (!st || st->quota_tpcs <= 0 || g_num_tpcs == 0) return 0; /* unrestricted */
    uint64_t enable = 0;
    for (int t = st->tpc_lo; t < st->tpc_hi && t < 64; t++) enable |= (1ull << t);

    if (allow_steal && g_lithos_cfg.enable_stealing) {
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
static uint64_t compute_disable_mask(StreamState* st, uint64_t now) {
    return compute_disable_mask_ex(st, now, 1);
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

static CUresult submit_launch_now(CUfunction f,
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

    int quota = -1, slot = -1, cur_tpc = 1, base = 0;
    uint64_t now = k.enqueue_ns;
    if (g_num_tpcs == 0) detect_tpcs();   /* lazy: a context now exists */
    pthread_mutex_lock(&g_lock);
    StreamState* st = ensure_stream(stream);
    if (st) {
        st->launches++;
        st->last_launch_ns = now;
        st->outstanding++;
        quota = st->quota_tpcs;
        slot = stream_slot(st);
        cur_tpc = stream_tpcs(st);
        base = (st->quota_tpcs > 0) ? st->tpc_lo : g_tpc_base;
        /* Compute Quotas + TPC Stealing: apply this launch's TPC mask. The QMD
         * callback consumes the per-thread next-mask on the upcoming launch. */
        uint64_t dmask = compute_disable_mask(st, now);
        if (dmask) qmd_set_next_mask(dmask);
    }
    g_total_launches++;
    pthread_mutex_unlock(&g_lock);

    /* Online latency prediction (§5.7): identify the operator by ordinal k on this
     * launch queue. Then (§5.5) right-size the TPC allocation, then predict the
     * duration at the *effective* allocation to drive atom sizing. */
    int op = 0, measure = 0, eff_tpc = cur_tpc;
    CUevent eva = NULL, evb = NULL;
    if (g_lithos_cfg.predict && slot >= 0) {
        predict_ensure();
        op = predict_next_op(slot);
        if (g_lithos_cfg.rightsize && cur_tpc >= 2) {
            int probe = 0;
            int rs = rightsize_tpcs((CUfunction)f, (int)(bx*by*bz), shmem, k.total_blocks,
                                    slot, op, cur_tpc, &probe);
            if (rs >= 1 && rs < cur_tpc) { qmd_set_next_mask(mask_first_tpcs(base, rs)); eff_tpc = rs; }
        }
        k.pred_us = predict_lookup(slot, op, eff_tpc, k.total_blocks);
        /* Outstanding-work throttle (§5.3): defer dispatch until the in-flight
         * backlog drops below the tunable limit (paper: 100us). */
        if (g_lithos_cfg.throttle) {
            int spins = 0;
            while (predict_outstanding_us() > g_lithos_cfg.outstanding_limit_us && spins++ < 100000)
                sched_yield();
        }
        measure = !stream_capturing(stream);
        if (measure && (eva = predict_evt_get())) cuEventRecord(eva, stream);
    }

    /* Kernel Atomization + dispatch to the GPU (Step 3/4). */
    int n = atomizer_dispatch(&k, quota);

    if (measure && eva) {
        if ((evb = predict_evt_get())) { cuEventRecord(evb, stream); predict_submit(slot, op, eff_tpc, eva, evb); }
        else predict_submit(slot, op, eff_tpc, eva, NULL);   /* returns eva to pool */
    }

    pthread_mutex_lock(&g_lock);
    if (st) st->atoms += n;
    g_total_atoms += n;
    pthread_mutex_unlock(&g_lock);
    return CUDA_SUCCESS;
}

/* ---- Dispatcher thread (§5.2, Fig.9 ①) ------------------------------------
 * With LITHOS_DISPATCH, cuLaunchKernel doesn't submit inline; it appends the
 * request to a launch queue and a single dispatcher thread pulls requests and
 * submits them to the GPU. This decouples the *scheduling decision* from the app
 * thread and gives one coherent point to apply global policy (ordering, the
 * outstanding-work throttle, stealing). The submitting thread waits for the
 * dispatcher to consume its request, so the app's kernel-arg memory stays valid
 * (a transparent interposer can't safely snapshot cuLaunchKernel's void** args --
 * their sizes are implicit -- so we keep them alive via this hand-off rather than
 * copy them). */
typedef struct DReq {
    CUfunction f; unsigned gx, gy, gz, bx, by, bz, shmem;
    CUstream s; void** params; void** extra;
    volatile int done; struct DReq* next;
} DReq;
static DReq* g_dq_head = NULL, *g_dq_tail = NULL;
static pthread_mutex_t g_dq_m  = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_dq_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_dq_dn = PTHREAD_COND_INITIALIZER;
static pthread_t g_disp; static int g_disp_started = 0; static CUcontext g_disp_ctx = NULL;

static void* disp_main(void* a) {
    (void)a;
    if (g_disp_ctx) cuCtxSetCurrent(g_disp_ctx);   /* submit in the app's context */
    pthread_mutex_lock(&g_dq_m);
    for (;;) {
        while (!g_dq_head) pthread_cond_wait(&g_dq_cv, &g_dq_m);
        DReq* r = g_dq_head; g_dq_head = r->next; if (!g_dq_head) g_dq_tail = NULL;
        pthread_mutex_unlock(&g_dq_m);
        submit_launch_now(r->f, r->gx, r->gy, r->gz, r->bx, r->by, r->bz, r->shmem,
                          r->s, r->params, r->extra);
        pthread_mutex_lock(&g_dq_m);
        r->done = 1; pthread_cond_broadcast(&g_dq_dn);
    }
    return NULL;
}
static void disp_ensure(void) {
    if (g_disp_started) return;
    g_disp_started = 1;
    cuCtxGetCurrent(&g_disp_ctx);
    pthread_create(&g_disp, NULL, disp_main, NULL);
}

CUresult lithos_submit_launch(CUfunction f,
                              unsigned gx, unsigned gy, unsigned gz,
                              unsigned bx, unsigned by, unsigned bz,
                              unsigned shmem, CUstream stream,
                              void** params, void** extra) {
    if (!g_lithos_cfg.dispatch)
        return submit_launch_now(f, gx, gy, gz, bx, by, bz, shmem, stream, params, extra);
    disp_ensure();
    DReq r = { f, gx, gy, gz, bx, by, bz, shmem, stream, params, extra, 0, NULL };
    pthread_mutex_lock(&g_dq_m);
    if (g_dq_tail) g_dq_tail->next = &r; else g_dq_head = &r;
    g_dq_tail = &r;
    pthread_cond_signal(&g_dq_cv);
    while (!r.done) pthread_cond_wait(&g_dq_dn, &g_dq_m);   /* args valid until consumed */
    pthread_mutex_unlock(&g_dq_m);
    return CUDA_SUCCESS;
}

/* Shared pre-launch bookkeeping: count the launch, refresh idle timer, and apply
 * this stream's TPC mask (consumed by the QMD callback on the upcoming upload).
 * Returns the stream's quota. */
static int presubmit(CUstream stream, int* slot_out, int* tpc_out, int special) {
    uint64_t now = lithos_now_ns();
    if (g_num_tpcs == 0) detect_tpcs();
    int quota = -1, slot = -1, tpc = g_num_tpcs > 0 ? (int)g_num_tpcs : 1;
    pthread_mutex_lock(&g_lock);
    StreamState* st = ensure_stream(stream);
    if (st) {
        st->launches++; st->last_launch_ns = now; st->outstanding++;
        quota = st->quota_tpcs; slot = stream_slot(st); tpc = stream_tpcs(st);
        /* Special kernels (cross-block-sync / persistent): give the EXACT quota
         * (no stealing) so the allocation matches the SM count they queried. */
        uint64_t dmask = compute_disable_mask_ex(st, now, !special);
        if (dmask) qmd_set_next_mask(dmask);
    }
    g_total_launches++;
    pthread_mutex_unlock(&g_lock);
    if (slot_out) *slot_out = slot;
    if (tpc_out) *tpc_out = tpc;
    return quota;
}

/* Does this Ex launch config carry a cooperative / thread-block-cluster attribute
 * (a cross-block-sync special kernel)? */
static int cfg_is_special(const CUlaunchConfig* cfg) {
    for (unsigned i = 0; i < cfg->numAttrs; i++)
        if (cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_COOPERATIVE ||
            cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) return 1;
    return 0;
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
    int slot = -1, tpc = 1;
    (void)presubmit(cfg->hStream, &slot, &tpc, cfg_is_special(cfg));
    int op = 0, measure = 0; CUevent eva = NULL, evb = NULL;
    if (g_lithos_cfg.predict && slot >= 0) {
        predict_ensure();
        op = predict_next_op(slot);
        uint64_t blocks = (uint64_t)cfg->gridDimX * cfg->gridDimY * cfg->gridDimZ;
        atomizer_set_ex_pred(predict_lookup(slot, op, tpc, blocks));
        if (g_lithos_cfg.throttle) { int sp = 0;
            while (predict_outstanding_us() > g_lithos_cfg.outstanding_limit_us && sp++ < 100000) sched_yield(); }
        measure = !stream_capturing(cfg->hStream);
        if (measure && (eva = predict_evt_get())) cuEventRecord(eva, cfg->hStream);
    } else atomizer_set_ex_pred(0);
    int n = atomizer_dispatch_ex(cfg, f, params, extra);
    if (measure && eva) {
        if ((evb = predict_evt_get())) { cuEventRecord(evb, cfg->hStream); predict_submit(slot, op, tpc, eva, evb); }
        else predict_submit(slot, op, tpc, eva, NULL);
    }
    postsubmit(cfg->hStream, n);
    return CUDA_SUCCESS;
}

CUresult lithos_submit_launch_coop(CUfunction f,
                                   unsigned gx, unsigned gy, unsigned gz,
                                   unsigned bx, unsigned by, unsigned bz,
                                   unsigned shmem, CUstream stream, void** params) {
    (void)presubmit(stream, NULL, NULL, 1);   /* cooperative kernel: special */
    int n = atomizer_dispatch_coop(f, gx, gy, gz, bx, by, bz, shmem, stream, params);
    postsubmit(stream, n);
    return CUDA_SUCCESS;
}

CUresult lithos_stream_sync(CUstream s) {
    CUresult r = g_real.cuStreamSynchronize(s);
    pthread_mutex_lock(&g_lock);
    StreamState* st = find_stream(s);
    int slot = stream_slot(st);
    if (st) st->outstanding = 0;   /* Tracker: work drained, sync queue cleared */
    pthread_mutex_unlock(&g_lock);
    /* A sync delimits a batch (§5.7): reset this queue's operator ordinal so the
     * next kernel is operator k=0 of the next batch. */
    if (g_lithos_cfg.predict && slot >= 0) predict_reset_op(slot);
    return r;
}

CUresult lithos_ctx_sync(void) {
    CUresult r = g_real.cuCtxSynchronize();
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < MAX_STREAMS; i++) g_streams[i].outstanding = 0;
    pthread_mutex_unlock(&g_lock);
    if (g_lithos_cfg.predict)
        for (int i = 0; i < MAX_STREAMS; i++) predict_reset_op(i);
    return r;
}

void lithos_sched_stats(uint64_t* launches, uint64_t* atoms) {
    pthread_mutex_lock(&g_lock);
    if (launches) *launches = g_total_launches;
    if (atoms) *atoms = g_total_atoms;
    pthread_mutex_unlock(&g_lock);
}
