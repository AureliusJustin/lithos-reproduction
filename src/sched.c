#define _GNU_SOURCE
/*
 * sched.c — the submit path: what happens between the app's launch call and the
 * kernel reaching the GPU.
 *
 * This is the spine that ties the scheduler's pieces together. For each launch:
 *
 *   1. find/create the stream's launch queue          (sched_stream.c, Fig.9 ①)
 *   2. apply the TPC mask: quota + stealing           (tpc_alloc.c,   Fig.9 ②)
 *   3. identify the operator and predict its duration (predict.c,     §5.7)
 *   4. optionally right-size the TPC allocation       (tpc_alloc.c,   Fig.9 ⑥)
 *   5. optionally wait on the outstanding-work throttle              (§5.3, ⑤)
 *   6. bracket the launch with events so the Tracker can measure it  (§5.7)
 *   7. hand off to the Kernel Atomizer, which splits and launches    (Fig.9 ③④)
 *
 * Three entry points mirror the three driver launch APIs: cuLaunchKernel (the
 * main path), cuLaunchKernelEx (used by the CUDA runtime, i.e. frameworks), and
 * cuLaunchCooperativeKernel. Stream/context synchronisation also lands here,
 * because a sync marks a batch boundary for the predictor and drains the
 * outstanding-work counters.
 *
 * See sched_internal.h for how the scheduler is split across files.
 */
#include "sched_internal.h"
#include "lithos_sched.h"
#include "real.h"
#include "atomizer.h"
#include "qmd.h"
#include "predict.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>

/* ---- helpers -------------------------------------------------------------- */

/* Is this stream currently capturing into a CUDA graph?
 *
 * Also latches a global capture guard. While ANY stream is capturing, the
 * predictor's Tracker thread must not issue CUDA calls: a call from another
 * thread on the capturing context can invalidate the in-progress capture (this
 * caused intermittent capture failures until it was fixed). We pause the Tracker
 * the first time we see a capturing launch and resume at the next sync. */
static int g_capture_latched = 0;
static int stream_capturing(CUstream s) {
    if (!g_real.cuStreamIsCapturing) return 0;
    CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
    if (g_real.cuStreamIsCapturing(s, &status) != CUDA_SUCCESS) return 0;

    int capturing = (status == CU_STREAM_CAPTURE_STATUS_ACTIVE);
    if (capturing && !g_capture_latched)       { g_capture_latched = 1; predict_capture_begin(); }
    else if (!capturing && g_capture_latched)  { g_capture_latched = 0; predict_capture_end(); }
    return capturing;
}

/* A sync cannot occur inside a capture, so any latched guard is stale by then. */
static void capture_guard_release(void) {
    if (g_capture_latched) { g_capture_latched = 0; predict_capture_end(); }
}

/* Bring up the predictor + Tracker thread on first use. */
static int g_predict_ready = 0;
static void predict_ensure(void) {
    if (g_predict_ready) return;
    predict_init();
    predict_start_tracker();
    g_predict_ready = 1;
}

/* Outstanding-work throttle (§5.3, Fig. 9 ⑤): "tracks outstanding work via sync
 * queues, throttling submissions until the backlog drops below a tunable
 * threshold" (the paper uses 100 us). The Tracker drains the counter as
 * completion events retire.
 *
 * WHERE THIS RUNS. In the paper the throttle sits in the DISPATCHER: the app
 * enqueues and returns, and the dispatcher simply defers submission while the
 * backlog is high. With LITHOS_DISPATCH=1 we match that — dispatch.c calls this
 * on the dispatcher thread, and submit_launch_now skips it (throttle_is_deferred
 * below). With the dispatcher off there is no separate thread to defer on, so the
 * calling thread waits here instead; that placement is the divergence, not the
 * policy.
 *
 * We sleep rather than spin: a throttled workload is waiting on the GPU, so
 * burning a core adds nothing. 20 us keeps the wait fine-grained against a
 * 100 us budget. The iteration bound stops a pathological case (host far
 * outrunning the GPU, or a lost completion) from blocking forever. */
void throttle_wait(void) {
    if (!g_lithos_cfg.throttle) return;
    struct timespec nap = { 0, 20000 };            /* 20 us */
    for (int i = 0; i < 5000; i++) {               /* <= ~100 ms hard cap */
        if (predict_outstanding_us() <= g_lithos_cfg.outstanding_limit_us) return;
        nanosleep(&nap, NULL);
    }
}

/* True when the dispatcher thread owns throttling, so the submit path must not
 * throttle again on the calling thread. */
static int throttle_is_deferred(void) { return g_lithos_cfg.dispatch != 0; }

/* Shared pre-launch bookkeeping for the Ex/cooperative paths: register the
 * stream, count the launch, refresh its idle timer, and arm its TPC mask.
 * `special` marks a cross-block-sync kernel, which gets its exact quota with no
 * stealing. Returns the stream's quota; reports its slot and TPC count. */
static int presubmit(CUstream stream, int* slot_out, int* tpc_out, int special) {
    uint64_t now = lithos_now_ns();
    if (g_num_tpcs == 0) detect_tpcs();       /* lazy: a context exists by now */

    int quota = -1, slot = -1, tpcs = g_num_tpcs > 0 ? (int)g_num_tpcs : 1;
    pthread_mutex_lock(&g_lock);
    StreamState* st = ensure_stream(stream);
    if (st) {
        st->launches++;
        st->last_launch_ns = now;
        st->outstanding++;
        quota = st->quota_tpcs;
        slot  = stream_slot(st);
        tpcs  = stream_tpcs(st);
        uint64_t dmask = compute_disable_mask_ex(st, now, !special);
        if (dmask) qmd_set_next_mask(dmask);
    }
    g_total_launches++;
    pthread_mutex_unlock(&g_lock);

    if (slot_out) *slot_out = slot;
    if (tpc_out)  *tpc_out  = tpcs;
    return quota;
}

/* Credit the atoms this launch expanded into. */
static void postsubmit(CUstream stream, int n_atoms) {
    pthread_mutex_lock(&g_lock);
    StreamState* st = find_stream(stream);
    if (st) st->atoms += n_atoms;
    g_total_atoms += n_atoms;
    pthread_mutex_unlock(&g_lock);
}

/* Does this Ex launch config carry a cooperative / thread-block-cluster
 * attribute? Those kernels synchronise across blocks, so they must not be split
 * and must not have TPCs added or removed under them (§6, "special kernels"). */
static int cfg_is_special(const CUlaunchConfig* cfg) {
    for (unsigned i = 0; i < cfg->numAttrs; i++)
        if (cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_COOPERATIVE ||
            cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) return 1;
    return 0;
}

/* ---- main submit path (cuLaunchKernel) ------------------------------------ */

/* Steps 1-7 from the file header, in order. Called either inline (default) or
 * from the dispatcher thread when LITHOS_DISPATCH is on. */
CUresult submit_launch_now(CUfunction f,
                           unsigned gx, unsigned gy, unsigned gz,
                           unsigned bx, unsigned by, unsigned bz,
                           unsigned shmem, CUstream stream,
                           void** params, void** extra) {
    /* Describe the launch for the atomizer. */
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

    /* (1)(2) Register the launch queue and arm this launch's quota/stealing mask.
     * `base` is the first TPC of this stream's own range — right-sizing lays its
     * smaller allocation out from there. */
    int quota = -1, slot = -1, cur_tpc = 1, base = 0;
    uint64_t now = k.enqueue_ns;
    if (g_num_tpcs == 0) detect_tpcs();
    pthread_mutex_lock(&g_lock);
    StreamState* st = ensure_stream(stream);
    if (st) {
        st->launches++;
        st->last_launch_ns = now;
        st->outstanding++;
        quota   = st->quota_tpcs;
        slot    = stream_slot(st);
        cur_tpc = stream_tpcs(st);
        base    = (st->quota_tpcs > 0) ? st->tpc_lo : g_tpc_base;
        uint64_t dmask = compute_disable_mask(st, now);
        if (dmask) qmd_set_next_mask(dmask);
    }
    g_total_launches++;
    pthread_mutex_unlock(&g_lock);

    /* Determine capture status ONCE here and pass it to the atomizer, which would
     * otherwise repeat the cuStreamIsCapturing driver call. */
    int capturing = stream_capturing(stream);
    atomizer_set_capture_hint(capturing);

    int op = 0, measure = 0, eff_tpc = cur_tpc;
    CUevent ev_done = NULL;

    if (g_lithos_cfg.predict && slot >= 0) {
        predict_ensure();

        /* (3) Identify the operator: the k-th kernel since this queue's last sync
         * (§5.7 — the same kernel function recurs with different shapes, so the
         * ordinal is what makes predictions meaningful). */
        op = predict_next_op(slot);

        /* (4) Right-size: possibly replace the quota mask with a smaller one. */
        if (g_lithos_cfg.rightsize && cur_tpc >= 2) {
            int probe = 0;
            int rs = rightsize_tpcs(f, (int)(bx * by * bz), shmem, k.total_blocks,
                                    slot, op, cur_tpc, &probe);
            if (rs >= 1 && rs < cur_tpc) {
                qmd_set_next_mask(mask_first_tpcs(base, rs));
                eff_tpc = rs;    /* predict + measure at the allocation we chose */
            }
        }

        /* Predicted duration at the effective allocation drives the atom count. */
        k.pred_us = predict_lookup(slot, op, eff_tpc, k.total_blocks);

        /* (5) Throttle before submitting — unless the dispatcher already did it. */
        if (!throttle_is_deferred()) throttle_wait();
        measure = !capturing;
    }

    /* (7) Kernel Atomization + dispatch to the GPU (Fig. 9 ③④). */
    int n_atoms = atomizer_dispatch(&k, quota);

    /* (6) Mark this launch's COMPLETION with a single event; the Tracker derives
     * the duration from the gap to the previous completion on this queue. */
    if (measure && (ev_done = predict_evt_get())) {
        cuEventRecord(ev_done, stream);
        predict_submit(slot, op, eff_tpc, ev_done);
    }

    postsubmit(stream, n_atoms);
    return CUDA_SUCCESS;
}

/* Public entry: either hand the launch to the dispatcher thread (§5.2) or run the
 * submit path inline on the calling thread. */
CUresult lithos_submit_launch(CUfunction f,
                              unsigned gx, unsigned gy, unsigned gz,
                              unsigned bx, unsigned by, unsigned bz,
                              unsigned shmem, CUstream stream,
                              void** params, void** extra) {
    if (g_lithos_cfg.dispatch)
        return dispatch_submit(f, gx, gy, gz, bx, by, bz, shmem, stream, params, extra);
    return submit_launch_now(f, gx, gy, gz, bx, by, bz, shmem, stream, params, extra);
}

/* ---- cuLaunchKernelEx (the CUDA runtime's path: PyTorch/TF/JAX/TensorRT) --- */

CUresult lithos_submit_launch_ex(const CUlaunchConfig* cfg, CUfunction f,
                                 void** params, void** extra) {
    int slot = -1, tpcs = 1;
    (void)presubmit(cfg->hStream, &slot, &tpcs, cfg_is_special(cfg));

    int capturing = stream_capturing(cfg->hStream);
    atomizer_set_capture_hint(capturing);

    int op = 0, measure = 0;
    CUevent ev_done = NULL;

    if (g_lithos_cfg.predict && slot >= 0) {
        predict_ensure();
        op = predict_next_op(slot);
        uint64_t blocks = (uint64_t)cfg->gridDimX * cfg->gridDimY * cfg->gridDimZ;
        /* The Ex path can't carry pred_us through a LithosKernel, so stash it for
         * the atomizer on this thread. */
        atomizer_set_ex_pred(predict_lookup(slot, op, tpcs, blocks));
        if (!throttle_is_deferred()) throttle_wait();
        measure = !capturing;
    } else {
        atomizer_set_ex_pred(0);
    }

    int n_atoms = atomizer_dispatch_ex(cfg, f, params, extra);

    /* One completion event; the Tracker derives the duration from the gap to the
     * previous completion on this queue (see predict.c). */
    if (measure && (ev_done = predict_evt_get())) {
        cuEventRecord(ev_done, cfg->hStream);
        predict_submit(slot, op, tpcs, ev_done);
    }

    postsubmit(cfg->hStream, n_atoms);
    return CUDA_SUCCESS;
}

/* ---- cuLaunchCooperativeKernel -------------------------------------------- */

/* Cooperative kernels synchronise across the whole grid, so they are never split
 * and always get their exact quota (special = 1). */
CUresult lithos_submit_launch_coop(CUfunction f,
                                   unsigned gx, unsigned gy, unsigned gz,
                                   unsigned bx, unsigned by, unsigned bz,
                                   unsigned shmem, CUstream stream, void** params) {
    (void)presubmit(stream, NULL, NULL, 1);
    atomizer_set_capture_hint(stream_capturing(stream));
    int n_atoms = atomizer_dispatch_coop(f, gx, gy, gz, bx, by, bz, shmem, stream, params);
    postsubmit(stream, n_atoms);
    return CUDA_SUCCESS;
}

/* ---- synchronisation ------------------------------------------------------ *
 * A sync drains outstanding work (clearing the throttle's view of it) and marks a
 * BATCH BOUNDARY: the predictor restarts its operator ordinal so the next kernel
 * is operator 0 of the next batch (§5.7). */

CUresult lithos_stream_sync(CUstream s) {
    capture_guard_release();
    CUresult r = g_real.cuStreamSynchronize(s);

    pthread_mutex_lock(&g_lock);
    StreamState* st = find_stream(s);
    int slot = stream_slot(st);
    if (st) st->outstanding = 0;
    pthread_mutex_unlock(&g_lock);

    if (g_lithos_cfg.predict && slot >= 0) predict_reset_op(slot);
    return r;
}

CUresult lithos_ctx_sync(void) {
    capture_guard_release();
    CUresult r = g_real.cuCtxSynchronize();

    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < MAX_STREAMS; i++) g_streams[i].outstanding = 0;
    pthread_mutex_unlock(&g_lock);

    if (g_lithos_cfg.predict)
        for (int i = 0; i < MAX_STREAMS; i++) predict_reset_op(i);
    return r;
}
