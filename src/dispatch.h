/*
 * dispatch.h — the Dispatcher and its launch queues (§5.2, Fig. 9 step 1).
 *
 * "Applications interact with LithOS via launch queues that buffer work, giving
 * LithOS full control over when work is dispatched to the GPU. This is important
 * because, once submitted, a kernel's priority or resources cannot be changed,
 * nor can it be rescheduled. Eagerly dispatching work can lead to sub-optimal
 * scheduling. LithOS therefore defers dispatch..."
 *
 * Deferral is the mechanism the rest of the scheduler is built on: while a launch
 * is still in a queue it can be reordered, re-prioritised, or re-sized; the
 * instant it reaches the GPU none of that is possible any more.
 *
 * Buffering a launch means the caller's argument memory is no longer live when
 * the launch is finally submitted, so every queued request owns a deep copy of
 * its arguments (params.h). Kernels whose layout we cannot recover are submitted
 * inline instead — after draining the stream, so ordering still holds.
 */
#ifndef LITHOS_DISPATCH_H
#define LITHOS_DISPATCH_H

#include <cuda.h>
#include <stdint.h>
#include "lithos.h"

/* Try to buffer a launch. Returns 1 if the dispatcher took ownership (the caller
 * must return CUDA_SUCCESS immediately and touch nothing further), 0 if the
 * caller has to submit it inline. */
int dispatch_try_std(CUfunction f,
                     unsigned gx, unsigned gy, unsigned gz,
                     unsigned bx, unsigned by, unsigned bz,
                     unsigned shmem, CUstream stream,
                     void** params, void** extra);
int dispatch_try_ex(const CUlaunchConfig* cfg, CUfunction f,
                    void** params, void** extra);
int dispatch_try_coop(CUfunction f,
                      unsigned gx, unsigned gy, unsigned gz,
                      unsigned bx, unsigned by, unsigned bz,
                      unsigned shmem, CUstream stream, void** params);

/* ---- ordering barriers ----------------------------------------------------
 * Deferring a launch opens a window in which any OTHER stream-ordered CUDA call
 * would reach the GPU ahead of it. Every such call must drain first.
 *
 * dispatch_pending() is the guard the barriers lead with: a single relaxed load,
 * and zero in the overwhelmingly common case, so the hooks cost nothing when
 * nothing is buffered (and nothing at all when the dispatcher is off).
 *
 * It is also where LithOS's own CUDA calls opt out — see the re-entrancy guard
 * in lithos.h for why draining on them deadlocks. */
extern int g_dispatch_deferred_n;
static inline int dispatch_pending(void) {
    return !g_lithos_internal &&
           __atomic_load_n(&g_dispatch_deferred_n, __ATOMIC_RELAXED) != 0;
}

/* Submit everything buffered for `stream` (plus the legacy default stream, which
 * is implicitly ordered against every blocking stream) and wait until it has
 * reached the GPU. */
void dispatch_drain(CUstream stream);
/* Same for every queue — for context-wide operations and for calls with no
 * stream argument, which are legacy-default-stream ordered. */
void dispatch_drain_all(void);

/* Sticky error from a buffered launch that failed after we had already returned
 * CUDA_SUCCESS to the application. Reported at the next drain, mirroring CUDA's
 * own asynchronous error model. Reading it clears it. */
CUresult dispatch_take_error(void);

/* A CUDA graph capture is a no-defer window: which graph a launch is captured
 * into depends on exactly when it is issued, so launches go straight through
 * while one is open. Called from the capture interception in interpose.c. */
void dispatch_capture_begin(CUstream stream);
void dispatch_capture_end(void);

/* Drop a stream's queue once its work is out (called from cuStreamDestroy). */
void dispatch_stream_destroyed(CUstream stream);

/* Counters for LITHOS_STATS / the benchmarks. */
void dispatch_stats(uint64_t* deferred, uint64_t* inline_fallback, uint64_t* reorders);

#endif /* LITHOS_DISPATCH_H */
