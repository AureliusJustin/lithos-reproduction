/* sched_internal.h — state shared between the TPC Scheduler's source files.
 *
 * The scheduler is split by concern (Fig. 9 step numbers in parentheses):
 *
 *   sched_stream.c  the stream registry = per-stream LAUNCH QUEUES (1), plus
 *                   TPC detection, quota assignment, and global counters
 *   tpc_alloc.c     turning a stream's quota into an actual SM-disable mask:
 *                   COMPUTE QUOTAS (2), TPC STEALING, per-atom/per-subgraph
 *                   slice masks, and hardware RIGHT-SIZING (6)
 *   dispatch.c      the optional DISPATCHER THREAD that owns submission (1)
 *   sched.c         the submit paths that tie it together: prediction hooks,
 *                   throttle, atomizer dispatch (3/4), and sync handling (5)
 *
 * Nothing here is part of the public interface — see lithos_sched.h for that.
 */
#ifndef LITHOS_SCHED_INTERNAL_H
#define LITHOS_SCHED_INTERNAL_H

#include <cuda.h>
#include <stdint.h>
#include <pthread.h>
#include "lithos.h"

#define MAX_STREAMS 256

#define SLOG(...) do { if (g_lithos_cfg.verbose) { \
    fprintf(stderr, "[sched] " __VA_ARGS__); fputc('\n', stderr); } } while (0)

/* One app stream == one LithOS launch queue (Fig. 9 step 1). */
typedef struct StreamState {
    CUstream s;                /* the app's stream handle (key)                 */
    int      priority;         /* stream priority as created                    */
    int      quota_tpcs;       /* guaranteed TPCs; <= 0 means unlimited         */
    int      tpc_lo, tpc_hi;   /* assigned contiguous TPC range [lo, hi)        */
    uint64_t launches, atoms;  /* counters for stats                            */
    uint64_t last_launch_ns;   /* last submit time — drives idle detection      */
    int      outstanding;      /* launches submitted but not yet drained        */
    int      in_use;           /* slot occupied                                 */
} StreamState;

/* ---- shared scheduler state (defined in sched_stream.c) ------------------- */
extern StreamState     g_streams[MAX_STREAMS];
extern pthread_mutex_t g_lock;        /* guards g_streams and the counters      */
extern uint64_t        g_total_launches, g_total_atoms;
extern uint32_t        g_num_tpcs;    /* TPCs on this device (0 = not yet known)*/
extern uint64_t        g_idle_ns;     /* no launch for this long => stream idle */
extern int             g_default_quota; /* LITHOS_QUOTA (-1 = none)             */
extern int             g_tpc_base;      /* LITHOS_TPC_BASE, this tenant's first TPC */
extern int             g_ranges_disjoint; /* any two streams own different TPCs?   */

/* ---- sched_stream.c ------------------------------------------------------- */
void         detect_tpcs(void);                  /* lazy; needs a live context  */
StreamState* find_stream(CUstream s);            /* caller holds g_lock         */
StreamState* ensure_stream(CUstream s);          /* caller holds g_lock         */
int          stream_slot(StreamState* st);       /* index, or -1                */
int          stream_tpcs(StreamState* st);       /* TPCs this stream runs on    */

/* ---- tpc_alloc.c ---------------------------------------------------------- */
/* Disable-mask for a stream (set bit = TPC disabled). `allow_steal` lets idle
 * streams' TPCs be borrowed; special (cross-block-sync) kernels pass 0 so they
 * get exactly their quota. Caller holds g_lock. */
uint64_t compute_disable_mask_ex(StreamState* st, uint64_t now, int allow_steal);
uint64_t compute_disable_mask(StreamState* st, uint64_t now);
/* Disable-mask enabling exactly `w` TPCs starting at `base`. */
uint64_t mask_first_tpcs(int base, int w);
/* Right-sizing (§5.5): fewest TPCs this kernel should get; *probe is set when the
 * scaling model wants this launch forced to a particular TPC count to sample it. */
int      rightsize_tpcs(CUfunction f, int block_threads, unsigned shmem, uint64_t blocks,
                        int slot, int op, int cur_tpc, int* probe);

/* ---- sched.c -------------------------------------------------------------- */
/* The real submit path. dispatch.c calls this from the dispatcher thread; with
 * the dispatcher disabled, lithos_submit_launch calls it inline. */
CUresult submit_launch_now(CUfunction f,
                           unsigned gx, unsigned gy, unsigned gz,
                           unsigned bx, unsigned by, unsigned bz,
                           unsigned shmem, CUstream stream,
                           void** params, void** extra);
/* Outstanding-work throttle (§5.3). Called by the DISPATCHER when one is running
 * (the paper's placement); the submit path calls it inline only when there is no
 * dispatcher thread to defer on. No-op unless LITHOS_THROTTLE is set. */
void throttle_wait(void);

/* ---- dispatch.c ----------------------------------------------------------- */
/* Hand a launch to the dispatcher thread and wait until it is submitted. */
CUresult dispatch_submit(CUfunction f,
                         unsigned gx, unsigned gy, unsigned gz,
                         unsigned bx, unsigned by, unsigned bz,
                         unsigned shmem, CUstream stream,
                         void** params, void** extra);

#endif /* LITHOS_SCHED_INTERNAL_H */
