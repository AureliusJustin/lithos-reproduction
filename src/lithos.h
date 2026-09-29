/*
 * LithOS reproduction -- core types shared across the interposition layer,
 * the Kernel Atomizer, and the TPC Scheduler.
 *
 * This is a from-scratch reimplementation of the mechanisms described in the
 * LithOS paper (SOSP). The original prototype is ~5000 lines of Rust; this
 * reproduction is written in C/C++/CUDA so it can reuse the QMD/TPC-masking
 * reverse-engineering already present in ../libsmctrl and be built directly
 * with the system nvcc. Mechanism-for-mechanism it mirrors Section 5 & 6:
 *
 *   - LibLithOS interposes the CUDA Driver API (Section 5.2 / 6).
 *   - Launch queues buffer work and decouple submission from execution.
 *   - The TPC Scheduler dispatches at TPC granularity (Section 5.3).
 *   - The Kernel Atomizer splits kernels into atoms (Section 5.4 / 6), fully
 *     transparently. NOTE: the paper reaches the atom's range check by patching
 *     the QMD program address to a "Prelude" kernel that JUMPS into the original;
 *     we instead SPLICE the range check into each kernel's own machine code so
 *     in-range blocks fall through. Same semantics, no control transfer — see
 *     docs/TECHNICAL_REPORT.md and src/atomize_splice.c.
 */
#ifndef LITHOS_H
#define LITHOS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Kernel Atomizer: the per-atom metadata the spliced prologue reads  */
/* ------------------------------------------------------------------ */
/*
 * Algorithm 1's range gate. The spliced-in prologue reads this from a FIXED
 * device address (baked into its SASS as a literal), checks whether its linear
 * block index falls inside [lo, hi), and EXITs if not; in-range blocks fall
 * through into the original kernel body.
 *
 * Only these two words exist on the device — the atomizer writes them with two
 * cuMemsetD32Async immediates before each atom's relaunch. The paper's Prelude
 * additionally needs the original kernel's entry point (it JUMPS there); our
 * fall-through splice needs no entry point, no generation counter, and no
 * out-of-range target, so those fields are deliberately absent. See
 * docs/TECHNICAL_REPORT.md for why the jump could not be reproduced.
 */
typedef struct AtomMetadata {
    uint32_t block_idx_lo;   /* first linear block index that does real work  */
    uint32_t block_idx_hi;   /* one past the last (exclusive)                 */
} AtomMetadata;

/* ------------------------------------------------------------------ */
/*  Deferred kernel launch (buffered in a launch queue, Fig. 9 Step 1) */
/* ------------------------------------------------------------------ */
typedef struct LithosKernel {
    /* Original cuLaunchKernel arguments */
    void*        func;            /* CUfunction                               */
    unsigned int gridDimX, gridDimY, gridDimZ;
    unsigned int blockDimX, blockDimY, blockDimZ;
    unsigned int sharedMemBytes;
    void*        stream;         /* CUstream                                  */
    void**       kernelParams;
    void**       extra;

    /* Derived / scheduling state */
    uint64_t     total_blocks;   /* gridDimX*Y*Z                              */
    uint64_t     enqueue_ns;     /* when the app enqueued this kernel         */
    double       pred_us;        /* predicted duration at the current TPC alloc;
                                  * 0 = unknown, atomizer falls back to a stub */
    double       atom_cost_us;   /* this operator's LEARNED marginal cost per atom
                                  * (§5.4 "monitors the effectiveness of the Kernel
                                  * Atomizer"); 0 = not yet known, use the default */
} LithosKernel;

/* ------------------------------------------------------------------ */
/*  Tunable parameters (Section 5.3 / 5.4)                            */
/* ------------------------------------------------------------------ */
typedef struct LithosConfig {
    double   atom_duration_us;      /* target atom length; paper uses 250-500us */
    double   slo_us;                /* LITHOS_SLO_US: max delay this process may impose
                                     * on a co-located tenant. Caps atom duration, since
                                     * an HP job waits behind at most one atom. 0 = off */
    double   atom_cost_us;          /* LITHOS_ATOM_COST_US: measured cost of one extra
                                     * atom (the early-exit full-grid relaunch)        */
    double   atom_max_overhead;     /* LITHOS_ATOM_MAX_OVERHEAD: cap splitting overhead
                                     * at this fraction of the kernel's runtime        */
    double   outstanding_limit_us;  /* sync-queue throttle; paper uses 100us  */
    int      min_blocks_to_atomize; /* skip atomization for tiny grids        */
    int      enable_atomizer;
    int      force_atoms;           /* LITHOS_FORCE_ATOMS: override atom count (0=auto) */
    int      max_atoms_inflight;    /* LITHOS_ATOMS_INFLIGHT: cap the atoms of one kernel
                                     * that may be outstanding at once (§5.3, "limits
                                     * outstanding atoms"). 0 = submit them all at once */
    int      atom_tpc_width;        /* LITHOS_ATOM_TPC: distinct W-TPC slice per atom (0=off) */
    int      atom_tpc_list[64];     /* LITHOS_ATOM_TPC_LIST: per-atom TPC widths (cycled) */
    int      atom_tpc_list_n;       /* number of entries in atom_tpc_list (0=off)        */
    int      graph_subgraphs;       /* LITHOS_GRAPH_SUBGRAPHS: partition graphs into K subgraphs (0=off) */
    int      enable_stealing;
    int      tpc_timers;       /* LITHOS_TPC_TIMERS: per-TPC busy timers gate steals */
    int      predict;             /* LITHOS_PREDICT: online latency prediction (§5.7) */
    int      predict_bracket;     /* LITHOS_PREDICT_BRACKET: time each launch with its
                                   * own start+stop pair instead of differencing against
                                   * the previous completion (see predict.c) */
    int      rightsize;           /* LITHOS_RIGHTSIZE: per-kernel TPC right-sizing (§5.5) */
    int      rightsize_occ;      /* LITHOS_RIGHTSIZE_OCC: use the stage-1 occupancy bound */
    int      dvfs;                /* LITHOS_DVFS: transparent power management (§5.6)   */
    double   dvfs_slip;           /* LITHOS_DVFS_SLIP: latency slip for DVFS (1.1 = 10%) */
    int      dvfs_min_samples;    /* LITHOS_DVFS_SAMPLES: samples/operator per phase     */
    double   dvfs_switch_ms;      /* LITHOS_DVFS_SWITCH_MS: min gap between transitions  */
    double   dvfs_probe_frac;     /* LITHOS_DVFS_PROBE: probe clock as a fraction of f_max */
    double   latency_slip;        /* LITHOS_SLIP: right-sizing latency-slip factor k    */
    int      throttle;            /* LITHOS_THROTTLE: enforce the outstanding-work limit */
    int      dispatch;            /* LITHOS_DISPATCH: buffer launches in launch queues, drained by a
                                   * dispatcher thread (§5.2) */
    int      dispatch_prio;       /* LITHOS_DISPATCH_PRIO: drain the queues by stream priority (1) or
                                   * in strict arrival order (0). The control for the experiment:
                                   * 0 keeps the deferral, drops the policy. */
    int      coordinator;         /* LITHOS_COORD: join the system-wide tenant table (§5.1) */
    int      priority;            /* LITHOS_PRIORITY: this tenant's priority (larger = higher) */
    int      perstream_quota;     /* LITHOS_PERSTREAM_QUOTA: give each stream its own disjoint TPC slice */
    int      verbose;
} LithosConfig;

extern LithosConfig g_lithos_cfg;

void lithos_config_init(void);       /* read env vars, set defaults          */
uint64_t lithos_now_ns(void);        /* monotonic clock helper               */

/* ------------------------------------------------------------------ */
/*  Re-entrancy guard: LithOS calling CUDA through its own interposers */
/* ------------------------------------------------------------------ */
/*
 * LithOS issues plenty of CUDA calls of its own — the atomizer writes each
 * atom's block range with cuMemsetD32Async, the predictor records and reaps
 * events, the graph scheduler re-instantiates execs. Those calls resolve to OUR
 * exported symbols, not the driver's, because within the library a direct call
 * to cuEventQuery binds to the definition in barrier.c.
 *
 * That matters once launches are buffered, because the ordering barriers drain
 * the launch queues before forwarding. LithOS's own calls must not do that:
 *
 *   - it deadlocks. The Tracker thread calls cuEventQuery, which would wait for
 *     the dispatcher to empty its queues; the dispatcher is waiting on the
 *     outstanding-work throttle, which only falls when the Tracker reaps a
 *     completion. Each waits for the other. (Observed, not hypothetical.)
 *   - it is unnecessary. LithOS makes these calls at points where it has already
 *     established the ordering it needs — the atom metadata write goes to the
 *     same stream immediately before the atom, the completion event immediately
 *     after the launch it times.
 *
 * So every entry point that runs LithOS's own CUDA work raises this thread-local
 * flag, and the barriers pass straight through while it is set. It counts rather
 * than toggles so nesting is safe.
 */
extern __thread int g_lithos_internal;
static inline void lithos_internal_begin(void) { g_lithos_internal++; }
static inline void lithos_internal_end(void)   { g_lithos_internal--; }

/* Per-atom TPC allocation: called by the atomizer before each atom's relaunch so
 * every atom of ONE kernel can be confined to a DISTINCT TPC set (the paper's
 * "TPC allocations can be dynamically adjusted throughout a kernel's execution").
 * With LITHOS_ATOM_TPC=W each atom gets its own W-TPC slice; otherwise it just
 * re-applies the stream's quota mask to every atom (the QMD mask is one-shot per
 * launch, so without this only the first atom would be confined). */
void lithos_apply_atom_mask(void* stream, int atom_idx, int n_atoms);

/* Compute (not apply) the TPC disable-mask for slice idx of n on this stream. */
uint64_t lithos_slice_mask(void* stream, int idx, int n);

/* Paper-model CUDA-graph scheduling (src/graphsched.c): partition an instantiated
 * graph into K subgraphs so the scheduler can allocate TPCs per subgraph. Returns
 * 1 and fills the interception if handled, 0 to fall back to the real call. */
int lithos_graph_instantiate(void* pExec, void* graph, unsigned long long flags);
int lithos_graph_launch(void* exec, void* stream);
int lithos_graph_exec_destroy(void* exec);

#ifdef __cplusplus
}
#endif

#endif /* LITHOS_H */
