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
 *   - The Kernel Atomizer splits kernels into atoms via a Prelude kernel and
 *     QMD program-address patching (Section 5.4 / 6), fully transparently.
 */
#ifndef LITHOS_H
#define LITHOS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Kernel Atomizer: metadata handed to the Prelude kernel            */
/* ------------------------------------------------------------------ */
/*
 * Algorithm 1 in the paper. The Prelude reads this struct from a fixed device
 * address, checks whether its linear block index falls inside [lo, hi), and if
 * so branches into the original kernel entry point (retaining the original
 * kernel's resources). Otherwise it early-exits.
 *
 * `kernel_entrypoint` is the device address of the original kernel's SASS as
 * read out of the QMD before we patch it.
 */
typedef struct AtomMetadata {
    uint32_t block_idx_lo;   /* first linear block index that does real work  */
    uint32_t block_idx_hi;   /* one past the last (exclusive)                 */
    uint32_t generation;     /* bumped every launch so device sees fresh data */
    uint32_t _pad;
    uint64_t kernel_entrypoint; /* original QMD program address (device VA)   */
    uint64_t noop_entrypoint;   /* trivial kernel for out-of-range blocks     */
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
    double       pred_us;        /* predicted duration at current TPC alloc   */
    int          n_atoms;        /* how many atoms to split into (>=1)        */
    int          atomize;        /* 1 if this kernel should be atomized       */
} LithosKernel;

/* ------------------------------------------------------------------ */
/*  Tunable parameters (Section 5.3 / 5.4)                            */
/* ------------------------------------------------------------------ */
typedef struct LithosConfig {
    double   atom_duration_us;      /* target atom length; 250-500us effective*/
    double   outstanding_limit_us;  /* sync-queue throttle; paper uses 100us  */
    int      max_outstanding_atoms; /* cap on in-flight atoms                 */
    int      min_blocks_to_atomize; /* skip atomization for tiny grids        */
    int      enable_atomizer;
    int      force_atoms;           /* LITHOS_FORCE_ATOMS: override atom count (0=auto) */
    int      atom_tpc_width;        /* LITHOS_ATOM_TPC: distinct W-TPC slice per atom (0=off) */
    int      atom_tpc_list[64];     /* LITHOS_ATOM_TPC_LIST: per-atom TPC widths (cycled) */
    int      atom_tpc_list_n;       /* number of entries in atom_tpc_list (0=off)        */
    int      graph_subgraphs;       /* LITHOS_GRAPH_SUBGRAPHS: partition graphs into K subgraphs (0=off) */
    int      enable_stealing;
    int      predict;             /* LITHOS_PREDICT: online latency prediction (§5.7) */
    int      rightsize;           /* LITHOS_RIGHTSIZE: per-kernel TPC right-sizing (§5.5) */
    double   latency_slip;        /* LITHOS_SLIP: right-sizing latency-slip factor k    */
    int      throttle;            /* LITHOS_THROTTLE: enforce the outstanding-work limit */
    int      dispatch;            /* LITHOS_DISPATCH: route launches through a dispatcher thread (§5.2) */
    int      perstream_quota;     /* LITHOS_PERSTREAM_QUOTA: give each stream its own disjoint TPC slice */
    int      enable_jump;         /* attempt the Prelude->original transfer      */
    int      use_brx;             /* 1 = patch CALL->BRX jump; 0 = keep the CALL  */
    int      verbose;
} LithosConfig;

extern LithosConfig g_lithos_cfg;

void lithos_config_init(void);       /* read env vars, set defaults          */
uint64_t lithos_now_ns(void);        /* monotonic clock helper               */

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
