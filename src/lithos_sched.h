/*
 * TPC Scheduler interface (Section 5.3).
 *
 * The interposition layer forwards buffered work here. The scheduler owns the
 * launch queues (Fig. 9 Step 1), applies compute quotas (Step 2), invokes the
 * Kernel Atomizer (Step 3), dispatches atoms onto assigned TPCs (Step 4), and
 * tracks outstanding work through sync queues (Step 5).
 */
#ifndef LITHOS_SCHED_H
#define LITHOS_SCHED_H

#include <cuda.h>
#include "lithos.h"

/* Lifecycle */
void lithos_sched_init(void);

/* SMs allocated to this tenant (quota x 2), or 0 if no quota. Used to spoof
 * CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT for special kernels (§6). */
int lithos_allocated_sms(void);

/* Stream <-> launch queue (Fig. 9, Step 1). A launch queue is created when an
 * application creates a stream. */
void lithos_stream_created(CUstream s, int priority);
void lithos_stream_destroyed(CUstream s);

/* An app called cuLaunchKernel. LithOS enqueues and returns control (Sec 5.2).
 * The scheduler decides when to atomize and dispatch to the GPU. */
CUresult lithos_submit_launch(CUfunction f,
                              unsigned gx, unsigned gy, unsigned gz,
                              unsigned bx, unsigned by, unsigned bz,
                              unsigned shmem, CUstream stream,
                              void** params, void** extra);

/* cuLaunchKernelEx / cuLaunchCooperativeKernel paths (used by the CUDA runtime,
 * i.e. PyTorch/TF/JAX/TensorRT). Same quota/mask bookkeeping, then the atomizer's
 * Ex / cooperative dispatch. */
CUresult lithos_submit_launch_ex(const CUlaunchConfig* cfg, CUfunction f,
                                 void** params, void** extra);
CUresult lithos_submit_launch_coop(CUfunction f,
                                   unsigned gx, unsigned gy, unsigned gz,
                                   unsigned bx, unsigned by, unsigned bz,
                                   unsigned shmem, CUstream stream, void** params);

/* Barriers must flush the relevant launch queues before waiting. */
CUresult lithos_stream_sync(CUstream s);
CUresult lithos_ctx_sync(void);

/* Per-application TPC quota (Step 2), expressed as a count of guaranteed TPCs.
 * <=0 means "no quota" (may use all TPCs). */
void lithos_set_quota(CUstream s, int n_tpcs);

#endif /* LITHOS_SCHED_H */
