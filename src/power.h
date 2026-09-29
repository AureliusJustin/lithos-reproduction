/*
 * power.h — transparent power management (LithOS §5.6, Figure 9 step 7).
 *
 * The paper's fourth mechanism, and the one this reproduction was missing. The
 * idea: a GPU enforces DVFS device-wide, so a multitenant system has to pick ONE
 * frequency for everything in flight. Picking it well means knowing how much each
 * kernel actually suffers from a lower clock — compute-bound kernels slow down
 * roughly in proportion to the frequency drop, memory-bound ones barely notice —
 * and weighting that by how much of the workload each kernel represents.
 *
 * THE MODEL (§5.6, reproduced exactly).
 *
 *   Each kernel gets a weight w: its share of the stream's cumulative runtime.
 *   Its slowdown at a lower frequency is approximated to first order as
 *
 *       slowdown = lat(f_th)/lat(f_max) - 1 = s * (f_max/f_th - 1)
 *
 *   which defines that kernel's SENSITIVITY s = slowdown / (f_max/f_th - 1).
 *   s ~ 1 is a kernel whose latency tracks the clock (compute-bound); s ~ 0 is one
 *   that does not care (memory-bound). The aggregate sensitivity is S = sum(w*s),
 *   and requiring the workload's total slowdown to stay within the latency slip
 *   budget `slip` gives
 *
 *       S * (f_max/f_final - 1) <= slip   =>   f_final = f_max / (1 + slip/S)
 *
 *   So a workload of compute-bound kernels (S -> 1) is held near f_max, while a
 *   memory-bound one (S -> 0) is pushed far down. The frequency actually applied
 *   is the closest supported clock at or below f_final.
 *
 * WHY IT LEARNS SLOWLY. Switching frequency costs about 50 ms on current GPUs
 * (§5.6), which is longer than most kernels. So the paper is deliberate about
 * being conservative: unseen kernels run at f_max, a probe at a lower frequency is
 * taken only once enough samples exist, and transitions are rate-limited. We do
 * the same — see power.c.
 *
 * PRIVILEGE. Setting a clock is a root-only NVML operation on this driver (an
 * unprivileged process is refused even after `nvidia-smi -acp UNRESTRICTED`, which
 * is deprecated and inert). LibLithOS therefore has to run privileged for this
 * mechanism to engage:
 *
 *     sudo env LD_PRELOAD=build/liblithos_full.so ./app     # plain sudo strips LD_*
 *
 * When it cannot set clocks, power management disables itself and says so once;
 * everything else in LithOS carries on unaffected.
 */
#ifndef LITHOS_POWER_H
#define LITHOS_POWER_H

#include <stdint.h>

/* Bring up NVML, enumerate the device's supported graphics clocks, and pin the
 * learning phase to f_max. Safe to call more than once; a failure (no NVML, no
 * privilege) latches the mechanism off. Needs a CUDA context to exist. */
void power_init(void);

/* One measured kernel completion, from the Tracker's reaping path: `slot` is the
 * launch queue, `op` the operator ordinal within the batch (§5.7), `us` the
 * measured duration. The frequency it ran at is whatever we had set, so power.c
 * does not need to be told. */
void power_note_measure(int slot, int op, double us);

/* Re-evaluate the model and, if warranted, change frequency. Rate-limited
 * internally, so it is safe to call often — the Tracker calls it on every pass. */
void power_maybe_update(void);

/* Restore the GPU to its default clocks. Called from the library destructor: a
 * tenant that exits must not leave the device pinned. */
void power_shutdown(void);

#endif /* LITHOS_POWER_H */
