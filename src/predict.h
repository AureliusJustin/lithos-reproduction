/* predict.h -- online latency prediction (LithOS §5.7).
 *
 * Learns each operator's execution time online (no offline profiling) so the
 * scheduler can size atoms (§5.4), right-size TPCs (§5.5), and drive stealing
 * timers / the outstanding-work throttle (§5.3). Operators are identified per
 * launch queue by an ORDINAL INDEX k (the k-th kernel since the last sync/batch
 * boundary), exactly as the paper does, since the same kernel function recurs
 * with different tensor sizes across a model's layers.
 */
#ifndef LITHOS_PREDICT_H
#define LITHOS_PREDICT_H
#include <cuda.h>
#include <stdint.h>

void   predict_init(void);

/* Per-queue operator ordinal: advance returns the current k then ++; reset zeroes
 * it (called at a sync/batch boundary). */
int    predict_next_op(int slot);
void   predict_reset_op(int slot);

/* Predicted latency (us) for operator (slot,op) at `tpcs` TPCs, given its grid
 * `blocks`. Returns 0 if there is no data yet (caller falls back to a default). */
double predict_lookup(int slot, int op, int tpcs, uint64_t blocks);

/* Right-sizing (§5.5): minimum TPCs that keeps latency within factor `slip` of the
 * all-TPC latency, using the l = m/t + b model fit from the all-TPC and 1-TPC
 * samples. Returns 0 if the model isn't ready (caller keeps the full allocation).
 * `want_probe` is set to a TPC count the caller should FORCE this launch to, to
 * collect a missing model point (all-TPC or 1-TPC), or 0 if no probe needed. */
int    predict_rightsize(int slot, int op, int all_tpcs, double slip, int* want_probe);

/* Event-measured feedback. sched.c records ONE event on the launch stream AFTER
 * the launch (a completion marker) and submits it; the Tracker reaps it, derives
 * the duration as the gap from the previous completion on the same queue, and
 * updates the table. One event per launch, not a start/stop pair — see predict.c. */
CUevent predict_evt_get(void);
void    predict_submit(int slot, int op, int tpcs, CUevent done_evt);

/* Outstanding work (§5.3): us of in-flight (submitted-not-yet-reaped) work. */
double  predict_outstanding_us(void);

/* Start the Tracker thread (reaps completions, feeds the table, decrements
 * outstanding). Idempotent. */
void    predict_start_tracker(void);

/* Park/unpark the Tracker around CUDA stream capture. While a capture is open,
 * CUDA calls from other threads on the same context can invalidate it, so the
 * Tracker must not touch events; pending measurements are reaped afterwards. */
void    predict_capture_begin(void);
void    predict_capture_end(void);

#endif
