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

/* The marginal cost of one extra atom for this operator, in microseconds, learned
 * from its split and unsplit measurements. 0 = not known yet, in which case the
 * caller falls back to the configured constant. See predict_record(). */
double predict_atom_cost(int slot, int op);

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
/* `start_evt` may be NULL: pass one only when this launch has no in-batch
 * predecessor (the first kernel after a sync), so it can still be timed. */
/* `n_atoms` is how many atoms this launch was split into. It is not bookkeeping:
 * §5.7 requires the predictor to account for "the granularity at which it is
 * atomized", because the measured duration of a split launch includes the cost of
 * the extra relaunches. Feeding that back unadjusted makes the estimate grow, which
 * splits the kernel further, which grows it again. */
void    predict_submit(int slot, int op, int tpcs, int n_atoms,
                       CUevent start_evt, CUevent done_evt);

/* Outstanding work (§5.3): us of in-flight (submitted-not-yet-reaped) work. */
double  predict_outstanding_us(void);
/* Number of launches submitted but not yet reaped. The throttle needs this so it
 * can always keep at least one launch in flight (see throttle_wait). */
int     predict_outstanding_n(void);

/* Start the Tracker thread (reaps completions, feeds the table, decrements
 * outstanding). Idempotent. */
void    predict_start_tracker(void);

/* Park/unpark the Tracker around CUDA stream capture. While a capture is open,
 * CUDA calls from other threads on the same context can invalidate it, so the
 * Tracker must not touch events; pending measurements are reaped afterwards. */
void    predict_capture_begin(void);
void    predict_capture_end(void);

#endif
