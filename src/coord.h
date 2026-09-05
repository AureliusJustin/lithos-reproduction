/* coord.h — the system-wide view of GPU state (LithOS §5.1, Figure 8).
 *
 * "As a GPU operating system, LithOS maintains a system-wide view of GPU state
 *  across applications with varying priorities, enabling efficient scheduling
 *  and management."  (§5.1)
 *
 * Figure 8 shows several applications, each linked against LibLithOS, all sitting
 * above ONE shared LithOS layer that owns the Scheduler / Atomizer / Right-sizing.
 * Everything else in this repo is per-process; this file is that shared layer.
 *
 * It is implemented as a POSIX shared-memory segment rather than a daemon: every
 * LibLithOS instance maps the same region, registers itself as a tenant, and both
 * publishes its own liveness and reads everyone else's. That gives the two things
 * the paper needs from a central scheduler without adding an RPC hop to the launch
 * path:
 *
 *   1. Compute quotas across APPLICATIONS (§5.2) — disjoint TPC ranges are handed
 *      out automatically, so tenants no longer need a hand-assigned
 *      LITHOS_TPC_BASE.
 *   2. TPC Stealing across APPLICATIONS (§5.3) — "the scheduler dynamically
 *      reassigns underutilized TPCs across applications". A tenant can borrow the
 *      TPCs of tenants that are currently idle, with the paper's priority
 *      safeguards: a higher-priority tenant's TPCs are never stolen, and stealing
 *      stops as soon as that tenant has runnable work again ("stealing is disabled
 *      for the latter's subsequent atoms once request B is submitted", §5.4).
 */
#ifndef LITHOS_COORD_H
#define LITHOS_COORD_H

#include <stdint.h>

#define COORD_MAX_TENANTS 64

/* Attach to (or create) the shared segment and register this process as a tenant.
 * `quota` is its guaranteed TPC count (<=0: unrestricted), `priority` is larger =
 * more important. Returns 1 if the coordinator is active, 0 to fall back to
 * per-process behaviour. Idempotent. */
int  coord_join(int n_tpcs, int quota, int priority);

/* This tenant's assigned contiguous range, chosen so tenants do not overlap.
 * Returns 0 if the coordinator is inactive. */
int  coord_my_range(int* lo, int* hi);

/* Publish liveness: called on every launch so other tenants can see that this one
 * has runnable work (drives the idle detection that stealing keys off). */
void coord_note_launch(uint64_t now_ns, int outstanding);

/* Mask of TPCs this tenant may borrow right now: owned by OTHER tenants that are
 * idle and not higher-priority. Bit t set = TPC t may be used. 0 = nothing to
 * borrow. `idle_ns` is how long a tenant must be quiet to count as idle. */
uint64_t coord_stealable_tpcs(uint64_t now_ns, uint64_t idle_ns);

/* ---- Per-TPC timers (§5.3) ------------------------------------------------
 *
 * "It maintains per-TPC timers informed by a latency prediction module,
 *  estimating kernel (and atom) durations at submission time. These timers help
 *  avoid stealing from long-running TPCs."
 *
 * Idle detection alone answers "is this TPC busy?"; these answer "how much longer
 * will it be busy?", which is what makes it safe to borrow. Without them a lender
 * that has just been handed a multi-millisecond kernel still looks idle for the
 * whole of g_idle_ns, and the borrower's work lands behind it — the head-of-line
 * blocking that Figure 10(b) is about.
 *
 * The array is device-wide and lives in the shared segment, because stealing is
 * cross-application: a timer set by one tenant has to be visible to every other.
 * When the coordinator is disabled it falls back to a process-local array, so
 * intra-process stealing (LITHOS_PERSTREAM_QUOTA=1) gets the same protection. */

/* Record that `tpc_mask`'s TPCs are expected to be busy until `until_ns`. Called
 * at submission time with the launch's predicted completion. Later deadlines win,
 * so overlapping launches extend rather than shorten a TPC's timer. */
void coord_mark_tpcs_busy(uint64_t tpc_mask, uint64_t until_ns);

/* Mask of TPCs whose timer has not yet expired — those a steal should avoid.
 * Bit t set = TPC t is predicted still busy at now_ns. */
uint64_t coord_busy_tpcs(uint64_t now_ns);

/* Number of live tenants (including this one); 0 when inactive. */
int  coord_tenant_count(void);

/* Human-readable dump of the shared table, for LITHOS_LOG_COORD / tests. */
void coord_dump(void);

#endif /* LITHOS_COORD_H */
