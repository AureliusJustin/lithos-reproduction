/* External admission gate (for the predictive arbiter in Tesis/Eksperimen/arbiter, "lithos" actuator).
 *
 * When LITHOS_GATE=<path> is set in a process (the best-effort tenant only), every kernel launch, atom and graph launch of
 * that process first waits while the gate's allow flag is 0. The gate is a 64-byte record written by the arbiter
 * (arbiter/live/gate.py documents the layout). With the atomizer and LITHOS_ATOMS_INFLIGHT=W the process cannot run more than
 * W atoms ahead of the GPU, so a closed gate takes effect within about W atoms (a few hundred microseconds) instead of whole
 * kernels.
 *
 * Fail open: no LITHOS_GATE, a missing file, a wrong magic, a record older than LITHOS_GATE_STALE_MS (500) or a wait longer
 * than LITHOS_GATE_MAX_MS (2000) all let the launch go. The gate never holds a launch during stream capture.
 * Host-only C, no CUDA: tests/test_gate.c builds and runs it without a GPU. */
#ifndef LITHOS_GATE_H
#define LITHOS_GATE_H

#include <stdint.h>

/* -1 = not yet initialised, 0 = no gate in this process, 1 = gated. Read inline on every launch, so an ungated process
 * (the latency-critical tenant) pays one load and a predictable branch per launch, not a call. */
extern int lithos_gate_state;
int lithos_gate_init(void);                 /* resolves the state once; returns it */
uint64_t lithos_gate_wait_slow(void);       /* the gated path */

/* non-zero when this process has a gate configured */
static inline int lithos_gate_enabled(void) {
    int s = __atomic_load_n(&lithos_gate_state, __ATOMIC_RELAXED);
    return __builtin_expect(s < 0, 0) ? lithos_gate_init() : s;
}

/* block while the gate is closed (bounded, fail open); returns the nanoseconds waited */
static inline uint64_t lithos_gate_wait(void) {
    return lithos_gate_enabled() ? lithos_gate_wait_slow() : 0;
}

/* counters for LITHOS_STATS / tests */
uint64_t lithos_gate_waits(void);
uint64_t lithos_gate_waited_ns(void);

#endif
