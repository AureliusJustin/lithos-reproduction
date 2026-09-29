#define _GNU_SOURCE
#include "lithos.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

LithosConfig g_lithos_cfg;
static pthread_once_t cfg_once = PTHREAD_ONCE_INIT;

/* Set while this thread is executing LithOS's own CUDA work — see lithos.h. */
__thread int g_lithos_internal;

static double envd(const char* k, double dflt) {
    const char* v = getenv(k);
    return v ? atof(v) : dflt;
}
static int envi(const char* k, int dflt) {
    const char* v = getenv(k);
    return v ? atoi(v) : dflt;
}

static void cfg_init_once(void) {
    /* Paper defaults: atom_duration 250-500us, 100us outstanding-work limit. */
    g_lithos_cfg.atom_duration_us      = envd("LITHOS_ATOM_US", 300.0);
    /* Derived atom-duration bounds (see effective_atom_us in atomizer.c).
     *
     * atom_cost_us is the cost of ONE extra atom: a full-grid relaunch whose
     * out-of-range blocks reach the prologue and exit. It is the constant that
     * decides how aggressively kernels are split, and it is hardware-dependent.
     *
     * The default was 5us, measured on an A6000 with a tiny test kernel, and it is
     * far too low for real framework kernels: measured on an A100 against PyTorch
     * training kernels it is 20-40us (bench/README.md shows the measurement). With
     * 5us the floor works out at 50us, which split a 4.4ms kernel into ~88 atoms
     * and cost 15x throughput -- exactly the failure the paper warns about, "if
     * this parameter is set too low, an atomized kernel may actually take longer
     * to complete". At the measured 30us the floor becomes 300us, which is where
     * the paper's own 250-500us guidance sits. Overhead is capped at 10% of the
     * kernel by default; SLO capping is opt-in. */
    g_lithos_cfg.slo_us                = envd("LITHOS_SLO_US", 0.0);
    g_lithos_cfg.atom_cost_us          = envd("LITHOS_ATOM_COST_US", 30.0);
    g_lithos_cfg.atom_max_overhead     = envd("LITHOS_ATOM_MAX_OVERHEAD", 0.10);
    g_lithos_cfg.outstanding_limit_us  = envd("LITHOS_OUTSTANDING_US", 100.0);
    g_lithos_cfg.min_blocks_to_atomize = envi("LITHOS_MIN_BLOCKS", 8);
    g_lithos_cfg.enable_atomizer       = envi("LITHOS_ATOMIZER", 1);
    g_lithos_cfg.force_atoms           = envi("LITHOS_FORCE_ATOMS", 0);
    /* Bound the atoms of one kernel that may be in flight (§5.3). Off by default:
     * it makes the submitting thread wait on the GPU inside the atom loop, which
     * costs throughput. Turning it on is what makes a kernel's allocation react to
     * arrivals mid-execution (Figure 10(c)) — see atom_gate_width in atomizer.c. */
    g_lithos_cfg.max_atoms_inflight    = envi("LITHOS_ATOMS_INFLIGHT", 0);
    g_lithos_cfg.atom_tpc_width        = envi("LITHOS_ATOM_TPC", 0);
    /* LITHOS_ATOM_TPC_LIST="1,2,3": atom i gets atom_tpc_list[i % n] TPCs (variable
     * per-atom widths, packed contiguously; overrides the uniform LITHOS_ATOM_TPC). */
    g_lithos_cfg.atom_tpc_list_n = 0;
    const char* lst = getenv("LITHOS_ATOM_TPC_LIST");
    if (lst && *lst) {
        int n = 0;
        for (const char* p = lst; *p && n < 64; ) {
            int v = atoi(p);
            if (v > 0) g_lithos_cfg.atom_tpc_list[n++] = v;
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
        g_lithos_cfg.atom_tpc_list_n = n;
    }
    g_lithos_cfg.graph_subgraphs = envi("LITHOS_GRAPH_SUBGRAPHS", 0);
    g_lithos_cfg.enable_stealing       = envi("LITHOS_STEALING", 1);
    /* Per-TPC timers (§5.3). On by default: without them a steal can land behind
     * a long kernel whose owner still reads as idle. Off = the older idle-only
     * heuristic, kept so the two can be compared directly. */
    g_lithos_cfg.tpc_timers            = envi("LITHOS_TPC_TIMERS", 1);
    g_lithos_cfg.predict               = envi("LITHOS_PREDICT", 1);
    /* How a launch is TIMED. Differencing against the previous completion on the
     * queue costs one event per launch instead of two, and is exact whenever the
     * stream is saturated — but it charges every host-side gap to the next kernel,
     * and frameworks are full of them. Measured on a PyTorch training step: single
     * "kernel" durations of 73 ms against a 66 ms whole iteration, which then drove
     * the atomizer to split kernels 1024 ways. Bracketing each launch measures GPU
     * time and nothing else, so it is the default; set 0 for the cheaper method on
     * workloads that keep the stream busy. */
    g_lithos_cfg.predict_bracket       = envi("LITHOS_PREDICT_BRACKET", 1);
    g_lithos_cfg.rightsize             = envi("LITHOS_RIGHTSIZE", 0);
    g_lithos_cfg.rightsize_occ         = envi("LITHOS_RIGHTSIZE_OCC", 1);
    g_lithos_cfg.latency_slip          = envd("LITHOS_SLIP", 1.1);
    /* Transparent power management (§5.6). Off by default for two reasons: it is
     * device-wide (one tenant's choice changes every tenant's clock) and it needs
     * privilege to set clocks at all. The paper's own experiments use a slip of
     * 1.1. Switching costs ~50 ms on current GPUs, so transitions are rate-limited
     * well above that and the learning period is deliberately long. */
    g_lithos_cfg.dvfs                  = envi("LITHOS_DVFS", 0);
    /* The sensitivities come from the predictor's measurements, so DVFS with the
     * predictor off would learn nothing and silently do nothing. Rather than fail
     * quietly, turn the predictor back on and say so. */
    if (g_lithos_cfg.dvfs && !g_lithos_cfg.predict) {
        g_lithos_cfg.predict = 1;
        fprintf(stderr, "[lithos] LITHOS_DVFS=1 needs the predictor; "
                        "overriding LITHOS_PREDICT=0\n");
    }
    g_lithos_cfg.dvfs_slip             = envd("LITHOS_DVFS_SLIP", 1.1);
    g_lithos_cfg.dvfs_min_samples      = envi("LITHOS_DVFS_SAMPLES", 8);
    g_lithos_cfg.dvfs_switch_ms        = envd("LITHOS_DVFS_SWITCH_MS", 500.0);
    g_lithos_cfg.dvfs_probe_frac       = envd("LITHOS_DVFS_PROBE", 0.75);
    g_lithos_cfg.dispatch              = envi("LITHOS_DISPATCH", 0);
    /* LITHOS_DISPATCH_PRIO=0 makes the dispatcher drain its queues in strict
     * arrival order instead of by priority. Deferral still happens; only the
     * policy that exploits it is switched off. That separates the COST of
     * buffering from the BENEFIT of scheduling out of the buffer, which is the
     * control condition the dispatcher experiment needs. */
    g_lithos_cfg.dispatch_prio         = envi("LITHOS_DISPATCH_PRIO", 1);
    /* The outstanding-work throttle and the dispatcher are one mechanism, not
     * two. The throttle is what makes work ACCUMULATE in the launch queues; the
     * dispatcher is what chooses from the accumulation. A dispatcher with no
     * throttle drains as fast as the GPU accepts work, so the queues stay empty
     * and there is never anything to choose between — deferral with nothing to
     * defer for. So turning the dispatcher on turns the throttle on unless the
     * caller has said otherwise explicitly. */
    g_lithos_cfg.throttle              = envi("LITHOS_THROTTLE",
                                              g_lithos_cfg.dispatch ? 1 : 0);
    g_lithos_cfg.perstream_quota       = envi("LITHOS_PERSTREAM_QUOTA", 0);
    /* Cross-application coordination (§5.1/5.2/5.3): on by default so quotas and
     * stealing span tenants, as the paper specifies. LITHOS_COORD=0 reverts to
     * purely per-process behaviour. */
    g_lithos_cfg.coordinator           = envi("LITHOS_COORD", 1);
    g_lithos_cfg.priority              = envi("LITHOS_PRIORITY", 0);
    g_lithos_cfg.verbose               = envi("LITHOS_VERBOSE", 0);
}

void lithos_config_init(void) { pthread_once(&cfg_once, cfg_init_once); }

uint64_t lithos_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
