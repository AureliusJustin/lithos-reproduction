#define _GNU_SOURCE
#include "lithos.h"
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
     * atom_cost_us default is the measured per-atom cost on an A6000; overhead is
     * capped at 10% of the kernel by default. SLO capping is opt-in. */
    g_lithos_cfg.slo_us                = envd("LITHOS_SLO_US", 0.0);
    g_lithos_cfg.atom_cost_us          = envd("LITHOS_ATOM_COST_US", 5.0);
    g_lithos_cfg.atom_max_overhead     = envd("LITHOS_ATOM_MAX_OVERHEAD", 0.10);
    g_lithos_cfg.outstanding_limit_us  = envd("LITHOS_OUTSTANDING_US", 100.0);
    g_lithos_cfg.min_blocks_to_atomize = envi("LITHOS_MIN_BLOCKS", 8);
    g_lithos_cfg.enable_atomizer       = envi("LITHOS_ATOMIZER", 1);
    g_lithos_cfg.force_atoms           = envi("LITHOS_FORCE_ATOMS", 0);
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
    g_lithos_cfg.rightsize             = envi("LITHOS_RIGHTSIZE", 0);
    g_lithos_cfg.latency_slip          = envd("LITHOS_SLIP", 1.1);
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
