#define _GNU_SOURCE
/*
 * power.c — transparent power management (LithOS §5.6). See power.h for the model
 * and for why this mechanism needs privilege.
 */
#include "power.h"
#include "lithos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <pthread.h>
#include <cuda.h>

#define PLOG(...) do { if (g_log) { \
    fprintf(stderr, "[dvfs] " __VA_ARGS__); fputc('\n', stderr); } } while (0)

/* ------------------------------------------------------------------ */
/*  NVML, loaded at runtime                                            */
/* ------------------------------------------------------------------ */
/* Resolved with dlopen rather than linked, so a machine without NVML still runs
 * everything else. The handful of entry points we need are declared locally to
 * avoid a build dependency on nvml.h. */
typedef void* nvmlDevice_t;
static void* g_nvml_lib;
static int   (*p_nvmlInit)(void);
static int   (*p_nvmlShutdown)(void);
static int   (*p_nvmlDeviceGetHandleByIndex)(unsigned, nvmlDevice_t*);
static int   (*p_nvmlDeviceGetSupportedMemoryClocks)(nvmlDevice_t, unsigned*, unsigned*);
static int   (*p_nvmlDeviceGetSupportedGraphicsClocks)(nvmlDevice_t, unsigned, unsigned*, unsigned*);
static int   (*p_nvmlDeviceSetGpuLockedClocks)(nvmlDevice_t, unsigned, unsigned);
static int   (*p_nvmlDeviceResetGpuLockedClocks)(nvmlDevice_t);

#define NVML_OK 0

/* ------------------------------------------------------------------ */
/*  State                                                              */
/* ------------------------------------------------------------------ */
#define PW_STREAMS 256
#define PW_OPS     2048
#define PW_MAX_CLOCKS 256

/* Fraction of observed runtime that must be accounted for by operators with
 * enough samples before a phase advances. Below 1.0 on purpose — see
 * learn_coverage(). */
#define PW_COVERAGE 0.8

/* Only step the clock down when the observed slowdown is comfortably inside the
 * budget. Stepping down at the boundary would oscillate across it. */
#define PW_HEADROOM 0.7

/* Per-operator scaling state. `runtime_us` accumulates so a kernel's WEIGHT is
 * its share of the stream's total time, exactly as §5.6 defines it — a kernel
 * that runs rarely or briefly cannot drag the whole workload's frequency down. */
typedef struct {
    double l_max;        /* mean latency at f_max (us); 0 = not yet measured   */
    long   n_max;        /* samples contributing to l_max                      */
    double l_probe;      /* mean latency at f_probe (us)                       */
    long   n_probe;
    double l_cur;        /* mean latency at the clock currently applied        */
    long   n_cur;        /* samples since the last frequency change            */
    double sens;         /* sensitivity s; < 0 = not yet derived               */
    double runtime_us;   /* cumulative observed runtime -> weight w            */
} PwOp;

static PwOp    g_ops[PW_STREAMS][PW_OPS];
static double  g_total_runtime_us;          /* denominator for the weights      */
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;

static nvmlDevice_t g_dev;
static unsigned g_clocks[PW_MAX_CLOCKS];    /* supported graphics clocks, desc   */
static int      g_n_clocks;
static unsigned g_f_max, g_f_probe, g_f_cur;
static int      g_enabled;                  /* 0 until init succeeds             */
static int      g_failed;                   /* latched: never try again          */
static int      g_log;
static uint64_t g_last_switch_ns;
static long     g_measures;                 /* measurements seen since init      */

/* Phases, in the paper's order: learn at f_max, then probe one lower clock to
 * derive sensitivities, then apply the model. */
enum { PH_LEARN = 0, PH_PROBE, PH_APPLY };
static int g_phase;

/* Tunables (see config.c). */
static double g_slip;         /* tolerated fractional slowdown, e.g. 0.1        */
static long   g_min_samples;  /* per-operator samples before a phase advances    */
static double g_min_switch_ms;/* rate limit on transitions (switching costs ~50ms)*/

/* ------------------------------------------------------------------ */
/*  Actuator                                                           */
/* ------------------------------------------------------------------ */
static int nvml_load(void) {
    g_nvml_lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!g_nvml_lib) g_nvml_lib = dlopen("libnvidia-ml.so", RTLD_LAZY);
    if (!g_nvml_lib) return -1;
#define SYM(v, n) do { *(void**)&v = dlsym(g_nvml_lib, n); if (!v) return -1; } while (0)
    SYM(p_nvmlInit,                            "nvmlInit_v2");
    SYM(p_nvmlShutdown,                        "nvmlShutdown");
    SYM(p_nvmlDeviceGetHandleByIndex,          "nvmlDeviceGetHandleByIndex_v2");
    SYM(p_nvmlDeviceGetSupportedMemoryClocks,  "nvmlDeviceGetSupportedMemoryClocks");
    SYM(p_nvmlDeviceGetSupportedGraphicsClocks,"nvmlDeviceGetSupportedGraphicsClocks");
    SYM(p_nvmlDeviceSetGpuLockedClocks,        "nvmlDeviceSetGpuLockedClocks");
    SYM(p_nvmlDeviceResetGpuLockedClocks,      "nvmlDeviceResetGpuLockedClocks");
#undef SYM
    return 0;
}

static int cmp_desc(const void* a, const void* b) {
    unsigned x = *(const unsigned*)a, y = *(const unsigned*)b;
    return x < y ? 1 : (x > y ? -1 : 0);
}

/* Apply a clock. Returns 0 on success. The first failure disables the whole
 * mechanism: on this driver an unprivileged process is refused, and retrying
 * once per decision would only produce a stream of identical errors. */
static int set_clock(unsigned mhz) {
    if (!g_enabled) return -1;
    int r = p_nvmlDeviceSetGpuLockedClocks(g_dev, mhz, mhz);
    if (r != NVML_OK) {
        fprintf(stderr, "[dvfs] cannot set GPU clocks (NVML error %d) — power "
                        "management disabled. It needs privilege: run as "
                        "`sudo env LD_PRELOAD=... ./app` (plain sudo strips LD_*).\n", r);
        g_enabled = 0; g_failed = 1;
        return -1;
    }
    /* Measurements taken at the old clock say nothing about the new one, so the
     * observation window restarts here. Without this the closed loop below reads
     * a mixture of two frequencies and never settles. */
    if (mhz != g_f_cur)
        for (int i = 0; i < PW_STREAMS; i++)
            for (int j = 0; j < PW_OPS; j++) { g_ops[i][j].l_cur = 0; g_ops[i][j].n_cur = 0; }
    g_f_cur = mhz;
    g_last_switch_ns = lithos_now_ns();
    return 0;
}

/* Closest supported clock at or BELOW `want`, so a rounding step never exceeds
 * the slowdown the caller budgeted for. */
static unsigned clock_at_or_below(double want) {
    unsigned best = g_clocks[g_n_clocks - 1];      /* the lowest we support */
    for (int i = 0; i < g_n_clocks; i++)
        if ((double)g_clocks[i] <= want) { best = g_clocks[i]; break; }
    return best;
}

void power_init(void) {
    if (g_enabled || g_failed) return;
    if (!g_lithos_cfg.dvfs) return;
    g_log          = getenv("LITHOS_LOG_DVFS") ? atoi(getenv("LITHOS_LOG_DVFS")) : 0;
    g_slip         = g_lithos_cfg.dvfs_slip > 1.0 ? g_lithos_cfg.dvfs_slip - 1.0 : 0.1;
    g_min_samples  = g_lithos_cfg.dvfs_min_samples;
    g_min_switch_ms= g_lithos_cfg.dvfs_switch_ms;

    if (nvml_load() != 0 || p_nvmlInit() != NVML_OK) {
        fprintf(stderr, "[dvfs] NVML unavailable — power management disabled\n");
        g_failed = 1; return;
    }
    CUdevice cudev = 0;
    cuCtxGetDevice(&cudev);                       /* NVML index == CUDA ordinal here */
    if (p_nvmlDeviceGetHandleByIndex((unsigned)cudev, &g_dev) != NVML_OK) {
        fprintf(stderr, "[dvfs] no NVML handle for device %d — disabled\n", (int)cudev);
        g_failed = 1; return;
    }

    /* Supported graphics clocks are enumerated per memory clock; datacenter parts
     * expose a single memory clock, so take the first and use its list. */
    unsigned nmem = PW_MAX_CLOCKS, mem[PW_MAX_CLOCKS];
    if (p_nvmlDeviceGetSupportedMemoryClocks(g_dev, &nmem, mem) != NVML_OK || nmem == 0) {
        fprintf(stderr, "[dvfs] cannot enumerate memory clocks — disabled\n");
        g_failed = 1; return;
    }
    unsigned ngfx = PW_MAX_CLOCKS;
    if (p_nvmlDeviceGetSupportedGraphicsClocks(g_dev, mem[0], &ngfx, g_clocks) != NVML_OK
            || ngfx == 0) {
        fprintf(stderr, "[dvfs] cannot enumerate graphics clocks — disabled\n");
        g_failed = 1; return;
    }
    qsort(g_clocks, ngfx, sizeof(unsigned), cmp_desc);
    g_n_clocks = (int)ngfx;
    g_f_max    = g_clocks[0];
    /* The probe clock is a real cost, not just a measurement: every kernel that
     * runs during it is slowed, and on a long-running tenant those samples are the
     * latency tail. Probing at the middle of the range (810 MHz here, a 43% drop)
     * nearly doubles latency for a compute-bound kernel to learn something a much
     * gentler drop reveals just as well — with s ~ 1 a 25% drop still produces a
     * 33% slowdown, far above measurement noise. So probe just below f_max by
     * default, and let it be tuned. */
    g_f_probe  = 0;
    {
        double want = (double)g_f_max * g_lithos_cfg.dvfs_probe_frac;
        for (int i = 0; i < g_n_clocks; i++)
            if ((double)g_clocks[i] <= want) { g_f_probe = g_clocks[i]; break; }
        if (!g_f_probe) g_f_probe = g_clocks[g_n_clocks - 1];
        if (g_f_probe == g_f_max && g_n_clocks > 1) g_f_probe = g_clocks[1];
    }

    for (int i = 0; i < PW_STREAMS; i++)
        for (int j = 0; j < PW_OPS; j++) g_ops[i][j].sens = -1.0;

    g_enabled = 1;
    /* §5.6: "Initially, LithOS collects per-kernel metadata at maximum frequency,
     * forcing unseen kernels to run at max frequency." */
    if (set_clock(g_f_max) != 0) return;
    g_phase = PH_LEARN;
    PLOG("enabled: %d clocks, f_max=%u MHz, probe=%u MHz, slip=%.0f%%, "
         "%ld samples/op", g_n_clocks, g_f_max, g_f_probe, g_slip * 100.0, g_min_samples);
}

/* ------------------------------------------------------------------ */
/*  Learning                                                           */
/* ------------------------------------------------------------------ */
void power_note_measure(int slot, int op, double us) {
    if (!g_enabled || us <= 0) return;
    if (slot < 0 || slot >= PW_STREAMS || op < 0 || op >= PW_OPS) return;

    pthread_mutex_lock(&g_mtx);
    PwOp* o = &g_ops[slot][op];
    o->runtime_us     += us;
    g_total_runtime_us += us;
    g_measures++;

    /* Attribute the sample to whichever clock is currently applied. Samples taken
     * at any other clock are ignored: mixing them would corrupt both means. */
    if (g_f_cur == g_f_max) {
        o->l_max = o->n_max ? (o->l_max * o->n_max + us) / (o->n_max + 1) : us;
        o->n_max++;
    } else if (g_f_cur != g_f_probe) {
        /* At an applied clock: this is the feedback signal for the closed loop. */
        o->l_cur = o->n_cur ? (o->l_cur * o->n_cur + us) / (o->n_cur + 1) : us;
        o->n_cur++;
    }
    if (g_f_cur == g_f_probe) {
        o->l_probe = o->n_probe ? (o->l_probe * o->n_probe + us) / (o->n_probe + 1) : us;
        o->n_probe++;
        if (o->n_probe >= g_min_samples && o->l_max > 0) {
            /* s = slowdown / (f_max/f_probe - 1), clamped to [0,1]: below 0 is
             * measurement noise (it "sped up"), above 1 means something other than
             * the clock changed, and neither is a scaling behaviour we should
             * extrapolate from. */
            double slowdown = o->l_probe / o->l_max - 1.0;
            double drop     = (double)g_f_max / (double)g_f_probe - 1.0;
            double s        = drop > 1e-9 ? slowdown / drop : 0.0;
            if (s < 0.0) s = 0.0;
            if (s > 1.0) s = 1.0;
            o->sens = s;
        }
    }
    pthread_mutex_unlock(&g_mtx);
}

/* Aggregate sensitivity S = sum(w * s) over operators that have one, where w is
 * the operator's share of observed runtime. Operators still without a sensitivity
 * are treated as fully sensitive (s = 1): the conservative direction, since it
 * holds the frequency up. Returns -1 if too little is known to decide. */
static double aggregate_sensitivity(double* coverage_out) {
    double S = 0, covered = 0;
    if (g_total_runtime_us <= 0) return -1;
    for (int i = 0; i < PW_STREAMS; i++) {
        for (int j = 0; j < PW_OPS; j++) {
            PwOp* o = &g_ops[i][j];
            if (o->runtime_us <= 0) continue;
            double w = o->runtime_us / g_total_runtime_us;
            if (o->sens >= 0.0) { S += w * o->sens; covered += w; }
            else                  S += w * 1.0;
        }
    }
    if (coverage_out) *coverage_out = covered;
    return S;
}

/* The aggregate slowdown actually observed at the clock now applied, relative to
 * each operator's latency at f_max, weighted by runtime. This is the measured
 * counterpart of the model's prediction — and unlike the prediction it needs no
 * extrapolation. `*cov` reports how much of the runtime had enough samples to
 * contribute. Returns -1 if nothing does yet. Caller holds g_mtx. */
static double observed_slowdown(double* cov) {
    double num = 0, wsum = 0;
    for (int i = 0; i < PW_STREAMS; i++)
        for (int j = 0; j < PW_OPS; j++) {
            PwOp* o = &g_ops[i][j];
            if (o->l_max <= 0 || o->l_cur <= 0 || o->n_cur < g_min_samples) continue;
            double w = o->runtime_us;
            num  += w * (o->l_cur / o->l_max - 1.0);
            wsum += w;
        }
    if (cov) *cov = g_total_runtime_us > 0 ? wsum / g_total_runtime_us : 0;
    return wsum > 0 ? num / wsum : -1.0;
}

/* Index of `mhz` in the descending clock list, or -1. */
static int clock_index(unsigned mhz) {
    for (int i = 0; i < g_n_clocks; i++) if (g_clocks[i] == mhz) return i;
    return -1;
}

/* Enough learned at f_max to move on?
 *
 * Weighted by runtime, not counted per operator. Requiring EVERY operator to
 * reach the sample threshold never becomes true in practice: a warm-up kernel, or
 * any operator that runs once and never again, keeps one unsatisfied entry in the
 * table forever and pins the system in the learning phase. What matters is that
 * the operators actually accounting for the workload's time are known — which is
 * the same criterion the probe phase uses for coverage. */
static double learn_coverage(void) {
    if (g_total_runtime_us <= 0) return 0;
    double covered = 0;
    for (int i = 0; i < PW_STREAMS; i++)
        for (int j = 0; j < PW_OPS; j++) {
            PwOp* o = &g_ops[i][j];
            if (o->runtime_us > 0 && o->n_max >= g_min_samples)
                covered += o->runtime_us / g_total_runtime_us;
        }
    return covered;
}

void power_maybe_update(void) {
    if (!g_enabled) return;

    uint64_t now = lithos_now_ns();
    /* §5.6: switching is expensive (~50 ms), so transitions are rate-limited and
     * the learning period is deliberately long. */
    if (g_last_switch_ns &&
        (double)(now - g_last_switch_ns) / 1e6 < g_min_switch_ms) return;

    pthread_mutex_lock(&g_mtx);
    int phase = g_phase;
    long measures = g_measures;

    if (phase == PH_LEARN) {
        double cov = learn_coverage();
        if (measures >= g_min_samples && cov >= PW_COVERAGE) {
            g_phase = PH_PROBE;
            pthread_mutex_unlock(&g_mtx);
            PLOG("learned at f_max: %ld measurements, %.0f%% of runtime covered; "
                 "probing at %u MHz", measures, cov * 100.0, g_f_probe);
            set_clock(g_f_probe);
            return;
        }
        pthread_mutex_unlock(&g_mtx);
        return;
    }

    if (phase == PH_PROBE) {
        double covered = 0;
        double S = aggregate_sensitivity(&covered);
        /* Wait until the operators that account for most of the runtime have a
         * sensitivity; a decision taken on 10% of the workload is a guess. */
        if (S < 0 || covered < PW_COVERAGE) { pthread_mutex_unlock(&g_mtx); return; }
        g_phase = PH_APPLY;
        pthread_mutex_unlock(&g_mtx);

        /* f_final = f_max / (1 + slip/S)  (§5.6). S -> 0 (memory-bound) drives the
         * frequency to the bottom of the range; S -> 1 holds it near the top. */
        double f_final = (S > 1e-6) ? (double)g_f_max / (1.0 + g_slip / S)
                                    : (double)g_clocks[g_n_clocks - 1];

        /* Do not JUMP to it. §5.6's operational description is incremental — "at
         * first, a kernel is assumed to scale linearly, and its frequency is
         * reduced based on the configured k. Depending on the observed
         * performance, LithOS either further lowers the frequency or stops" — and
         * the linear assumption (s = 1, so S = 1) is the most conservative target
         * there is: f_max/(1+slip). Starting there and descending under
         * observation costs a few hundred milliseconds of convergence; starting at
         * an unvalidated extrapolation costs correctness. Measured on a real
         * kernel mix, the extrapolated target overshot a 10% latency budget by
         * six times. So take whichever of the two is the HIGHER clock and let the
         * closed loop below walk down from it. */
        double f_linear = (double)g_f_max / (1.0 + g_slip);
        double f_start  = f_final > f_linear ? f_final : f_linear;
        unsigned target = clock_at_or_below(f_start);
        PLOG("S=%.3f (coverage %.0f%%) slip=%.0f%% -> model %.0f MHz, "
             "linear-scaling %.0f MHz -> starting at %u MHz",
             S, covered * 100.0, g_slip * 100.0, f_final, f_linear, target);
        set_clock(target);
        return;
    }

    /* PH_APPLY: close the loop.
     *
     * The model's target is a FIRST-ORDER extrapolation from one probe, and §5.6
     * says so. Extrapolating it far from that probe is where it breaks: on a real
     * kernel mix, a target derived from a 25% clock drop overshot a 10% latency
     * budget by six times, because a mix of kernels does not slow down linearly
     * all the way down. Worse, re-deriving the aggregate sensitivity from samples
     * taken at the applied clock feeds the output back into the input, and the
     * clock walks steadily downwards (measured: S drifting 0.122 -> 0.100 over
     * successive decisions, dragging the clock with it).
     *
     * So the model is used ONCE, to pick a starting point, and from then on the
     * decision is made from what actually happened — which is what the paper
     * describes: "Depending on the observed performance, LithOS either further
     * lowers the frequency or stops after confirming linear behavior." We compare
     * the observed aggregate slowdown against the budget and step one supported
     * clock at a time: up if we have overshot, down only if there is real headroom.
     * Stepping rather than re-solving keeps the loop stable, and the rate limit
     * above bounds how fast it can move. */
    double cov = 0;
    double obs = observed_slowdown(&cov);
    pthread_mutex_unlock(&g_mtx);
    if (obs < 0 || cov < PW_COVERAGE) return;    /* not enough evidence yet */

    int idx = clock_index(g_f_cur);
    if (idx < 0) return;
    if (obs > g_slip && idx > 0) {
        PLOG("observed slowdown %.0f%% > budget %.0f%% -> raising to %u MHz",
             obs * 100.0, g_slip * 100.0, g_clocks[idx - 1]);
        set_clock(g_clocks[idx - 1]);
    } else if (obs < g_slip * PW_HEADROOM && idx + 1 < g_n_clocks) {
        PLOG("observed slowdown %.0f%% < budget %.0f%% -> lowering to %u MHz",
             obs * 100.0, g_slip * 100.0, g_clocks[idx + 1]);
        set_clock(g_clocks[idx + 1]);
    }
}

/* A tenant that exits must not leave the GPU pinned to its chosen clock — the
 * next process on the machine would silently inherit it. */
__attribute__((destructor)) static void power_atexit(void) { power_shutdown(); }

void power_shutdown(void) {
    if (!g_enabled) return;
    g_enabled = 0;
    if (p_nvmlDeviceResetGpuLockedClocks) p_nvmlDeviceResetGpuLockedClocks(g_dev);
    PLOG("reset GPU clocks to default");
    if (p_nvmlShutdown) p_nvmlShutdown();
}
