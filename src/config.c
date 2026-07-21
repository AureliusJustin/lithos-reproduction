#define _GNU_SOURCE
#include "lithos.h"
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

LithosConfig g_lithos_cfg;
static pthread_once_t cfg_once = PTHREAD_ONCE_INIT;

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
    g_lithos_cfg.outstanding_limit_us  = envd("LITHOS_OUTSTANDING_US", 100.0);
    g_lithos_cfg.max_outstanding_atoms = envi("LITHOS_MAX_ATOMS", 8);
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
    g_lithos_cfg.enable_stealing       = envi("LITHOS_STEALING", 1);
    /* The transparent Prelude redirect is proven, but the Prelude->original
     * tail-transfer needs a compiler-emitted JUMP (see docs); gated off so the
     * library stays correct by default. */
    g_lithos_cfg.enable_jump           = envi("LITHOS_ATOM_JUMP", 0);
    /* On datacenter GPUs (A100/H100) the plain CALL-based transfer is expected
     * to work; the BRX byte-patch is a GA102 workaround attempt (and does not
     * actually function). Default to the CALL. */
    g_lithos_cfg.use_brx               = envi("LITHOS_BRX", 0);
    g_lithos_cfg.verbose               = envi("LITHOS_VERBOSE", 0);
}

void lithos_config_init(void) { pthread_once(&cfg_once, cfg_init_once); }

uint64_t lithos_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
