#define _GNU_SOURCE
/*
 * barrier.c — generated forwarders that keep buffered launches in order.
 *
 * Each entry in barrier.def becomes a driver entry point that drains the
 * relevant launch queue and then calls the real one. See barrier.def for why
 * each call is on the list.
 *
 * RESOLVING THE REAL FUNCTION. Two sources, in priority order:
 *
 *   1. whatever cuGetProcAddress handed back for this symbol. This matters for
 *      the per-thread-default-stream ABI: a process built with
 *      --default-stream per-thread asks the driver for "cuMemcpyAsync" and gets
 *      cuMemcpyAsync_ptsz, which treats stream 0 as the per-thread stream rather
 *      than the legacy one. Forwarding to the plain symbol instead would quietly
 *      change which stream the call lands on.
 *   2. failing that, dlsym — the LD_PRELOAD path, where the application linked
 *      against the symbol directly and gets whichever ABI it was built for.
 */
#include "barrier.h"
#include "dispatch.h"
#include "real.h"
#include "lithos.h"
#include <cuda.h>
#include <stddef.h>
#include <string.h>

/* One record per generated forwarder, so cuGetProcAddress can both hand out our
 * wrapper and stash the real pointer it was about to return. */
static BarrierFn g_fns[] = {
#define BARRIER_S(sym, params, args, strm) { #sym, NULL, NULL },
#define BARRIER_A(sym, params, args)       { #sym, NULL, NULL },
#include "barrier.def"
#undef BARRIER_S
#undef BARRIER_A
};
static const int g_n_fns = (int)(sizeof(g_fns) / sizeof(g_fns[0]));

/* Index in g_fns, assigned in the same order the table above was built. */
enum {
#define BARRIER_S(sym, params, args, strm) BIDX_##sym,
#define BARRIER_A(sym, params, args)       BIDX_##sym,
#include "barrier.def"
#undef BARRIER_S
#undef BARRIER_A
    BIDX__COUNT
};

/* Real function for slot `i`: the one cuGetProcAddress gave us, else dlsym. */
static void* real_of(int i) {
    void* p = __atomic_load_n(&g_fns[i].real, __ATOMIC_ACQUIRE);
    if (!p) {
        p = lithos_real_sym(g_fns[i].name);
        __atomic_store_n(&g_fns[i].real, p, __ATOMIC_RELEASE);
    }
    return p;
}

/* The forwarders. dispatch_pending() is a single relaxed load and false whenever
 * nothing is buffered — which is always, with the dispatcher off — so an
 * un-deferred process pays nothing measurable for any of this. */
#define BARRIER_S(sym, params, args, strm)                                     \
    CUresult sym params {                                                      \
        if (dispatch_pending()) dispatch_drain(strm);                          \
        CUresult (*real) params = (CUresult (*) params)real_of(BIDX_##sym);    \
        return real ? real args : CUDA_ERROR_NOT_SUPPORTED;                    \
    }
#define BARRIER_A(sym, params, args)                                           \
    CUresult sym params {                                                      \
        if (dispatch_pending()) dispatch_drain_all();                          \
        CUresult (*real) params = (CUresult (*) params)real_of(BIDX_##sym);    \
        return real ? real args : CUDA_ERROR_NOT_SUPPORTED;                    \
    }
#include "barrier.def"
#undef BARRIER_S
#undef BARRIER_A

/* Fill in each record's wrapper pointer once, now that the functions exist. */
static void bind_wrappers(void) {
#define BARRIER_S(sym, params, args, strm) g_fns[BIDX_##sym].wrapper = (void*)sym;
#define BARRIER_A(sym, params, args)       g_fns[BIDX_##sym].wrapper = (void*)sym;
#include "barrier.def"
#undef BARRIER_S
#undef BARRIER_A
}

BarrierFn* barrier_lookup(const char* name) {
    static int bound;
    if (!bound) { bind_wrappers(); bound = 1; }

    for (int i = 0; i < g_n_fns; i++)
        if (strcmp(g_fns[i].name, name) == 0) return &g_fns[i];

    /* cuGetProcAddress is asked for the UNVERSIONED name ("cuMemcpyHtoD") and
     * the driver picks the ABI from the requested CUDA version; our table holds
     * the exported, versioned symbol. Try the versioned form before giving up. */
    size_t n = strlen(name);
    if (n && n < 120) {
        char buf[128];
        memcpy(buf, name, n);
        memcpy(buf + n, "_v2", 4);
        for (int i = 0; i < g_n_fns; i++)
            if (strcmp(g_fns[i].name, buf) == 0) return &g_fns[i];
    }
    return NULL;
}
