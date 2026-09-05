#define _GNU_SOURCE
/*
 * params.c — kernel argument layout recovery and deep-copy (see params.h).
 *
 * cuFuncGetParamInfo walks a CUfunction's parameters one index at a time,
 * reporting each one's offset and size, and fails on the first index past the
 * end. That gives us both the count and the total footprint. Measured on an
 * A100 / CUDA 12.8 the query costs a few hundred ns, so it is done once per
 * CUfunction and cached; the hot path is a single hash probe.
 */
#include "params.h"
#include "real.h"
#include "lithos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

/* cuFuncGetParamInfo arrived in CUDA 12.4. Resolve it dynamically so the library
 * still loads (and simply declines to buffer launches) against older drivers. */
static CUresult (*g_get_param_info)(CUfunction, size_t, size_t*, size_t*);
static int      g_probe_done;

/* ---- cache ---------------------------------------------------------------- *
 * Open-addressed, insert-only, keyed by CUfunction — the same shape the
 * atomizer's function sets use. A process loads at most a few thousand distinct
 * kernels, so 16k slots keep the table under a quarter full. */
#define PC_BITS 14
#define PC_SIZE (1u << PC_BITS)

typedef struct PEntry {
    CUfunction  f;          /* NULL = empty slot                              */
    int         ok;         /* 1 = layout valid, 0 = this kernel can't be
                             * buffered (negative results are cached too, or we
                             * would re-query it on every launch)             */
    ParamLayout lay;
} PEntry;

static PEntry*         g_cache;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static inline unsigned pc_hash(CUfunction f) {
    return (unsigned)(((uintptr_t)f * 11400714819323198485ull) >> (64 - PC_BITS));
}

/* Find the slot for `f`: either its entry or the empty one it would occupy.
 * Caller holds g_lock. Returns NULL only if the table is full. */
static PEntry* pc_slot(CUfunction f) {
    unsigned i = pc_hash(f);
    for (unsigned n = 0; n < PC_SIZE; n++, i = (i + 1) & (PC_SIZE - 1))
        if (!g_cache[i].f || g_cache[i].f == f) return &g_cache[i];
    return NULL;
}

/* Ask the driver for every parameter of `f`. Returns 1 and fills `lay` on
 * success; 0 if the kernel is unbufferable (too many parameters, or too many
 * bytes).
 *
 * The query reports one parameter per index and fails on the first index past
 * the end, which is also exactly what it does when the kernel has no parameters
 * at all — and, for that matter, if the driver simply declines to answer. Those
 * three cases are indistinguishable from here, so n == 0 is reported as-is and
 * the CALLER decides: it knows whether the application actually passed an
 * argument array, and refuses to buffer if the two disagree. Guessing "zero
 * parameters" for a kernel that has some would silently launch it with garbage. */
static int query_layout(CUfunction f, ParamLayout* lay) {
    memset(lay, 0, sizeof(*lay));

    for (int i = 0; i < LITHOS_MAX_PARAMS; i++) {
        size_t off = 0, sz = 0;
        if (g_get_param_info(f, (size_t)i, &off, &sz) != CUDA_SUCCESS) {
            lay->n = i;
            return 1;
        }
        if (off + sz > LITHOS_MAX_PARAM_BYTES) return 0;
        lay->off[i]  = off;
        lay->size[i] = sz;
        /* Parameters come back in increasing offset order, but take the max
         * rather than assuming it — `total` must cover the last byte written. */
        if (off + sz > lay->total) lay->total = off + sz;
    }
    return 0;                      /* more parameters than we are willing to copy */
}

const ParamLayout* params_layout(CUfunction f) {
    if (!f) return NULL;

    if (!g_probe_done) {
        pthread_mutex_lock(&g_lock);
        if (!g_probe_done) {
            g_get_param_info = (CUresult (*)(CUfunction, size_t, size_t*, size_t*))
                               lithos_real_sym("cuFuncGetParamInfo");
            g_cache = calloc(PC_SIZE, sizeof(PEntry));
            if (!g_get_param_info && g_lithos_cfg.verbose)
                fprintf(stderr, "[params] cuFuncGetParamInfo unavailable — "
                                "launches cannot be buffered\n");
            g_probe_done = 1;
        }
        pthread_mutex_unlock(&g_lock);
    }
    if (!g_get_param_info || !g_cache) return NULL;

    /* Fast path: an already-cached entry is immutable, so read it unlocked. The
     * `f` field is published after `lay`/`ok` are written (see below), so a
     * non-NULL `f` means the rest of the entry is complete. */
    PEntry* e = pc_slot(f);
    if (!e) return NULL;
    if (__atomic_load_n(&e->f, __ATOMIC_ACQUIRE) == f) return e->ok ? &e->lay : NULL;

    ParamLayout lay;
    int ok = query_layout(f, &lay);

    pthread_mutex_lock(&g_lock);
    e = pc_slot(f);                                  /* re-probe under the lock */
    if (e && !e->f) {
        e->lay = lay;
        e->ok  = ok;
        __atomic_store_n(&e->f, f, __ATOMIC_RELEASE);   /* publish last */
    }
    pthread_mutex_unlock(&g_lock);

    return ok ? &e->lay : NULL;
}

void params_pack(const ParamLayout* lay, void** params, void* buf, void** out_ptrs) {
    unsigned char* p = (unsigned char*)buf;
    for (int i = 0; i < lay->n; i++) {
        memcpy(p + lay->off[i], params[i], lay->size[i]);
        out_ptrs[i] = p + lay->off[i];
    }
}

/* ---- the `extra` path ----------------------------------------------------- *
 * An alternative to kernelParams: `extra` is a NULL-terminated list of
 * key/value pairs carrying one pre-packed argument blob and its size. Because
 * the size is stated explicitly, this form needs no layout query — but both the
 * pointer and the size must be present for the blob to be copyable. */

long params_extra_size(void** extra) {
    if (!extra) return 0;

    void*   ptr  = NULL;
    size_t  size = 0;
    int     have_ptr = 0, have_size = 0;

    for (int i = 0; extra[i] != CU_LAUNCH_PARAM_END; i++) {
        if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) { ptr  = extra[++i]; have_ptr = 1; }
        else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            size = *(size_t*)extra[++i]; have_size = 1;
        } else return -1;                 /* unknown key: don't guess, don't buffer */
    }

    if (!have_ptr && !have_size) return 0;                   /* empty list */
    if (!have_ptr || !have_size || !ptr) return -1;          /* unusable    */
    if (size > LITHOS_MAX_PARAM_BYTES)   return -1;          /* too large   */
    return (long)size;
}

void params_extra_pack(void** extra, size_t size, void* buf, void** out_extra) {
    void* src = NULL;
    for (int i = 0; extra[i] != CU_LAUNCH_PARAM_END; i++) {
        if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) src = extra[++i];
        else i++;
    }
    if (src) memcpy(buf, src, size);

    /* CU_LAUNCH_PARAM_BUFFER_SIZE points AT a size_t rather than carrying the
     * value, so that cell must outlive the enqueuing call as well. Park it just
     * past the blob at natural alignment — LITHOS_EXTRA_BYTES reserves the room. */
    size_t* size_cell = (size_t*)((unsigned char*)buf + ((size + 7u) & ~(size_t)7u));
    *size_cell = size;

    out_extra[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
    out_extra[1] = buf;
    out_extra[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
    out_extra[3] = size_cell;
    out_extra[4] = CU_LAUNCH_PARAM_END;
}
