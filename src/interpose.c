/*
 * LibLithOS -- CUDA Driver API interposition (Section 5.2, Section 6).
 *
 * LithOS "interposes at the driver level, providing a dynamically linked
 * library, LibLithOS, that mimics the native CUDA library." Applications
 * interact with LithOS instead of the driver, while CUDA call semantics are
 * preserved. LithOS implements only a small subset of the Driver API (e.g.
 * cuLaunchKernel, cuStreamCreate); everything else is forwarded verbatim.
 *
 * Two interposition paths are supported so the layer is transparent to the
 * whole ML stack:
 *   (a) Symbol interposition: apps/tools linked directly against libcuda find
 *       our exported cuLaunchKernel/etc. first (LD_PRELOAD or wrapper install).
 *   (b) cuGetProcAddress interception: the CUDA *runtime* (PyTorch, TF, ...)
 *       resolves driver functions through cuGetProcAddress_v2; we return our
 *       wrappers for the calls we override and the real pointer otherwise.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>

#include "real.h"
#include "lithos_sched.h"
#include "lithos.h"
#include "atomizer.h"
#include "predict.h"

#define LOG(...) do { if (g_lithos_cfg.verbose) { \
    fprintf(stderr, "[lithos] " __VA_ARGS__); fputc('\n', stderr); } } while (0)

/* LithOS "builds on top of MPS" (Section 6) so tenants execute concurrently
 * across contexts. Ensure the per-user MPS control daemon is running BEFORE this
 * process creates its CUDA context (this runs from ensure_init, i.e. on the first
 * intercepted driver call, well before cuCtxCreate). Idempotent and shared: it
 * reuses an existing daemon (default pipe /tmp/nvidia-mps) and only starts one if
 * none is up. Disable with LITHOS_MPS=0. */
static void lithos_mps_ensure(void) {
    if (getenv("LITHOS_MPS") && atoi(getenv("LITHOS_MPS")) == 0) return;
    /* Never recurse into the MPS control/server processes themselves. */
    char comm[64] = {0}; FILE* cf = fopen("/proc/self/comm", "r");
    if (cf) { if (!fgets(comm, sizeof(comm), cf)) comm[0] = 0; fclose(cf); }
    if (strstr(comm, "mps")) return;

    setenv("CUDA_DEVICE_MAX_CONNECTIONS", "8", 0);   /* enough MPS hw channels */
    const char* pipe = getenv("CUDA_MPS_PIPE_DIRECTORY");
    char ctl[600]; snprintf(ctl, sizeof(ctl), "%s/control", pipe ? pipe : "/tmp/nvidia-mps");
    if (access(ctl, F_OK) == 0) { LOG("MPS already running (%s)", ctl); return; }

    int rc = system("nvidia-cuda-mps-control -d >/dev/null 2>&1");
    (void)rc;
    for (int i = 0; i < 100 && access(ctl, F_OK) != 0; i++) usleep(20000);  /* wait <=2s */
    LOG("MPS control daemon %s (%s)", access(ctl, F_OK) == 0 ? "started" : "NOT started", ctl);
}

static void ensure_init(void) {
    static int done = 0;
    if (__atomic_load_n(&done, __ATOMIC_ACQUIRE)) return;
    lithos_config_init();
    lithos_mps_ensure();
    lithos_real_init();
    lithos_sched_init();
    __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
    LOG("LibLithOS initialised (atomizer=%d stealing=%d atom_dur=%.0fus)",
        g_lithos_cfg.enable_atomizer, g_lithos_cfg.enable_stealing,
        g_lithos_cfg.atom_duration_us);
}

/* ------------------------------------------------------------------ */
/*  Overridden Driver API entry points                                */
/* ------------------------------------------------------------------ */

CUresult cuInit(unsigned int flags) {
    ensure_init();
    CUresult r = g_real.cuInit ? g_real.cuInit(flags) : CUDA_ERROR_NOT_INITIALIZED;
    LOG("cuInit(%u) = %d", flags, r);
    return r;
}

CUresult cuStreamCreate(CUstream* phStream, unsigned int flags) {
    ensure_init();
    CUresult r = g_real.cuStreamCreate(phStream, flags);
    if (r == CUDA_SUCCESS) {
        /* "A launch queue is created when an application creates a stream via
         *  cuStreamCreate." (Section 5.2) */
        lithos_stream_created(*phStream, 0);
        LOG("cuStreamCreate -> %p (launch queue created)", (void*)*phStream);
    }
    return r;
}

CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) {
    ensure_init();
    CUresult r = g_real.cuStreamCreateWithPriority(phStream, flags, priority);
    if (r == CUDA_SUCCESS) {
        lithos_stream_created(*phStream, priority);
        LOG("cuStreamCreateWithPriority -> %p (prio %d)", (void*)*phStream, priority);
    }
    return r;
}

CUresult cuStreamDestroy_v2(CUstream hStream) {
    ensure_init();
    lithos_stream_sync(hStream);        /* drain the launch queue first */
    lithos_stream_destroyed(hStream);
    return g_real.cuStreamDestroy(hStream);
}
/* cuda.h #defines cuStreamDestroy -> cuStreamDestroy_v2, so the _v2 definition
 * above already services callers of cuStreamDestroy. */

CUresult cuLaunchKernel(CUfunction f,
                        unsigned gx, unsigned gy, unsigned gz,
                        unsigned bx, unsigned by, unsigned bz,
                        unsigned shmem, CUstream stream,
                        void** params, void** extra) {
    ensure_init();
    /* "On asynchronous CUDA calls like cuLaunchKernel, LithOS enqueues the
     *  kernel and returns control to the application." (Section 5.2) */
    return lithos_submit_launch(f, gx, gy, gz, bx, by, bz, shmem, stream, params, extra);
}
/* Per-thread-default-stream alias used by the runtime. */
CUresult cuLaunchKernel_ptsz(CUfunction f,
                             unsigned gx, unsigned gy, unsigned gz,
                             unsigned bx, unsigned by, unsigned bz,
                             unsigned shmem, CUstream stream,
                             void** params, void** extra) {
    return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shmem, stream, params, extra);
}

/* ------------------------------------------------------------------ */
/*  Module loading -- the Kernel Atomizer's transparent hook           */
/* ------------------------------------------------------------------ */
/* We splice a per-block range-check prologue into every kernel of the loaded
 * cubin (atomize_splice.c), so a later per-atom relaunch runs only the atom's
 * blocks and skips the rest -- fall-through, no control transfer. If the image
 * isn't a raw cubin (fatbin/PTX) or splicing fails, we load it verbatim. */
static CUresult load_data_common(CUmodule* m, const void* image,
                                 unsigned int n, CUjit_option* o, void** ov) {
    void* spl = NULL; size_t splz = 0; int atomized = 0; void* names = NULL;
    atomizer_intercept_cubin(image, &spl, &splz, &atomized, &names);
    const void* use = atomized ? spl : image;
    CUresult r = o ? g_real.cuModuleLoadDataEx(m, use, n, o, ov)
                   : g_real.cuModuleLoadData(m, use);
    if (atomized) {
        if (r == CUDA_SUCCESS) { atomizer_register_module(*m, names);
            LOG("cuModuleLoadData: atomized module %p", (void*)*m); }
        else { /* spliced image rejected -> fall back to the original, unatomized */
            r = o ? g_real.cuModuleLoadDataEx(m, image, n, o, ov)
                  : g_real.cuModuleLoadData(m, image);
            LOG("cuModuleLoadData: spliced load failed, loaded original");
        }
        free(spl);
    }
    return r;
}

CUresult cuModuleLoadData(CUmodule* m, const void* image) {
    ensure_init();
    return load_data_common(m, image, 0, NULL, NULL);
}
CUresult cuModuleLoadDataEx(CUmodule* m, const void* image, unsigned int n,
                            CUjit_option* o, void** ov) {
    ensure_init();
    return load_data_common(m, image, n, o, ov);
}
CUresult cuModuleLoad(CUmodule* m, const char* fname) {
    ensure_init();
    /* Read the file and route through the cubin splicer. */
    FILE* f = fopen(fname, "rb");
    if (!f) return g_real.cuModuleLoad(m, fname);
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = malloc(sz + 1);           /* +1: PTX images must be NUL-terminated */
    if (!buf || fread(buf, 1, sz, f) != (size_t)sz) { if(buf) free(buf); fclose(f);
        return g_real.cuModuleLoad(m, fname); }
    buf[sz] = 0;
    fclose(f);
    CUresult r = load_data_common(m, buf, 0, NULL, NULL);
    free(buf);
    return r;
}

CUresult cuModuleGetFunction(CUfunction* pf, CUmodule m, const char* name) {
    ensure_init();
    CUresult r = g_real.cuModuleGetFunction(pf, m, name);
    if (r == CUDA_SUCCESS) atomizer_note_get_function(*pf, m, name);
    return r;
}

CUresult cuModuleLoadFatBinary(CUmodule* m, const void* fatCubin) {
    ensure_init();
    void* spl = NULL; size_t splz = 0; int atomized = 0; void* names = NULL;
    atomizer_intercept_cubin(fatCubin, &spl, &splz, &atomized, &names);
    CUresult r;
    if (atomized) {
        r = g_real.cuModuleLoadData(m, spl);          /* spliced -> raw cubin */
        if (r == CUDA_SUCCESS) { atomizer_register_module(*m, names);
            LOG("cuModuleLoadFatBinary: atomized module %p", (void*)*m); }
        else r = g_real.cuModuleLoadFatBinary(m, fatCubin);
        free(spl);
    } else r = g_real.cuModuleLoadFatBinary(m, fatCubin);
    return r;
}

/* ------------------------------------------------------------------ */
/*  CUDA 12 Library API -- the CUDA runtime's actual loading path       */
/* ------------------------------------------------------------------ */
CUresult cuLibraryLoadData(CUlibrary* lib, const void* code,
                           CUjit_option* jo, void** jv, unsigned int nj,
                           CUlibraryOption* lo, void** lv, unsigned int nl) {
    ensure_init();
    void* spl = NULL; size_t splz = 0; int atomized = 0; void* names = NULL;
    atomizer_intercept_cubin(code, &spl, &splz, &atomized, &names);
    const void* use = atomized ? spl : code;
    CUresult r = g_real.cuLibraryLoadData(lib, use, jo, jv, nj, lo, lv, nl);
    if (atomized) {
        if (r == CUDA_SUCCESS) { atomizer_register_library(*lib, names);
            LOG("cuLibraryLoadData: atomized library %p", (void*)*lib); }
        else { r = g_real.cuLibraryLoadData(lib, code, jo, jv, nj, lo, lv, nl);
            LOG("cuLibraryLoadData: spliced load failed, loaded original"); }
        free(spl);
    }
    return r;
}
CUresult cuLibraryLoadFromFile(CUlibrary* lib, const char* fn,
                               CUjit_option* jo, void** jv, unsigned int nj,
                               CUlibraryOption* lo, void** lv, unsigned int nl) {
    ensure_init();
    FILE* f = fopen(fn, "rb");
    if (!f) return g_real.cuLibraryLoadFromFile(lib, fn, jo, jv, nj, lo, lv, nl);
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = malloc(sz + 1);
    if (!buf || fread(buf, 1, sz, f) != (size_t)sz) { if(buf) free(buf); fclose(f);
        return g_real.cuLibraryLoadFromFile(lib, fn, jo, jv, nj, lo, lv, nl); }
    buf[sz] = 0; fclose(f);
    CUresult r = cuLibraryLoadData(lib, buf, jo, jv, nj, lo, lv, nl);
    free(buf);
    return r;
}
CUresult cuLibraryGetKernel(CUkernel* pk, CUlibrary lib, const char* name) {
    ensure_init();
    CUresult r = g_real.cuLibraryGetKernel(pk, lib, name);
    if (r == CUDA_SUCCESS) atomizer_note_get_kernel(*pk, lib, name);
    return r;
}
CUresult cuKernelGetFunction(CUfunction* pf, CUkernel k) {
    ensure_init();
    CUresult r = g_real.cuKernelGetFunction(pf, k);
    if (r == CUDA_SUCCESS) { atomizer_note_kernel_function(*pf, k);
        LOG("cuKernelGetFunction kernel=%p -> func=%p", (void*)k, (void*)*pf); }
    return r;
}
CUresult cuLibraryGetModule(CUmodule* pm, CUlibrary lib) {
    ensure_init();
    CUresult r = g_real.cuLibraryGetModule(pm, lib);
    if (r == CUDA_SUCCESS) atomizer_note_library_module(*pm, lib);
    return r;
}

/* ------------------------------------------------------------------ */
/*  Ex / cooperative launches                                          */
/* ------------------------------------------------------------------ */
CUresult cuLaunchKernelEx(const CUlaunchConfig* cfg, CUfunction f,
                          void** params, void** extra) {
    ensure_init();
    return lithos_submit_launch_ex(cfg, f, params, extra);
}
CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig* cfg, CUfunction f,
                               void** params, void** extra) {
    return cuLaunchKernelEx(cfg, f, params, extra);
}
CUresult cuLaunchCooperativeKernel(CUfunction f,
                                   unsigned gx, unsigned gy, unsigned gz,
                                   unsigned bx, unsigned by, unsigned bz,
                                   unsigned shmem, CUstream stream, void** params) {
    ensure_init();
    return lithos_submit_launch_coop(f, gx, gy, gz, bx, by, bz, shmem, stream, params);
}
CUresult cuLaunchCooperativeKernel_ptsz(CUfunction f,
                                        unsigned gx, unsigned gy, unsigned gz,
                                        unsigned bx, unsigned by, unsigned bz,
                                        unsigned shmem, CUstream stream, void** params) {
    return cuLaunchCooperativeKernel(f, gx, gy, gz, bx, by, bz, shmem, stream, params);
}

/* The CUDA runtime resolves its internal launch/load entry points through
 * cuGetExportTable (opaque per-UUID function tables), bypassing the public API.
 * Log every table it requests so we can locate the launch/load slots to hook. */
CUresult cuGetExportTable(const void** ppTable, const CUuuid* id) {
    ensure_init();
    CUresult r = g_real.cuGetExportTable(ppTable, id);
    if (getenv("LITHOS_TRACE_ET") && r == CUDA_SUCCESS) {
        const unsigned char* u = (const unsigned char*)id->bytes;
        fprintf(stderr, "[et] table %p uuid ", *ppTable);
        for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", u[i]);
        fprintf(stderr, "\n");
    }
    return r;
}

CUresult cuStreamSynchronize(CUstream hStream) {
    ensure_init();
    return lithos_stream_sync(hStream);
}
CUresult cuStreamSynchronize_ptsz(CUstream hStream) { return cuStreamSynchronize(hStream); }

CUresult cuCtxSynchronize(void) {
    ensure_init();
    return lithos_ctx_sync();
}

/* ------------------------------------------------------------------ */
/*  CUDA graph subgraph-scheduling interception (LITHOS_GRAPH_SUBGRAPHS)*/
/* ------------------------------------------------------------------ */
/* cuda.h version-mangles these (e.g. cuGraphInstantiate -> cuGraphInstantiateWithFlags,
 * cuGraphLaunch -> __CUDA_API_PTSZ(...)); undo so we can define each ABI by name. */
#undef cuGraphInstantiate
#undef cuGraphInstantiateWithFlags
#undef cuGraphLaunch
#undef cuGraphExecDestroy
/* Real graph entry points, resolved lazily (we forward when subgraph mode is off
 * or an exec isn't one we partitioned). */
static CUresult (*rg_instantiate)(CUgraphExec*, CUgraph, unsigned long long);
static CUresult (*rg_instantiate_v2)(CUgraphExec*, CUgraph, CUgraphNode*, char*, size_t);
static CUresult (*rg_launch)(CUgraphExec, CUstream);
static CUresult (*rg_exec_destroy)(CUgraphExec);
static void graph_resolve(void) {
    if (rg_launch) return;
    rg_instantiate    = (typeof(rg_instantiate))lithos_real_sym("cuGraphInstantiateWithFlags");
    rg_instantiate_v2 = (typeof(rg_instantiate_v2))lithos_real_sym("cuGraphInstantiate_v2");
    rg_launch         = (typeof(rg_launch))lithos_real_sym("cuGraphLaunch");
    rg_exec_destroy   = (typeof(rg_exec_destroy))lithos_real_sym("cuGraphExecDestroy");
}

CUresult cuGraphInstantiateWithFlags(CUgraphExec* pExec, CUgraph graph, unsigned long long flags) {
    ensure_init(); graph_resolve();
    if (lithos_graph_instantiate(pExec, graph, flags)) return CUDA_SUCCESS;
    return rg_instantiate(pExec, graph, flags);
}
/* cuGraphInstantiate (the base name) maps to different ABIs across CUDA versions;
 * cover the two we see: the flags form and the v2 (node/log-buffer) form. */
CUresult cuGraphInstantiate(CUgraphExec* pExec, CUgraph graph, unsigned long long flags) {
    ensure_init(); graph_resolve();
    if (lithos_graph_instantiate(pExec, graph, flags)) return CUDA_SUCCESS;
    return rg_instantiate(pExec, graph, flags);
}
CUresult cuGraphInstantiate_v2(CUgraphExec* pExec, CUgraph graph, CUgraphNode* errNode, char* logBuf, size_t bufSz) {
    ensure_init(); graph_resolve();
    if (lithos_graph_instantiate(pExec, graph, 0)) return CUDA_SUCCESS;
    return rg_instantiate_v2(pExec, graph, errNode, logBuf, bufSz);
}
CUresult cuGraphLaunch(CUgraphExec exec, CUstream stream) {
    ensure_init(); graph_resolve();
    if (lithos_graph_launch(exec, stream)) return CUDA_SUCCESS;
    return rg_launch(exec, stream);
}
CUresult cuGraphLaunch_ptsz(CUgraphExec exec, CUstream stream) { return cuGraphLaunch(exec, stream); }
CUresult cuGraphExecDestroy(CUgraphExec exec) {
    ensure_init(); graph_resolve();
    if (lithos_graph_exec_destroy(exec)) return CUDA_SUCCESS;
    return rg_exec_destroy(exec);
}

/* ---- CUDA graph capture window -------------------------------------------
 * The predictor's Tracker thread must issue no CUDA calls while a capture is
 * open, or it can invalidate it. Watching launches is not enough: the window
 * opens at cuStreamBeginCapture, which may be followed by app work we never see.
 * Intercepting the capture calls themselves closes the window exactly. */
#undef cuStreamBeginCapture
#undef cuStreamEndCapture
static CUresult (*rg_begin_capture)(CUstream, CUstreamCaptureMode);
static CUresult (*rg_end_capture)(CUstream, CUgraph*);

CUresult cuStreamBeginCapture_v2(CUstream s, CUstreamCaptureMode mode) {
    ensure_init();
    if (!rg_begin_capture)
        rg_begin_capture = (typeof(rg_begin_capture))lithos_real_sym("cuStreamBeginCapture_v2");
    predict_capture_begin();                 /* park the Tracker for the duration */
    return rg_begin_capture(s, mode);
}
CUresult cuStreamBeginCapture(CUstream s, CUstreamCaptureMode mode) {
    return cuStreamBeginCapture_v2(s, mode);
}
CUresult cuStreamEndCapture(CUstream s, CUgraph* g) {
    ensure_init();
    if (!rg_end_capture)
        rg_end_capture = (typeof(rg_end_capture))lithos_real_sym("cuStreamEndCapture");
    CUresult r = rg_end_capture(s, g);
    predict_capture_end();                   /* Tracker may resume */
    return r;
}

/* Special-kernel support (§6): report the tenant's ALLOCATED SM count for
 * CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, so cross-block-sync / persistent
 * kernels that size themselves to the device see only their TPC partition.
 * Every other attribute is forwarded unchanged. */
CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
    ensure_init();
    CUresult r = g_real.cuDeviceGetAttribute(pi, attrib, dev);
    if (r == CUDA_SUCCESS && pi && attrib == CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) {
        int alloc = lithos_allocated_sms();
        if (alloc > 0 && alloc < *pi) {
            if (getenv("LITHOS_VERBOSE")) fprintf(stderr, "[interpose] MULTIPROCESSOR_COUNT %d -> %d (tenant partition)\n", *pi, alloc);
            *pi = alloc;
        }
    }
    return r;
}

/* ------------------------------------------------------------------ */
/*  cuGetProcAddress interception (transparent to the CUDA runtime)   */
/* ------------------------------------------------------------------ */

struct override { const char* name; void* fn; };
#undef cuGetProcAddress
CUresult cuGetProcAddress_v2(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
CUresult cuGetProcAddress(const char*, void**, int, cuuint64_t);
static const struct override overrides[] = {
    /* Return OURSELVES for the resolver: the CUDA runtime asks the driver for
     * cuGetProcAddress once, then routes every other lookup through whatever it
     * gets back. If we hand it the real resolver, all launch/load calls bypass
     * us; handing it our own wrapper keeps every subsequent lookup interposed. */
    { "cuGetProcAddress",           (void*)cuGetProcAddress },
    { "cuGetProcAddress_v2",        (void*)cuGetProcAddress_v2 },
    { "cuInit",                     (void*)cuInit },
    { "cuStreamCreate",             (void*)cuStreamCreate },
    { "cuStreamCreateWithPriority", (void*)cuStreamCreateWithPriority },
    { "cuStreamDestroy",            (void*)cuStreamDestroy },
    { "cuStreamDestroy_v2",         (void*)cuStreamDestroy_v2 },
    { "cuLaunchKernel",             (void*)cuLaunchKernel },
    { "cuLaunchKernel_ptsz",        (void*)cuLaunchKernel_ptsz },
    { "cuStreamSynchronize",        (void*)cuStreamSynchronize },
    { "cuStreamSynchronize_ptsz",   (void*)cuStreamSynchronize_ptsz },
    { "cuCtxSynchronize",           (void*)cuCtxSynchronize },
    { "cuGetExportTable",           (void*)cuGetExportTable },
    { "cuModuleLoad",               (void*)cuModuleLoad },
    { "cuModuleLoadData",           (void*)cuModuleLoadData },
    { "cuModuleLoadDataEx",         (void*)cuModuleLoadDataEx },
    { "cuModuleLoadFatBinary",      (void*)cuModuleLoadFatBinary },
    { "cuModuleGetFunction",        (void*)cuModuleGetFunction },
    { "cuLibraryLoadData",          (void*)cuLibraryLoadData },
    { "cuLibraryLoadFromFile",      (void*)cuLibraryLoadFromFile },
    { "cuLibraryGetKernel",         (void*)cuLibraryGetKernel },
    { "cuLibraryGetModule",         (void*)cuLibraryGetModule },
    { "cuKernelGetFunction",        (void*)cuKernelGetFunction },
    { "cuLaunchKernelEx",           (void*)cuLaunchKernelEx },
    { "cuLaunchKernelEx_ptsz",      (void*)cuLaunchKernelEx_ptsz },
    { "cuLaunchCooperativeKernel",  (void*)cuLaunchCooperativeKernel },
    { "cuLaunchCooperativeKernel_ptsz", (void*)cuLaunchCooperativeKernel_ptsz },
    { "cuGraphInstantiate",         (void*)cuGraphInstantiate },
    { "cuGraphInstantiateWithFlags",(void*)cuGraphInstantiateWithFlags },
    { "cuGraphInstantiate_v2",      (void*)cuGraphInstantiate_v2 },
    { "cuGraphLaunch",              (void*)cuGraphLaunch },
    { "cuGraphExecDestroy",         (void*)cuGraphExecDestroy },
    { "cuDeviceGetAttribute",       (void*)cuDeviceGetAttribute },
    { "cuStreamBeginCapture",       (void*)cuStreamBeginCapture },
    { "cuStreamBeginCapture_v2",    (void*)cuStreamBeginCapture_v2 },
    { "cuStreamEndCapture",         (void*)cuStreamEndCapture },
    { NULL, NULL },
};

CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion,
                             cuuint64_t flags, CUdriverProcAddressQueryResult* st) {
    ensure_init();
    /* First let the driver resolve normally so unknown symbols still work and
     * so `st` is filled in correctly. */
    CUresult r = g_real.cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, st);
    if (r != CUDA_SUCCESS || !pfn) return r;
    if (getenv("LITHOS_TRACE_GPA")) fprintf(stderr, "[gpa] %s\n", symbol);
    for (const struct override* o = overrides; o->name; o++) {
        if (strcmp(symbol, o->name) == 0) {
            *pfn = o->fn;   /* hand the app *our* wrapper */
            LOG("cuGetProcAddress(%s) -> LithOS wrapper", symbol);
            break;
        }
    }
    return r;
}

/* The v1 ABI (no symbolStatus) forwards into the v2 path. cuda.h #defines
 * cuGetProcAddress -> cuGetProcAddress_v2, so undo that to expose both ABIs. */
#undef cuGetProcAddress
CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion,
                          cuuint64_t flags) {
    return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, NULL);
}
