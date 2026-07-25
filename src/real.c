#define _GNU_SOURCE
#include "real.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

RealDriver g_real;
static void* real_libcuda = NULL;
static pthread_once_t real_once = PTHREAD_ONCE_INIT;

/* Open the genuine libcuda.so.1 even if a LithOS wrapper occupies that name.
 * When LD_PRELOAD-ed, RTLD_NEXT works; if not, fall back to the on-disk lib. */
static void open_real_libcuda(void) {
    if (real_libcuda) return;
    /* The distro symlink points at the real driver; our wrapper (if installed
     * as libcuda.so.1) is bypassed by opening the versioned .so directly. */
    const char* candidates[] = {
        "/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "libcuda.so.1",
        NULL,
    };
    for (int i = 0; candidates[i]; i++) {
        real_libcuda = dlopen(candidates[i], RTLD_NOW | RTLD_GLOBAL);
        if (real_libcuda) return;
    }
}

void* lithos_real_sym(const char* name) {
    /* Preferred: the next definition in the link chain (real driver ahead of
     * us when we are LD_PRELOAD-ed). */
    void* p = dlsym(RTLD_NEXT, name);
    if (p) return p;
    /* Fallback: explicitly loaded real driver. */
    open_real_libcuda();
    if (real_libcuda) {
        p = dlsym(real_libcuda, name);
        if (p) return p;
    }
    /* Last resort: ask the driver's own resolver (post-11.3 symbol table). */
    if (g_real.cuGetProcAddress_v2) {
        void* fn = NULL;
        if (g_real.cuGetProcAddress_v2(name, &fn, CUDA_VERSION, 0, NULL) == CUDA_SUCCESS)
            return fn;
    }
    return NULL;
}

static void real_init_once(void) {
    memset(&g_real, 0, sizeof(g_real));
    /* Resolve the resolver first so lithos_real_sym can use it as a fallback. */
    g_real.cuGetProcAddress_v2 =
        (typeof(g_real.cuGetProcAddress_v2))dlsym(RTLD_NEXT, "cuGetProcAddress_v2");
    if (!g_real.cuGetProcAddress_v2) {
        open_real_libcuda();
        if (real_libcuda)
            g_real.cuGetProcAddress_v2 =
                (typeof(g_real.cuGetProcAddress_v2))dlsym(real_libcuda, "cuGetProcAddress_v2");
    }

    g_real.cuLaunchKernel            = (typeof(g_real.cuLaunchKernel))lithos_real_sym("cuLaunchKernel");
    g_real.cuStreamCreate            = (typeof(g_real.cuStreamCreate))lithos_real_sym("cuStreamCreate");
    g_real.cuStreamCreateWithPriority= (typeof(g_real.cuStreamCreateWithPriority))lithos_real_sym("cuStreamCreateWithPriority");
    g_real.cuStreamDestroy           = (typeof(g_real.cuStreamDestroy))lithos_real_sym("cuStreamDestroy_v2");
    if (!g_real.cuStreamDestroy)
        g_real.cuStreamDestroy       = (typeof(g_real.cuStreamDestroy))lithos_real_sym("cuStreamDestroy");
    g_real.cuStreamSynchronize       = (typeof(g_real.cuStreamSynchronize))lithos_real_sym("cuStreamSynchronize");
    g_real.cuCtxSynchronize          = (typeof(g_real.cuCtxSynchronize))lithos_real_sym("cuCtxSynchronize");
    g_real.cuInit                    = (typeof(g_real.cuInit))lithos_real_sym("cuInit");
    g_real.cuModuleLoad              = (typeof(g_real.cuModuleLoad))lithos_real_sym("cuModuleLoad");
    g_real.cuModuleLoadData          = (typeof(g_real.cuModuleLoadData))lithos_real_sym("cuModuleLoadData");
    g_real.cuModuleLoadDataEx        = (typeof(g_real.cuModuleLoadDataEx))lithos_real_sym("cuModuleLoadDataEx");
    g_real.cuModuleLoadFatBinary     = (typeof(g_real.cuModuleLoadFatBinary))lithos_real_sym("cuModuleLoadFatBinary");
    g_real.cuModuleGetFunction       = (typeof(g_real.cuModuleGetFunction))lithos_real_sym("cuModuleGetFunction");
    g_real.cuLibraryLoadData         = (typeof(g_real.cuLibraryLoadData))lithos_real_sym("cuLibraryLoadData");
    g_real.cuLibraryLoadFromFile     = (typeof(g_real.cuLibraryLoadFromFile))lithos_real_sym("cuLibraryLoadFromFile");
    g_real.cuLibraryGetKernel        = (typeof(g_real.cuLibraryGetKernel))lithos_real_sym("cuLibraryGetKernel");
    g_real.cuLibraryGetModule        = (typeof(g_real.cuLibraryGetModule))lithos_real_sym("cuLibraryGetModule");
    g_real.cuKernelGetFunction       = (typeof(g_real.cuKernelGetFunction))lithos_real_sym("cuKernelGetFunction");
    g_real.cuLaunchKernelEx          = (typeof(g_real.cuLaunchKernelEx))lithos_real_sym("cuLaunchKernelEx");
    g_real.cuLaunchCooperativeKernel = (typeof(g_real.cuLaunchCooperativeKernel))lithos_real_sym("cuLaunchCooperativeKernel");
    g_real.cuGetExportTable          = (typeof(g_real.cuGetExportTable))lithos_real_sym("cuGetExportTable");
    g_real.cuStreamIsCapturing       = (typeof(g_real.cuStreamIsCapturing))lithos_real_sym("cuStreamIsCapturing");
    g_real.cuDeviceGetAttribute      = (typeof(g_real.cuDeviceGetAttribute))lithos_real_sym("cuDeviceGetAttribute");
}

void lithos_real_init(void) {
    pthread_once(&real_once, real_init_once);
}
