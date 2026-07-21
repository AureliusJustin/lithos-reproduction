/*
 * libcuda.so.1 wrapper glue (Section 6, "Interposition Architecture").
 *
 * When LibLithOS is installed *as* libcuda.so.1 (or placed ahead on
 * LD_LIBRARY_PATH), the CUDA runtime's dlopen("libcuda.so.1") resolves to us,
 * and dlsym for driver functions returns our overrides for the interposed
 * subset and the real driver's symbols (via our NEEDED dependency, patched to
 * the real libcuda.so) for everything else. This mirrors libsmctrl's wrapper.
 *
 * Only compiled into the wrapper build (-DLITHOS_WRAPPER).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <stdlib.h>

/* Some apps/libraries dlopen("libcuda.so") directly; redirect those to the
 * real driver so we don't recurse into ourselves. */
void* dlopen(const char* filename, int flags) {
    if (filename && (strcmp(filename, "libcuda.so") == 0 ||
                     strcmp(filename, "libcuda.so.1") == 0)) {
        /* Load the genuine driver by absolute path, bypassing our wrapper. */
        void* real = dlmopen(LM_ID_BASE, "/lib/x86_64-linux-gnu/libcuda.so.1", flags);
        if (real) return real;
    }
    return dlmopen(LM_ID_BASE, filename, flags);
}

/* Force CUDA to expose enough hardware channels under MPS (see paper Sec 6 /
 * libsmctrl): without this, MPS defaults to 2 connections and serialises. */
__attribute__((constructor)) static void lithos_wrapper_setup(void) {
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", "8", 0);
}
