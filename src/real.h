/*
 * Resolver for the *real* CUDA driver entry points that LibLithOS forwards to.
 *
 * LithOS interposes at the CUDA Driver API -- "the lowest common denominator"
 * (Section 6). We obtain the genuine driver functions two ways:
 *   1. dlsym(RTLD_NEXT, ...) when LibLithOS is LD_PRELOAD-ed ahead of libcuda.
 *   2. the driver's own cuGetProcAddress_v2, which is how the CUDA *runtime*
 *      resolves driver functions on CUDA 11.3+ (so intercepting it is how we
 *      stay transparent to PyTorch/TF/etc).
 */
#ifndef LITHOS_REAL_H
#define LITHOS_REAL_H

#include <cuda.h>

/* Real driver function pointers, lazily resolved. */
typedef struct RealDriver {
    CUresult (*cuGetProcAddress_v2)(const char*, void**, int, cuuint64_t,
                                    CUdriverProcAddressQueryResult*);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                               unsigned, unsigned, unsigned, unsigned,
                               CUstream, void**, void**);
    CUresult (*cuStreamCreate)(CUstream*, unsigned int);
    CUresult (*cuStreamCreateWithPriority)(CUstream*, unsigned int, int);
    CUresult (*cuStreamDestroy)(CUstream);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuCtxSynchronize)(void);
    CUresult (*cuInit)(unsigned int);
    CUresult (*cuModuleLoad)(CUmodule*, const char*);
    CUresult (*cuModuleLoadData)(CUmodule*, const void*);
    CUresult (*cuModuleLoadDataEx)(CUmodule*, const void*, unsigned int,
                                   CUjit_option*, void**);
    CUresult (*cuModuleLoadFatBinary)(CUmodule*, const void*);
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
    /* CUDA 12 Library API + Ex/cooperative launches -- the path the CUDA runtime
     * (PyTorch/TF/JAX/TensorRT) actually uses. */
    CUresult (*cuLibraryLoadData)(CUlibrary*, const void*, CUjit_option*, void**,
                                  unsigned int, CUlibraryOption*, void**, unsigned int);
    CUresult (*cuLibraryLoadFromFile)(CUlibrary*, const char*, CUjit_option*, void**,
                                      unsigned int, CUlibraryOption*, void**, unsigned int);
    CUresult (*cuLibraryGetKernel)(CUkernel*, CUlibrary, const char*);
    CUresult (*cuLibraryGetModule)(CUmodule*, CUlibrary);
    CUresult (*cuKernelGetFunction)(CUfunction*, CUkernel);
    CUresult (*cuLaunchKernelEx)(const CUlaunchConfig*, CUfunction, void**, void**);
    CUresult (*cuLaunchCooperativeKernel)(CUfunction, unsigned, unsigned, unsigned,
                                          unsigned, unsigned, unsigned, unsigned,
                                          CUstream, void**);
    CUresult (*cuGetExportTable)(const void**, const CUuuid*);
    CUresult (*cuStreamIsCapturing)(CUstream, CUstreamCaptureStatus*);
} RealDriver;

extern RealDriver g_real;

/* Resolve a single symbol from the real driver (dlsym RTLD_NEXT, then the
 * driver's cuGetProcAddress). Returns NULL if unavailable. */
void* lithos_real_sym(const char* name);

/* Populate g_real (idempotent). */
void lithos_real_init(void);

#endif /* LITHOS_REAL_H */
