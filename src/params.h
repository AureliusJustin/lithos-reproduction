/*
 * params.h — recovering a kernel's argument layout so launches can be BUFFERED.
 *
 * §5.2 requires that "on asynchronous CUDA calls like cuLaunchKernel, LithOS
 * enqueues the kernel and returns control to the application". Returning control
 * is the hard part: cuLaunchKernel takes `void** kernelParams`, an array of
 * POINTERS to argument values whose sizes appear nowhere in the call. They are
 * implicit in the kernel's signature. The driver only reads them during the call,
 * so once we return, the application is free to overwrite or free them — which
 * means a queued launch that still points at app memory is a use-after-free
 * waiting to happen.
 *
 * The way out is cuFuncGetParamInfo (CUDA 12.4+), which reports the byte offset
 * and size of each parameter of an already-loaded CUfunction. The driver knows
 * this from the cubin's .nv.info, so it works for closed-source kernels too —
 * cuBLAS, cuDNN, whatever the framework loaded. With the layout in hand we can
 * deep-copy the argument VALUES into the queue entry and hand the dispatcher a
 * rebuilt `void**` that points at our own storage.
 *
 * Everything is cached per CUfunction: the query is only cheap enough to do once.
 */
#ifndef LITHOS_PARAMS_H
#define LITHOS_PARAMS_H

#include <cuda.h>
#include <stddef.h>

/* Upper bounds on what we will buffer. Kernels past either limit fall back to
 * inline submission rather than being copied — correctness first, and both are
 * far above anything real (CUDA's own parameter limit is 4 KB pre-12.1). */
#define LITHOS_MAX_PARAMS     128
#define LITHOS_MAX_PARAM_BYTES 8192

typedef struct ParamLayout {
    int    n;                            /* number of parameters                */
    size_t total;                        /* bytes needed to hold them all       */
    size_t off[LITHOS_MAX_PARAMS];       /* byte offset of each in that buffer  */
    size_t size[LITHOS_MAX_PARAMS];      /* byte size of each                   */
} ParamLayout;

/* Layout for `f`, or NULL if it cannot be determined (driver too old, kernel too
 * large). The returned pointer is owned by the cache and stays valid for the
 * process lifetime. Thread-safe.
 *
 * A returned layout with n == 0 is AMBIGUOUS — it means the driver failed at
 * parameter 0, which happens both for a genuinely argument-less kernel and for a
 * kernel whose signature the driver would not report. Callers must treat n == 0
 * as "safe only if the application passed no arguments either". */
const ParamLayout* params_layout(CUfunction f);

/* Copy the argument VALUES that `params` points at into `buf` (>= layout->total
 * bytes) and fill `out_ptrs` with pointers into `buf`. After this the launch no
 * longer references any application memory. */
void params_pack(const ParamLayout* lay, void** params, void* buf, void** out_ptrs);

/* The `extra` path (CU_LAUNCH_PARAM_BUFFER_POINTER / _SIZE) passes one packed
 * blob with an EXPLICIT size, so it needs no layout query. Report that size, or
 * 0 if `extra` is NULL/empty. Returns -1 if `extra` is present but not in the
 * copyable pointer+size form, meaning the launch must not be buffered. */
long params_extra_size(void** extra);

/* Bytes `buf` must hold for params_extra_pack: the blob, plus an aligned cell
 * for the size (CU_LAUNCH_PARAM_BUFFER_SIZE points AT a size_t, so that value
 * has to outlive the enqueuing call too). */
#define LITHOS_EXTRA_BYTES(size) ((((size) + 7u) & ~(size_t)7u) + sizeof(size_t))

/* Rebuild an `extra` list into out_extra[5] (two key/value pairs plus the END
 * terminator), copying the original blob into `buf`. */
void params_extra_pack(void** extra, size_t size, void* buf, void** out_extra);

#endif /* LITHOS_PARAMS_H */
