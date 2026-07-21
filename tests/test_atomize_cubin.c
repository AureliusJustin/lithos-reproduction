/* End-to-end Kernel Atomizer test through the full LithOS library.
 *
 * Loads a real CUBIN via cuModuleLoad, so LibLithOS's module hook splices a
 * per-block range-check prologue into the kernel (atomize_splice.c). Each
 * cuLaunchKernel is then transparently split into atoms -- disjoint block ranges
 * relaunched over the full grid -- and the kernel writes out[b]=b+1000 for every
 * block. If any atom dropped or duplicated a block the check below catches it.
 *
 * Run under: LD_PRELOAD=build/liblithos_full.so, arg1=grid, arg2=cubin path.
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define CK(x) do{CUresult r=(x); if(r){const char*s;cuGetErrorName(r,&s); \
    fprintf(stderr,"%s: %s\n",#x,s); return 1;}}while(0)

int main(int argc, char** argv) {
    int G = argc > 1 ? atoi(argv[1]) : 256;
    const char* cubin = argc > 2 ? argv[2] : "build/atomize_mark.cubin";
    int T = 32;

    CK(cuInit(0));
    CUdevice d; CK(cuDeviceGet(&d, 0));
    CUcontext ctx; CK(cuCtxCreate(&ctx, 0, d));
    CUmodule m; CK(cuModuleLoad(&m, cubin));
    CUfunction fn; CK(cuModuleGetFunction(&fn, m, "mark"));
    CUdeviceptr out; CK(cuMemAlloc(&out, (size_t)G * 4));

    for (int rep = 0; rep < 4; rep++) {
        CK(cuMemsetD32(out, 0xffffffff, G));
        void* a[] = { &out };
        CK(cuLaunchKernel(fn, G,1,1, T,1,1, 0, 0, a, NULL));
        CK(cuCtxSynchronize());
        int* h = malloc((size_t)G * 4); CK(cuMemcpyDtoH(h, out, (size_t)G * 4));
        int bad = 0;
        for (int b = 0; b < G; b++)
            if (h[b] != b + 1000) { if (bad < 5) fprintf(stderr,"  rep%d out[%d]=%d want %d\n",rep,b,h[b],b+1000); bad++; }
        free(h);
        if (bad) { printf("test_atomize_cubin: FAIL rep %d: %d/%d blocks wrong\n", rep, bad, G); return 1; }
    }
    printf("test_atomize_cubin: PASS (%d blocks, cubin spliced+atomized, all correct across 4 reps)\n", G);
    return 0;
}
