/* Reverse-engineering probe: dump the QMD for a launch so we can locate the
 * program-address field on this GPU. Two kernels in one module let us confirm
 * the field by cross-checking against their (adjacent) entry addresses. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "qmd.h"

#define CK(x) do { CUresult r=(x); if(r){const char*s;cuGetErrorName(r,&s); \
    fprintf(stderr,"%s failed: %s\n",#x,s); exit(1);} } while(0)

static const char* ptx =
".version 6.0\n.target sm_50\n.address_size 64\n"
".visible .entry k1(.param .u64 p){ .reg .b64 %rd; ld.param.u64 %rd,[p]; st.global.u32 [%rd], 1; ret; }\n"
".visible .entry k2(.param .u64 p){ .reg .b64 %rd; ld.param.u64 %rd,[p]; st.global.u32 [%rd], 2; ret; }\n";

int main() {
    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev,0));
    CUcontext ctx; CK(cuCtxCreate(&ctx,0,dev));
    CUmodule mod; CK(cuModuleLoadData(&mod, ptx));
    CUfunction f1,f2; CK(cuModuleGetFunction(&f1,mod,"k1")); CK(cuModuleGetFunction(&f2,mod,"k2"));
    CUdeviceptr d; CK(cuMemAlloc(&d,16));
    fprintf(stderr, "[probe] dptr arg = %#llx\n", (unsigned long long)d);
    qmd_init();
    void* args[]={&d};

    // Distinctive grid dims to anchor the layout: gridX=7 gridY=3 gridZ=2
    fprintf(stderr,"=== k1, grid {7,3,2} block {5,1,1} shmem 0 ===\n");
    qmd_dump_next();
    CK(cuLaunchKernel(f1, 7,3,2, 5,1,1, 0, 0, args, NULL));
    CK(cuCtxSynchronize());

    fprintf(stderr,"=== k2, grid {9,1,1} block {5,1,1} ===\n");
    qmd_dump_next();
    CK(cuLaunchKernel(f2, 9,1,1, 5,1,1, 0, 0, args, NULL));
    CK(cuCtxSynchronize());
    return 0;
}
