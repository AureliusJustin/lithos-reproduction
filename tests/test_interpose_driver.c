/* Driver-API interposition test. Uses the CUDA Driver API directly, so calls
 * go through the PLT and are interposed by LD_PRELOAD-ed LibLithOS. Loads a
 * kernel from an embedded PTX module and launches it through LithOS. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { CUresult r=(x); if(r){const char*s;cuGetErrorName(r,&s); \
    fprintf(stderr,"%s failed: %s\n",#x,s); exit(1);} } while(0)

// saxpy in PTX (sm_50 baseline, JIT-compiled by the driver).
static const char* ptx =
".version 6.0\n.target sm_50\n.address_size 64\n"
".visible .entry saxpy(.param .u64 py, .param .u64 px, .param .f32 pa, .param .u32 pn){\n"
"  .reg .pred %p; .reg .f32 %f<4>; .reg .b32 %r<6>; .reg .b64 %rd<7>;\n"
"  ld.param.u64 %rd1, [py]; ld.param.u64 %rd2, [px];\n"
"  ld.param.f32 %f1, [pa]; ld.param.u32 %r2, [pn];\n"
"  mov.u32 %r3, %ctaid.x; mov.u32 %r4, %ntid.x; mov.u32 %r5, %tid.x;\n"
"  mad.lo.s32 %r1, %r3, %r4, %r5;\n"
"  setp.ge.s32 %p, %r1, %r2; @%p bra DONE;\n"
"  mul.wide.s32 %rd3, %r1, 4;\n"
"  cvta.to.global.u64 %rd4, %rd2; add.s64 %rd5, %rd4, %rd3; ld.global.f32 %f2, [%rd5];\n"
"  cvta.to.global.u64 %rd6, %rd1; add.s64 %rd6, %rd6, %rd3; ld.global.f32 %f3, [%rd6];\n"
"  fma.rn.f32 %f3, %f1, %f2, %f3; st.global.f32 [%rd6], %f3;\n"
"DONE: ret;\n}\n";

int main() {
    const int n = 1 << 20; size_t bytes = n * sizeof(float);
    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CK(cuCtxCreate(&ctx, 0, dev));
    CUmodule mod; CK(cuModuleLoadData(&mod, ptx));
    CUfunction fn; CK(cuModuleGetFunction(&fn, mod, "saxpy"));

    CUdeviceptr dx, dy; CK(cuMemAlloc(&dx, bytes)); CK(cuMemAlloc(&dy, bytes));
    float* x = malloc(bytes); float* y = malloc(bytes);
    for (int i = 0; i < n; i++) { x[i] = 1.0f; y[i] = 2.0f; }
    CK(cuMemcpyHtoD(dx, x, bytes)); CK(cuMemcpyHtoD(dy, y, bytes));

    CUstream s; CK(cuStreamCreate(&s, 0));
    float a = 2.0f; int nn = n;
    void* args[] = { &dy, &dx, &a, &nn };
    int threads = 256, blocks = (n + threads - 1) / threads;
    for (int rep = 0; rep < 4; rep++)
        CK(cuLaunchKernel(fn, blocks,1,1, threads,1,1, 0, s, args, NULL));
    CK(cuStreamSynchronize(s));
    CK(cuMemcpyDtoH(y, dy, bytes));

    float expect = 2.0f + 4 * (2.0f * 1.0f);
    int bad = 0; for (int i = 0; i < n; i++) if (y[i] != expect) bad++;
    if (bad) { printf("FAIL: %d mismatches\n", bad); return 1; }
    printf("test_interpose_driver: PASS (y==%.1f)\n", expect);
    return 0;
}
