/* Small-scale atomizer correctness test through the full LithOS library.
 * A kernel writes out[block]=block+1000 for each of 64 blocks. Atomized, every
 * block must still run exactly once. Run under LD_PRELOAD of liblithos_full. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define CK(x) do{CUresult r=(x); if(r){const char*s;cuGetErrorName(r,&s); \
    fprintf(stderr,"%s: %s\n",#x,s); exit(1);} }while(0)

static const char* ptx =
".version 6.0\n.target sm_50\n.address_size 64\n"
".visible .entry mark(.param .u64 po){\n"
"  .reg .b32 %r<3>; .reg .b64 %rd<5>;\n"
"  ld.param.u64 %rd1,[po];\n"
"  mov.u32 %r1,%ctaid.x;\n"
"  add.u32 %r2,%r1,1000;\n"
"  mul.wide.u32 %rd2,%r1,4;\n"
"  cvta.to.global.u64 %rd3,%rd1; add.s64 %rd4,%rd3,%rd2;\n"
"  st.global.u32 [%rd4],%r2;\n"
"  ret;\n}\n";

int main(int argc, char** argv) {
    int G = argc>1?atoi(argv[1]):64;
    int T = argc>2?atoi(argv[2]):32;
    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev,0));
    CUcontext ctx; CK(cuCtxCreate(&ctx,0,dev));
    CUmodule mod; CK(cuModuleLoadData(&mod,ptx));
    CUfunction fn; CK(cuModuleGetFunction(&fn,mod,"mark"));
    CUdeviceptr d; CK(cuMemAlloc(&d,G*sizeof(int)));
    int* h = malloc(G*sizeof(int));
    CUstream s; CK(cuStreamCreate(&s,0));
    void* args[]={&d};
    // Several launches so atomization engages after the first (capture) launch.
    for (int rep=0; rep<4; rep++) {
        for (int i=0;i<G;i++) h[i]=-1;
        CK(cuMemcpyHtoD(d,h,G*sizeof(int)));
        CK(cuLaunchKernel(fn, G,1,1, T,1,1, 0, s, args, NULL));
        CK(cuStreamSynchronize(s));
        CK(cuMemcpyDtoH(h,d,G*sizeof(int)));
        int bad=0; for(int b=0;b<G;b++) if(h[b]!=b+1000){ if(bad<5)fprintf(stderr,"  rep%d out[%d]=%d\n",rep,b,h[b]); bad++; }
        if (bad) { printf("FAIL rep %d: %d/%d wrong\n", rep, bad, G); return 1; }
    }
    printf("test_atomize_small: PASS (%d blocks x %d threads, all correct across 4 reps)\n", G, T);
    return 0;
}
