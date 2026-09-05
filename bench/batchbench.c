/* A BATCHED workload: KN kernels launched back-to-back, then one sync (a "batch",
 * as in a model's forward pass). Contrast with heavybench.c, which syncs after
 * every launch. The predictor identifies operators by their ordinal within a
 * batch, so this is the shape it is designed for. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

static const char* bench_cubin(const char* name) {
    const char* e = getenv("LITHOS_BENCH_CUBIN");
    if (e && *e) return e;
    static char buf[512];
    snprintf(buf, sizeof buf, "build/kernels/%s", name);
    return buf;
}
#define CK(x) do{CUresult r=(x);if(r){fprintf(stderr,"ERR@%d\n",__LINE__);return 2;}}while(0)

int main(int c, char** v) {
    int KN = c > 1 ? atoi(v[1]) : 8;      /* kernels per batch */
    int IT = c > 2 ? atoi(v[2]) : 20;     /* batches           */
    int G  = c > 3 ? atoi(v[3]) : 256;
    CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d,0)); CUcontext x; CK(cuCtxCreate(&x,0,d));
    FILE* f = fopen(bench_cubin("work.cubin"), "rb");
    if(!f){fprintf(stderr,"no cubin\n");return 2;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char* b=malloc(n); if(fread(b,1,n,f)!=(size_t)n){return 2;} fclose(f);
    CUmodule m; CK(cuModuleLoadData(&m,b)); CUfunction fn; CK(cuModuleGetFunction(&fn,m,"work"));
    CUdeviceptr buf; CK(cuMemAlloc(&buf,(size_t)G*128*4)); CK(cuMemsetD32(buf,0x3f800000,(size_t)G*128));
    CUstream s; CK(cuStreamCreate(&s,0));
    /* kernel k does k-dependent work, so each ordinal has a distinct true latency */
    for (int it = 0; it < IT; it++) {
        for (int k = 0; k < KN; k++) {
            int w = 2000 + k * 3000;
            void* a[] = { &buf, &k, &w };
            CK(cuLaunchKernel(fn, G,1,1, 128,1,1, 0, s, a, 0));
        }
        CK(cuStreamSynchronize(s));       /* batch boundary */
    }
    printf("ran %d batches x %d kernels\n", IT, KN);
    return 0;
}
