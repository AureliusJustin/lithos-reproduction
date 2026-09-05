/* Does TPC stealing improve throughput, not just change the mask?
 *
 * Two streams with DISJOINT per-stream quotas (LITHOS_PERSTREAM_QUOTA=1). Stream A
 * runs flat out; stream B is bursty and mostly idle. With stealing enabled A
 * should borrow B's idle TPCs and finish more work. Reports A's throughput. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

static const char* bench_cubin(const char* name) {
    const char* e = getenv("LITHOS_BENCH_CUBIN");
    if (e && *e) return e;
    static char buf[512];
    snprintf(buf, sizeof buf, "build/kernels/%s", name);
    return buf;
}
#define CK(x) do{CUresult r=(x);if(r){fprintf(stderr,"ERR@%d\n",__LINE__);exit(2);}}while(0)
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}

static CUfunction fn; static CUcontext ctx;
static CUdeviceptr bufA, bufB; static CUstream sA, sB;
static double DUR = 4.0; static int G = 128, W = 20000;
static volatile int stop_b = 0;

/* B: rare short bursts, long idle gaps -> its TPCs are idle most of the time. */
static void* bursty(void* _) {
    (void)_; cuCtxSetCurrent(ctx);
    int k = 1; void* a[] = { &bufB, &k, &W };
    while (!stop_b) {
        for (int i = 0; i < 2; i++) cuLaunchKernel(fn, G,1,1, 128,1,1, 0, sB, a, 0);
        cuStreamSynchronize(sB);
        struct timespec ts = {0, 200*1000*1000};   /* idle 200 ms */
        nanosleep(&ts, NULL);
    }
    return NULL;
}

int main(int c, char** v) {
    if (c > 1) DUR = atof(v[1]);
    CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d,0)); CK(cuCtxCreate(&ctx,0,d));
    FILE* f = fopen(bench_cubin("work.cubin"), "rb");
    if(!f){fprintf(stderr,"no cubin\n");return 2;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char* b=malloc(n); if(fread(b,1,n,f)!=(size_t)n) return 2; fclose(f);
    CUmodule m; CK(cuModuleLoadData(&m,b)); CK(cuModuleGetFunction(&fn,m,"work"));
    CK(cuMemAlloc(&bufA,(size_t)G*128*4)); CK(cuMemAlloc(&bufB,(size_t)G*128*4));
    CK(cuMemsetD32(bufA,0x3f800000,(size_t)G*128)); CK(cuMemsetD32(bufB,0x3f800000,(size_t)G*128));
    CK(cuStreamCreate(&sA,0)); CK(cuStreamCreate(&sB,0));   /* order fixes their quotas */

    pthread_t th; pthread_create(&th, NULL, bursty, NULL);
    /* let B register and then fall idle so A can see it as lendable */
    struct timespec ts={0,400*1000*1000}; nanosleep(&ts,NULL);

    int k = 0; void* a[] = { &bufA, &k, &W };
    long cnt = 0; double t0 = now(), tend = t0 + DUR;
    while (now() < tend) {
        for (int i = 0; i < 8; i++) CK(cuLaunchKernel(fn, G,1,1, 128,1,1, 0, sA, a, 0));
        CK(cuStreamSynchronize(sA)); cnt += 8;
    }
    double el = now() - t0;
    stop_b = 1; pthread_join(th, NULL);
    printf("A: %.0f it/s\n", cnt/el);
    return 0;
}
