#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Locate a benchmark cubin: $LITHOS_BENCH_CUBIN wins, else build/kernels/<name>.
 * Lets every harness run from the repo root without hardcoding a path. */
static const char* bench_cubin(const char* name) {
    const char* e = getenv("LITHOS_BENCH_CUBIN");
    if (e && *e) return e;
    static char buf[512];
    snprintf(buf, sizeof buf, "build/kernels/%s", name);
    return buf;
}
#define CK(x) do{CUresult r=(x);if(r){fprintf(stderr,"ERR@%d\n",__LINE__);return 2;}}while(0)
static double us(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
int main(int c,char**v){int G=c>1?atoi(v[1]):1,M=c>2?atoi(v[2]):2000;
CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuCtxCreate(&x,0,d));
FILE*f=fopen(bench_cubin("nullk.cubin"),"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*b=malloc(n);fread(b,1,n,f);fclose(f);
CUmodule m;CK(cuModuleLoadData(&m,b));CUfunction fn;CK(cuModuleGetFunction(&fn,m,"nullk"));
CUdeviceptr p;CK(cuMemAlloc(&p,4096));CUstream s;CK(cuStreamCreate(&s,0));void*a[]={&p};
for(int i=0;i<200;i++)CK(cuLaunchKernel(fn,G,1,1,32,1,1,0,s,a,0));CK(cuStreamSynchronize(s));
double t0=us();for(int i=0;i<M;i++){CK(cuLaunchKernel(fn,G,1,1,32,1,1,0,s,a,0));if((i&127)==127)CK(cuStreamSynchronize(s));}CK(cuStreamSynchronize(s));double t1=us();
printf("async=%.2f us/launch\n",(t1-t0)/M);return 0;}
