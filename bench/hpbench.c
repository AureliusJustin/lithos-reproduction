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
int cmp(const void*a,const void*b){double x=*(double*)a-*(double*)b;return x<0?-1:x>0;}
int main(int c,char**v){int G=c>1?atoi(v[1]):16,W=c>2?atoi(v[2]):200,IT=c>3?atoi(v[3]):400;
/* 4th arg: inter-request gap in us. Gaps > the coordinator idle threshold (1ms)
 * make this tenant look idle to peers, which is what enables TPC stealing. */
long period_us=c>4?atol(v[4]):300;
CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuCtxCreate(&x,0,d));
FILE*f=fopen(bench_cubin("work.cubin"),"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*b=malloc(n);fread(b,1,n,f);fclose(f);
CUmodule m;CK(cuModuleLoadData(&m,b));CUfunction fn;CK(cuModuleGetFunction(&fn,m,"work"));
CUdeviceptr buf;CK(cuMemAlloc(&buf,(size_t)G*128*4));CK(cuMemsetD32(buf,0x3f800000,(size_t)G*128));
CUstream s;CK(cuStreamCreate(&s,0));int kk=0;void*a[]={&buf,&kk,&W};
for(int i=0;i<20;i++){CK(cuLaunchKernel(fn,G,1,1,128,1,1,0,s,a,0));CK(cuStreamSynchronize(s));}
double*sm=malloc(IT*8);
for(int i=0;i<IT;i++){double t0=us();CK(cuLaunchKernel(fn,G,1,1,128,1,1,0,s,a,0));CK(cuStreamSynchronize(s));sm[i]=us()-t0;
 struct timespec ts={period_us/1000000, (period_us%1000000)*1000};nanosleep(&ts,0);}
qsort(sm,IT,8,cmp);
printf("p50=%.1f p99=%.1f p999=%.1f us\n",sm[IT/2],sm[(int)(IT*0.99)],sm[(int)(IT*0.999)]);return 0;}
