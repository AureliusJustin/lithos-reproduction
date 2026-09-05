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
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
int cmp(const void*a,const void*b){double x=*(double*)a-*(double*)b;return x<0?-1:x>0;}
int main(int c,char**v){const char*tag=c>1?v[1]:"T";int G=c>2?atoi(v[2]):128,W=c>3?atoi(v[3]):4000;double DUR=c>4?atof(v[4]):4.0;
CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuCtxCreate(&x,0,d));
FILE*f=fopen(bench_cubin("work.cubin"),"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*b=malloc(n);fread(b,1,n,f);fclose(f);
CUmodule m;CK(cuModuleLoadData(&m,b));CUfunction fn;CK(cuModuleGetFunction(&fn,m,"work"));
CUdeviceptr buf;CK(cuMemAlloc(&buf,(size_t)G*128*4));CK(cuMemsetD32(buf,0x3f800000,(size_t)G*128));
CUstream s;CK(cuStreamCreate(&s,0));int kk=0;void*a[]={&buf,&kk,&W};
for(int i=0;i<10;i++)CK(cuLaunchKernel(fn,G,1,1,128,1,1,0,s,a,0));CK(cuStreamSynchronize(s));
int cap=200000;double*lat=malloc(cap*8);int cnt=0;double t0=now(),tend=t0+DUR;
while(now()<tend&&cnt<cap){double a0=now();CK(cuLaunchKernel(fn,G,1,1,128,1,1,0,s,a,0));CK(cuStreamSynchronize(s));lat[cnt++]=(now()-a0)*1e3;}
double el=now()-t0;qsort(lat,cnt,8,cmp);
printf("[%s] %.0f it/s  p50=%.3f p99=%.3f ms\n",tag,cnt/el,lat[cnt/2],lat[(int)(cnt*0.99)]);return 0;}
