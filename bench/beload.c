#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

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
int main(int c,char**v){int G=c>1?atoi(v[1]):20000,W=c>2?atoi(v[2]):40000,SEC=c>3?atoi(v[3]):8;
CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuCtxCreate(&x,0,d));
FILE*f=fopen(bench_cubin("work.cubin"),"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*b=malloc(n);fread(b,1,n,f);fclose(f);
CUmodule m;CK(cuModuleLoadData(&m,b));CUfunction fn;CK(cuModuleGetFunction(&fn,m,"work"));
CUdeviceptr buf;CK(cuMemAlloc(&buf,(size_t)G*128*4));CK(cuMemsetD32(buf,0x3f800000,(size_t)G*128));
CUstream s;CK(cuStreamCreate(&s,0));int kk=0;void*a[]={&buf,&kk,&W};long cnt=0;time_t end=time(0)+SEC;
while(time(0)<end){for(int i=0;i<20;i++)CK(cuLaunchKernel(fn,G,1,1,128,1,1,0,s,a,0));CK(cuStreamSynchronize(s));cnt+=20;}
printf("BE %ld kernels\n",cnt);return 0;}
