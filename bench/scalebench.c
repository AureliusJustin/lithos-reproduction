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
static double ms(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
int cmp(const void*a,const void*b){double x=*(double*)a-*(double*)b;return x<0?-1:x>0;}
int main(int c,char**v){int N=c>1?atoi(v[1]):1024,IT=c>2?atoi(v[2]):40;
CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuCtxCreate(&x,0,d));
FILE*f=fopen(bench_cubin("mm.cubin"),"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*b=malloc(n);fread(b,1,n,f);fclose(f);
CUmodule m;CK(cuModuleLoadData(&m,b));CUfunction fn;CK(cuModuleGetFunction(&fn,m,"mm"));
size_t sz=(size_t)N*N*4;CUdeviceptr A,B,C;CK(cuMemAlloc(&A,sz));CK(cuMemAlloc(&B,sz));CK(cuMemAlloc(&C,sz));
CK(cuMemsetD32(A,0x3f000000,N*N));CK(cuMemsetD32(B,0x3f000000,N*N));CUstream s;CK(cuStreamCreate(&s,0));
void*a[]={&A,&B,&C,&N};int g=N/16;
for(int i=0;i<8;i++)CK(cuLaunchKernel(fn,g,g,1,16,16,1,0,s,a,0));CK(cuStreamSynchronize(s));
double*sm=malloc(IT*8);for(int i=0;i<IT;i++){double t0=ms();CK(cuLaunchKernel(fn,g,g,1,16,16,1,0,s,a,0));CK(cuStreamSynchronize(s));sm[i]=ms()-t0;}
qsort(sm,IT,8,cmp);printf("%.3f\n",sm[IT/2]);return 0;}
