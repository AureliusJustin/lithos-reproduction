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
int main(int c,char**v){int KN=c>1?atoi(v[1]):8,W=c>2?atoi(v[2]):200,IT=c>3?atoi(v[3]):300;
CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuCtxCreate(&x,0,d));
FILE*f=fopen(bench_cubin("work.cubin"),"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*b=malloc(n);fread(b,1,n,f);fclose(f);
CUmodule m;CK(cuModuleLoadData(&m,b));CUfunction fn;CK(cuModuleGetFunction(&fn,m,"work"));
int G=256,T=128;CUdeviceptr buf;CK(cuMemAlloc(&buf,(size_t)G*T*4));CK(cuMemsetD32(buf,0x3f800000,(size_t)G*T));
CUstream s;CK(cuStreamCreate(&s,0));int kids[512];int kw=0;void*wa[]={&buf,&kw,&W};
CK(cuLaunchKernel(fn,G,1,1,T,1,1,0,s,wa,0));CK(cuStreamSynchronize(s));
double tc0=ms();CK(cuStreamBeginCapture(s,CU_STREAM_CAPTURE_MODE_GLOBAL));
for(int k=0;k<KN;k++){kids[k]=k;void*a[]={&buf,&kids[k],&W};CK(cuLaunchKernel(fn,G,1,1,T,1,1,0,s,a,0));}
CUgraph g;CK(cuStreamEndCapture(s,&g));double tc=ms()-tc0;
double ti0=ms();CUgraphExec e;CK(cuGraphInstantiateWithFlags(&e,g,0));double ti=ms()-ti0;
CK(cuGraphLaunch(e,s));CK(cuStreamSynchronize(s));
double*sm=malloc(IT*8);for(int i=0;i<IT;i++){double a=ms();CK(cuGraphLaunch(e,s));CK(cuStreamSynchronize(s));sm[i]=ms()-a;}
qsort(sm,IT,8,cmp);printf("capture=%.2fms instantiate=%.2fms replay=%.1fus\n",tc,ti,sm[IT/2]*1000);return 0;}
