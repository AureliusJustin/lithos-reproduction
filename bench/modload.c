#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define CK(x) do{CUresult r=(x);if(r){fprintf(stderr,"ERR@%d\n",__LINE__);return 2;}}while(0)
static double us(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
int main(int c,char**v){const char*p=c>1?v[1]:"nullk.cubin";int M=c>2?atoi(v[2]):200;
CK(cuInit(0));CUdevice d;CK(cuDeviceGet(&d,0));CUcontext x;CK(cuCtxCreate(&x,0,d));
FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*b=malloc(n+1);fread(b,1,n,f);b[n]=0;fclose(f);
CUmodule m;CK(cuModuleLoadData(&m,b));CK(cuModuleUnload(m));
double t0=us();for(int i=0;i<M;i++){CK(cuModuleLoadData(&m,b));CK(cuModuleUnload(m));}double t1=us();
printf("%.1f\n",(t1-t0)/M);return 0;}
