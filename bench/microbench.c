#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static unsigned char* rd(const char*p,size_t*n){FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);*n=ftell(f);fseek(f,0,SEEK_SET);unsigned char*b=malloc(*n);if(fread(b,1,*n,f)!=*n){} fclose(f);return b;}
int main(int argc,char**argv){
  int grid = argc>1?atoi(argv[1]):1;
  cuInit(0); CUdevice d; cuDeviceGet(&d,0); CUcontext c; cuCtxCreate(&c,0,d);
  size_t csz; unsigned char* cub = rd(getenv("NULLK"), &csz);
  int preload = argc>2?atoi(argv[2]):0;
  for(int i=0;i<preload;i++){ CUmodule pm; if(cuModuleLoadData(&pm,cub)) break; CUfunction pf; cuModuleGetFunction(&pf,pm,"nul"); }
  CUmodule m; cuModuleLoadData(&m,cub); CUfunction f; cuModuleGetFunction(&f,m,"nul"); /* f is gated LAST -> tail of scan */
  CUdeviceptr out; cuMemAlloc(&out, (size_t)grid*4);
  void* args[]={&out};
  /* warm */ for(int i=0;i<100;i++) cuLaunchKernel(f,grid,1,1,1,1,1,0,0,args,0); cuCtxSynchronize();

  /* Bench 1: async launch host throughput (enqueue cost) */
  int N=200000; double t0=now();
  for(int i=0;i<N;i++) cuLaunchKernel(f,grid,1,1,1,1,1,0,0,args,0);
  cuCtxSynchronize(); double t1=now();
  printf("  async-launch   : %7.3f us/launch  (host enqueue, grid=%d)\n",(t1-t0)/N*1e6,grid);

  /* Bench 2: launch+sync round-trip latency */
  int N2=20000; double t2=now();
  for(int i=0;i<N2;i++){ cuLaunchKernel(f,grid,1,1,1,1,1,0,0,args,0); cuCtxSynchronize(); }
  double t3=now();
  printf("  launch+sync    : %7.3f us/op      (round-trip latency)\n",(t3-t2)/N2*1e6);

  /* Bench 3: module load (+ splice under LithOS) */
  int N3=300; double t4=now();
  for(int i=0;i<N3;i++){ CUmodule mm; cuModuleLoadData(&mm,cub); cuModuleUnload(mm); }
  double t5=now();
  printf("  module-load    : %7.1f us/load    (incl. splice under LithOS)\n",(t5-t4)/N3*1e6);
  return 0;
}
