#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>
__global__ void work(float* out,int iters){
  float x=threadIdx.x*1e-3f+blockIdx.x*1e-6f;
  for(int i=0;i<iters;i++) x=fmaf(x,1.0000001f,1e-7f);
  if(x==123.456789f) out[blockIdx.x]=x;
}
double now(){ struct timeval t; gettimeofday(&t,0); return t.tv_sec+t.tv_usec*1e-6; }
int main(int c,char**v){
  const char* tag=c>1?v[1]:"P"; int iters=c>2?atoi(v[2]):3000000; int G=2048;
  float* d; cudaMalloc(&d,G*4);
  work<<<G,256>>>(d,iters); cudaDeviceSynchronize();   // warmup
  double t0=now(); work<<<G,256>>>(d,iters); cudaDeviceSynchronize(); double t1=now();
  printf("[%s] iters=%d wall=%.2fs\n",tag,iters,t1-t0); return 0; }
