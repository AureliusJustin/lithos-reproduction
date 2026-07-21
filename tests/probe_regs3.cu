#include <cuda.h>
#include <cstdio>
extern "C"{void qmd_init();void qmd_dump_next();}
template<int N> __global__ void kern(int* o){
  int a[N];
  #pragma unroll
  for(int i=0;i<N;i++) a[i]=o[blockIdx.x*N+i]*(i+7)^(i<<2);
  int s=0;
  #pragma unroll
  for(int i=0;i<N;i++){ s+=a[i]; s^=a[(i*3+1)%N]+s; }
  o[blockIdx.x]=s;
}
int main(){ int*o; cudaMalloc(&o,65536*sizeof(int)); cudaFree(0); qmd_init();
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa,(const void*)kern<2>); fprintf(stderr,"A regs=%d\n",fa.numRegs);
  qmd_dump_next(); kern<2><<<8,64>>>(o); cudaDeviceSynchronize();
  cudaFuncGetAttributes(&fa,(const void*)kern<24>); fprintf(stderr,"B regs=%d\n",fa.numRegs);
  qmd_dump_next(); kern<24><<<8,64>>>(o); cudaDeviceSynchronize();
  return 0; }
