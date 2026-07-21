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
#define CK(x) do{cudaError_t e=(x);if(e){printf("%s\n",cudaGetErrorString(e));return 1;}}while(0)
template<int N> int run(int*o){
  cudaFuncAttributes fa; cudaFuncGetAttributes(&fa,(const void*)kern<N>);
  fprintf(stderr,"[regs=%d] ",fa.numRegs); qmd_dump_next(); kern<N><<<8,64>>>(o); return cudaDeviceSynchronize();
}
int main(){ int*o; cudaMalloc(&o,65536*sizeof(int)); cudaFree(0); qmd_init();
  run<2>(o); run<8>(o); run<16>(o); run<24>(o); run<32>(o); return 0; }
