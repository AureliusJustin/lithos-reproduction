/* cuFFT: another closed-source library shipping fatbin kernels, and a different
 * launch pattern from GEMM. Round-trip FFT->IFFT must return the input. */
#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
int main(){
  int N=4096; cufftComplex *d; cudaMalloc(&d,N*sizeof(cufftComplex));
  cufftComplex* h=(cufftComplex*)malloc(N*sizeof(cufftComplex));
  for(int i=0;i<N;i++){h[i].x=sinf(i*0.01f); h[i].y=0;}
  cudaMemcpy(d,h,N*sizeof(cufftComplex),cudaMemcpyHostToDevice);
  cufftHandle p; if(cufftPlan1d(&p,N,CUFFT_C2C,1)!=CUFFT_SUCCESS){printf("plan FAIL\n");return 1;}
  cufftExecC2C(p,d,d,CUFFT_FORWARD);
  cufftExecC2C(p,d,d,CUFFT_INVERSE);
  cudaDeviceSynchronize();
  cufftComplex* o=(cufftComplex*)malloc(N*sizeof(cufftComplex));
  cudaMemcpy(o,d,N*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
  int bad=0; for(int i=0;i<N;i++){ float v=o[i].x/N; if(fabsf(v-h[i].x)>1e-3f) bad++; }
  printf("cuFFT C2C round-trip N=%d: wrong=%d [%s]\n",N,bad,bad?"FAIL":"PASS");
  return bad?1:0;
}
