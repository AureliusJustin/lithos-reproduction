/* cuBLAS through LithOS: closed-source kernels, loaded as fatbins via the CUDA
 * runtime — the same path PyTorch/TF use for GEMM. Checked against a CPU ref. */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
int main(){
  int N=512; size_t sz=(size_t)N*N*sizeof(float);
  float *hA=(float*)malloc(sz),*hB=(float*)malloc(sz),*hC=(float*)malloc(sz);
  for(int i=0;i<N*N;i++){hA[i]=(i%7)*0.1f; hB[i]=(i%5)*0.2f;}
  float *dA,*dB,*dC; cudaMalloc(&dA,sz);cudaMalloc(&dB,sz);cudaMalloc(&dC,sz);
  cudaMemcpy(dA,hA,sz,cudaMemcpyHostToDevice);cudaMemcpy(dB,hB,sz,cudaMemcpyHostToDevice);
  cublasHandle_t h; if(cublasCreate(&h)){printf("cublasCreate FAIL\n");return 1;}
  float a=1.f,b=0.f;
  for(int r=0;r<3;r++) cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&a,dA,N,dB,N,&b,dC,N);
  cudaDeviceSynchronize();
  cudaMemcpy(hC,dC,sz,cudaMemcpyDeviceToHost);
  /* spot-check a few entries (column-major: C = B*A in row-major terms) */
  int bad=0;
  for(int t=0;t<8;t++){int i=(t*37)%N,j=(t*53)%N;double acc=0;
    for(int k=0;k<N;k++) acc+=(double)hA[k*N+i]*hB[j*N+k];
    if(fabs(hC[j*N+i]-acc)>1e-1*fabs(acc)+1e-1) bad++;}
  printf("cuBLAS SGEMM N=%d: wrong=%d [%s]\n",N,bad,bad?"FAIL":"PASS");
  return bad?1:0;
}
