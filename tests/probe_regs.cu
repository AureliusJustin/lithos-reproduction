// Locate the register-count field in the QMD by dumping it for two kernels
// with very different register usage.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
extern "C" { void qmd_init(); void qmd_dump_next(); }

__global__ void few(int* o){ if(threadIdx.x==0) o[blockIdx.x]=1; }
// Force high register pressure with many simultaneously-live values.
__global__ void many(int* o){
    int a[16];
    #pragma unroll
    for(int i=0;i<16;i++) a[i]=o[blockIdx.x*16+i]*(i+1);
    int s=0;
    #pragma unroll
    for(int i=0;i<16;i++) s+=a[i]^(a[(i+1)&15]+i);
    o[blockIdx.x]=s;
}
#define CK(x) do{cudaError_t e=(x); if(e){printf("%s:%s\n",#x,cudaGetErrorString(e));exit(1);}}while(0)
int main(){
    int* o; CK(cudaMalloc(&o, 4096*sizeof(int)));
    CK(cudaFree(0)); qmd_init();
    CUfunction f;
    cuModuleGetFunction; // link
    int r1=0,r2=0;
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa,(const void*)few);  r1=fa.numRegs;
    cudaFuncGetAttributes(&fa,(const void*)many);  r2=fa.numRegs;
    fprintf(stderr,"[probe] few regs=%d  many regs=%d\n", r1, r2);
    fprintf(stderr,"=== few ===\n");  qmd_dump_next(); few<<<8,64>>>(o);  CK(cudaDeviceSynchronize());
    fprintf(stderr,"=== many ===\n"); qmd_dump_next(); many<<<8,64>>>(o); CK(cudaDeviceSynchronize());
    return 0;
}
