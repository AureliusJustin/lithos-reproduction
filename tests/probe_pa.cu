#include <cuda.h>
#include <cstdio>
extern "C"{void qmd_init();void qmd_dump_next();}
__global__ void k1(int*o){o[blockIdx.x]=1;}
__global__ void k2(int*o){o[blockIdx.x]=2;}
int main(){int*o;cudaMalloc(&o,4096*sizeof(int));cudaFree(0);qmd_init();
 fprintf(stderr,"K1\n");qmd_dump_next();k1<<<8,32>>>(o);cudaDeviceSynchronize();
 fprintf(stderr,"K2\n");qmd_dump_next();k2<<<8,32>>>(o);cudaDeviceSynchronize();return 0;}
