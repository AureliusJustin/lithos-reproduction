extern "C" __global__ void probe(int* out){int b=blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;int sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));if(threadIdx.x==0)out[b]=sm;}
