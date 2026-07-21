extern "C" __global__ void nul(int* x){ int b=blockIdx.x; if(threadIdx.x>1000000) x[b]=b; }
