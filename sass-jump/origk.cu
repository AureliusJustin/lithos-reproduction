extern "C" __global__ void orig(unsigned* out){
  unsigned b = blockIdx.z*gridDim.y*gridDim.x + blockIdx.y*gridDim.x + blockIdx.x;
  if(threadIdx.x==0) out[b] = b + 100;
}
