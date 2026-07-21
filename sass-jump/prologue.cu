struct M{unsigned lo,hi;};
extern "C" __global__ void probe(unsigned* sink){
  unsigned long long b=(unsigned long long)blockIdx.z*gridDim.y*gridDim.x
    +(unsigned long long)blockIdx.y*gridDim.x+blockIdx.x;
  M* a=(M*)0x700000001000ULL;
  if(b<a->lo||b>=a->hi) return;      // @P EXIT out-of-range
  asm volatile("mov.u32 %0, 0xDEADBEEF;":"=r"(sink[0])::"memory"); // marker = cut point
}
