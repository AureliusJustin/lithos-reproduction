/* SM-identity probe for test_stealing: each block records the SM it ran on. */
extern "C" __global__ void probe(int* out){
  int b = blockIdx.x;
  int sm; asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
  if (threadIdx.x == 0) out[b] = sm;
}
