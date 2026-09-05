/* SM-identity probe for test_stealing: each block records the SM it ran on. */
extern "C" __global__ void probe(int* out){
  int b = blockIdx.x;
  int sm; asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
  if (threadIdx.x == 0) out[b] = sm;
}

/* A deliberately long kernel, for tests/test_tpc_timers.c: it makes a stream that
 * has submitted real work but stopped launching — the case the coarse idle test
 * cannot tell apart from a genuinely idle one. */
extern "C" __global__ void spin(unsigned long long cycles){
  unsigned long long t0 = clock64();
  while (clock64() - t0 < cycles) { }
}
