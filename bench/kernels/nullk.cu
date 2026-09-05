/* Minimal kernel for launch-latency measurement: does essentially no work, so the
 * measured time is host-side launch overhead rather than GPU execution. */
extern "C" __global__ void nullk(int* p){ if (p && threadIdx.x == 0) p[blockIdx.x] = blockIdx.x; }
