/* membound.cu — a bandwidth-bound kernel, the counterpart to work.cu.
 *
 * §5.6's model only earns its keep if it can tell the two apart: a compute-bound
 * kernel's latency tracks the clock (sensitivity ~1) and must hold the frequency
 * up, while a memory-bound one barely notices a lower clock (sensitivity ~0) and
 * can be run far down the curve for the same latency budget. work.cu is the first
 * case; this is the second.
 *
 * To be genuinely memory-bound rather than cache-bound, the working set has to
 * exceed the L2 (40 MB on an A100), which the caller arranges by launching enough
 * blocks: the buffer is gridDim.x * blockDim.x floats. Each iteration walks a
 * large stride so successive passes miss the cache, while threads within a warp
 * stay contiguous so the loads still coalesce — bandwidth-limited, not
 * latency-limited by poor access patterns.
 *
 * Same signature as work.cu so the existing harnesses can load it via
 * LITHOS_BENCH_CUBIN with no changes.
 */
extern "C" __global__ void work(float* buf, int kid, int iters) {
    long i     = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)gridDim.x * blockDim.x;
    long mask  = total - 1;             /* caller uses a power-of-two thread count */
    float v = 0.f;
    for (int it = 0; it < iters; it++)
        v += buf[(i + (long)it * 4099) & mask];
    buf[i] = v + (float)kid * 1e-6f;
}
