/* randacc.cu — a latency-bound kernel: scattered DRAM reads.
 *
 * The low-sensitivity end of §5.6's spectrum. A streaming kernel is still partly
 * clock-bound (address arithmetic, and enough L2 reuse to keep the SMs fed), but
 * scattered reads across a working set far larger than the L2 stall on DRAM
 * latency, which does not shrink when the SM clock does. That is the case the
 * paper's model must push to a low frequency: little is lost, and the power saving
 * is real.
 *
 * Same signature as work.cu so the existing harnesses can load it unchanged.
 */
extern "C" __global__ void work(float* buf, int kid, int iters) {
    long i     = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)gridDim.x * blockDim.x;
    long mask  = total - 1;              /* caller uses a power-of-two thread count */
    long idx   = i;
    float acc  = 0.f;
    for (int it = 0; it < iters; it++) {
        /* A cheap high-stride hash: successive indices land in unrelated pages, so
         * neither the cache nor the TLB helps and each load is a DRAM round trip. */
        idx = (idx * 2654435761L + 1013904223L) & mask;
        acc += buf[idx];
    }
    buf[i] = acc + (float)kid * 1e-6f;
}
