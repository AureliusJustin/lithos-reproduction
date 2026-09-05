/*
 * Kernels for tests/test_dispatch_order.c — the ordering regression for
 * buffered launches (§5.2).
 *
 * Each kernel is deliberately SLOW (a spin on clock64) so that a launch which
 * was wrongly allowed to be overtaken fails deterministically rather than
 * racily: the reader gets the pre-kernel value every time, not once in a while.
 *
 * Only block 0 / thread 0 does the update, so the result is identical however
 * many atoms the Kernel Atomizer splits the grid into — each block still runs
 * exactly once across the atoms.
 */

extern "C" __global__ void bump(unsigned int* p, unsigned long long spin) {
    unsigned long long t0 = clock64();
    while (clock64() - t0 < spin) { }
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(p, 1u);
}

/* Copies *src into *dst — used to check a cross-stream event dependency really
 * observed the producer kernel's write. */
extern "C" __global__ void relay(const unsigned int* src, unsigned int* dst,
                                 unsigned long long spin) {
    unsigned long long t0 = clock64();
    while (clock64() - t0 < spin) { }
    if (blockIdx.x == 0 && threadIdx.x == 0) *dst = *src;
}
