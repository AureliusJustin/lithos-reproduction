/*
 * test_dispatch_mt.cu — the MULTI-WORKER dispatcher (§5.2), correctness.
 *
 * tests/test_dispatch_order.c proves a buffered launch is never overtaken. It
 * does not prove that a buffered launch actually RAN, and that is a different
 * failure: a dispatcher worker that drops every launch it takes still preserves
 * ordering perfectly among the ones it does not drop.
 *
 * The bug this test exists for: submit_one() cached "the context I already made
 * current" in a process-wide static, but cuCtxSetCurrent binds PER THREAD. Only
 * the first worker ever called it; the others skipped a bind they had never
 * performed and submitted with no current context, so their launches failed with
 * CUDA_ERROR_INVALID_CONTEXT. A deferred launch has already returned success to
 * the application, so nothing surfaced except wrong results much later — in
 * PyTorch, gradients that were silently zero.
 *
 * This must be a CUDA-RUNTIME app, not a driver-API one. Under the driver API
 * the worker threads turn out to have a usable context anyway and the bug stays
 * hidden; it is the runtime path (the one PyTorch/TF/JAX use, via the
 * libcuda.so.1 wrapper) that exposes it. Run it the way frameworks are run:
 *
 *   LITHOS_DISPATCH=1 LITHOS_DISPATCH_THREADS=4 \
 *   LD_LIBRARY_PATH=build build/test_dispatch_mt
 *
 * Several streams give the workers independent launch queues so they run
 * concurrently, and the launch count per stream is high enough that no single
 * worker absorbs them all. Each kernel bumps its own stream's counter, so a
 * dropped launch is an exact short count rather than a racy value.
 *
 * Stream 0 is deliberately the LEGACY DEFAULT stream, and that is what makes the
 * test sensitive. A launch on an explicit stream still succeeds with no current
 * context because the driver infers the context from the stream handle -- so a
 * test using only created streams passes even with the bug present. The default
 * stream carries no such handle, so it is the case that actually fails, and it
 * is the stream frameworks put most of their work on.
 */
#include <cstdio>
#include <cuda_runtime.h>

#define NSTREAMS 4
#define NLAUNCH  256

#define CK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) {        \
        fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call,        \
                cudaGetErrorString(_e)); return 2; } } while (0)

/* Only block 0 / thread 0 updates, so the result is identical however many
 * atoms the Kernel Atomizer splits the grid into. */
__global__ void bump(unsigned int* p, unsigned long long spin) {
    unsigned long long t0 = clock64();
    while (clock64() - t0 < spin) { }
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(p, 1u);
}

int main() {
    cudaStream_t  s[NSTREAMS];
    unsigned int* d[NSTREAMS];
    for (int i = 0; i < NSTREAMS; i++) {
        if (i == 0) s[i] = 0;                    /* legacy default stream */
        else CK(cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking));
        CK(cudaMalloc(&d[i], sizeof(unsigned int)));
        CK(cudaMemset(d[i], 0, sizeof(unsigned int)));
    }

    /* Launch VOLUME is the point (it spreads work across the workers), not
     * making any single kernel slow. */
    for (int n = 0; n < NLAUNCH; n++)
        for (int i = 0; i < NSTREAMS; i++)
            bump<<<4, 32, 0, s[i]>>>(d[i], 2000ull);

    int fail = 0;
    for (int i = 0; i < NSTREAMS; i++) {
        unsigned int got = 0;
        CK(cudaStreamSynchronize(s[i]));
        CK(cudaMemcpy(&got, d[i], sizeof(got), cudaMemcpyDeviceToHost));
        if (got != NLAUNCH) {
            printf("  FAIL: stream %d ran %u/%d launches (%d dropped)\n",
                   i, got, NLAUNCH, NLAUNCH - (int)got);
            fail = 1;
        }
    }
    if (!fail) printf("  PASS: all %d launches ran on every one of %d streams\n",
                      NLAUNCH, NSTREAMS);

    printf("DISPATCH_MT: %s\n", fail ? "FAIL" : "PASS (0 failures)");
    return fail;
}
