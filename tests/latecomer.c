/* latecomer.c — a tenant that is idle when its neighbour starts, and becomes
 * busy part-way through the neighbour's kernel.
 *
 * This is the arrival in LithOS Figure 10(c): while this process is quiet its
 * TPCs are lent away (§5.3 TPC Stealing); once it has work again the borrower is
 * supposed to give them back for its *subsequent atoms*. It must stay alive
 * throughout — a process that exits releases its coordinator slot, which would
 * change the borrower's mask for an entirely different reason.
 *
 *   latecomer <idle-seconds> <busy-seconds>
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define CK(x) do{ CUresult r=(x); if(r){ fprintf(stderr,"latecomer: ERR @%d\n",__LINE__); return 2; } }while(0)

static const char* bench_cubin(const char* name) {
    const char* e = getenv("LITHOS_BENCH_CUBIN");
    if (e && *e) return e;
    static char buf[512];
    snprintf(buf, sizeof buf, "build/kernels/%s", name);
    return buf;
}
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char** argv) {
    double idle = argc > 1 ? atof(argv[1]) : 2.0;
    double busy = argc > 2 ? atof(argv[2]) : 4.0;

    CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d, 0));
    CUcontext c; CK(cuCtxCreate(&c, 0, d));
    FILE* f = fopen(bench_cubin("work.cubin"), "rb");
    if (!f) { fprintf(stderr, "latecomer: no cubin\n"); return 2; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* b = malloc(n);
    if (!b || fread(b, 1, n, f) != (size_t)n) return 2;
    fclose(f);
    CUmodule m; CK(cuModuleLoadData(&m, b));
    CUfunction fn; CK(cuModuleGetFunction(&fn, m, "work"));
    CUdeviceptr buf; CK(cuMemAlloc(&buf, 256 * 128 * 4));
    CK(cuMemsetD32(buf, 0x3f800000, 256 * 128));
    CUstream s; CK(cuStreamCreate(&s, 0));
    int kid = 0, w = 200; void* a[] = { &buf, &kid, &w };

    /* Register as a tenant, then go quiet: this is the lender. */
    CK(cuLaunchKernel(fn, 8,1,1, 128,1,1, 0, s, a, NULL));
    CK(cuStreamSynchronize(s));
    printf("latecomer: registered, idle for %.1fs\n", idle); fflush(stdout);
    usleep((useconds_t)(idle * 1e6));

    /* Busy means CONTINUOUSLY busy, by every measure the scheduler uses, and
     * that constrains the shape of this loop more than it first appears.
     * A tenant counts as idle (tpc_alloc.c, coord.c) if ANY of these holds:
     *
     *   - it has not SUBMITTED a launch in the last g_idle_ns (1 ms), or
     *   - it has no outstanding work.
     *
     * So a loop that queues deeply and then waits fails the first test (the host
     * submits rarely), and a launch-then-synchronize loop fails the second (zero
     * outstanding at every sync). What satisfies both is what the paper's
     * arriving request actually is: a stream of SHORT kernels submitted
     * back to back. Async launches return immediately, so the submit cadence
     * stays far under a millisecond, and the queue is only drained once per
     * batch — a window microseconds wide, far narrower than the neighbour's
     * per-atom cadence. */
    printf("latecomer: BUSY now\n"); fflush(stdout);
    int wb = 20000; void* ab[] = { &buf, &kid, &wb };
    double end = now() + busy;
    while (now() < end) {
        for (int i = 0; i < 200; i++)
            CK(cuLaunchKernel(fn, 64,1,1, 128,1,1, 0, s, ab, NULL));
        CK(cuStreamSynchronize(s));
    }
    printf("latecomer: done\n"); fflush(stdout);
    return 0;
}
