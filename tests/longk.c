/* longk.c — the best-effort tenant of Figure 10(c): ONE long kernel, atomized.
 *
 * The point is that the kernel is long enough for the world to change while it
 * runs. With LITHOS_ATOMS_INFLIGHT set, its atoms are submitted a few at a time
 * rather than all at once, so each atom's TPC allocation is decided while the
 * kernel is already executing — which is what lets a neighbour's arrival take
 * the borrowed TPCs back mid-kernel.
 *
 *   longk <blocks> <iters>
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CK(x) do{ CUresult r=(x); if(r){ fprintf(stderr,"longk: ERR @%d\n",__LINE__); return 2; } }while(0)

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
    int G = argc > 1 ? atoi(argv[1]) : 2048;
    int W = argc > 2 ? atoi(argv[2]) : 40000;

    CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d, 0));
    CUcontext c; CK(cuCtxCreate(&c, 0, d));
    FILE* f = fopen(bench_cubin("work.cubin"), "rb");
    if (!f) { fprintf(stderr, "longk: no cubin\n"); return 2; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* b = malloc(n);
    if (!b || fread(b, 1, n, f) != (size_t)n) return 2;
    fclose(f);
    CUmodule m; CK(cuModuleLoadData(&m, b));
    CUfunction fn; CK(cuModuleGetFunction(&fn, m, "work"));
    CUdeviceptr buf; CK(cuMemAlloc(&buf, (size_t)G * 128 * 4));
    CK(cuMemsetD32(buf, 0x3f800000, (size_t)G * 128));
    CUstream s; CK(cuStreamCreate(&s, 0));
    int kid = 0; void* a[] = { &buf, &kid, &W };

    double t0 = now();
    CK(cuLaunchKernel(fn, G,1,1, 128,1,1, 0, s, a, NULL));
    CK(cuStreamSynchronize(s));
    printf("longk: %d blocks x %d iters in %.3fs\n", G, W, now() - t0);
    return 0;
}
