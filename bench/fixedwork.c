/* fixedwork.c — run a FIXED amount of work, so energy comparisons are fair.
 *
 * §8.3: "We run experiments for a fixed number of requests or training epochs for
 * a fair energy comparison." That detail is not incidental. Energy is average
 * power times wall time, so measuring over a fixed *duration* compares nothing but
 * average power — a configuration that runs slower looks better simply because it
 * did less work in the window. Fixing the iteration count instead makes the two
 * halves of the trade visible at once: a lower clock draws less power but takes
 * longer, and the product is what decides whether it was worth it.
 *
 *   fixedwork <tag> <grid> <work> <iterations>
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static const char* bench_cubin(const char* name) {
    const char* e = getenv("LITHOS_BENCH_CUBIN");
    if (e && *e) return e;
    static char buf[512];
    snprintf(buf, sizeof buf, "build/kernels/%s", name);
    return buf;
}
#define CK(x) do{ CUresult r=(x); if(r){ fprintf(stderr,"fixedwork: ERR @%d\n",__LINE__); return 2; } }while(0)
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec/1e9; }
static int cmpd(const void* a, const void* b){ double x=*(const double*)a-*(const double*)b; return x<0?-1:x>0; }

int main(int argc, char** argv) {
    const char* tag = argc > 1 ? argv[1] : "FW";
    int G  = argc > 2 ? atoi(argv[2]) : 256;
    int W  = argc > 3 ? atoi(argv[3]) : 20000;
    long N = argc > 4 ? atol(argv[4]) : 100000;

    CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d, 0));
    CUcontext c; CK(cuCtxCreate(&c, 0, d));
    FILE* f = fopen(bench_cubin("work.cubin"), "rb");
    if (!f) { fprintf(stderr, "fixedwork: no cubin\n"); return 2; }
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

    for (int i = 0; i < 10; i++) CK(cuLaunchKernel(fn, G,1,1, 128,1,1, 0, s, a, NULL));
    CK(cuStreamSynchronize(s));

    double* lat = malloc((size_t)N * sizeof(double));
    if (!lat) return 2;
    double t0 = now();
    for (long i = 0; i < N; i++) {
        double a0 = now();
        CK(cuLaunchKernel(fn, G,1,1, 128,1,1, 0, s, a, NULL));
        CK(cuStreamSynchronize(s));
        lat[i] = (now() - a0) * 1e3;
    }
    double elapsed = now() - t0;
    qsort(lat, N, sizeof(double), cmpd);
    printf("[%s] %ld iters in %.3f s  p50=%.3f p99=%.3f ms\n",
           tag, N, elapsed, lat[N/2], lat[(long)(N * 0.99)]);
    return 0;
}
