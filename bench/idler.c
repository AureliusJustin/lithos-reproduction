/* idler.c — a tenant that registers with the coordinator, does one launch so it
 * is visibly "runnable", then goes idle for a while WITHOUT exiting.
 *
 * This is the other half of a cross-application stealing test: the lender has to
 * still be alive (holding its quota) but quiet, which is exactly the situation
 * §5.3 describes — "idle TPCs are lent to other tasks". A tenant that exits
 * releases its slot, so it cannot be the lender. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#define CK(x) do{CUresult r=(x);if(r){fprintf(stderr,"ERR@%d\n",__LINE__);return 2;}}while(0)

static const char* bench_cubin(const char* name) {
    const char* e = getenv("LITHOS_BENCH_CUBIN");
    if (e && *e) return e;
    static char buf[512];
    snprintf(buf, sizeof buf, "build/kernels/%s", name);
    return buf;
}

int main(int argc, char** argv) {
    int secs = argc > 1 ? atoi(argv[1]) : 6;
    CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d, 0));
    CUcontext c; CK(cuCtxCreate(&c, 0, d));
    FILE* f = fopen(bench_cubin("work.cubin"), "rb");
    if (!f) { fprintf(stderr, "idler: no cubin\n"); return 2; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* b = malloc(n); if (fread(b, 1, n, f) != (size_t)n) return 2; fclose(f);
    CUmodule m; CK(cuModuleLoadData(&m, b));
    CUfunction fn; CK(cuModuleGetFunction(&fn, m, "work"));
    CUdeviceptr buf; CK(cuMemAlloc(&buf, 256 * 128 * 4));
    CK(cuMemsetD32(buf, 0x3f800000, 256 * 128));
    CUstream s; CK(cuStreamCreate(&s, 0));
    int kid = 0, w = 200; void* a[] = { &buf, &kid, &w };
    CK(cuLaunchKernel(fn, 8,1,1, 128,1,1, 0, s, a, 0));   /* register as a tenant */
    CK(cuStreamSynchronize(s));
    printf("idler: registered, idling %ds\n", secs); fflush(stdout);
    sleep(secs);                                           /* alive but quiet */
    return 0;
}
