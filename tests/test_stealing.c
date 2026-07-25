/* test_stealing.c — verifies TPC Stealing (§5.3) actually lends idle TPCs.
 *
 * "To improve work conservation, the scheduler dynamically reassigns underutilized
 * TPCs across applications... idle TPCs are lent to other tasks."
 *
 * Setup: two streams with DISJOINT quotas (LITHOS_PERSTREAM_QUOTA=1), e.g.
 * A owns TPCs [0,q) and B owns [q,2q). B is left idle, then A launches. With
 * stealing enabled A should run on BOTH ranges; with it disabled, only its own.
 *
 * We measure the SM count each launch actually used via %smid, which is the
 * ground truth that the mask took effect (2 SMs per TPC on Ampere/Ada).
 *
 * Run:
 *   LITHOS_QUOTA=4 LITHOS_PERSTREAM_QUOTA=1 LITHOS_STEALING=1 \
 *     LD_PRELOAD=build/liblithos_full.so build/test_stealing
 * Expect ~2x the SMs with stealing on vs off.
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define CK(x) do { CUresult r=(x); if (r) { const char* s; cuGetErrorString(r,&s); \
    fprintf(stderr, "CUDA error %s at line %d\n", s, __LINE__); return 2; } } while (0)

/* Records the SM each block ran on, so we can count distinct SMs. */
static const char* PROBE_SRC =
    "extern \"C\" __global__ void probe(int* out){\n"
    "  int b = blockIdx.x;\n"
    "  int sm; asm volatile(\"mov.u32 %0, %%smid;\":\"=r\"(sm));\n"
    "  if (threadIdx.x == 0) out[b] = sm;\n"
    "}\n";

int main(void) {
    const char* cubin_path = getenv("STEAL_CUBIN");
    if (!cubin_path) cubin_path = "build/steal_probe.cubin";

    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CK(cuCtxCreate(&ctx, 0, dev));

    FILE* f = fopen(cubin_path, "rb");
    if (!f) { fprintf(stderr, "test_stealing: cannot open %s\n", cubin_path); return 77; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* img = malloc(n); if (fread(img, 1, n, f) != (size_t)n) { fclose(f); return 2; }
    fclose(f);

    CUmodule mod; CK(cuModuleLoadData(&mod, img));
    CUfunction fn; CK(cuModuleGetFunction(&fn, mod, "probe"));

    const int G = 64;
    CUdeviceptr out; CK(cuMemAlloc(&out, G * sizeof(int)));
    CUstream A, B;
    CK(cuStreamCreate(&A, 0));
    CK(cuStreamCreate(&B, 0));      /* B exists so it owns a quota, then goes idle */

    void* args[] = { &out };
    /* Touch B once so it is registered with a range, then let it fall idle. */
    CK(cuLaunchKernel(fn, 8,1,1, 32,1,1, 0, B, args, 0));
    CK(cuStreamSynchronize(B));
    usleep(50000);                  /* exceed the idle threshold */

    /* A launches while B is idle: stealing (if on) should widen A's mask. */
    CK(cuMemsetD32(out, 0xffffffffu, G));
    CK(cuLaunchKernel(fn, G,1,1, 32,1,1, 0, A, args, 0));
    CK(cuStreamSynchronize(A));

    int* h = malloc(G * sizeof(int));
    CK(cuMemcpyDtoH(h, out, G * sizeof(int)));

    int seen[256] = {0}, distinct = 0;
    for (int i = 0; i < G; i++) {
        int sm = h[i];
        if (sm >= 0 && sm < 256 && !seen[sm]) { seen[sm] = 1; distinct++; }
    }

    const char* steal = getenv("LITHOS_STEALING");
    int stealing_on = !steal || atoi(steal) != 0;
    const char* q = getenv("LITHOS_QUOTA");
    int quota = q ? atoi(q) : -1;

    printf("test_stealing: stealing=%s quota=%d -> %d distinct SMs\n",
           stealing_on ? "on" : "off", quota, distinct);

    /* With a quota of q TPCs, an unstolen launch sees ~2q SMs; stealing one idle
     * peer's equal-sized range should roughly double that. */
    if (quota > 0) {
        int own = quota * 2;
        if (stealing_on) {
            if (distinct > own) printf("  PASS: borrowed idle TPCs (%d SMs > own %d)\n", distinct, own);
            else { printf("  FAIL: no TPCs borrowed (%d SMs, expected > %d)\n", distinct, own); return 1; }
        } else {
            if (distinct <= own) printf("  PASS: confined to its own %d SMs\n", own);
            else { printf("  FAIL: exceeded its quota (%d SMs > %d)\n", distinct, own); return 1; }
        }
    }
    return 0;
}
