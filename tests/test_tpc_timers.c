/* test_tpc_timers.c — per-TPC timers gate TPC Stealing (§5.3).
 *
 * "It maintains per-TPC timers informed by a latency prediction module,
 *  estimating kernel (and atom) durations at submission time. These timers help
 *  avoid stealing from long-running TPCs."
 *
 * tests/test_stealing.c proves an IDLE peer's TPCs are lent. This proves the
 * opposite case, which the idle test alone gets wrong.
 *
 * The hole: a stream counts as idle if it "hasn't launched within g_idle_ns"
 * (1 ms). A stream that submits ONE long kernel and then goes quiet satisfies
 * that after 1 ms — while its kernel is still running for milliseconds more. The
 * coarse test sees "no recent launch" and lends its TPCs; the borrower's work then
 * lands behind that kernel, which is the head-of-line blocking of Figure 10(b).
 * The timer knows the kernel's predicted end and refuses the borrow.
 *
 * Setup: B owns a disjoint quota and launches a long spin, then goes quiet for
 * longer than the idle threshold but far less than the kernel's runtime. A then
 * launches and we count the distinct SMs it actually ran on (%smid is ground
 * truth that the mask took effect; 2 SMs per TPC on Ampere).
 *
 *   LITHOS_TPC_TIMERS=0  -> A steals B's range   (SM count ~doubles)  [the bug]
 *   LITHOS_TPC_TIMERS=1  -> A keeps its own      (SM count ~= quota*2)
 *
 * The predictor must have SEEN this kernel before it can time it, so B's spin is
 * run to convergence first; that warm-up is part of the mechanism, not scaffolding
 * (§5.7: predictions are learned online, never profiled offline).
 *
 *   LITHOS_QUOTA=4 LITHOS_PERSTREAM_QUOTA=1 LITHOS_STEALING=1 \
 *     STEAL_CUBIN=build/steal_probe.cubin \
 *     LD_PRELOAD=build/liblithos_full.so build/test_tpc_timers
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define CK(x) do { CUresult r=(x); if (r) { const char* s; cuGetErrorString(r,&s); \
    fprintf(stderr, "CUDA error %s at line %d\n", s, __LINE__); return 2; } } while (0)

/* Long enough that B is unambiguously still running when A launches, and well
 * past the 1 ms idle threshold. A100 boost ~1.4 GHz => ~14 ms. */
#define SPIN_CYCLES 20000000ull
#define QUIET_US    3000        /* > 1 ms idle threshold, << spin duration */

int main(void) {
    const char* cubin_path = getenv("STEAL_CUBIN");
    if (!cubin_path) cubin_path = "build/steal_probe.cubin";

    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CK(cuCtxCreate(&ctx, 0, dev));

    FILE* f = fopen(cubin_path, "rb");
    if (!f) { fprintf(stderr, "test_tpc_timers: cannot open %s\n", cubin_path); return 77; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* img = malloc(n); if (fread(img, 1, n, f) != (size_t)n) { fclose(f); return 2; }
    fclose(f);

    CUmodule mod; CK(cuModuleLoadData(&mod, img));
    CUfunction probe, spin;
    CK(cuModuleGetFunction(&probe, mod, "probe"));
    CK(cuModuleGetFunction(&spin,  mod, "spin"));

    const int G = 64;
    CUdeviceptr out; CK(cuMemAlloc(&out, G * sizeof(int)));
    CUstream A, B;
    CK(cuStreamCreate(&A, 0));
    CK(cuStreamCreate(&B, 0));

    unsigned long long cycles = SPIN_CYCLES;
    void* spin_args[] = { &cycles };
    void* probe_args[] = { &out };

    /* Warm the predictor on B's operator: it learns durations online, so the
     * first few launches have no estimate and publish no timer. Each iteration is
     * synced so the operator ordinal resets to 0 and every sample lands on the
     * same operator slot. */
    for (int i = 0; i < 6; i++) {
        CK(cuLaunchKernel(spin, 8,1,1, 32,1,1, 0, B, spin_args, 0));
        CK(cuStreamSynchronize(B));
    }

    /* Clear the output BEFORE B's spin is in flight. cuMemsetD32 runs on the
     * legacy NULL stream, which implicitly synchronizes with every blocking
     * stream — issuing it later would block until B finished and the timing below
     * would measure that wait instead of the steal. */
    CK(cuMemsetD32(out, 0xffffffffu, G));

    /* B submits real work, then goes quiet for longer than the idle threshold.
     * Its kernel is still running throughout. */
    CK(cuLaunchKernel(spin, 8,1,1, 32,1,1, 0, B, spin_args, 0));
    usleep(QUIET_US);

    /* A launches into that window. What is asserted is the ALLOCATION DECISION,
     * not a latency win: this probe is trivial, so even when it wrongly borrows
     * B's occupied TPCs it co-schedules into their free warp slots instead of
     * queueing, and both configurations complete in ~0.03 ms. Timing was measured
     * and deliberately dropped rather than reported as a benefit it does not show
     * (see docs/TECHNICAL_REPORT.md). */
    CK(cuLaunchKernel(probe, G,1,1, 32,1,1, 0, A, probe_args, 0));
    CK(cuStreamSynchronize(A));
    CK(cuStreamSynchronize(B));

    int* h = malloc(G * sizeof(int));
    CK(cuMemcpyDtoH(h, out, G * sizeof(int)));

    int seen[256] = {0}, distinct = 0;
    for (int i = 0; i < G; i++) {
        int sm = h[i];
        if (sm >= 0 && sm < 256 && !seen[sm]) { seen[sm] = 1; distinct++; }
    }

    const char* t = getenv("LITHOS_TPC_TIMERS");
    int timers_on = !t || atoi(t) != 0;
    const char* q = getenv("LITHOS_QUOTA");
    int quota = q ? atoi(q) : -1;
    int own = quota > 0 ? quota * 2 : 0;

    printf("test_tpc_timers: timers=%s quota=%d -> %d distinct SMs (own=%d)\n",
           timers_on ? "on" : "off", quota, distinct, own);

    if (quota <= 0) { printf("  SKIP: needs LITHOS_QUOTA\n"); return 0; }

    if (timers_on) {
        if (distinct <= own) {
            printf("  PASS: did not steal from a TPC still running work (%d <= %d SMs)\n",
                   distinct, own);
        } else {
            printf("  FAIL: stole into a busy TPC (%d SMs > own %d)\n", distinct, own);
            return 1;
        }
    } else {
        /* The control: without timers the same setup DOES steal. If this stops
         * being true the test has lost its teeth and the PASS above proves
         * nothing, so it is a failure here too. */
        if (distinct > own) {
            printf("  PASS (control): idle-only heuristic stole the busy range (%d > %d SMs)\n",
                   distinct, own);
        } else {
            printf("  FAIL (control): expected the idle-only heuristic to steal "
                   "(%d SMs, wanted > %d) — the test no longer discriminates\n", distinct, own);
            return 1;
        }
    }
    return 0;
}
