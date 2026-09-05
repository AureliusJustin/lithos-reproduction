/*
 * test_dispatch_order.c — buffered launches must not be overtaken (§5.2).
 *
 * When LithOS "enqueues the kernel and returns control to the application", the
 * kernel exists only in a LithOS launch queue: the driver has not seen it and
 * the stream looks empty. Every other stream-ordered call therefore has to drain
 * that queue before it is forwarded (src/barrier.def), or it lands on the GPU
 * ahead of a kernel the application already issued.
 *
 * Each case below reads back a value the deferred kernel writes. If the reader
 * were allowed to overtake the kernel it would observe the PREVIOUS value —
 * deterministically, not occasionally, because the kernels spin long enough that
 * there is no race to lose. That is what makes this a regression test rather
 * than a flaky one: a missing barrier fails it every run.
 *
 * Run it both ways — with the dispatcher off it must still pass, which confirms
 * the harness itself is sound.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ITERS 64
#define SPIN  200000ull        /* ~0.1-0.2 ms per kernel on a modern GPU */

static int g_fail;

#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("    FAIL: "); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static void cu(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
        const char* s = NULL; cuGetErrorName(r, &s);
        fprintf(stderr, "%s failed: %s\n", what, s ? s : "?");
        exit(2);
    }
}

static CUfunction f_bump, f_relay;
static CUdeviceptr d_a, d_b;

/* Start each case from a clean slate. Without this a case that bails out early
 * leaves work in flight, and the next one's counters start wrong — which reads
 * as a second, phantom failure. */
static void reset(void) {
    cuCtxSynchronize();
    cuMemsetD32(d_a, 0, 1);
    cuMemsetD32(d_b, 0, 1);
    cuCtxSynchronize();
}

/* Launch `bump` on `s`, incrementing *d_a. */
static void launch_bump(CUstream s, CUdeviceptr p) {
    unsigned long long spin = SPIN;
    void* args[] = { &p, &spin };
    cu(cuLaunchKernel(f_bump, 64, 1, 1, 128, 1, 1, 0, s, args, NULL), "launch bump");
}

/* ---- 1. async device-to-host copy on the same stream ---------------------- */
static void test_async_memcpy(CUstream s) {
    printf("  async D2H after a deferred launch\n");
    reset();
    for (int i = 0; i < ITERS; i++) {
        launch_bump(s, d_a);
        unsigned int h = 0xdeadbeef;
        cu(cuMemcpyDtoHAsync(&h, d_a, 4, s), "D2H async");
        cu(cuStreamSynchronize(s), "sync");
        CHECK(h == (unsigned)(i + 1), "iter %d: read %u, expected %u "
              "(the copy overtook the kernel)", i, h, i + 1);
        if (g_fail) return;
    }
}

/* ---- 2. blocking copy, which is ordered against the legacy stream --------- */
static void test_blocking_memcpy(CUstream s) {
    printf("  blocking D2H after a deferred launch\n");
    reset();
    for (int i = 0; i < ITERS; i++) {
        launch_bump(s, d_a);
        unsigned int h = 0xdeadbeef;
        cu(cuMemcpyDtoH(&h, d_a, 4), "D2H");
        CHECK(h == (unsigned)(i + 1), "iter %d: read %u, expected %u", i, h, i + 1);
        if (g_fail) return;
    }
}

/* ---- 3. cross-stream dependency expressed with an event ------------------- *
 * The producer is buffered on s1; the event must be recorded AFTER it, so the
 * consumer on s2 waits for the real thing. Recording the event early would let
 * the relay read a stale value even though the program said otherwise. */
static void test_event_dependency(CUstream s1, CUstream s2) {
    printf("  event dependency across streams\n");
    CUevent ev;
    cu(cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING), "event create");
    reset();

    for (int i = 0; i < ITERS; i++) {
        launch_bump(s1, d_a);
        cu(cuEventRecord(ev, s1), "event record");
        cu(cuStreamWaitEvent(s2, ev, 0), "wait event");

        unsigned long long spin = SPIN / 4;
        void* args[] = { &d_a, &d_b, &spin };
        cu(cuLaunchKernel(f_relay, 64, 1, 1, 128, 1, 1, 0, s2, args, NULL), "launch relay");

        unsigned int h = 0xdeadbeef;
        cu(cuMemcpyDtoHAsync(&h, d_b, 4, s2), "D2H async");
        cu(cuStreamSynchronize(s2), "sync s2");
        CHECK(h == (unsigned)(i + 1), "iter %d: relayed %u, expected %u", i, h, i + 1);
        if (g_fail) break;
    }
    cuEventDestroy(ev);
}

/* ---- 4. cuStreamQuery must not claim an empty stream is finished ---------- */
static void test_stream_query(CUstream s) {
    printf("  cuStreamQuery vs buffered work\n");
    reset();
    for (int i = 0; i < 16; i++) {
        launch_bump(s, d_a);
        /* Spin until the stream reports idle, then read WITHOUT syncing. If the
         * query had answered before the kernel was even submitted, this read
         * would see the old value. */
        while (cuStreamQuery(s) == CUDA_ERROR_NOT_READY) { }
        unsigned int h = 0xdeadbeef;
        cu(cuMemcpyDtoH(&h, d_a, 4), "D2H");
        CHECK(h == (unsigned)(i + 1), "iter %d: read %u after query said idle, expected %u",
              i, h, i + 1);
        if (g_fail) return;
    }
}

/* ---- 5. two streams at different priorities both finish ------------------- *
 * Priority decides ORDER between queues, never whether something runs. A
 * starved low-priority queue would show up here as a wrong final count. */
static void test_priority_progress(CUstream hi, CUstream lo) {
    printf("  both priorities make progress\n");
    reset();
    for (int i = 0; i < ITERS; i++) {
        launch_bump(lo, d_b);
        launch_bump(hi, d_a);
    }
    cu(cuStreamSynchronize(hi), "sync hi");
    cu(cuStreamSynchronize(lo), "sync lo");

    unsigned int a = 0, b = 0;
    cu(cuMemcpyDtoH(&a, d_a, 4), "D2H a");
    cu(cuMemcpyDtoH(&b, d_b, 4), "D2H b");
    CHECK(a == ITERS, "high-priority stream ran %u/%d", a, ITERS);
    CHECK(b == ITERS, "low-priority stream ran %u/%d (starved?)", b, ITERS);
}

int main(int argc, char** argv) {
    const char* cubin = argc > 1 ? argv[1] : "build/order_probe.cubin";

    cu(cuInit(0), "cuInit");
    CUdevice dev; CUcontext ctx;
    cu(cuDeviceGet(&dev, 0), "cuDeviceGet");
    cu(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");

    CUmodule mod;
    cu(cuModuleLoad(&mod, cubin), "cuModuleLoad");
    cu(cuModuleGetFunction(&f_bump, mod, "bump"), "get bump");
    cu(cuModuleGetFunction(&f_relay, mod, "relay"), "get relay");
    cu(cuMemAlloc(&d_a, 4), "alloc a");
    cu(cuMemAlloc(&d_b, 4), "alloc b");

    /* A high- and a low-priority stream, so the dispatcher has something to
     * choose between. CUDA's convention is that lower means higher priority.
     *
     * Flags are 0 — BLOCKING streams — deliberately. A CU_STREAM_NON_BLOCKING
     * stream is defined not to order against the legacy default stream, so the
     * blocking-copy case below would have nothing to test and would fail on
     * stock CUDA too. Blocking streams are also what applications get by
     * default from cudaStreamCreate. */
    int lo_p = 0, hi_p = 0;
    cuCtxGetStreamPriorityRange(&hi_p, &lo_p);
    CUstream s_hi, s_lo;
    cu(cuStreamCreateWithPriority(&s_hi, 0, hi_p), "stream hi");
    cu(cuStreamCreateWithPriority(&s_lo, 0, lo_p), "stream lo");

    printf("dispatch ordering tests (LITHOS_DISPATCH=%s)\n",
           getenv("LITHOS_DISPATCH") ? getenv("LITHOS_DISPATCH") : "unset");

    test_async_memcpy(s_hi);
    test_blocking_memcpy(s_hi);
    test_event_dependency(s_hi, s_lo);
    test_stream_query(s_hi);
    test_priority_progress(s_hi, s_lo);

    cuStreamDestroy(s_hi);
    cuStreamDestroy(s_lo);

    printf("ORDERING: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
