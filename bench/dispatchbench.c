/*
 * dispatchbench.c — what the Dispatcher is for (§5.2).
 *
 * "Once submitted, a kernel's priority or resources cannot be changed, nor can
 * it be rescheduled. Eagerly dispatching work can lead to sub-optimal
 * scheduling. LithOS therefore defers dispatch to minimize outstanding work on
 * the GPU."
 *
 * The scenario is Figure 10's: a latency-sensitive stream sharing a GPU with a
 * batch stream that submits as fast as it can. One process, two streams, because
 * a dispatcher governs the launch queues of one address space.
 *
 *   BE thread  floods long kernels on a low-priority stream.
 *   HP thread  every period, launches one short kernel and times launch->sync.
 *
 * What the three configurations isolate:
 *
 *   dispatch off      every BE launch goes straight to the GPU. By the time an
 *                     HP kernel arrives there is a deep hardware-side backlog in
 *                     front of it, and nothing can reorder it — that is the
 *                     "cannot be rescheduled" the paper is pointing at.
 *   dispatch, FIFO    work is buffered and the GPU backlog is bounded, but the
 *                     queues drain in arrival order. This is the control: it
 *                     charges the full COST of buffering and takes none of the
 *                     benefit, so the difference from the next line is the
 *                     policy's contribution rather than the mechanism's.
 *   dispatch, prio    the dispatcher waits for GPU capacity and only THEN picks,
 *                     so an HP kernel that arrived during the wait goes first.
 *
 * Reports HP p50/p99/p999 and BE throughput, because the interesting question is
 * what the tail improvement costs the batch job.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

static const char* bench_cubin(const char* name) {
    const char* e = getenv("LITHOS_BENCH_CUBIN");
    if (e && *e) return e;
    static char buf[512];
    snprintf(buf, sizeof buf, "build/kernels/%s", name);
    return buf;
}
#define CK(x) do { CUresult r_=(x); if (r_) { \
    fprintf(stderr,"ERR %d @%d\n",(int)r_,__LINE__); exit(2);} } while (0)

static double us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6 + t.tv_nsec/1e3;
}
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a - *(const double*)b; return x<0?-1:x>0;
}

static CUfunction g_fn;
static CUdeviceptr g_be_buf, g_hp_buf;
static CUstream g_be_s, g_hp_s;
static volatile int g_stop;
static long g_be_count;

/* Grid/iteration counts: BE kernels are big and long, HP kernels small and
 * short — the asymmetry is the point, since a short kernel stuck behind a long
 * one is exactly the head-of-line blocking the design targets. */
static int BE_G = 4096, BE_W = 6000;
static int HP_G = 8,    HP_W = 200;

/* How many BE kernels are issued back-to-back before syncing. This is the knob
 * that decides whether the experiment has anything to say: a burst is precisely
 * the situation the paper describes, where eager submission commits a pile of
 * work the scheduler can no longer touch. With a burst of 1 there is never a
 * backlog to reorder and every configuration must look the same. */
static int BE_BURST = 256;

static void* be_main(void* arg) {
    (void)arg;
    int kid = 0;
    void* a[] = { &g_be_buf, &kid, &BE_W };
    while (!g_stop) {
        for (int i = 0; i < BE_BURST && !g_stop; i++)
            CK(cuLaunchKernel(g_fn, BE_G,1,1, 128,1,1, 0, g_be_s, a, NULL));
        CK(cuStreamSynchronize(g_be_s));
        __atomic_add_fetch(&g_be_count, BE_BURST, __ATOMIC_RELAXED);
    }
    return NULL;
}

int main(int argc, char** argv) {
    int iters     = argc > 1 ? atoi(argv[1]) : 400;
    int period_us = argc > 2 ? atoi(argv[2]) : 2000;
    /* BE kernel shape is tunable because the throttle's behaviour depends
     * entirely on how a kernel compares to the 100 us outstanding-work budget:
     * below it the backlog bound is meaningful, far above it a single kernel
     * already blows the budget and the throttle degenerates into "one kernel at
     * a time". Both regimes are worth measuring. */
    if (argc > 3) BE_G = atoi(argv[3]);
    if (argc > 4) BE_W = atoi(argv[4]);
    if (argc > 5) BE_BURST = atoi(argv[5]);

    CK(cuInit(0));
    CUdevice d; CK(cuDeviceGet(&d,0));
    CUcontext ctx; CK(cuCtxCreate(&ctx,0,d));

    FILE* f = fopen(bench_cubin("work.cubin"),"rb");
    if (!f) { fprintf(stderr,"no work.cubin (make bench)\n"); return 2; }
    fseek(f,0,SEEK_END); long n = ftell(f); fseek(f,0,SEEK_SET);
    char* img = malloc(n); if (fread(img,1,n,f)!=(size_t)n) return 2; fclose(f);
    CUmodule m; CK(cuModuleLoadData(&m,img));
    CK(cuModuleGetFunction(&g_fn,m,"work"));

    CK(cuMemAlloc(&g_be_buf,(size_t)BE_G*128*4));
    CK(cuMemsetD32(g_be_buf,0x3f800000,(size_t)BE_G*128));
    CK(cuMemAlloc(&g_hp_buf,(size_t)HP_G*128*4));
    CK(cuMemsetD32(g_hp_buf,0x3f800000,(size_t)HP_G*128));

    /* Declare the priorities the way an application would: CUDA's own stream
     * priorities, which LithOS reads straight through as launch-queue priority.
     * Nothing here is LithOS-specific — the same binary runs unmodified in every
     * configuration, and without LithOS at all. */
    int hi = 0, lo = 0;
    cuCtxGetStreamPriorityRange(&hi,&lo);
    CK(cuStreamCreateWithPriority(&g_hp_s, CU_STREAM_NON_BLOCKING, hi));
    CK(cuStreamCreateWithPriority(&g_be_s, CU_STREAM_NON_BLOCKING, lo));

    int kid = 0;
    void* hp_args[] = { &g_hp_buf, &kid, &HP_W };

    /* Warm up: module load, JIT, first-touch allocation. */
    for (int i = 0; i < 20; i++) {
        CK(cuLaunchKernel(g_fn, HP_G,1,1, 128,1,1, 0, g_hp_s, hp_args, NULL));
        CK(cuStreamSynchronize(g_hp_s));
    }

    pthread_t be;
    pthread_create(&be, NULL, be_main, NULL);

    double t_start = us();
    double* lat = malloc(iters * sizeof(double));
    /* Split the measurement: time spent INSIDE the launch call (admission into a
     * launch queue) versus time waiting for the result (dispatch + GPU). Without
     * this split a bad tail is unattributable — it could equally be the buffer
     * refusing to accept the launch or the GPU refusing to run it. */
    double* enq = malloc(iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        double t0 = us();
        CK(cuLaunchKernel(g_fn, HP_G,1,1, 128,1,1, 0, g_hp_s, hp_args, NULL));
        double t1 = us();
        CK(cuStreamSynchronize(g_hp_s));
        lat[i] = us() - t0;
        enq[i] = t1 - t0;

        double spare = period_us - (us() - t0);
        if (spare > 0) {
            struct timespec ts = { 0, (long)(spare * 1000) };
            nanosleep(&ts, NULL);
        }
    }
    double elapsed = (us() - t_start) / 1e6;

    g_stop = 1;
    pthread_join(be, NULL);

    qsort(lat, iters, sizeof(double), cmpd);
    qsort(enq, iters, sizeof(double), cmpd);
    long be_done = __atomic_load_n(&g_be_count, __ATOMIC_RELAXED);
    printf("HP p50=%.1f p90=%.1f p99=%.1f p999=%.1f us   BE=%.0f kernels/s"
           "   [enqueue p50=%.1f p99=%.1f]\n",
           lat[iters/2], lat[(int)(iters*0.90)], lat[(int)(iters*0.99)],
           lat[(int)(iters*0.999)], be_done / elapsed,
           enq[iters/2], enq[(int)(iters*0.99)]);
    return 0;
}
