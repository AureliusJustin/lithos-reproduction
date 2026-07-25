#define _GNU_SOURCE
/*
 * dispatch.c — the Dispatcher thread (§5.2, Fig. 9 step 1).
 *
 * The paper decouples submission from the app: "on asynchronous CUDA calls like
 * cuLaunchKernel, LithOS enqueues the kernel and returns control to the
 * application", and dispatcher threads later drain those launch queues onto the
 * GPU. That gives one place where global policy (ordering, the outstanding-work
 * throttle, stealing decisions) is applied.
 *
 * WHAT WE DO AND WHY IT DIFFERS. With LITHOS_DISPATCH=1 every cuLaunchKernel is
 * appended to a queue and submitted by a single dispatcher thread — so the
 * "one authority submits everything" role is reproduced. But the enqueue is a
 * HAND-OFF, not fire-and-forget: the calling thread blocks until the dispatcher
 * has consumed its request.
 *
 * The reason is kernel arguments. cuLaunchKernel takes `void** kernelParams`, an
 * array of POINTERS to argument values, and their sizes are nowhere in the call —
 * they're implicit in the kernel's signature (recoverable only from the cubin's
 * .nv.info EIATTR_KPARAM_INFO). A transparent interposer therefore cannot safely
 * deep-copy them, and if we returned immediately the app would be free to
 * overwrite those buffers before the dispatcher read them. Blocking until consume
 * keeps the caller's argument memory alive without copying it.
 *
 * Consequence: this gives the central-dispatch role but not the CPU-latency
 * decoupling (the app thread still waits), and it costs ~45 us per launch in
 * cross-thread wakeups — which is why it is opt-in and off by default. See
 * docs/FIDELITY.md.
 */
#include "sched_internal.h"
#include <stdio.h>
#include <pthread.h>

/* One queued launch. Allocated on the CALLER's stack: safe precisely because the
 * caller blocks until `done`, so the record (and the argument arrays it points
 * at) outlive the dispatcher's use of them. */
typedef struct DReq {
    CUfunction   f;
    unsigned     gx, gy, gz;      /* grid dimensions  */
    unsigned     bx, by, bz;      /* block dimensions */
    unsigned     shmem;
    CUstream     s;
    void**       params;          /* app-owned; NOT copied (see header comment) */
    void**       extra;
    volatile int done;            /* set by the dispatcher once submitted       */
    struct DReq* next;
} DReq;

/* Singly-linked FIFO of pending launches. */
static DReq*           g_queue_head = NULL;
static DReq*           g_queue_tail = NULL;
static pthread_mutex_t g_queue_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_work_ready = PTHREAD_COND_INITIALIZER;  /* head != NULL  */
static pthread_cond_t  g_req_done   = PTHREAD_COND_INITIALIZER;  /* a req finished */

static pthread_t g_thread;
static int       g_started = 0;
static CUcontext g_ctx = NULL;     /* the app's context, adopted by the thread */

/* Drain the queue forever: pop the oldest request, submit it through the normal
 * scheduler path, then wake whoever is waiting on it. */
static void* dispatcher_main(void* arg) {
    (void)arg;
    if (g_ctx) cuCtxSetCurrent(g_ctx);   /* submit into the app's CUDA context */

    pthread_mutex_lock(&g_queue_lock);
    for (;;) {
        while (!g_queue_head) pthread_cond_wait(&g_work_ready, &g_queue_lock);

        DReq* req = g_queue_head;
        g_queue_head = req->next;
        if (!g_queue_head) g_queue_tail = NULL;

        /* Submit outside the queue lock so other threads can keep enqueuing. */
        pthread_mutex_unlock(&g_queue_lock);

        /* Outstanding-work throttle (§5.3, Fig. 9 ⑤) belongs HERE: the paper
         * throttles *submission* from the dispatcher, deferring work while the
         * GPU backlog is above the threshold. The submit path skips its own
         * throttle when a dispatcher is running, so this is the only wait. */
        throttle_wait();

        submit_launch_now(req->f, req->gx, req->gy, req->gz,
                          req->bx, req->by, req->bz, req->shmem,
                          req->s, req->params, req->extra);
        pthread_mutex_lock(&g_queue_lock);

        req->done = 1;
        pthread_cond_broadcast(&g_req_done);
    }
    return NULL;
}

/* Start the dispatcher on first use, adopting the caller's CUDA context. */
static void dispatcher_ensure(void) {
    if (g_started) return;
    g_started = 1;
    cuCtxGetCurrent(&g_ctx);
    pthread_create(&g_thread, NULL, dispatcher_main, NULL);
}

CUresult dispatch_submit(CUfunction f,
                         unsigned gx, unsigned gy, unsigned gz,
                         unsigned bx, unsigned by, unsigned bz,
                         unsigned shmem, CUstream stream,
                         void** params, void** extra) {
    dispatcher_ensure();

    DReq req = { f, gx, gy, gz, bx, by, bz, shmem, stream, params, extra, 0, NULL };

    pthread_mutex_lock(&g_queue_lock);
    if (g_queue_tail) g_queue_tail->next = &req;
    else              g_queue_head = &req;
    g_queue_tail = &req;
    pthread_cond_signal(&g_work_ready);

    /* Block until the dispatcher has submitted it — this is what keeps `params`
     * (and this stack record) valid without copying them. */
    while (!req.done) pthread_cond_wait(&g_req_done, &g_queue_lock);
    pthread_mutex_unlock(&g_queue_lock);

    return CUDA_SUCCESS;
}
