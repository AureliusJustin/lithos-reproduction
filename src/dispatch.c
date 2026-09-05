#define _GNU_SOURCE
/*
 * dispatch.c — the Dispatcher thread and the launch queues it drains
 * (§5.2 "Kernel Submission", §5.3 "Operation"; Fig. 9 step 1).
 *
 * WHY DEFER AT ALL. Submission to the GPU is irreversible: "once submitted, a
 * kernel's priority or resources cannot be changed, nor can it be rescheduled".
 * Everything the TPC Scheduler can do — reorder by priority, resize the TPC
 * allocation, hold work back so a latency-critical arrival is not stuck behind a
 * batch job — is only possible while the launch is still on THIS side of the
 * driver. So the dispatcher's job is not to make any single launch faster; it is
 * to preserve the option to schedule.
 *
 * THE LOOP. The order of the two waits is the whole design:
 *
 *     1. block until some queue has work
 *     2. wait for GPU capacity          (outstanding-work throttle, §5.3)
 *     3. ONLY THEN choose what to send  (highest-priority queue, FIFO within it)
 *
 * Choosing after the capacity wait rather than before is what makes buffering
 * worth anything. Work that arrives while we are waiting in step 2 joins the
 * queues and is considered in step 3, so a high-priority launch that shows up
 * during the wait goes out first instead of behind whatever happened to be at
 * the head when the wait began. Picking first and then waiting would pin the
 * decision to the oldest information available — the "eager dispatch" the paper
 * warns about, just moved onto another thread.
 *
 * ORDER WITHIN A STREAM IS NEVER CHANGED. CUDA streams are FIFO, so each stream
 * gets its own queue and is drained in arrival order. Reordering happens only
 * BETWEEN streams, where CUDA guarantees no ordering to begin with, so it cannot
 * change any program's meaning.
 *
 * ARGUMENT LIFETIME. Returning control to the application means its argument
 * buffers may be gone by the time we launch, so each request owns a deep copy
 * made through params.h. Kernels whose layout the driver won't report are not
 * buffered at all — the caller drains the stream and submits them inline.
 *
 * ORDERING AGAINST EVERYTHING ELSE. A buffered launch has not reached the stream
 * yet, so any other stream-ordered call (a memcpy, an event record, a free) would
 * overtake it. Those calls drain first; see dispatch_drain and the barrier hooks
 * in interpose.c.
 */
#include "sched_internal.h"
#include "dispatch.h"
#include "params.h"
#include "lithos_sched.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* One queue per stream. MAX_STREAMS is the scheduler's own stream ceiling. */
#define DQ_MAX   MAX_STREAMS
#define DQ_WORDS ((DQ_MAX + 63) / 64)

/* Reused request buffers start at least this big, so the pool stops allocating
 * almost immediately: a process launches the same handful of kernel signatures
 * over and over, and 256 bytes covers essentially all of them. */
#define DREQ_MIN_BYTES 256

#define MAX_EX_ATTRS 8      /* CUlaunchConfig attributes we are willing to copy */

enum { KIND_STD = 0, KIND_EX, KIND_COOP };

/* A buffered launch. Owns everything it needs: after enqueue it references no
 * application memory whatsoever. */
typedef struct DReq {
    struct DReq* next;
    size_t       cap;              /* bytes available in `buf`                  */

    int          kind;
    CUcontext    ctx;              /* context to submit under                   */
    CUfunction   f;
    unsigned     gx, gy, gz, bx, by, bz, shmem;
    CUstream     s;
    uint64_t     enqueue_ns;       /* when the APP called, not when we launch    */
    uint64_t     seq;              /* global arrival order (FIFO tie-break)      */

    void**       params;           /* -> ptrs, or NULL                          */
    void**       extra;            /* -> xtra, or NULL                          */
    void*        ptrs[LITHOS_MAX_PARAMS];
    void*        xtra[5];

    /* KIND_EX only: the config is copied, with its attribute array following. */
    CUlaunchConfig   cfg;
    CUlaunchAttribute attrs[MAX_EX_ATTRS];

    unsigned char buf[];           /* deep-copied argument values               */
} DReq;

typedef struct DQueue {
    CUstream s;
    int      in_use;
    int      prio;                 /* lower value = higher priority (CUDA's own
                                    * convention, so cuStreamCreateWithPriority
                                    * values carry straight through)            */
    DReq*    head;
    DReq*    tail;
    int      n;
    int      busy;                 /* a worker is submitting from this queue.
                                    * At most one at a time, which is what keeps
                                    * the stream's FIFO order intact while other
                                    * workers run other streams concurrently.   */
} DQueue;

static DQueue          g_q[DQ_MAX];
static uint64_t        g_nonempty[DQ_WORDS];   /* bitmap of queues with work    */
static pthread_mutex_t g_dlock  = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_work   = PTHREAD_COND_INITIALIZER;  /* a queue filled  */
static pthread_cond_t  g_room   = PTHREAD_COND_INITIALIZER;  /* depth dropped   */
static pthread_cond_t  g_drained= PTHREAD_COND_INITIALIZER;  /* a submit ended  */

static DReq*     g_pool;               /* free list of request buffers          */
static uint64_t  g_seq;                /* arrival counter                       */
static int       g_legacy_q = -1;      /* queue index of the legacy default stream */

/* "The TPC Scheduler uses dispatcher threadS to monitor launch queues" (§5.3).
 * The plural is load-bearing. cuLaunchKernel BLOCKS once the driver's per-stream
 * command buffer fills, which a best-effort stream bursting work does constantly
 * — and with a single dispatcher, that block stalls every other stream behind it.
 * Measured on the co-located benchmark: one thread left the HP tail at 2.8 ms
 * with priority fully enabled, because the dispatcher was sitting inside a BE
 * launch and could not act on the choice it had already made. */
#define MAX_WORKERS 8
static pthread_t g_workers[MAX_WORKERS];
static int       g_nworkers = 2;       /* LITHOS_DISPATCH_THREADS               */
static int       g_started;
static int       g_max_depth = 1024;   /* LITHOS_DISPATCH_DEPTH, per queue       */
static int       g_capture;            /* >0 while any stream capture is open   */

int g_dispatch_deferred_n;             /* read by dispatch_pending()            */

/* Stats. */
static uint64_t  g_n_deferred, g_n_inline, g_n_reorder;
static CUresult  g_err = CUDA_SUCCESS;

/* ---- bitmap helpers ------------------------------------------------------- */
static inline void bm_set(int i)   { g_nonempty[i >> 6] |=  (1ull << (i & 63)); }
static inline void bm_clear(int i) { g_nonempty[i >> 6] &= ~(1ull << (i & 63)); }
static inline int  bm_any(void) {
    for (int w = 0; w < DQ_WORDS; w++) if (g_nonempty[w]) return 1;
    return 0;
}

/* ---- queue lookup --------------------------------------------------------- */

/* The legacy default stream is implicitly ordered against every blocking stream,
 * so work on it and work anywhere else are mutually visible. NULL and
 * CU_STREAM_LEGACY name it; CU_STREAM_PER_THREAD deliberately does not. */
static inline int is_legacy(CUstream s) {
    return s == NULL || s == (CUstream)CU_STREAM_LEGACY;
}

/* Find (or create) the queue for `s`. Caller holds g_dlock. `prio` is resolved by
 * the caller beforehand, since that needs the scheduler's lock. */
static DQueue* dq_for(CUstream s, int prio) {
    int free_slot = -1;
    for (int i = 0; i < DQ_MAX; i++) {
        if (g_q[i].in_use) { if (g_q[i].s == s) return &g_q[i]; }
        else if (free_slot < 0) free_slot = i;
    }
    if (free_slot < 0) return NULL;              /* table full: submit inline */

    DQueue* q = &g_q[free_slot];
    memset(q, 0, sizeof(*q));
    q->in_use = 1;
    q->s      = s;
    q->prio   = prio;
    if (is_legacy(s)) g_legacy_q = free_slot;
    return q;
}

/* ---- request pool --------------------------------------------------------- */

/* Take a request with room for `need` argument bytes. Caller holds g_dlock. */
static DReq* req_get(size_t need) {
    if (need < DREQ_MIN_BYTES) need = DREQ_MIN_BYTES;
    if (g_pool && g_pool->cap >= need) {
        DReq* r = g_pool;
        g_pool = r->next;
        return r;
    }
    DReq* r = malloc(sizeof(DReq) + need);
    if (!r) return NULL;
    r->cap = need;
    return r;
}
static void req_put(DReq* r) {          /* caller holds g_dlock */
    r->next = g_pool;
    g_pool  = r;
}

/* ---- enqueue -------------------------------------------------------------- */

/* Link `r` onto its queue and wake the dispatcher. Caller holds g_dlock. */
static void enqueue(DQueue* q, DReq* r) {
    int i = (int)(q - g_q);
    r->next = NULL;
    r->seq  = g_seq++;
    if (q->tail) q->tail->next = r; else q->head = r;
    q->tail = r;
    if (q->n++ == 0) bm_set(i);
    __atomic_add_fetch(&g_dispatch_deferred_n, 1, __ATOMIC_RELEASE);
    g_n_deferred++;
    pthread_cond_signal(&g_work);
}

/* ---- the dispatcher ------------------------------------------------------- */

/* Choose the next request: the highest-priority non-empty queue, oldest arrival
 * first among equals. Caller holds g_dlock, and bm_any() is true.
 *
 * Ties broken by global arrival order mean that when every stream shares one
 * priority, selection is by arrival order alone — which is what makes "priority
 * off" a clean control condition for the experiment. Note that this is a STRICT
 * global FIFO only with a single worker: the busy check below skips any queue
 * another worker is draining, so with several workers a younger request is picked
 * whenever the oldest one's queue is taken, priority or no priority. */
static DReq* pick(int* q_index) {
    int best = -1;
    for (int w = 0; w < DQ_WORDS; w++) {
        uint64_t m = g_nonempty[w];
        while (m) {
            int i = (w << 6) + __builtin_ctzll(m);
            m &= m - 1;
            if (g_q[i].busy) continue;   /* another worker owns this stream */
            if (best < 0) { best = i; continue; }
            /* With LITHOS_DISPATCH_PRIO=0 every queue compares equal, so the
             * arrival-order tie-break below decides everything — deferral
             * without the policy. */
            int dp = g_lithos_cfg.dispatch_prio ? g_q[i].prio - g_q[best].prio : 0;
            if (dp < 0 || (dp == 0 && g_q[i].head->seq < g_q[best].head->seq))
                best = i;
        }
    }
    if (best < 0) return NULL;

    DQueue* q = &g_q[best];
    DReq*   r = q->head;
    q->head = r->next;
    if (!q->head) { q->tail = NULL; bm_clear(best); }
    q->n--;

    /* A reorder is any pick that is not the globally oldest buffered launch —
     * i.e. a case where priority actually changed the outcome. */
    for (int w = 0; w < DQ_WORDS; w++) {
        uint64_t m = g_nonempty[w];
        while (m) {
            int i = (w << 6) + __builtin_ctzll(m);
            m &= m - 1;
            if (g_q[i].head->seq < r->seq) { g_n_reorder++; w = DQ_WORDS; break; }
        }
    }

    *q_index = best;
    return r;
}

/* Is there anything a free worker could take right now? Non-empty is not enough:
 * a queue another worker is already draining is off limits. */
static int pickable(void) {
    for (int w = 0; w < DQ_WORDS; w++) {
        uint64_t m = g_nonempty[w];
        while (m) {
            int i = (w << 6) + __builtin_ctzll(m);
            m &= m - 1;
            if (!g_q[i].busy) return 1;
        }
    }
    return 0;
}

static void submit_one(DReq* r) {
    /* Adopt the enqueuing thread's context. Cached rather than asking the driver
     * each time, since virtually every process has a single context and the call
     * is pure overhead after the first.
     *
     * The cache MUST be thread-local: cuCtxSetCurrent binds per thread, so a
     * process-wide cache would let the first worker record the context and every
     * other worker then skip the bind it never performed. Those workers submit
     * with no current context, and every launch they take fails with
     * CUDA_ERROR_INVALID_CONTEXT -- silently, because a deferred launch has
     * already returned success to the application. */
    static __thread CUcontext s_ctx;
    if (r->ctx && r->ctx != s_ctx) { cuCtxSetCurrent(r->ctx); s_ctx = r->ctx; }

    CUresult rc;
    switch (r->kind) {
    case KIND_EX:
        rc = submit_launch_ex_now(&r->cfg, r->f, r->params, r->extra, 1);
        break;
    case KIND_COOP:
        rc = submit_launch_coop_now(r->f, r->gx, r->gy, r->gz,
                                    r->bx, r->by, r->bz, r->shmem, r->s,
                                    r->params, 1);
        break;
    default:
        rc = submit_launch_now(r->f, r->gx, r->gy, r->gz, r->bx, r->by, r->bz,
                               r->shmem, r->s, r->params, r->extra,
                               r->enqueue_ns, 1);
        break;
    }
    /* We already told the application CUDA_SUCCESS, so a failure here can only
     * be reported later — the same way CUDA surfaces asynchronous errors. That
     * delay also hides bugs in the dispatcher itself (a mis-bound context once
     * failed every launch on one worker while the app saw only wrong results),
     * so say something immediately when asked to be verbose. */
    if (rc != CUDA_SUCCESS) {
        if (g_lithos_cfg.verbose)
            fprintf(stderr, "[lithos] deferred launch failed: CUresult=%d\n", (int)rc);
        if (g_err == CUDA_SUCCESS) g_err = rc;
    }
}

static void* dispatcher_main(void* arg) {
    (void)arg;
    /* Everything this thread does is LithOS's own work, so its CUDA calls never
     * drain the launch queues — it IS the thing that drains them. */
    lithos_internal_begin();
    for (;;) {
        pthread_mutex_lock(&g_dlock);
        while (!pickable()) pthread_cond_wait(&g_work, &g_dlock);
        pthread_mutex_unlock(&g_dlock);

        /* Wait for room on the GPU BEFORE deciding what to send. Anything that
         * arrives during this wait gets to compete in pick() below. */
        throttle_wait();

        pthread_mutex_lock(&g_dlock);
        int qi = -1;
        DReq* r = pick(&qi);
        if (!r) { pthread_mutex_unlock(&g_dlock); continue; }   /* taken under us */
        g_q[qi].busy = 1;              /* this stream is ours until we are done */
        pthread_mutex_unlock(&g_dlock);

        /* Outside the lock: cuLaunchKernel can block here for a long time when
         * the driver's command buffer for this stream is full, and holding the
         * lock through that would stall every other worker and every drain. */
        submit_one(r);

        pthread_mutex_lock(&g_dlock);
        g_q[qi].busy = 0;
        req_put(r);
        __atomic_sub_fetch(&g_dispatch_deferred_n, 1, __ATOMIC_RELEASE);
        pthread_cond_broadcast(&g_work);     /* the queue is selectable again */
        pthread_cond_broadcast(&g_drained);
        pthread_cond_broadcast(&g_room);
        pthread_mutex_unlock(&g_dlock);
    }
    return NULL;
}

static void dispatcher_ensure(void) {   /* caller holds g_dlock */
    if (g_started) return;
    g_started = 1;
    const char* d = getenv("LITHOS_DISPATCH_DEPTH");
    if (d && atoi(d) > 0) g_max_depth = atoi(d);
    const char* t = getenv("LITHOS_DISPATCH_THREADS");
    if (t && atoi(t) > 0) g_nworkers = atoi(t);
    if (g_nworkers > MAX_WORKERS) g_nworkers = MAX_WORKERS;
    for (int i = 0; i < g_nworkers; i++)
        pthread_create(&g_workers[i], NULL, dispatcher_main, NULL);
}

/* ---- admission ------------------------------------------------------------ */

/* Common front half of every dispatch_try_*: decide whether this launch may be
 * buffered at all and, if so, take a request with the arguments already copied
 * into it. Returns NULL when the caller must submit inline.
 *
 * On success the request is NOT yet queued and g_dlock IS held, so the caller can
 * finish filling it in and call enqueue(). */
static DReq* admit(CUfunction f, CUstream s, void** params, void** extra,
                   DQueue** q_out) {
    if (!g_lithos_cfg.dispatch) return NULL;

    /* How many argument bytes must we own? At most one of the two forms is in
     * use; a kernel we cannot size is never buffered. */
    size_t need = 0;
    const ParamLayout* lay = NULL;
    long xsz = params_extra_size(extra);
    if (xsz < 0) { g_n_inline++; return NULL; }   /* opaque `extra` -> inline */

    if (xsz > 0) {
        need = LITHOS_EXTRA_BYTES((size_t)xsz);
    } else if (params) {
        lay = params_layout(f);
        /* n == 0 is ambiguous (params.h): the driver reports the same thing for
         * an argument-less kernel and for one it won't describe. The application
         * just handed us an argument array, so the two disagree — submit inline
         * rather than launch this kernel with nothing. */
        if (!lay || lay->n == 0) { g_n_inline++; return NULL; }
        need = lay->total;
    }

    /* Resolve the stream's priority before taking g_dlock: it needs the
     * scheduler's lock, and nesting the two would invite a lock-order bug. */
    int prio = lithos_stream_prio(s);
    CUcontext ctx = NULL;
    cuCtxGetCurrent(&ctx);

    pthread_mutex_lock(&g_dlock);

    /* Checked under the lock, and set under the lock by dispatch_capture_begin,
     * so a capture can never open between the test and the enqueue. */
    if (g_capture) { pthread_mutex_unlock(&g_dlock); g_n_inline++; return NULL; }

    dispatcher_ensure();

    DQueue* q = dq_for(s, prio);

    /* Bounded buffering, PER QUEUE. Unbounded queues would let a host thread
     * that never synchronises grow them without limit, so at the cap the
     * enqueuing thread waits for the dispatcher to make room.
     *
     * The cap is per-queue rather than global because a global one starves the
     * work it is most important not to starve: a best-effort thread bursting
     * launches fills the shared budget, and a latency-critical launch then
     * blocks at the ADMISSION gate — before its priority is ever looked at.
     * Measured, on the co-located benchmark: a global cap left the HP tail at
     * 4.5 ms with priority fully enabled, because the choice never got to
     * happen. Per-queue, each stream's backpressure is its own. */
    while (q && q->n >= g_max_depth)
        pthread_cond_wait(&g_room, &g_dlock);

    DReq* r = q ? req_get(need) : NULL;
    if (!r) {                                    /* no queue slot or no memory */
        pthread_mutex_unlock(&g_dlock);
        g_n_inline++;
        return NULL;
    }

    if (xsz > 0) {
        params_extra_pack(extra, (size_t)xsz, r->buf, r->xtra);
        r->extra  = r->xtra;
        r->params = NULL;
    } else if (lay) {
        params_pack(lay, params, r->buf, r->ptrs);
        r->params = r->ptrs;
        r->extra  = NULL;
    } else {
        r->params = NULL;                        /* genuinely argument-less */
        r->extra  = NULL;
    }

    r->ctx        = ctx;
    r->f          = f;
    r->s          = s;
    r->enqueue_ns = lithos_now_ns();
    *q_out = q;
    return r;
}

int dispatch_try_std(CUfunction f,
                     unsigned gx, unsigned gy, unsigned gz,
                     unsigned bx, unsigned by, unsigned bz,
                     unsigned shmem, CUstream stream,
                     void** params, void** extra) {
    DQueue* q = NULL;
    DReq* r = admit(f, stream, params, extra, &q);
    if (!r) return 0;

    r->kind = KIND_STD;
    r->gx = gx; r->gy = gy; r->gz = gz;
    r->bx = bx; r->by = by; r->bz = bz;
    r->shmem = shmem;
    enqueue(q, r);
    pthread_mutex_unlock(&g_dlock);
    return 1;
}

int dispatch_try_ex(const CUlaunchConfig* cfg, CUfunction f,
                    void** params, void** extra) {
    /* The config and its attribute array are the caller's, so both are copied. */
    if (cfg->numAttrs > MAX_EX_ATTRS) return 0;

    DQueue* q = NULL;
    DReq* r = admit(f, cfg->hStream, params, extra, &q);
    if (!r) return 0;

    r->kind = KIND_EX;
    r->cfg  = *cfg;
    for (unsigned i = 0; i < cfg->numAttrs; i++) r->attrs[i] = cfg->attrs[i];
    r->cfg.attrs = cfg->numAttrs ? r->attrs : NULL;
    enqueue(q, r);
    pthread_mutex_unlock(&g_dlock);
    return 1;
}

int dispatch_try_coop(CUfunction f,
                      unsigned gx, unsigned gy, unsigned gz,
                      unsigned bx, unsigned by, unsigned bz,
                      unsigned shmem, CUstream stream, void** params) {
    DQueue* q = NULL;
    DReq* r = admit(f, stream, params, NULL, &q);
    if (!r) return 0;

    r->kind = KIND_COOP;
    r->gx = gx; r->gy = gy; r->gz = gz;
    r->bx = bx; r->by = by; r->bz = bz;
    r->shmem = shmem;
    enqueue(q, r);
    pthread_mutex_unlock(&g_dlock);
    return 1;
}

/* ---- draining ------------------------------------------------------------- */

/* Wait until nothing is buffered for queue `i` (or anywhere, if i < 0) and the
 * dispatcher is not mid-submit for it. Caller holds g_dlock. */
static void wait_drained(int i) {
    for (;;) {
        int busy = 0;
        if (i < 0) {
            busy = bm_any();
            for (int k = 0; !busy && k < DQ_MAX; k++) busy = g_q[k].busy;
        } else {
            busy = (g_q[i].n > 0) || g_q[i].busy;
        }
        if (!busy) return;
        pthread_cond_wait(&g_drained, &g_dlock);
    }
}

void dispatch_drain(CUstream stream) {
    if (!dispatch_pending()) return;

    /* Anything ordered against the legacy default stream is ordered against
     * everything, so a legacy-stream barrier drains the lot. */
    if (is_legacy(stream)) { dispatch_drain_all(); return; }

    pthread_mutex_lock(&g_dlock);
    for (int i = 0; i < DQ_MAX; i++)
        if (g_q[i].in_use && g_q[i].s == stream) { wait_drained(i); break; }
    /* ...and the legacy queue too: its work is implicitly ordered before ours. */
    if (g_legacy_q >= 0) wait_drained(g_legacy_q);
    pthread_mutex_unlock(&g_dlock);
}

void dispatch_drain_all(void) {
    if (!dispatch_pending()) return;
    pthread_mutex_lock(&g_dlock);
    wait_drained(-1);
    pthread_mutex_unlock(&g_dlock);
}

CUresult dispatch_take_error(void) {
    if (g_err == CUDA_SUCCESS) return CUDA_SUCCESS;
    pthread_mutex_lock(&g_dlock);
    CUresult e = g_err;
    g_err = CUDA_SUCCESS;
    pthread_mutex_unlock(&g_dlock);
    return e;
}

/* ---- stream capture ------------------------------------------------------- *
 * Capture is scoped to the capturing stream, and which graph a launch lands in
 * depends on when it is issued. Buffering a launch across that boundary would
 * put it in the wrong graph — or in none. So the whole window is a no-defer
 * zone: the flag goes up FIRST (under the same lock admit() tests it under, so
 * no launch can slip between the test and the enqueue), then whatever was
 * already buffered is drained. Captures are rare and short, so nothing is lost
 * by being blunt here. */
void dispatch_capture_begin(CUstream s) {
    if (!g_lithos_cfg.dispatch) return;
    pthread_mutex_lock(&g_dlock);
    g_capture++;
    pthread_mutex_unlock(&g_dlock);
    dispatch_drain(s);
}
void dispatch_capture_end(void) {
    if (!g_lithos_cfg.dispatch) return;
    pthread_mutex_lock(&g_dlock);
    if (g_capture > 0) g_capture--;
    pthread_mutex_unlock(&g_dlock);
}

void dispatch_stream_destroyed(CUstream s) {
    if (!g_lithos_cfg.dispatch) return;
    dispatch_drain(s);
    pthread_mutex_lock(&g_dlock);
    for (int i = 0; i < DQ_MAX; i++)
        if (g_q[i].in_use && g_q[i].s == s) {
            g_q[i].in_use = 0;
            if (g_legacy_q == i) g_legacy_q = -1;
            break;
        }
    pthread_mutex_unlock(&g_dlock);
}

void dispatch_stats(uint64_t* deferred, uint64_t* inline_fallback, uint64_t* reorders) {
    if (deferred)        *deferred        = g_n_deferred;
    if (inline_fallback) *inline_fallback = g_n_inline;
    if (reorders)        *reorders        = g_n_reorder;
}
