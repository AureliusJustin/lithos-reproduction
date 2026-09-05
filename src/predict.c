#define _GNU_SOURCE
/* predict.c -- online latency prediction + the Tracker thread (LithOS §5.7, §5.3).
 * See predict.h. */
#include "predict.h"
#include "real.h"
#include "lithos.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#define P_STREAMS 256
#define P_OPS     2048     /* kernels tracked per batch (ordinal k range) */

/* Per-operator learned state. Two named samples power the §5.5 scaling model;
 * `ema` is the running estimate at whatever TPC count it last ran (for atom
 * sizing / stealing timers when the full model isn't needed). */
typedef struct {
    double l_all, l_one;   /* latency (us) at all-TPC / at 1-TPC; 0 = unknown */
    int    t_all;          /* the "all-TPC" count used for l_all              */
    double ema;            /* EMA of observed latency (us)                    */
    int    ema_tpc;        /* TPC count the EMA was observed at               */
    int    probe_state;    /* 0=need l_all, 1=need l_one, 2=model ready       */
    long   seen;
} OpPred;

static OpPred g_tab[P_STREAMS][/*P_OPS*/ 2048];
static int    g_op_idx[P_STREAMS];
static pthread_mutex_t g_pmtx = PTHREAD_MUTEX_INITIALIZER;

/* Pending measurements (ring) the Tracker reaps.
 *
 * ONE event per launch, recorded AFTER it: the event marks that launch's
 * COMPLETION. Because a stream executes its launches serially, a kernel's
 * duration is the gap between its own completion and the previous one on the same
 * launch queue:  duration(k) = completion(k) - completion(k-1).
 *
 * That halves the per-launch event cost versus bracketing each kernel with a
 * start/stop pair, while still tracking EVERY task's completion — which the paper
 * requires, since the same Tracker signal clears the sync queues (the
 * outstanding-work throttle) and updates the stealing timers, not just the
 * predictor ("as tasks complete, sync queues are cleared and timers updated,
 * potentially refining predictions", §5.3/§5.7).
 *
 * The previous completion event is held per slot in g_prev_evt. It is dropped at
 * a sync/batch boundary so the first kernel of a new batch never measures the
 * idle gap across the sync. */
typedef struct {
    int     slot, op, tpcs;
    unsigned gen;        /* batch generation this launch belongs to */
    double  predicted_us;/* what we predicted for THIS launch (0 = no prediction) */
    CUevent start_evt;   /* NULL unless this launch had no in-batch predecessor */
    CUevent done_evt;
    int     used;
} Pend;
#define P_PEND 4096
static Pend g_pend[P_PEND];
static int  g_pend_head, g_pend_tail;
static pthread_mutex_t g_pendmtx = PTHREAD_MUTEX_INITIALIZER;

/* Batch generation per launch queue, bumped by predict_reset_op at every sync.
 * Each launch is tagged with the generation current when it was submitted, so the
 * Tracker can tell whether two consecutive completions belong to the SAME batch.
 * Differencing is only valid within a batch: across a sync the gap is idle time,
 * not kernel time. Tagging (rather than a "skip next" flag) is exact regardless of
 * how far the Tracker lags behind the app. */
static volatile unsigned g_batch_gen[P_STREAMS];

/* The previous completion on each queue — the reference for the next delta —
 * together with its generation. Owned by the Tracker thread. */
static CUevent  g_prev_evt[P_STREAMS];
static unsigned g_prev_gen[P_STREAMS];
static int      g_prev_valid[P_STREAMS];

/* event pool */
#define P_EVPOOL 256
static CUevent g_evpool[P_EVPOOL]; static int g_evpool_n;
static pthread_mutex_t g_evmtx = PTHREAD_MUTEX_INITIALIZER;

static double g_outstanding_us = 0;   /* in-flight work (us)                    */
static int    g_outstanding_n  = 0;   /* in-flight LAUNCHES (see the throttle)   */
static int    g_tracker_started = 0;
static pthread_t g_tracker;

void predict_init(void) {
    memset(g_tab, 0, sizeof(g_tab));
    memset(g_op_idx, 0, sizeof(g_op_idx));
    g_pend_head = g_pend_tail = g_evpool_n = 0;
    g_outstanding_us = 0;
    g_outstanding_n = 0;
}

int predict_next_op(int slot) {
    if (slot < 0 || slot >= P_STREAMS) return 0;
    int k = g_op_idx[slot];
    if (k >= P_OPS - 1) k = P_OPS - 1;         /* clamp; long batches share the last slot */
    g_op_idx[slot] = k + 1;
    return k;
}
/* A sync ends the batch. Reset the ordinal AND drop this queue's completion
 * reference: the next batch's first kernel must not measure the idle gap across
 * the sync. (The event object is leaked back to the pool lazily by the Tracker if
 * it is still in flight; marking it stale here is enough for correctness.) */
void predict_reset_op(int slot) {
    if (slot < 0 || slot >= P_STREAMS) return;
    g_op_idx[slot] = 0;
    __atomic_add_fetch(&g_batch_gen[slot], 1, __ATOMIC_RELAXED);   /* new batch */
}

double predict_lookup(int slot, int op, int tpcs, uint64_t blocks) {
    if (slot < 0 || slot >= P_STREAMS || op < 0 || op >= P_OPS || tpcs < 1) return 0;
    pthread_mutex_lock(&g_pmtx);
    OpPred* o = &g_tab[slot][op];
    double pred = 0;
    if (o->probe_state == 2 && o->t_all > 1) {
        /* l = m/t + b from the two samples */
        double m = (o->l_one - o->l_all) * o->t_all / (double)(o->t_all - 1);
        double b = o->l_one - m;
        pred = m / tpcs + b;
    } else if (o->ema > 0) {
        /* conservative optimal-linear scaling from the last observation */
        pred = o->ema * (double)o->ema_tpc / (double)tpcs;
    }
    pthread_mutex_unlock(&g_pmtx);
    (void)blocks;
    return pred > 0 ? pred : 0;
}

int predict_rightsize(int slot, int op, int all_tpcs, double slip, int* want_probe) {
    *want_probe = 0;
    if (slot < 0 || slot >= P_STREAMS || op < 0 || op >= P_OPS || all_tpcs < 2) return 0;
    pthread_mutex_lock(&g_pmtx);
    OpPred* o = &g_tab[slot][op];
    int t_min = 0;
    if (o->probe_state == 0)      *want_probe = all_tpcs;   /* collect l_all first */
    else if (o->probe_state == 1) *want_probe = 1;          /* then l_one          */
    else if (o->probe_state == 2 && o->t_all > 1) {
        double m = (o->l_one - o->l_all) * o->t_all / (double)(o->t_all - 1);
        double b = o->l_one - m;
        double denom = slip * o->l_all - b;
        if (denom > 1e-9) {
            double t = m / denom;
            t_min = (int)(t + 0.999);              /* ceil */
            if (t_min < 1) t_min = 1;
            if (t_min > all_tpcs) t_min = all_tpcs;
        } else t_min = all_tpcs;
    }
    pthread_mutex_unlock(&g_pmtx);
    return t_min;
}

static void predict_record(int slot, int op, int tpcs, double us) {
    if (slot < 0 || slot >= P_STREAMS || op < 0 || op >= P_OPS || us <= 0) return;
    pthread_mutex_lock(&g_pmtx);
    OpPred* o = &g_tab[slot][op];
    o->seen++;
    o->ema = o->ema > 0 ? 0.7 * o->ema + 0.3 * us : us;
    o->ema_tpc = tpcs;
    /* fill the scaling-model points as the right-sizer probes them */
    if (o->probe_state == 0)      { o->l_all = us; o->t_all = tpcs; o->probe_state = 1; }
    else if (o->probe_state == 1 && tpcs == 1) { o->l_one = us; o->probe_state = 2; }
    else if (o->probe_state == 1 && o->t_all && tpcs == o->t_all) o->l_all = 0.7*o->l_all + 0.3*us;
    double ema = o->ema; long seen = o->seen;
    pthread_mutex_unlock(&g_pmtx);
    if (getenv("LITHOS_LOG_PREDICT"))
        fprintf(stderr, "[predict] op(slot=%d,k=%d) @%dTPC measured=%.1fus ema=%.1fus seen=%ld\n",
                slot, op, tpcs, us, ema, seen);
}

/* ---- capture guard -------------------------------------------------------
 * CUDA stream capture is fragile: while a thread is between cuStreamBeginCapture
 * and cuStreamEndCapture, CUDA calls issued on the same context from OTHER threads
 * (our Tracker's cuEventQuery / cuEventCreate / cuEventDestroy) can invalidate the
 * capture ("operation failed due to a previous error during capture") or crash it.
 * sched.c raises this flag around capture so the Tracker parks itself; the pending
 * measurements simply stay queued and are reaped afterwards. */
/* A simple flag is NOT enough: the Tracker could test it, be descheduled, and then
 * issue its CUDA calls after a capture has opened. So capture_begin also WAITS for
 * the Tracker to leave any in-progress CUDA section (g_track_busy), and the
 * Tracker only enters that section while no capture is open. Both sides take
 * g_capmtx, which makes the two states mutually exclusive. */
static int             g_capture_active = 0;   /* open captures (guarded by g_capmtx) */
static int             g_track_busy     = 0;   /* Tracker inside CUDA calls           */
static pthread_mutex_t g_capmtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_capcv  = PTHREAD_COND_INITIALIZER;

void predict_capture_begin(void) {
    pthread_mutex_lock(&g_capmtx);
    g_capture_active++;
    while (g_track_busy) pthread_cond_wait(&g_capcv, &g_capmtx);  /* drain the Tracker */
    pthread_mutex_unlock(&g_capmtx);
}
void predict_capture_end(void) {
    pthread_mutex_lock(&g_capmtx);
    if (g_capture_active > 0) g_capture_active--;
    pthread_cond_broadcast(&g_capcv);
    pthread_mutex_unlock(&g_capmtx);
}
/* Try to enter the Tracker's CUDA section; 0 means "a capture is open, back off". */
static int track_enter(void) {
    pthread_mutex_lock(&g_capmtx);
    if (g_capture_active > 0) { pthread_mutex_unlock(&g_capmtx); return 0; }
    g_track_busy = 1;
    pthread_mutex_unlock(&g_capmtx);
    return 1;
}
static void track_leave(void) {
    pthread_mutex_lock(&g_capmtx);
    g_track_busy = 0;
    pthread_cond_broadcast(&g_capcv);
    pthread_mutex_unlock(&g_capmtx);
}

/* ---- event pool ---- */
CUevent predict_evt_get(void) {
    CUevent e = NULL;
    pthread_mutex_lock(&g_evmtx);
    if (g_evpool_n > 0) e = g_evpool[--g_evpool_n];
    pthread_mutex_unlock(&g_evmtx);
    if (!e) { if (cuEventCreate(&e, CU_EVENT_DEFAULT) != CUDA_SUCCESS) e = NULL; }
    return e;
}
static void evt_put(CUevent e) {
    if (!e) return;
    pthread_mutex_lock(&g_evmtx);
    if (g_evpool_n < P_EVPOOL) g_evpool[g_evpool_n++] = e; else cuEventDestroy(e);
    pthread_mutex_unlock(&g_evmtx);
}

/* Queue one launch's completion event for the Tracker. `done_evt` must already
 * have been recorded on the launch stream AFTER the launch. */
void predict_submit(int slot, int op, int tpcs, CUevent start_evt, CUevent done_evt) {
    if (!done_evt) { evt_put(start_evt); return; }
    /* Credit the estimated work immediately so the throttle sees this launch as
     * in flight right away; the estimate is corrected when the event is reaped. */
    double est_at_submit = predict_lookup(slot, op, tpcs, 0);   /* scored on reap */
    double est = est_at_submit > 0 ? est_at_submit : 20.0;  /* unknown: ~20us in flight */

    pthread_mutex_lock(&g_pendmtx);
    int next = (g_pend_head + 1) % P_PEND;
    if (next == g_pend_tail) {          /* ring full: drop this measurement */
        pthread_mutex_unlock(&g_pendmtx);
        evt_put(start_evt); evt_put(done_evt);
        return;
    }
    unsigned gen = (slot >= 0 && slot < P_STREAMS)
                 ? __atomic_load_n(&g_batch_gen[slot], __ATOMIC_RELAXED) : 0;
    g_pend[g_pend_head] = (Pend){ slot, op, tpcs, gen, est_at_submit, start_evt, done_evt, 1 };
    g_pend_head = next;
    pthread_mutex_unlock(&g_pendmtx);

    pthread_mutex_lock(&g_pmtx);
    g_outstanding_us += est;
    g_outstanding_n++;
    pthread_mutex_unlock(&g_pmtx);
}

double predict_outstanding_us(void) {
    pthread_mutex_lock(&g_pmtx); double v = g_outstanding_us; pthread_mutex_unlock(&g_pmtx);
    return v;
}
int predict_outstanding_n(void) {
    pthread_mutex_lock(&g_pmtx); int v = g_outstanding_n; pthread_mutex_unlock(&g_pmtx);
    return v;
}

/* Reap any completed measurement (FIFO): if the stop event is done, compute the
 * elapsed time, update the table, and release the events + outstanding credit. */
static int reap_one(void) {
    pthread_mutex_lock(&g_pendmtx);
    if (g_pend_tail == g_pend_head) { pthread_mutex_unlock(&g_pendmtx); return 0; }
    Pend p = g_pend[g_pend_tail];
    pthread_mutex_unlock(&g_pendmtx);
    if (!p.used) return 0;

    /* Enter the CUDA section: refuses (and we back off) while a capture is open,
     * and blocks capture_begin from returning until we leave. */
    if (!track_enter()) return 0;

    /* Has this launch finished? If not, leave it queued and retry later. */
    if (cuEventQuery(p.done_evt) != CUDA_SUCCESS) { track_leave(); return 0; }

    /* Duration = time since the PREVIOUS completion on this launch queue. With no
     * predecessor (first kernel of a batch) we can't derive a duration, but the
     * completion itself still counts — this event simply becomes the reference
     * for the next one. */
    int valid_slot = (p.slot >= 0 && p.slot < P_STREAMS);
    int scored = 0;

    /* A launch with no in-batch predecessor carries its own start event, so it can
     * still be timed directly. Without this, workloads that sync after EVERY
     * launch (latency-critical inference) would never be measured at all — every
     * launch is the first of its batch, so there is no previous completion to
     * difference against. */
    if (p.start_evt) {
        float ms = 0;
        if (cuEventElapsedTime(&ms, p.start_evt, p.done_evt) == CUDA_SUCCESS && ms >= 0) {
            double actual_us = ms * 1000.0;
            if (p.predicted_us > 0 && getenv("LITHOS_PREDICT_ACC"))
                fprintf(stderr, "[acc] slot=%d op=%d tpc=%d pred=%.1f actual=%.1f err=%.1f\n",
                        p.slot, p.op, p.tpcs, p.predicted_us, actual_us, p.predicted_us - actual_us);
            predict_record(p.slot, p.op, p.tpcs, actual_us);
            scored = 1;
        }
        evt_put(p.start_evt);
    }

    if (!scored && valid_slot && g_prev_valid[p.slot]) {
        /* Only difference against the previous completion if it belongs to the
         * SAME batch — otherwise the gap spans a sync and is idle time. */
        if (g_prev_gen[p.slot] == p.gen) {
            float ms = 0;
            if (cuEventElapsedTime(&ms, g_prev_evt[p.slot], p.done_evt) == CUDA_SUCCESS && ms >= 0) {
                double actual_us = ms * 1000.0;
                /* Prediction accuracy, scored the way the paper does: compare the
                 * prediction MADE FOR THIS LAUNCH against the measured duration.
                 * Emitted as one line per launch so an external script can compute
                 * misprediction rate and error percentiles. */
                if (p.predicted_us > 0 && getenv("LITHOS_PREDICT_ACC"))
                    fprintf(stderr, "[acc] slot=%d op=%d tpc=%d pred=%.1f actual=%.1f err=%.1f\n",
                            p.slot, p.op, p.tpcs, p.predicted_us, actual_us,
                            p.predicted_us - actual_us);
                predict_record(p.slot, p.op, p.tpcs, actual_us);
            }
        }
        evt_put(g_prev_evt[p.slot]);         /* recycle the old reference */
    }
    if (valid_slot) {
        g_prev_evt[p.slot]   = p.done_evt;   /* becomes the next reference */
        g_prev_gen[p.slot]   = p.gen;
        g_prev_valid[p.slot] = 1;
    } else {
        evt_put(p.done_evt);
    }
    track_leave();   /* done with CUDA; a waiting capture may now proceed */

    /* Release this launch's share of the outstanding-work counter (§5.3). */
    double est = predict_lookup(p.slot, p.op, p.tpcs, 0);
    if (est <= 0) est = 20.0;
    pthread_mutex_lock(&g_pmtx);
    g_outstanding_us -= est;
    if (g_outstanding_us < 0) g_outstanding_us = 0;
    if (g_outstanding_n > 0) g_outstanding_n--;
    pthread_mutex_unlock(&g_pmtx);

    pthread_mutex_lock(&g_pendmtx);
    g_pend_tail = (g_pend_tail + 1) % P_PEND;
    pthread_mutex_unlock(&g_pendmtx);
    return 1;
}

static CUcontext g_track_ctx = NULL;
static void* tracker_main(void* arg) {
    (void)arg;
    /* The Tracker's cuEventQuery/cuEventElapsedTime resolve to LithOS's own
     * ordering barriers, which would drain the launch queues — and deadlock,
     * since the dispatcher is waiting on the outstanding-work count that only
     * this thread can lower. Mark the thread internal so they pass through.
     * See the re-entrancy guard in lithos.h. */
    lithos_internal_begin();
    if (g_track_ctx) cuCtxSetCurrent(g_track_ctx);   /* events are context-bound */
    for (;;) {
        int did = 0;
        while (reap_one()) did = 1;
        if (!did) usleep(50);       /* idle nap; keeps reap latency low */
    }
    return NULL;
}

void predict_start_tracker(void) {
    pthread_mutex_lock(&g_pmtx);
    int go = !g_tracker_started; g_tracker_started = 1;
    pthread_mutex_unlock(&g_pmtx);
    if (go) {
        cuCtxGetCurrent(&g_track_ctx);   /* capture the app's context for the tracker */
        pthread_create(&g_tracker, NULL, tracker_main, NULL);
    }
}
