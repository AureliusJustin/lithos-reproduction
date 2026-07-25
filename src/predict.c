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

/* pending measurements (ring) the tracker reaps */
typedef struct { int slot, op, tpcs; CUevent a, b; int used; } Pend;
#define P_PEND 4096
static Pend g_pend[P_PEND];
static int  g_pend_head, g_pend_tail;
static pthread_mutex_t g_pendmtx = PTHREAD_MUTEX_INITIALIZER;

/* event pool */
#define P_EVPOOL 256
static CUevent g_evpool[P_EVPOOL]; static int g_evpool_n;
static pthread_mutex_t g_evmtx = PTHREAD_MUTEX_INITIALIZER;

static double g_outstanding_us = 0;   /* in-flight work (us) */
static int    g_tracker_started = 0;
static pthread_t g_tracker;

void predict_init(void) {
    memset(g_tab, 0, sizeof(g_tab));
    memset(g_op_idx, 0, sizeof(g_op_idx));
    g_pend_head = g_pend_tail = g_evpool_n = 0;
    g_outstanding_us = 0;
}

int predict_next_op(int slot) {
    if (slot < 0 || slot >= P_STREAMS) return 0;
    int k = g_op_idx[slot];
    if (k >= P_OPS - 1) k = P_OPS - 1;         /* clamp; long batches share the last slot */
    g_op_idx[slot] = k + 1;
    return k;
}
void predict_reset_op(int slot) { if (slot >= 0 && slot < P_STREAMS) g_op_idx[slot] = 0; }

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

void predict_submit(int slot, int op, int tpcs, CUevent a, CUevent b) {
    if (!a || !b) { evt_put(a); evt_put(b); return; }
    /* estimate the work now so the throttle sees it immediately; corrected on reap */
    double est = predict_lookup(slot, op, tpcs, 0);
    if (est <= 0) est = 20.0;   /* unknown: assume ~20us in flight */
    pthread_mutex_lock(&g_pendmtx);
    int nxt = (g_pend_head + 1) % P_PEND;
    if (nxt == g_pend_tail) { pthread_mutex_unlock(&g_pendmtx); evt_put(a); evt_put(b); return; }
    g_pend[g_pend_head] = (Pend){slot, op, tpcs, a, b, 1};
    g_pend_head = nxt;
    pthread_mutex_unlock(&g_pendmtx);
    pthread_mutex_lock(&g_pmtx); g_outstanding_us += est; pthread_mutex_unlock(&g_pmtx);
}

double predict_outstanding_us(void) {
    pthread_mutex_lock(&g_pmtx); double v = g_outstanding_us; pthread_mutex_unlock(&g_pmtx);
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
    if (cuEventQuery(p.b) != CUDA_SUCCESS) return 0;   /* not finished; try later */
    float ms = 0; double us = 0;
    if (cuEventElapsedTime(&ms, p.a, p.b) == CUDA_SUCCESS) us = ms * 1000.0;
    predict_record(p.slot, p.op, p.tpcs, us);
    double est = predict_lookup(p.slot, p.op, p.tpcs, 0); if (est <= 0) est = 20.0;
    pthread_mutex_lock(&g_pmtx);
    g_outstanding_us -= est; if (g_outstanding_us < 0) g_outstanding_us = 0;
    pthread_mutex_unlock(&g_pmtx);
    evt_put(p.a); evt_put(p.b);
    pthread_mutex_lock(&g_pendmtx); g_pend_tail = (g_pend_tail + 1) % P_PEND; pthread_mutex_unlock(&g_pendmtx);
    return 1;
}

static CUcontext g_track_ctx = NULL;
static void* tracker_main(void* arg) {
    (void)arg;
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
