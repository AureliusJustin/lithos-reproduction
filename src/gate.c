/* External admission gate: see gate.h. Host-only, no CUDA. */
#define _GNU_SOURCE
#include "gate.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define GATE_MAGIC 0x4152424754450001ULL
#define GATE_SIZE 64

/* the record written by arbiter/live/gate.py (little-endian, naturally aligned) */
typedef struct {
    volatile uint64_t magic;
    volatile uint64_t seq;
    volatile int32_t  allow;
    volatile int32_t  budget;
    volatile double   t;          /* writer's CLOCK_MONOTONIC, seconds */
} GateRec;

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
int lithos_gate_state = -1;
static const GateRec* g_rec;      /* NULL: no gate (or not yet mapped) */
static const char*    g_path;
static double g_stale_s = 0.5, g_max_s = 2.0, g_poll_s = 50e-6;
static uint64_t g_waits, g_waited_ns;

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double g_next_try;          /* the file may appear later (arbiter started after the engine): retry, but rarely */

static double now_s(void);
static void gate_map(void) {
    if (g_rec || !g_path) return;
    double now = now_s();
    if (now < g_next_try) return;     /* at most one open() per 100 ms, not one per launch */
    g_next_try = now + 0.1;
    int fd = open(g_path, O_RDONLY);
    if (fd < 0) return;                                   /* the arbiter has not created it yet: retry later */
    void* p = mmap(NULL, GATE_SIZE, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p != MAP_FAILED) g_rec = (const GateRec*)p;
}

static void gate_init(void) {
    g_path = getenv("LITHOS_GATE");
    if (!g_path || !*g_path) { g_path = NULL; return; }
    const char* v;
    if ((v = getenv("LITHOS_GATE_STALE_MS"))) g_stale_s = atof(v) / 1e3;
    if ((v = getenv("LITHOS_GATE_MAX_MS")))   g_max_s   = atof(v) / 1e3;
    if ((v = getenv("LITHOS_GATE_POLL_US")))  g_poll_s  = atof(v) / 1e6;
    gate_map();
}

int lithos_gate_init(void) {
    pthread_once(&g_once, gate_init);
    int s = g_path != NULL;
    __atomic_store_n(&lithos_gate_state, s, __ATOMIC_RELEASE);
    return s;
}

/* 1 = closed and fresh (must wait), 0 = open / invalid / stale (go) */
static int gate_closed(double now) {
    if (!g_rec) { gate_map(); if (!g_rec) return 0; }
    if (g_rec->magic != GATE_MAGIC) return 0;
    double age = now - g_rec->t;
    if (age > g_stale_s || age < -1.0) return 0;
    return g_rec->allow == 0;
}

/* Gate waits on THIS thread (launch paths compare it around a dispatch: a launch
 * held at the gate must not teach the predictor that the gate time was kernel time). */
__thread uint64_t lithos_gate_tls_waits;

uint64_t lithos_gate_wait_slow(void) {
    double t0 = now_s(), t = t0;
    if (!gate_closed(t)) return 0;
    struct timespec poll = { 0, (long)(g_poll_s * 1e9) };
    while (gate_closed(t) && t - t0 < g_max_s) {
        nanosleep(&poll, NULL);
        t = now_s();
    }
    uint64_t ns = (uint64_t)((t - t0) * 1e9);
    lithos_gate_tls_waits++;
    __atomic_fetch_add(&g_waits, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&g_waited_ns, ns, __ATOMIC_RELAXED);
    return ns;
}

uint64_t lithos_gate_waits(void)     { return __atomic_load_n(&g_waits, __ATOMIC_RELAXED); }
uint64_t lithos_gate_waited_ns(void) { return __atomic_load_n(&g_waited_ns, __ATOMIC_RELAXED); }
