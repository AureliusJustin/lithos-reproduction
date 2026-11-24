/* Host-only test of the external admission gate (src/gate.c); no GPU needed.
 * build+run: make test_gate   (or: cc -O2 -pthread tests/test_gate.c src/gate.c -o build/test_gate && build/test_gate) */
#define _GNU_SOURCE
#include "../src/gate.h"

#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

typedef struct { uint64_t magic, seq; int32_t allow, budget; double t; char pad[32]; } Rec;
static volatile Rec* rec;

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static void put(int allow, double age) { rec->allow = allow; rec->t = now_s() - age; rec->seq++; rec->magic = 0x4152424754450001ULL; }

static void* opener(void* arg) {            /* reopens the gate after 50 ms, refreshing the heartbeat meanwhile */
    (void)arg; double t0 = now_s();
    while (now_s() - t0 < 0.05) { put(0, 0); usleep(1000); }
    put(1, 0); return NULL;
}

int main(void) {
    char path[] = "/tmp/lithos_gate_testXXXXXX";
    int fd = mkstemp(path); assert(fd >= 0);
    assert(ftruncate(fd, 64) == 0);
    rec = mmap(NULL, 64, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); assert(rec != MAP_FAILED);
    memset((void*)rec, 0, 64);
    setenv("LITHOS_GATE", path, 1); setenv("LITHOS_GATE_MAX_MS", "300", 1);
    assert(lithos_gate_enabled());

    put(1, 0);                               /* open: no wait */
    assert(lithos_gate_wait() == 0);

    put(0, 0);                               /* closed, kept fresh by a thread that opens it after 50 ms */
    pthread_t th; pthread_create(&th, NULL, opener, NULL);
    uint64_t ns = lithos_gate_wait(); pthread_join(th, NULL);
    printf("closed then opened: waited %.1f ms\n", ns / 1e6);
    assert(ns > 40e6 && ns < 150e6);

    put(0, 1.0);                             /* closed but stale (writer gone): fail open */
    assert(lithos_gate_wait() == 0);

    put(0, 0);                               /* closed and never reopened: bounded by LITHOS_GATE_MAX_MS ... but the record */
    ns = lithos_gate_wait();                 /* goes stale after 500 ms, so the wait ends at min(max, stale) = 300 ms */
    printf("closed, never reopened: waited %.1f ms\n", ns / 1e6);
    assert(ns > 250e6 && ns < 400e6);

    rec->magic = 0; put(0, 0); rec->magic = 7; /* wrong magic: fail open */
    assert(lithos_gate_wait() == 0);

    printf("gate test OK (%llu waits)\n", (unsigned long long)lithos_gate_waits());
    unlink(path); return 0;
}
