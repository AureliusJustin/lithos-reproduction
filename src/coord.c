#define _GNU_SOURCE
/*
 * coord.c — the shared LithOS layer of Figure 8: one system-wide view of GPU
 * state that every LibLithOS instance maps. See coord.h for the design rationale.
 *
 * Layout: a single POSIX shm segment holding a small table of tenant slots. Each
 * process owns exactly one slot, writes only its own fields, and reads everyone
 * else's. A mutex guards slot allocation and range assignment; the per-launch hot
 * path (publishing liveness, computing what may be stolen) takes no lock — it
 * reads plain 64-bit fields, which is safe here because a torn or slightly stale
 * read can only make a tenant look momentarily busier or idler than it is, and
 * both mistakes are self-correcting on the next launch.
 *
 * Crash safety matters: a tenant that dies leaves its slot behind, so slots carry
 * the owner's pid and are reclaimed when that pid is gone.
 */
#include "coord.h"
#include "lithos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <errno.h>

/* Bumped when the segment layout changes (LITH2 added the per-TPC timers), so a
 * stale segment from an older build is rejected rather than misread. */
#define COORD_MAGIC   0x4C49544832ULL   /* "LITH2" */
#define COORD_SHM     "/lithos_coord"

typedef struct {
    int      pid;              /* owner; 0 = free slot                        */
    int      priority;         /* larger = more important                     */
    int      quota;            /* guaranteed TPCs (<=0 = unrestricted)        */
    int      tpc_lo, tpc_hi;   /* assigned contiguous range [lo,hi)           */
    uint64_t last_launch_ns;   /* liveness: when this tenant last submitted   */
    int      outstanding;      /* its in-flight launches                      */
    int      _pad;
} CoordTenant;

#define COORD_MAX_TPCS 64      /* the disable mask is 64 bits (see tpc_alloc.c) */

typedef struct {
    uint64_t        magic;
    int             n_tpcs;                  /* device TPC count (first writer wins) */
    int             _pad;
    pthread_mutex_t lock;                    /* process-shared                */
    CoordTenant     tenant[COORD_MAX_TENANTS];
    /* Per-TPC timers (§5.3): expected-completion time of the work submitted to
     * each TPC. Device-wide and lock-free — see coord_mark_tpcs_busy. */
    uint64_t        tpc_busy_until[COORD_MAX_TPCS];
} CoordShm;

static CoordShm* g_shm;        /* mapped segment, NULL when inactive */
static int       g_slot = -1;  /* our index into tenant[]            */
static int       g_active;

#define CLOG(...) do { if (getenv("LITHOS_LOG_COORD")) { \
    fprintf(stderr, "[coord] " __VA_ARGS__); fputc('\n', stderr); } } while (0)

/* Is a pid still alive? Used to reclaim slots left by crashed tenants.
 * EPERM means it exists but belongs to another user — still alive. */
static int pid_alive(int pid) {
    if (pid <= 0) return 0;
    return kill(pid, 0) == 0 || errno == EPERM;
}

/* Drop slots whose owning process is gone. Caller holds the lock. */
static void reap_dead(void) {
    for (int i = 0; i < COORD_MAX_TENANTS; i++) {
        CoordTenant* t = &g_shm->tenant[i];
        if (t->pid && !pid_alive(t->pid)) {
            CLOG("reaping dead tenant pid=%d slot=%d", t->pid, i);
            memset(t, 0, sizeof(*t));
        }
    }
}

/* Assign this tenant a contiguous range that overlaps no live tenant.
 * Tenants are packed in slot order; if the device runs out of TPCs we wrap, which
 * degrades to sharing rather than failing the launch. Caller holds the lock. */
static void assign_range(int slot) {
    CoordTenant* me = &g_shm->tenant[slot];
    int n = g_shm->n_tpcs > 0 ? g_shm->n_tpcs : 1;
    if (me->quota <= 0) { me->tpc_lo = 0; me->tpc_hi = n; return; }

    int cursor = 0;
    for (int i = 0; i < COORD_MAX_TENANTS; i++) {
        CoordTenant* t = &g_shm->tenant[i];
        if (i == slot || !t->pid || t->quota <= 0) continue;
        if (t->tpc_hi > cursor) cursor = t->tpc_hi;   /* pack after existing tenants */
    }
    if (cursor + me->quota > n) cursor = 0;           /* oversubscribed: wrap */
    me->tpc_lo = cursor;
    me->tpc_hi = cursor + me->quota;
    if (me->tpc_hi > n) me->tpc_hi = n;
}

int coord_join(int n_tpcs, int quota, int priority) {
    if (g_active) return 1;
    if (!g_lithos_cfg.coordinator) return 0;          /* LITHOS_COORD=0 */

    /* Exactly one process may initialise the segment, and O_EXCL is what decides
     * which. Creating with a plain O_CREAT and then inferring "I created it" from
     * the size being zero is a race: two tenants starting together both see size
     * 0, both ftruncate, and both memset -- so the second silently erases the
     * first's registration. Each then sees an empty table, and both are assigned
     * the same TPC range, which is precisely the overlap the coordinator exists
     * to prevent. It is intermittent, so it survives casual testing. */
    int creator = 1;
    int fd = shm_open(COORD_SHM, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0 && errno == EEXIST) { creator = 0; fd = shm_open(COORD_SHM, O_RDWR, 0600); }
    if (fd < 0) { CLOG("shm_open failed; coordinator disabled"); return 0; }

    if (creator && ftruncate(fd, sizeof(CoordShm)) != 0) {
        close(fd); shm_unlink(COORD_SHM);      /* do not leave an unusable stub */
        return 0;
    }
    if (!creator) {
        /* The creator sizes the segment before it publishes the magic; mapping it
         * before the ftruncate lands would fault on first touch. */
        struct stat st;
        for (int i = 0; i < 2000; i++) {
            if (fstat(fd, &st) == 0 && (size_t)st.st_size >= sizeof(CoordShm)) break;
            usleep(1000);
        }
        if (fstat(fd, &st) != 0 || (size_t)st.st_size < sizeof(CoordShm)) {
            close(fd); CLOG("shm never sized; coordinator disabled"); return 0;
        }
    }

    void* p = mmap(NULL, sizeof(CoordShm), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { CLOG("mmap failed; coordinator disabled"); return 0; }
    g_shm = p;

    if (creator) {
        /* Zeroed by the kernel already; set up the process-shared mutex and only
         * then publish the magic, so a joiner that sees the magic sees a fully
         * initialised segment. */
        pthread_mutexattr_t a;
        pthread_mutexattr_init(&a);
        pthread_mutexattr_setpshared(&a, PTHREAD_PROCESS_SHARED);
        pthread_mutexattr_setrobust(&a, PTHREAD_MUTEX_ROBUST);   /* survive a crash */
        pthread_mutex_init(&g_shm->lock, &a);
        pthread_mutexattr_destroy(&a);
        __atomic_store_n(&g_shm->magic, COORD_MAGIC, __ATOMIC_RELEASE);
    } else {
        int ready = 0;
        for (int i = 0; i < 2000 && !ready; i++) {
            if (__atomic_load_n(&g_shm->magic, __ATOMIC_ACQUIRE) == COORD_MAGIC) ready = 1;
            else usleep(1000);
        }
        if (!ready) { munmap(p, sizeof(CoordShm)); g_shm = NULL;
            CLOG("shm never initialised; coordinator disabled"); return 0; }
    }

    int rc = pthread_mutex_lock(&g_shm->lock);
    if (rc == EOWNERDEAD) pthread_mutex_consistent(&g_shm->lock);  /* previous holder died */
    if (g_shm->n_tpcs == 0) g_shm->n_tpcs = n_tpcs;
    reap_dead();
    for (int i = 0; i < COORD_MAX_TENANTS; i++) {
        if (g_shm->tenant[i].pid) continue;
        g_slot = i;
        CoordTenant* me = &g_shm->tenant[i];
        memset(me, 0, sizeof(*me));
        me->pid = (int)getpid();
        me->quota = quota;
        me->priority = priority;
        assign_range(i);
        break;
    }
    pthread_mutex_unlock(&g_shm->lock);

    if (g_slot < 0) { munmap(g_shm, sizeof(CoordShm)); g_shm = NULL; return 0; }
    g_active = 1;
    CLOG("joined slot=%d pid=%d quota=%d prio=%d range=[%d,%d) of %d TPCs",
         g_slot, (int)getpid(), quota, priority,
         g_shm->tenant[g_slot].tpc_lo, g_shm->tenant[g_slot].tpc_hi, g_shm->n_tpcs);
    return 1;
}

int coord_my_range(int* lo, int* hi) {
    if (!g_active) return 0;
    if (lo) *lo = g_shm->tenant[g_slot].tpc_lo;
    if (hi) *hi = g_shm->tenant[g_slot].tpc_hi;
    return 1;
}

void coord_note_launch(uint64_t now_ns, int outstanding) {
    if (!g_active) return;
    CoordTenant* me = &g_shm->tenant[g_slot];
    me->last_launch_ns = now_ns;      /* published for other tenants' idle checks */
    me->outstanding    = outstanding;
}

/* TPCs we may borrow: those owned by OTHER live tenants that are currently idle.
 *
 * The paper's safeguards (§5.3, §5.4):
 *   - never steal from a tenant of HIGHER priority — its quota is a guarantee;
 *   - stop stealing the moment that tenant has runnable work again, which falls
 *     out of the idle test since it publishes last_launch_ns on every launch.
 */
uint64_t coord_stealable_tpcs(uint64_t now_ns, uint64_t idle_ns) {
    if (!g_active) return 0;
    CoordTenant* me = &g_shm->tenant[g_slot];
    uint64_t borrow = 0;

    for (int i = 0; i < COORD_MAX_TENANTS; i++) {
        CoordTenant* t = &g_shm->tenant[i];
        if (i == g_slot || !t->pid || t->quota <= 0) continue;
        if (t->priority > me->priority) continue;          /* never rob a higher priority */

        int idle = (t->last_launch_ns == 0) ||
                   (now_ns > t->last_launch_ns && now_ns - t->last_launch_ns > idle_ns) ||
                   (t->outstanding == 0);
        if (!idle) continue;

        for (int c = t->tpc_lo; c < t->tpc_hi && c < 64; c++) borrow |= (1ull << c);
    }
    return borrow;
}

/* ---- Per-TPC timers (§5.3) ------------------------------------------------ */

/* Fallback when the coordinator is off, so intra-process stealing still gets
 * timers. Same shape as the shared array, so both paths share all the logic. */
static uint64_t g_local_busy[COORD_MAX_TPCS];

/* Ceiling on how far ahead a TPC may be declared busy (see coord_mark_tpcs_busy).
 * Generous next to an atom (hundreds of microseconds) and next to the 100 us
 * outstanding-work limit, but short enough that a bad prediction cannot park a
 * TPC for a whole training step. */
#define COORD_BUSY_HORIZON_NS  (200ull * 1000 * 1000)   /* 200 ms */

static uint64_t* busy_array(void) {
    return (g_active && g_shm) ? g_shm->tpc_busy_until : g_local_busy;
}

/* Lock-free by design. These are 64-bit aligned words written with a plain
 * relaxed store, and the launch path already reads the tenant table this way: a
 * torn or stale read can only make a TPC look busier or freer than it is for one
 * launch, and either mistake is corrected by the next submission. Taking the
 * coordinator mutex here would put a cross-process lock on every launch. */
/* A queued launch EXTENDS a TPC's busy horizon; it does not merely propose a
 * deadline of its own.
 *
 * The distinction decides whether the timers work at all. A launch is timed at
 * SUBMISSION, but it does not start then — it starts when the work already
 * queued ahead of it on that TPC has drained. Publishing `now + duration` (the
 * original form) therefore describes a kernel that runs immediately, which is
 * true only for an empty queue. Submit 200 short kernels in a burst and every
 * one of them claims the TPC for the same quarter-millisecond starting now, so
 * the whole 50 ms of queued work reads as ~0.25 ms of busy — and a neighbour
 * scanning a moment later sees a free TPC and steals it, landing behind all of
 * it. That is precisely the head-of-line blocking §5.3 introduces these timers
 * to avoid.
 *
 * Accumulating instead — start = max(now, current horizon), new horizon =
 * start + duration — makes the horizon track the depth of the queue, which is
 * what "estimating kernel (and atom) durations at submission time" has to mean
 * for the estimate to be usable.
 *
 * The horizon is capped so a tenant whose predictions run long cannot pin its
 * TPCs indefinitely: a stale timer only ever costs work conservation, and the
 * cap bounds how long that can last. */
void coord_mark_tpcs_busy(uint64_t tpc_mask, uint64_t now_ns, uint64_t dur_ns) {
    uint64_t* busy = busy_array();
    uint64_t  cap  = now_ns + COORD_BUSY_HORIZON_NS;
    while (tpc_mask) {
        int t = __builtin_ctzll(tpc_mask);
        tpc_mask &= tpc_mask - 1;
        if (t >= COORD_MAX_TPCS) break;
        uint64_t cur   = __atomic_load_n(&busy[t], __ATOMIC_RELAXED);
        uint64_t start = cur > now_ns ? cur : now_ns;   /* queued behind the rest */
        uint64_t until = start + dur_ns;
        if (until > cap) until = cap;
        if (until > cur) __atomic_store_n(&busy[t], until, __ATOMIC_RELAXED);
    }
}

uint64_t coord_busy_tpcs(uint64_t now_ns) {
    uint64_t* busy = busy_array();
    uint64_t m = 0;
    for (int t = 0; t < COORD_MAX_TPCS; t++)
        if (__atomic_load_n(&busy[t], __ATOMIC_RELAXED) > now_ns) m |= (1ull << t);
    return m;
}

int coord_tenant_count(void) {
    if (!g_active) return 0;
    int n = 0;
    for (int i = 0; i < COORD_MAX_TENANTS; i++) if (g_shm->tenant[i].pid) n++;
    return n;
}

void coord_dump(void) {
    if (!g_active) { fprintf(stderr, "[coord] inactive\n"); return; }
    fprintf(stderr, "[coord] %d TPCs, tenants:\n", g_shm->n_tpcs);
    for (int i = 0; i < COORD_MAX_TENANTS; i++) {
        CoordTenant* t = &g_shm->tenant[i];
        if (!t->pid) continue;
        fprintf(stderr, "[coord]   slot=%d pid=%d prio=%d quota=%d range=[%d,%d)%s\n",
                i, t->pid, t->priority, t->quota, t->tpc_lo, t->tpc_hi,
                i == g_slot ? "  <- me" : "");
    }
}

/* Release our slot on exit so a restarting tenant reuses it immediately. */
__attribute__((destructor)) static void coord_leave(void) {
    if (!g_active) return;
    int rc = pthread_mutex_lock(&g_shm->lock);
    if (rc == EOWNERDEAD) pthread_mutex_consistent(&g_shm->lock);
    memset(&g_shm->tenant[g_slot], 0, sizeof(CoordTenant));
    pthread_mutex_unlock(&g_shm->lock);
    g_active = 0;
}
