/*
 * Kernel Atomizer (Section 5.4) -- prologue-splice implementation.
 *
 * The paper's Prelude (Algorithm 1) redirects a launch to a small kernel that
 * range-checks the block index and transfers control into the original kernel.
 * The original ends in EXIT, and *every* register-indirect control transfer we
 * could byte-patch out of ptxas output (CALL / BRX) sets up a return- or
 * convergence-state that the callee's EXIT then violates (INVALID_PC at warp
 * retirement). LithOS gets a clean jump for free from its Rust/LLVM guaranteed
 * tail calls; we cannot reproduce that by patching ptxas output (see the SASS
 * reverse-engineering log in the project memory and sass-jump/).
 *
 * This implementation SIDESTEPS the transfer entirely. Rather than a separate
 * Prelude that must jump into the original, we PREPEND the range-check directly
 * into each app kernel's own .text.<fn> section (via cubin ELF surgery at module
 * load time), so in-range blocks FALL THROUGH into the original code. There is
 * no CALL, no BRX, no BSSY convergence barrier -- the original's EXIT is its own
 * clean top-level exit. Functionally identical to Algorithm 1 (a per-block range
 * gate that runs the original for in-range blocks and skips the rest), achieved
 * transparently with no app source or PTX.
 *
 * Flow:
 *   - cuModuleLoad* is intercepted; atomizer_intercept_cubin() splices the
 *     range-check prologue into every kernel and returns a modified cubin.
 *   - The prologue reads AtomMetadata{lo,hi} from a fixed device address baked
 *     in as a literal (a const-bank global would read garbage: the const bank is
 *     the app's, gridDim excepted, so we use a literal absolute address).
 *   - At launch, the grid is split into atoms; before each atom's full-grid
 *     relaunch we write {lo,hi} to that address with a stream-ordered async copy,
 *     so every block of atom_i observes meta_i before atom_{i+1} overwrites it.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <elf.h>
#include <unistd.h>
#include <time.h>
#include "atomizer.h"
#include "qmd.h"
#include "real.h"
#include "dispatch.h"

#define AZLOG(...) do { if (g_lithos_cfg.verbose) { \
    fprintf(stderr, "[atomizer] " __VA_ARGS__); fputc('\n', stderr); } } while (0)

/* core splicer (atomize_splice.c) */
int atomize_build_prologue(int sm_major, int sm_minor, uint64_t meta_va,
                           unsigned char** out, size_t* len);
int atomize_splice_cubin(const void* cubin, size_t sz, const unsigned char* pro,
                         size_t prolen, void** out, size_t* outsz,
                         char*** names_out, int* n_names_out);

/* Wall-clock helper for the LITHOS_DIAG splice timing below. */
static double az_now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
/* fatbin.c: turn an ELF/fatbin/PTX image into a raw cubin (0 ok, -1 = verbatim).
 * *out==image with *outsz==0 means "already a cubin"; otherwise *out is malloc'd. */
int atomize_image_to_cubin(const void* image, int want_sm, void** out, size_t* outsz);

/* ------------------------------------------------------------------ */
/*  Shared metadata buffer + prologue (one per process)               */
/* ------------------------------------------------------------------ */
/* The spliced-in prologue reads AtomMetadata{lo,hi} from ONE fixed device address
 * (its VA is baked into the prologue's SASS as a literal), so there is exactly one
 * metadata buffer per process. Before each atom's relaunch we write that atom's
 * [lo,hi) there, stream-ordered ahead of the launch. */
static CUdeviceptr     g_meta;            /* device AtomMetadata{lo,hi}            */
static CUevent         g_atom_ev;         /* orders g_meta writes across streams   */
static int             g_atom_ev_valid;   /* has g_atom_ev been recorded yet?      */
static unsigned char*  g_prologue;        /* the range-check SASS bytes (padded)   */
static size_t          g_prolen;          /* its length                            */
static int             g_prologue_ready;  /* prologue + g_meta built?              */
static int             g_dev_sm;          /* running device SM, e.g. 86            */
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;   /* guards all of the above */

/* The Ex launch path has no LithosKernel to carry pred_us, so sched.c stashes the
 * predicted duration here (per thread) just before calling atomizer_dispatch_ex. */
static __thread double g_ex_pred_us = 0;
void atomizer_set_ex_pred(double us) { g_ex_pred_us = us; }

/* sched.c already has to know whether the stream is capturing, and
 * cuStreamIsCapturing is a driver call — so it passes the answer down here rather
 * than making us ask again. -1 means "unknown, look it up yourself". */
static __thread int g_capture_hint = -1;
void atomizer_set_capture_hint(int capturing) { g_capture_hint = capturing; }

/* ------------------------------------------------------------------ */
/*  Gating: which functions may be atomized                           */
/* ------------------------------------------------------------------ */
/* Every launched function falls into one of three classes, and getting this right
 * is what keeps results correct when a module is only partially spliceable:
 *
 *   ATOM     the function's kernel WAS spliced, so it carries the range-check and
 *            may be split into atoms (we set [lo,hi) per atom).
 *   FULL     the function came from a module we spliced, but this kernel's name
 *            wasn't in the spliced set (splice-skipped, or a handle we couldn't
 *            match by name). It MAY still carry a prologue, so it must be launched
 *            with an explicit FULL range — otherwise it would read whatever [lo,hi)
 *            the previous atom left in g_meta and silently drop blocks.
 *   VERBATIM the function is from an un-spliced module: launch it untouched.
 *
 * We record membership by handle. Containers (CUmodule/CUlibrary) remember the
 * kernel NAMES we spliced, and functions resolved from them are classified by name.
 */
#define MAX_MODS  8192
#define MAX_FUNCS 262144

typedef struct {
    void*  handle;    /* CUmodule or CUlibrary                        */
    char** names;     /* kernel names successfully spliced in it      */
    int    n;
} Container;
typedef struct SplicedNames { char** names; int n; } SplicedNames;

static Container g_containers[MAX_MODS];
static int       g_containers_n;

static void* g_atom_funcs[MAX_FUNCS];   /* class ATOM: spliced -> may be split   */
static int   g_atom_funcs_n;
static void* g_full_funcs[MAX_FUNCS];   /* class FULL: force the full range      */
static int   g_full_funcs_n;
static void* g_verb_funcs[MAX_FUNCS];   /* class VERBATIM: distinct un-atomized  */
static int   g_verb_funcs_n;

/* ---- pointer hash set -----------------------------------------------------
 * Function handles are looked up on EVERY launch, and a big app can register a
 * lot of them (TensorRT loads 11k+ kernels), so the membership test is open-
 * addressed rather than a linear scan: O(1) instead of O(n).
 *
 * The table is sized to the next power of two above MAX_FUNCS*2 so it never
 * exceeds 50% load, which keeps probe chains short. Entries are only ever added
 * (a module's kernels stay registered for the process lifetime), so there is no
 * deletion/tombstone handling to get wrong. */
#define FSET_BITS 20                      /* 1M slots for up to 262k entries */
#define FSET_SIZE (1u << FSET_BITS)
#define FSET_MASK (FSET_SIZE - 1)

typedef struct { void** slot; } FuncSet;  /* open-addressed table of pointers */

/* Fibonacci hashing: multiply by 2^64/phi and take the high bits. Handles are
 * allocator pointers whose low bits are mostly alignment zeros, so the high bits
 * of the product mix far better than a plain mask of the address. */
static inline unsigned fset_hash(void* h) {
    return (unsigned)(((uintptr_t)h * 11400714819323198485ull) >> (64 - FSET_BITS));
}
static int fset_has(void** table, void* h) {
    unsigned i = fset_hash(h);
    for (;;) {
        void* cur = table[i];
        if (!cur)     return 0;           /* empty slot: not present */
        if (cur == h) return 1;
        i = (i + 1) & FSET_MASK;          /* linear probe */
    }
}
static void fset_add(void** table, void* h) {
    unsigned i = fset_hash(h);
    for (;;) {
        void* cur = table[i];
        if (cur == h) return;             /* already present */
        if (!cur) { table[i] = h; return; }
        i = (i + 1) & FSET_MASK;
    }
}

/* Membership tables, parallel to the g_*_funcs arrays (which stay as the ordered
 * record used for stats). Allocated lazily so a process that never atomizes pays
 * nothing. */
static void** g_atom_set;
static void** g_full_set;
static void** g_verb_set;
static void** fset_alloc(void) { return calloc(FSET_SIZE, sizeof(void*)); }


/* ------------------------------------------------------------------ */
/*  Coverage statistics (LITHOS_STATS=1 dumps them at exit)           */
/* ------------------------------------------------------------------ */
static void stats_tick(void);

static long g_st_split;      /* launches split into >1 atom      */
static long g_st_atom1;      /* gated launches that stayed 1 atom */
static long g_st_full;       /* launches forced to the full range */
static long g_st_verbatim;   /* launches passed through untouched */
/* Coverage below 100% on a closed-source library is only actionable if you know
 * WHICH kernel escaped. cuFuncGetName (CUDA 12.3+) reports it whichever API handed
 * out the handle -- including paths that never call cuModuleGetFunction -- but it
 * needs a live context, so the name is captured here at the launch rather than in
 * the atexit stats dump. */
static const char* g_verb_names[MAX_FUNCS];
static int         g_verb_names_n;
static void note_verbatim_name(void* f) {
    static CUresult (*get_name)(const char**, CUfunction);
    static int tried;
    if (!tried) { tried = 1;
        get_name = (CUresult(*)(const char**, CUfunction))lithos_real_sym("cuFuncGetName"); }
    const char* nm = NULL;
    if (get_name && get_name(&nm, (CUfunction)f) == CUDA_SUCCESS && nm &&
        g_verb_names_n < MAX_FUNCS)
        g_verb_names[g_verb_names_n++] = nm;   /* driver-owned, lives as long as the module */
}

static void stat_verbatim(void* f) {
    __atomic_fetch_add(&g_st_verbatim, 1, __ATOMIC_RELAXED);
    pthread_mutex_lock(&g_mtx);
    if (!g_verb_set) g_verb_set = fset_alloc();
    if (g_verb_set && !fset_has(g_verb_set, f)) {
        fset_add(g_verb_set, f);
        if (g_verb_funcs_n < MAX_FUNCS) g_verb_funcs[g_verb_funcs_n++] = f;
        if (getenv("LITHOS_DIAG")) note_verbatim_name(f);
    }
    pthread_mutex_unlock(&g_mtx);
    stats_tick();
}
static void write_stats(FILE* fp) {
    long atomized = g_st_split + g_st_atom1 + g_st_full;
    long total = atomized + g_st_verbatim;
    fprintf(fp,
        "[stats] distinct kernels: gated(atomizable)=%d full-range=%d un-atomized=%d\n"
        "[stats] launches: split=%ld single-atom=%ld full-range=%ld un-atomized=%ld\n"
        "[stats] atomization coverage: %ld/%ld launches = %.1f%%\n",
        g_atom_funcs_n, g_full_funcs_n, g_verb_funcs_n,
        g_st_split, g_st_atom1, g_st_full, g_st_verbatim,
        atomized, total, total ? 100.0 * atomized / total : 0.0);

    for (int i = 0; i < g_verb_names_n; i++)
        fprintf(fp, "[diag] un-atomized kernel: %s\n", g_verb_names[i]);

    /* Dispatcher coverage (§5.2). Buffered vs submitted inline matters as much
     * as atomization coverage does: a launch that fell back to inline submission
     * was never schedulable, so it is invisible to every policy above it. */
    if (g_lithos_cfg.dispatch) {
        uint64_t buf = 0, inl = 0, reord = 0;
        dispatch_stats(&buf, &inl, &reord);
        fprintf(fp, "[stats] dispatcher: buffered=%llu inline-fallback=%llu (%.1f%% buffered) "
                    "reordered=%llu\n",
                (unsigned long long)buf, (unsigned long long)inl,
                (buf + inl) ? 100.0 * (double)buf / (double)(buf + inl) : 0.0,
                (unsigned long long)reord);
    }
}
/* Periodic dump to LITHOS_STATS_FILE.<pid> so coverage survives a subprocess
 * (e.g. vLLM's EngineCore) killed rather than exited -- and so the launch-free
 * parent doesn't overwrite the worker's numbers (each pid gets its own file). */
static void write_stats_file(void) {
    const char* base = getenv("LITHOS_STATS_FILE"); if (!base) return;
    char path[512]; snprintf(path, sizeof(path), "%s.%d", base, (int)getpid());
    FILE* fp = fopen(path, "w"); if (fp) { write_stats(fp); fclose(fp); }
}
static void stats_tick(void) {
    static _Atomic long ticks;
    if ((__atomic_add_fetch(&ticks, 1, __ATOMIC_RELAXED) & 0xff) != 0) return;  /* every 256 */
    write_stats_file();
}
__attribute__((destructor)) static void atom_stats_dump(void) {
    if (getenv("LITHOS_STATS")) write_stats(stderr);
    write_stats_file();
}

static Container* find_container(void* h) {
    for (int i = 0; i < g_containers_n; i++) if (g_containers[i].handle == h) return &g_containers[i];
    return NULL;
}
static int names_have(char** names, int n, const char* name) {
    for (int i = 0; i < n; i++) if (!strcmp(names[i], name)) return 1;
    return 0;
}
static void add_atom_func(void* h) {
    if (!g_atom_set) g_atom_set = fset_alloc();
    if (!g_atom_set) return;
    if (fset_has(g_atom_set, h)) return;
    if (g_atom_funcs_n < MAX_FUNCS) g_atom_funcs[g_atom_funcs_n++] = h;
    fset_add(g_atom_set, h);
}
static void add_full_func(void* h) {
    if (!g_full_set) g_full_set = fset_alloc();
    if (!g_full_set) return;
    if (fset_has(g_full_set, h)) return;
    if (g_full_funcs_n < MAX_FUNCS) g_full_funcs[g_full_funcs_n++] = h;
    fset_add(g_full_set, h);
}

void atomizer_init(void) {
    qmd_init();   /* still used by the scheduler for TPC masking */
}

/* Lazily build the shared prologue + metadata buffer (needs a live context). */
static int ensure_prologue(void) {
    if (g_prologue_ready) return 0;
    if (cuMemAlloc(&g_meta, 4096) != CUDA_SUCCESS) return -1;
    uint32_t init[2] = { 0, 0xffffffffu };        /* default: pass ALL blocks */
    cuMemcpyHtoD(g_meta, init, sizeof(init));
    cuEventCreate(&g_atom_ev, CU_EVENT_DISABLE_TIMING);   /* cross-stream metadata order */
    int maj = 0, min = 0; CUdevice dev;
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) return -1;
    cuDeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    g_dev_sm = maj * 10 + min;
    if (atomize_build_prologue(maj, min, (uint64_t)g_meta, &g_prologue, &g_prolen) != 0)
        return -1;
    g_prologue_ready = 1;
    AZLOG("prologue ready: %zu bytes (sm_%d%d), meta buffer @ %#llx",
          g_prolen, maj, min, (unsigned long long)g_meta);
    return 0;
}

/* Total byte span of an ELF cubin image (cuModuleLoadData carries no length). */
static size_t elf_image_size(const unsigned char* b) {
    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)b;
    size_t end = eh->e_shoff + (size_t)eh->e_shnum * eh->e_shentsize;
    size_t pend = eh->e_phoff + (size_t)eh->e_phnum * eh->e_phentsize;
    if (pend > end) end = pend;
    const Elf64_Shdr* sh = (const Elf64_Shdr*)(b + eh->e_shoff);
    for (int i = 0; i < eh->e_shnum; i++) {
        if (sh[i].sh_type == SHT_NOBITS) continue;
        size_t se = sh[i].sh_offset + sh[i].sh_size;
        if (se > end) end = se;
    }
    return end;
}

/* Intercept a cubin at module-load time. On success *out is a malloc'd spliced
 * image the caller loads then frees, *atomized=1, and *names is an opaque token
 * (the set of spliced kernel names) to pass to atomizer_register_*. Otherwise
 * *out==image and *atomized=0. */
int atomizer_intercept_cubin(const void* image, void** out, size_t* outsz,
                             int* atomized, void** names) {
    *atomized = 0; *out = (void*)image; *outsz = 0; *names = NULL;
    if (!g_lithos_cfg.enable_atomizer || !image) return 0;

    pthread_mutex_lock(&g_mtx);
    if (ensure_prologue() != 0) { pthread_mutex_unlock(&g_mtx); return 0; }
    int want_sm = g_dev_sm;
    pthread_mutex_unlock(&g_mtx);

    /* Unwrap: raw cubin -> used directly; fatbin -> matching cubin (LZ4-decoded
     * if needed); PTX / PTX-only fatbin -> JIT'd to a cubin. -1 => load verbatim. */
    void* cub = NULL; size_t cubsz = 0;
    if (atomize_image_to_cubin(image, want_sm, &cub, &cubsz) != 0) return 0;
    int cub_is_image = (cub == image);
    if (cubsz == 0) cubsz = elf_image_size(cub);     /* image was already a cubin */

    pthread_mutex_lock(&g_mtx);
    void* spl = NULL; size_t splz = 0; char** nm = NULL; int nn = 0;
    double t_splice0 = az_now_ms();
    int rc = atomize_splice_cubin(cub, cubsz, g_prologue, g_prolen, &spl, &splz, &nm, &nn);
    if (getenv("LITHOS_DIAG"))
        fprintf(stderr, "[diag] splice: %d kernels, %.2f ms (cubin %zu -> %zu bytes)\n",
                nn, az_now_ms() - t_splice0, cubsz, splz);
    pthread_mutex_unlock(&g_mtx);
    if (!cub_is_image) free(cub);
    if (rc != 0 || !spl) return 0;                   /* nothing spliced: load verbatim */

    SplicedNames* sn = malloc(sizeof(*sn)); sn->names = nm; sn->n = nn;
    *out = spl; *outsz = splz; *atomized = 1; *names = sn;
    AZLOG("atomized image -> cubin %zu bytes (spliced %zu, %d kernel(s))%s",
          cubsz, splz, nn, cub_is_image ? "" : " [unwrapped fatbin/PTX]");
    return 0;
}

/* Associate a loaded container (CUmodule/CUlibrary) with its spliced-name set. */
static void register_container(void* h, void* names) {
    if (!h || !names) return;
    SplicedNames* sn = names;
    pthread_mutex_lock(&g_mtx);
    if (!find_container(h) && g_containers_n < MAX_MODS) {
        g_containers[g_containers_n].handle = h;
        g_containers[g_containers_n].names = sn->names;
        g_containers[g_containers_n].n = sn->n;
        g_containers_n++;
    }
    pthread_mutex_unlock(&g_mtx);
    free(sn);   /* the name arrays are now owned by the container entry */
}
void atomizer_register_module(CUmodule m, void* names)    { register_container(m, names); }
void atomizer_register_library(CUlibrary lib, void* names){ register_container(lib, names); }

/* library -> module (cuLibraryGetModule): the module shares the library's name set. */
void atomizer_note_library_module(CUmodule m, CUlibrary lib) {
    pthread_mutex_lock(&g_mtx);
    Container* c = find_container(lib);
    if (c && !find_container(m) && g_containers_n < MAX_MODS) {
        g_containers[g_containers_n].handle = m;
        g_containers[g_containers_n].names = c->names;   /* shared (not freed twice) */
        g_containers[g_containers_n].n = c->n;
        g_containers_n++;
    }
    pthread_mutex_unlock(&g_mtx);
}
/* Why a kernel ended up in the class it did. Coverage is the headline number in
 * docs/BENCHMARKS.md, and when a closed-source library reports 0% the question is
 * always the same: did we never see the module, or did we see it and skip the
 * kernel? This answers it by name. */
static void diag_class(const char* api, const char* name, int have_container, int gated) {
    if (!getenv("LITHOS_DIAG")) return;
    fprintf(stderr, "[diag] %s %-48s -> %s\n", api, name ? name : "(null)",
            !have_container ? "VERBATIM (container not spliced)"
                            : (gated ? "gated" : "full-range (name not in splice set)"));
}

/* module -> function (cuModuleGetFunction): gate iff this kernel name was spliced. */
void atomizer_note_get_function(CUfunction f, CUmodule m, const char* name) {
    pthread_mutex_lock(&g_mtx);
    Container* c = find_container(m);
    int gated = 0;
    if (c) { if ((gated = names_have(c->names, c->n, name))) add_atom_func(f); else add_full_func(f); }
    pthread_mutex_unlock(&g_mtx);
    diag_class("cuModuleGetFunction", name, c != NULL, gated);
}
/* library -> kernel (cuLibraryGetKernel): a CUkernel can be launched directly. */
void atomizer_note_get_kernel(CUkernel k, CUlibrary lib, const char* name) {
    pthread_mutex_lock(&g_mtx);
    Container* c = find_container(lib);
    int gated = 0;
    if (c) { if ((gated = names_have(c->names, c->n, name))) add_atom_func(k); else add_full_func(k); }
    pthread_mutex_unlock(&g_mtx);
    diag_class("cuLibraryGetKernel", name, c != NULL, gated);
}
/* kernel -> function (cuKernelGetFunction): inherit the kernel handle's class. */
void atomizer_note_kernel_function(CUfunction f, CUkernel k) {
    pthread_mutex_lock(&g_mtx);
    if (g_atom_set && fset_has(g_atom_set, (void*)k)) add_atom_func(f);
    else if (g_full_set && fset_has(g_full_set, (void*)k)) add_full_func(f);
    pthread_mutex_unlock(&g_mtx);
}

/* Last-resort classification, from the function handle alone.
 *
 * The notifiers above cover every API that hands out a kernel handle BY NAME
 * (cuModuleGetFunction, cuLibraryGetKernel, cuKernelGetFunction). CUDA 12 also
 * offers bulk enumeration -- cuLibraryEnumerateKernels / cuLibraryGetKernelCount,
 * which cuFFT uses -- that returns handles with no names at all, so a library
 * using it would launch every kernel VERBATIM even though we spliced its module.
 * Chasing each new enumeration API is a losing game; instead ask the driver which
 * module a handle belongs to and what it is called. If the module is one we
 * spliced, the kernel classifies exactly as it would have on the named path.
 *
 * Both queries need a live context, which a launch guarantees. Called once per
 * distinct handle (the caller memoises via g_atom_set/g_full_set/g_seen_set).
 * Caller must NOT hold g_mtx. */
static int classify_by_handle(CUfunction f) {
    static CUresult (*get_mod)(CUmodule*, CUfunction);
    static CUresult (*get_name)(const char**, CUfunction);
    static int tried;
    if (!tried) { tried = 1;
        get_mod  = (CUresult(*)(CUmodule*, CUfunction))lithos_real_sym("cuFuncGetModule");
        get_name = (CUresult(*)(const char**, CUfunction))lithos_real_sym("cuFuncGetName"); }
    if (!get_mod || !get_name) return 0;

    CUmodule m = NULL; const char* nm = NULL;
    if (get_mod(&m, f) != CUDA_SUCCESS || !m) return 0;
    if (get_name(&nm, f) != CUDA_SUCCESS || !nm) return 0;

    int gated = 0;
    pthread_mutex_lock(&g_mtx);
    Container* c = find_container(m);
    if (c) { if ((gated = names_have(c->names, c->n, nm))) add_atom_func(f); else add_full_func(f); }
    pthread_mutex_unlock(&g_mtx);
    if (c) diag_class("byHandle", nm, 1, gated);
    return c != NULL;
}

/* Handles already put through classify_by_handle, so an un-spliced kernel costs
 * the two driver queries once rather than on every launch. */
static void** g_seen_set;
static int   handle_classified(CUfunction f) {
    pthread_mutex_lock(&g_mtx);
    if (!g_seen_set) g_seen_set = fset_alloc();
    int seen = g_seen_set && fset_has(g_seen_set, (void*)f);
    if (!seen && g_seen_set) fset_add(g_seen_set, (void*)f);
    pthread_mutex_unlock(&g_mtx);
    if (seen) return 0;
    return classify_by_handle(f);
}

static int is_atomized_func(CUfunction f) {
    return g_atom_set && fset_has(g_atom_set, (void*)f);
}
/* From an atomized module but not individually gated -> force full-range meta. */
static int is_full_func(CUfunction f) {
    return g_full_set && fset_has(g_full_set, (void*)f);
}

/* Number of atoms = ceil(predicted_duration / atom_duration), clamped to
 * [1, blocks] (§5.4). `pred_us` is the online predictor's estimate (§5.7) when
 * available; otherwise we fall back to the `blocks x 0.5us` stub. For very large
 * grids the atom_duration is scaled up to bound early-exit thread-block traffic
 * (the paper's aggressiveness control). */
/* Choose the atom duration for this launch.
 *
 * The paper treats atom_duration as a hand-tuned constant ("limits of 250-500us
 * are effective") and only WARNS that "if this parameter is set too low, an
 * atomized kernel may actually take longer to complete" — leaving it to the
 * operator to avoid that. Both bounds can be derived instead of guessed:
 *
 *  FLOOR — from the overhead we actually pay. Every atom is a full-grid relaunch
 *     whose out-of-range blocks reach the prologue and exit, so splitting into n
 *     atoms costs about n x atom_cost_us. Requiring that cost to stay under a
 *     fraction f of the kernel's own runtime gives atom_us >= atom_cost_us / f.
 *     This turns the paper's caveat into a guarantee: splitting can never add more
 *     than f of the kernel's duration, whatever the operator configured.
 *
 *  CEILING — from the latency budget. A co-located latency-critical tenant waits
 *     behind at most ONE atom of this kernel (measured: HP tail latency tracked
 *     atom size almost exactly, see docs/BENCHMARKS.md §4), so if this process
 *     must not delay others by more than LITHOS_SLO_US, atoms must not exceed it.
 *     The paper has no equivalent: its fixed 250-500us is unrelated to any SLO.
 *
 * Plus the paper's own aggressiveness control: very large grids get a longer atom
 * duration, to bound the extra thread-block traffic from early-exiting blocks. */
static double effective_atom_us(uint64_t blocks) {
    double atom_us = g_lithos_cfg.atom_duration_us;

    /* Paper: "for kernels with a large number of thread blocks, the Kernel
     * Atomizer dynamically adjusts the atom_duration parameter." */
    if (blocks > 4096) atom_us *= 2.0;

    /* FLOOR: keep splitting overhead under max_overhead of the kernel's runtime. */
    double frac = g_lithos_cfg.atom_max_overhead;
    if (frac > 0.0 && g_lithos_cfg.atom_cost_us > 0.0) {
        double floor_us = g_lithos_cfg.atom_cost_us / frac;
        if (atom_us < floor_us) atom_us = floor_us;
    }

    /* CEILING: never make another tenant wait longer than its budget. */
    if (g_lithos_cfg.slo_us > 0.0 && atom_us > g_lithos_cfg.slo_us)
        atom_us = g_lithos_cfg.slo_us;

    return atom_us > 1.0 ? atom_us : 1.0;
}

static int decide_atoms(uint64_t blocks, double pred_us) {
    if (!g_lithos_cfg.enable_atomizer) return 1;
    if (g_lithos_cfg.graph_subgraphs > 1) return 1;  /* graphs: subgraph is the unit, kernels run whole */
    if (g_lithos_cfg.force_atoms > 0) {   /* explicit override (testing/policy) */
        int n = g_lithos_cfg.force_atoms;
        return (uint64_t)n > blocks ? (int)blocks : n;
    }
    if (blocks < (uint64_t)g_lithos_cfg.min_blocks_to_atomize) return 1;

    double atom_us = effective_atom_us(blocks);
    double dur = (pred_us > 0) ? pred_us : (double)blocks * 0.5;   /* predictor, or stub */
    int n = (int)(dur / atom_us);
    if (n < 1) n = 1;
    if ((uint64_t)n > blocks) n = (int)blocks;
    return n;
}
int atomizer_num_atoms(const LithosKernel* k, double pred_us) {
    return decide_atoms(k->total_blocks, pred_us);
}

/* Write {lo,hi} to the shared metadata address, ordered on the launch stream
 * ahead of the (full-grid) relaunch. Uses two cuMemsetD32Async immediate writes
 * rather than a host->device copy, for two reasons: (1) a tiny async HtoD copy
 * is often executed synchronously, which is ILLEGAL during CUDA-graph capture
 * and invalidates it; (2) memset bakes the value into the recorded graph node,
 * so a captured atom subgraph REPLAYS the correct range -- no host staging, no
 * per-atom persistent buffers, no stale-value hazard. Works identically for
 * eager launches and graph capture. */
/* Last range written to g_meta, so a redundant write can be skipped. Valid only
 * while a single stream is doing atomized work and we are not capturing: with
 * several streams the interleaving is decided by the GPU, and during capture the
 * write must be *recorded* as a node every time even if the value repeats. */
static uint32_t g_meta_lo, g_meta_hi;
static int      g_meta_known;

static void meta_write(CUstream s, uint32_t lo, uint32_t hi) {
    cuMemsetD32Async(g_meta,     lo, 1, s);
    cuMemsetD32Async(g_meta + 4, hi, 1, s);
    g_meta_lo = lo; g_meta_hi = hi; g_meta_known = 1;
}

/* Write the range only if the buffer does not already hold it. Saves two driver
 * calls on the common path where consecutive launches use the same range — e.g.
 * back-to-back single-atom (n=1) launches, which all request the full grid.
 * `cacheable` is 0 during capture or once multiple streams are in play. */
static void meta_write_cached(CUstream s, uint32_t lo, uint32_t hi, int cacheable) {
    if (cacheable && g_meta_known && g_meta_lo == lo && g_meta_hi == hi) return;
    meta_write(s, lo, hi);
    if (!cacheable) g_meta_known = 0;   /* value may be reordered/replayed: forget it */
}

/* The atom range lives in a single per-process device buffer (g_meta), so two
 * atomized launches on DIFFERENT streams of the SAME process could interleave
 * their metadata writes and reads. (There is no cross-*process* race -- each
 * process has its own g_meta.) We serialize atomized launches across streams
 * with a CUDA event: each waits for the previous atomized launch to finish
 * before writing metadata. For a single-stream app this is a same-stream wait =
 * a no-op, so the common inference path pays nothing; multi-stream frameworks
 * (e.g. JAX/XLA) stay correct. Skipped during graph capture (the captured stream
 * is self-ordered, and waiting on an event recorded outside the capture is
 * illegal). Caller holds g_mtx. */
static int stream_capturing(CUstream s) {
    if (g_capture_hint >= 0) return g_capture_hint;   /* sched.c already asked */
    if (!g_real.cuStreamIsCapturing) return 0;
    CUstreamCaptureStatus st = CU_STREAM_CAPTURE_STATUS_NONE;
    if (g_real.cuStreamIsCapturing(s, &st) != CUDA_SUCCESS) return 0;
    return st == CU_STREAM_CAPTURE_STATUS_ACTIVE;
}

/* Cross-stream serialization is only needed once MORE THAN ONE stream issues
 * atomized launches: with a single stream the launches are already ordered, so the
 * wait/record pair is a guaranteed no-op that still costs two driver calls.
 *
 * We therefore track the first stream we see and stay on a fast path until a
 * second one appears. This FAILS SAFE: the moment a second stream shows up the
 * fast path is disabled permanently (g_multi_stream latches), and the very first
 * launch on that new stream takes the slow path, so no window exists where two
 * streams both skip serialization. Single-stream inference — the common case —
 * then pays nothing, while multi-stream frameworks (JAX/XLA) keep full ordering. */
static CUstream g_first_atom_stream;
static int      g_have_first_stream;
static int      g_multi_stream;          /* latched: never returns to the fast path */

/* Returns 1 if this launch may skip the wait/record pair. Caller holds g_mtx. */
static int serial_can_skip(CUstream s) {
    if (g_multi_stream) return 0;
    if (!g_have_first_stream) { g_first_atom_stream = s; g_have_first_stream = 1; return 1; }
    if (g_first_atom_stream == s) return 1;
    g_multi_stream = 1;                  /* a second stream: serialize from now on */
    g_meta_known   = 0;                  /* the cached range is no longer trustworthy */
    return 0;
}

/* `capturing` is passed in by the caller, which already had to determine it —
 * this avoids a second cuStreamIsCapturing driver call per launch. */
static void atom_serial_begin_ex(CUstream s, int capturing, int skip) {
    if (!capturing && !skip && g_atom_ev_valid) cuStreamWaitEvent(s, g_atom_ev, 0);
}
static void atom_serial_end_ex(CUstream s, int capturing, int skip) {
    if (!capturing && !skip && g_atom_ev) { cuEventRecord(g_atom_ev, s); g_atom_ev_valid = 1; }
}

static int launch_original(LithosKernel* k) {
    g_real.cuLaunchKernel((CUfunction)k->func,
                          k->gridDimX, k->gridDimY, k->gridDimZ,
                          k->blockDimX, k->blockDimY, k->blockDimZ,
                          k->sharedMemBytes, (CUstream)k->stream,
                          k->kernelParams, k->extra);
    return 1;
}

/* The main atomization path (cuLaunchKernel). Splits the grid into N contiguous
 * block ranges and relaunches the UNMODIFIED grid once per range; the spliced-in
 * prologue makes each launch execute only the blocks in the current [lo,hi).
 * Returns the number of launches actually issued. */
int atomizer_dispatch(LithosKernel* k, int quota_tpcs) {
    (void)quota_tpcs;
    CUfunction f      = (CUfunction)k->func;
    uint64_t   blocks = k->total_blocks;
    CUstream   s      = (CUstream)k->stream;

    /* --- classes VERBATIM and FULL (see the gating comment near the top) --- */
    /* Unknown handle: it may still come from a module we spliced but have been
     * obtained through a bulk-enumeration API that carries no name (see
     * classify_by_handle). Ask the driver once before giving up on it. */
    if (!is_atomized_func(f) && !is_full_func(f)) handle_classified(f);
    if (!is_atomized_func(f)) {
        if (!is_full_func(f)) {                 /* VERBATIM: not ours, pass through */
            stat_verbatim(f);
            return launch_original(k);
        }
        /* FULL: may carry a prologue but isn't individually gated, so pin the range
         * to the whole grid before launching — never let it inherit a stale one. */
        pthread_mutex_lock(&g_mtx);
        int capturing = stream_capturing(s);
        int skip = !capturing && serial_can_skip(s);
        atom_serial_begin_ex(s, capturing, skip);
        meta_write_cached(s, 0, (uint32_t)(blocks > 0xffffffffu ? 0xffffffffu : blocks),
                          skip && !capturing);
        launch_original(k);
        atom_serial_end_ex(s, capturing, skip);
        pthread_mutex_unlock(&g_mtx);
        __atomic_fetch_add(&g_st_full, 1, __ATOMIC_RELAXED);
        stats_tick();
        return 1;
    }

    /* --- class ATOM: split the grid ---------------------------------------- */
    int      n_atoms       = decide_atoms(blocks, k->pred_us);
    uint64_t blocks_per_atom = (blocks + n_atoms - 1) / n_atoms;   /* ceil-divide */
    int      launched      = 0;

    pthread_mutex_lock(&g_mtx);
    /* g_meta is process-wide, so concurrent streams must not interleave their
     * metadata writes; this serializes atomized launches across streams. With a
     * single stream the launches are already ordered, so the pair is skipped
     * (serial_can_skip); it is also skipped during graph capture. */
    int capturing = stream_capturing(s);
    int skip = !capturing && serial_can_skip(s);
    atom_serial_begin_ex(s, capturing, skip);

    for (int i = 0; i < n_atoms; i++) {
        uint64_t lo = (uint64_t)i * blocks_per_atom;
        uint64_t hi = lo + blocks_per_atom;
        if (hi > blocks) hi = blocks;
        if (lo >= hi) break;                     /* ragged tail: no blocks left */

        /* Publish this atom's range, stream-ordered ahead of its launch. */
        meta_write_cached(s, (uint32_t)lo, (uint32_t)hi, skip && !capturing);

        /* Give THIS atom its own TPC allocation. The QMD next-mask is one-shot
         * (consumed per launch), so it must be re-armed before every atom —
         * that is also what makes distinct per-atom allocations possible. */
        lithos_apply_atom_mask(s, i, n_atoms);

        launch_original(k);
        launched++;
    }
    if (launched == 0) { launch_original(k); launched = 1; }   /* safety net */

    atom_serial_end_ex(s, capturing, skip);
    pthread_mutex_unlock(&g_mtx);

    __atomic_fetch_add(launched > 1 ? &g_st_split : &g_st_atom1, 1, __ATOMIC_RELAXED);
    stats_tick();
    AZLOG("atomized func %p: %d atom(s) over %lu blocks", (void*)f, launched, blocks);
    return launched;
}

/* cuLaunchKernelEx: same atom split, replaying the Ex call (preserves launch
 * attributes such as clusters). */
int atomizer_dispatch_ex(const CUlaunchConfig* cfg, CUfunction f, void** params, void** extra) {
    int atom = is_atomized_func(f), full = !atom && is_full_func(f);
    if (!atom && !full) { stat_verbatim((void*)f); return (g_real.cuLaunchKernelEx(cfg, f, params, extra), 1); }
    uint64_t blocks = (uint64_t)cfg->gridDimX * cfg->gridDimY * cfg->gridDimZ;
    CUstream s = cfg->hStream;
    /* Cooperative / cluster launches must NOT be split: a grid- or cluster-wide
     * barrier needs every block live. Run once, full-range. Same for ungated
     * (full) kernels: force the full range, never split. */
    int nosplit = full;
    for (unsigned i = 0; i < cfg->numAttrs; i++)
        if (cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_COOPERATIVE ||
            cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) { nosplit = 1; break; }
    int n = nosplit ? 1 : decide_atoms(blocks, g_ex_pred_us);
    uint64_t per = (blocks + n - 1) / n;
    int launched = 0;
    pthread_mutex_lock(&g_mtx);
    int cap  = stream_capturing(s);
    int skip = !cap && serial_can_skip(s);
    int cacheable = skip && !cap;
    atom_serial_begin_ex(s, cap, skip);
    if (n <= 1) {
        meta_write_cached(s, 0, (uint32_t)(blocks > 0xffffffffu ? 0xffffffffu : blocks), cacheable);
        g_real.cuLaunchKernelEx(cfg, f, params, extra); launched = 1;
    } else for (int i = 0; i < n; i++) {
        uint64_t lo = (uint64_t)i * per, hi = lo + per; if (hi > blocks) hi = blocks;
        if (lo >= hi) break;
        meta_write_cached(s, (uint32_t)lo, (uint32_t)hi, cacheable);
        g_real.cuLaunchKernelEx(cfg, f, params, extra);
        launched++;
    }
    atom_serial_end_ex(s, cap, skip);
    pthread_mutex_unlock(&g_mtx);
    __atomic_fetch_add(full ? &g_st_full : (launched > 1 ? &g_st_split : &g_st_atom1), 1, __ATOMIC_RELAXED); stats_tick();
    AZLOG("atomized (Ex) func %p: %d atom(s) over %lu blocks", (void*)f, launched, blocks);
    return launched;
}

/* Cooperative kernels are NEVER split -- a grid-wide barrier needs every block
 * live at once. A spliced cooperative kernel still needs full-range metadata so
 * its range-check passes all blocks. */
int atomizer_dispatch_coop(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                           unsigned bx, unsigned by, unsigned bz, unsigned shmem,
                           CUstream stream, void** params) {
    if (is_atomized_func(f) || is_full_func(f)) {
        uint64_t blocks = (uint64_t)gx * gy * gz;
        pthread_mutex_lock(&g_mtx);
        int cap  = stream_capturing(stream);
        int skip = !cap && serial_can_skip(stream);
        atom_serial_begin_ex(stream, cap, skip);
        meta_write_cached(stream, 0, (uint32_t)(blocks > 0xffffffffu ? 0xffffffffu : blocks),
                          skip && !cap);
        g_real.cuLaunchCooperativeKernel(f, gx, gy, gz, bx, by, bz, shmem, stream, params);
        atom_serial_end_ex(stream, cap, skip);
        pthread_mutex_unlock(&g_mtx);
        __atomic_fetch_add(&g_st_full, 1, __ATOMIC_RELAXED); stats_tick();
        return 1;
    }
    stat_verbatim((void*)f);
    g_real.cuLaunchCooperativeKernel(f, gx, gy, gz, bx, by, bz, shmem, stream, params);
    return 1;
}