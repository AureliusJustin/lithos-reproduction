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
#include "atomizer.h"
#include "qmd.h"
#include "real.h"

#define AZLOG(...) do { if (g_lithos_cfg.verbose) { \
    fprintf(stderr, "[atomizer] " __VA_ARGS__); fputc('\n', stderr); } } while (0)

/* core splicer (atomize_splice.c) */
int atomize_build_prologue(int sm_major, int sm_minor, uint64_t meta_va,
                           unsigned char** out, size_t* len);
int atomize_splice_cubin(const void* cubin, size_t sz, const unsigned char* pro,
                         size_t prolen, void** out, size_t* outsz,
                         char*** names_out, int* n_names_out);
/* fatbin.c: turn an ELF/fatbin/PTX image into a raw cubin (0 ok, -1 = verbatim).
 * *out==image with *outsz==0 means "already a cubin"; otherwise *out is malloc'd. */
int atomize_image_to_cubin(const void* image, int want_sm, void** out, size_t* outsz);

/* ------------------------------------------------------------------ */
/*  Shared metadata buffer + prologue (one per process)               */
/* ------------------------------------------------------------------ */
static CUdeviceptr    g_meta;             /* device {lo,hi} the prologue reads   */
static CUevent        g_atom_ev;          /* cross-stream serialization of g_meta */
static int            g_atom_ev_valid;
static unsigned char* g_prologue;         /* range-check SASS (padded)           */
static size_t         g_prolen;
static int            g_prologue_ready;
static int            g_dev_sm;           /* running device SM (e.g. 86)         */
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;
/* predicted duration for the NEXT Ex dispatch on this thread (set by sched.c) */
static __thread double g_ex_pred_us = 0;
void atomizer_set_ex_pred(double us) { g_ex_pred_us = us; }

/* ------------------------------------------------------------------ */
/*  Which loaded objects were successfully atomized                   */
/* ------------------------------------------------------------------ */
/* Per-container (CUmodule/CUlibrary) set of the kernel names we actually spliced
 * -- only those get gated at launch, so a module with one un-spliceable kernel
 * still atomizes the rest and the odd one runs verbatim. Functions/kernels
 * resolved from a container are gated by NAME membership. */
#define MAX_MODS  8192
#define MAX_FUNCS 262144
typedef struct { void* handle; char** names; int n; } Container;
static Container  g_containers[MAX_MODS]; static int g_containers_n;
static void*      g_atom_funcs[MAX_FUNCS]; static int g_atom_funcs_n;  /* spliced -> split */
/* Functions resolved from an atomized module whose NAME wasn't in the spliced set
 * (splice-skipped, or a handle/name we couldn't match). They may still carry the
 * range-check prologue, so before launching them we write a serialized FULL range
 * to g_meta -- otherwise they'd read a previous atom's stale [lo,hi] and drop
 * blocks. This closes the shared-metadata hole for imperfect gating. */
static void*      g_full_funcs[MAX_FUNCS]; static int g_full_funcs_n;
typedef struct SplicedNames { char** names; int n; } SplicedNames;

/* Atomization coverage stats (LITHOS_STATS=1 dumps them at exit). */
static int handle_in(void** set, int n, void* h);
static long g_st_split, g_st_atom1, g_st_full, g_st_verbatim;
static void* g_verb_funcs[MAX_FUNCS]; static int g_verb_funcs_n; /* distinct un-atomized funcs */
static void stats_tick(void);
static void stat_verbatim(void* f) {
    __atomic_fetch_add(&g_st_verbatim, 1, __ATOMIC_RELAXED);
    pthread_mutex_lock(&g_mtx);
    if (!handle_in(g_verb_funcs, g_verb_funcs_n, f) && g_verb_funcs_n < MAX_FUNCS)
        g_verb_funcs[g_verb_funcs_n++] = f;
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

static int handle_in(void** set, int n, void* h) {
    for (int i = 0; i < n; i++) if (set[i] == h) return 1;
    return 0;
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
    if (!handle_in(g_atom_funcs, g_atom_funcs_n, h) && g_atom_funcs_n < MAX_FUNCS)
        g_atom_funcs[g_atom_funcs_n++] = h;
}
static void add_full_func(void* h) {
    if (!handle_in(g_full_funcs, g_full_funcs_n, h) && g_full_funcs_n < MAX_FUNCS)
        g_full_funcs[g_full_funcs_n++] = h;
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
    int rc = atomize_splice_cubin(cub, cubsz, g_prologue, g_prolen, &spl, &splz, &nm, &nn);
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
/* module -> function (cuModuleGetFunction): gate iff this kernel name was spliced. */
void atomizer_note_get_function(CUfunction f, CUmodule m, const char* name) {
    pthread_mutex_lock(&g_mtx);
    Container* c = find_container(m);
    if (c) { if (names_have(c->names, c->n, name)) add_atom_func(f); else add_full_func(f); }
    pthread_mutex_unlock(&g_mtx);
}
/* library -> kernel (cuLibraryGetKernel): a CUkernel can be launched directly. */
void atomizer_note_get_kernel(CUkernel k, CUlibrary lib, const char* name) {
    pthread_mutex_lock(&g_mtx);
    Container* c = find_container(lib);
    if (c) { if (names_have(c->names, c->n, name)) add_atom_func(k); else add_full_func(k); }
    pthread_mutex_unlock(&g_mtx);
}
/* kernel -> function (cuKernelGetFunction): inherit the kernel handle's class. */
void atomizer_note_kernel_function(CUfunction f, CUkernel k) {
    pthread_mutex_lock(&g_mtx);
    if (handle_in(g_atom_funcs, g_atom_funcs_n, (void*)k)) add_atom_func(f);
    else if (handle_in(g_full_funcs, g_full_funcs_n, (void*)k)) add_full_func(f);
    pthread_mutex_unlock(&g_mtx);
}

static int is_atomized_func(CUfunction f) {
    return handle_in(g_atom_funcs, g_atom_funcs_n, (void*)f);
}
/* From an atomized module but not individually gated -> force full-range meta. */
static int is_full_func(CUfunction f) {
    return handle_in(g_full_funcs, g_full_funcs_n, (void*)f);
}

/* Number of atoms = ceil(predicted_duration / atom_duration), clamped to
 * [1, blocks] (§5.4). `pred_us` is the online predictor's estimate (§5.7) when
 * available; otherwise we fall back to the `blocks x 0.5us` stub. For very large
 * grids the atom_duration is scaled up to bound early-exit thread-block traffic
 * (the paper's aggressiveness control). */
static int decide_atoms(uint64_t blocks, double pred_us) {
    if (!g_lithos_cfg.enable_atomizer) return 1;
    if (g_lithos_cfg.graph_subgraphs > 1) return 1;  /* graphs: subgraph is the unit, kernels run whole */
    if (g_lithos_cfg.force_atoms > 0) {   /* explicit override (testing/policy) */
        int n = g_lithos_cfg.force_atoms;
        return (uint64_t)n > blocks ? (int)blocks : n;
    }
    if (blocks < (uint64_t)g_lithos_cfg.min_blocks_to_atomize) return 1;
    double atom_us = g_lithos_cfg.atom_duration_us;
    if (blocks > 4096) atom_us *= 2.0;
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
static void meta_write(CUstream s, uint32_t lo, uint32_t hi) {
    cuMemsetD32Async(g_meta,     lo, 1, s);
    cuMemsetD32Async(g_meta + 4, hi, 1, s);
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
    if (!g_real.cuStreamIsCapturing) return 0;
    CUstreamCaptureStatus st = CU_STREAM_CAPTURE_STATUS_NONE;
    if (g_real.cuStreamIsCapturing(s, &st) != CUDA_SUCCESS) return 0;
    return st == CU_STREAM_CAPTURE_STATUS_ACTIVE;
}

static int atom_serial_begin(CUstream s) {
    int cap = stream_capturing(s);
    if (!cap && g_atom_ev_valid) cuStreamWaitEvent(s, g_atom_ev, 0);
    return cap;
}
static void atom_serial_end(CUstream s, int cap) {
    if (!cap && g_atom_ev) { cuEventRecord(g_atom_ev, s); g_atom_ev_valid = 1; }
}

static int launch_original(LithosKernel* k) {
    g_real.cuLaunchKernel((CUfunction)k->func,
                          k->gridDimX, k->gridDimY, k->gridDimZ,
                          k->blockDimX, k->blockDimY, k->blockDimZ,
                          k->sharedMemBytes, (CUstream)k->stream,
                          k->kernelParams, k->extra);
    return 1;
}

int atomizer_dispatch(LithosKernel* k, int quota_tpcs) {
    (void)quota_tpcs;
    CUfunction f = (CUfunction)k->func;
    uint64_t blocks = k->total_blocks;
    CUstream s = (CUstream)k->stream;
    if (!is_atomized_func(f)) {
        if (!is_full_func(f)) { stat_verbatim(f); return launch_original(k); }  /* verbatim */
        /* spliced-but-ungated: serialized full-range so it never drops blocks */
        pthread_mutex_lock(&g_mtx);
        int cp = atom_serial_begin(s);
        meta_write(s, 0, (uint32_t)(blocks > 0xffffffffu ? 0xffffffffu : blocks));
        launch_original(k);
        atom_serial_end(s, cp);
        pthread_mutex_unlock(&g_mtx);
        __atomic_fetch_add(&g_st_full, 1, __ATOMIC_RELAXED); stats_tick();
        return 1;
    }

    int n = decide_atoms(blocks, k->pred_us);
    uint64_t per = (blocks + n - 1) / n;
    int launched = 0;
    pthread_mutex_lock(&g_mtx);
    int cap = atom_serial_begin(s);
    for (int i = 0; i < n; i++) {
        uint64_t lo = (uint64_t)i * per, hi = lo + per; if (hi > blocks) hi = blocks;
        if (lo >= hi) break;
        meta_write(s, (uint32_t)lo, (uint32_t)hi);
        /* give THIS atom its own TPC allocation (distinct slice under
           LITHOS_ATOM_TPC, else the stream quota mask) -- the QMD next-mask is
           one-shot, so it must be re-set before every atom's launch */
        lithos_apply_atom_mask(s, i, n);
        launch_original(k);
        launched++;
    }
    if (launched == 0) { launch_original(k); launched = 1; }
    atom_serial_end(s, cap);
    pthread_mutex_unlock(&g_mtx);
    __atomic_fetch_add(launched > 1 ? &g_st_split : &g_st_atom1, 1, __ATOMIC_RELAXED); stats_tick();
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
    int cap = atom_serial_begin(s);
    if (n <= 1) {
        meta_write(s, 0, (uint32_t)(blocks > 0xffffffffu ? 0xffffffffu : blocks));
        g_real.cuLaunchKernelEx(cfg, f, params, extra); launched = 1;
    } else for (int i = 0; i < n; i++) {
        uint64_t lo = (uint64_t)i * per, hi = lo + per; if (hi > blocks) hi = blocks;
        if (lo >= hi) break;
        meta_write(s, (uint32_t)lo, (uint32_t)hi);
        g_real.cuLaunchKernelEx(cfg, f, params, extra);
        launched++;
    }
    atom_serial_end(s, cap);
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
        int cap = atom_serial_begin(stream);
        meta_write(stream, 0, (uint32_t)(blocks > 0xffffffffu ? 0xffffffffu : blocks));
        g_real.cuLaunchCooperativeKernel(f, gx, gy, gz, bx, by, bz, shmem, stream, params);
        atom_serial_end(stream, cap);
        pthread_mutex_unlock(&g_mtx);
        __atomic_fetch_add(&g_st_full, 1, __ATOMIC_RELAXED); stats_tick();
        return 1;
    }
    stat_verbatim((void*)f);
    g_real.cuLaunchCooperativeKernel(f, gx, gy, gz, bx, by, bz, shmem, stream, params);
    return 1;
}