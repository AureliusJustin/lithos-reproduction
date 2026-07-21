/*
 * Kernel Atomizer (Section 5.4) -- real implementation.
 *
 * Transparently splits a kernel's grid into atoms by:
 *   1. capturing the original kernel's SASS entry VA from the QMD (qmd.c),
 *   2. launching the ORIGINAL kernel but patching the QMD program address to a
 *      Prelude (Algorithm 1) so the GPU runs the Prelude instead,
 *   3. the Prelude early-exits blocks outside the atom's [lo,hi) range and
 *      tail-calls the original entry for in-range blocks.
 *
 * The Prelude reads its AtomMetadata from a fixed device address baked in as a
 * literal (the driver does NOT set up the Prelude's constant bank when it runs
 * under the original's QMD, so a const-bank global would read garbage). We JIT
 * a Prelude per launch queue with NVRTC, baking that queue's metadata address.
 *
 * Per-atom metadata is written with a stream-ordered async copy, so within a
 * stream the sequence [write meta_i, launch atom_i, write meta_{i+1}, ...] is
 * serialised by the GPU: every block of atom_i observes meta_i before meta_{i+1}
 * is written. No host synchronisation is required.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <elf.h>
#include "atomizer.h"
#include "qmd.h"
#include "real.h"

#define AZLOG(...) do { if (g_lithos_cfg.verbose) { \
    fprintf(stderr, "[atomizer] " __VA_ARGS__); fputc('\n', stderr); } } while (0)

/* ------- per-launch-queue (stream) atomization context ------- */
#define MAX_QUEUES 256
#define META_RING  512
typedef struct AtomCtx {
    CUstream    stream;
    int         in_use;
    CUdeviceptr meta_dev;      /* device AtomMetadata storage (ring)   */
    AtomMetadata* meta_host;   /* pinned host staging ring             */
    CUmodule    mod;           /* JIT-ed Prelude module                */
    uint64_t    prelude_entry; /* captured Prelude program address     */
    uint64_t    noop_entry;    /* captured no-op kernel entry          */
    uint8_t     prelude_qmd[256]; /* Prelude's code-identity QMD fields */
    uint8_t     noop_qmd[256];    /* no-op's QMD (diagnostic)          */
    int         prelude_regs;  /* Prelude register requirement         */
    long        call_off;      /* in-function byte offset of the patched jump */
    uint64_t    ring_idx;      /* monotonically increasing atom index  */
} AtomCtx;

static AtomCtx g_ctx[MAX_QUEUES];
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;

/* CUfunction -> original entry VA cache */
#define MAX_FUNCS 4096
static struct { void* func; uint64_t entry; } g_entry_cache[MAX_FUNCS];
static int g_entry_n;

void atomizer_init(void) {
    qmd_init();
    memset(g_ctx, 0, sizeof(g_ctx));
}

/* Patch the Prelude's tail CALL into a register-indirect JUMP (BRX) in the
 * compiled cubin. Returns the CALL's in-function byte offset (needed to compute
 * the PC-relative jump target), or -1 if not found.
 *
 * sm_80 instructions are 16 bytes. The CALL.REL.NOINC R2 encoding begins with
 * bytes 44 73 00 02 (opcode 0x344 + register R2); we overwrite the whole 16
 * bytes with BRX R2, imm=0 (49 79 00 02 ... , control word from a real BRX). */
static const uint8_t CALL_SIG[4] = {0x44, 0x73, 0x00, 0x02};
/* We overwrite the CALL with BRX R2 (ptxas encoding from a switch, register
 * field set to R2). BRX is a register-indirect PC-relative branch:
 *   target = PC_of_BRX + imm + R2   (bytes, no scaling)
 * It pushes no return PC, so the original kernel's EXIT is clean. The fixed
 * immediate here is -0xc0; it is absorbed into K when computing the relative
 * target (theoretical K = -0xc0, i.e. base = prelude_entry + call_off - 0xc0). */
static const uint8_t BRX_INSN[16] = {
    0x49, 0x79, 0x00, 0x02, 0x40, 0xff, 0xff, 0xff,   /* BRX R2, imm=-0xc0 */
    0xff, 0xff, 0x83, 0x03, 0x00, 0xea, 0x0f, 0x00,   /* control word      */
};
static long patch_call_to_brx(uint8_t* cub, size_t sz) {
    const Elf64_Ehdr* eh = (const Elf64_Ehdr*)cub;
    if (sz < sizeof(*eh) || memcmp(cub, ELFMAG, SELFMAG) != 0) return -1;
    const Elf64_Shdr* sh = (const Elf64_Shdr*)(cub + eh->e_shoff);
    for (int i = 0; i < eh->e_shnum; i++) {
        if (sh[i].sh_type != SHT_PROGBITS) continue;
        uint8_t* base = cub + sh[i].sh_offset;
        size_t n = sh[i].sh_size;
        if (sh[i].sh_offset + n > sz) continue;
        for (size_t o = 0; o + 16 <= n; o += 16) {
            if (memcmp(base + o, CALL_SIG, 4) == 0) {
                memcpy(base + o, BRX_INSN, 16);   /* CALL -> BRX */
                return (long)o;                   /* in-section = in-function */
            }
        }
    }
    return -1;
}

/* Number of atoms = ceil(predicted_us / atom_duration), clamped to [1, blocks].
 * For kernels with many blocks the atom_duration is scaled up to limit the
 * early-exit thread-block traffic (Section 5.4, "Performance Optimizations"). */
int atomizer_num_atoms(const LithosKernel* k, double pred_us) {
    if (!g_lithos_cfg.enable_atomizer) return 1;
    if (k->total_blocks < (uint64_t)g_lithos_cfg.min_blocks_to_atomize) return 1;
    double atom_us = g_lithos_cfg.atom_duration_us;
    if (k->total_blocks > 4096) atom_us *= 2.0;   /* dampen aggressiveness */
    int n = (int)(pred_us / atom_us);
    if (n < 1) n = 1;
    if ((uint64_t)n > k->total_blocks) n = (int)k->total_blocks;
    return n;
}

static uint64_t lookup_entry(void* func) {
    for (int i = 0; i < g_entry_n; i++)
        if (g_entry_cache[i].func == func) return g_entry_cache[i].entry;
    return 0;
}
static void store_entry(void* func, uint64_t entry) {
    if (g_entry_n < MAX_FUNCS) { g_entry_cache[g_entry_n].func = func;
        g_entry_cache[g_entry_n].entry = entry; g_entry_n++; }
}

/* Build + JIT a Prelude whose AtomMetadata lives at `meta_va` (Algorithm 1).
 *
 * The transfer into the original entry must be a single UNCONDITIONAL, branch-
 * free tail call: any conditional branch before the call makes ptxas wrap the
 * call in a BSSY/BSYNC reconvergence barrier, and since the original ends in
 * EXIT (never returns) the dangling BSYNC faults with INVALID_PC. We therefore
 * select the target branchlessly -- the original entry for in-range blocks, a
 * trivial no-op kernel for out-of-range blocks -- and call it unconditionally.
 * The Prelude reads its metadata from a literal absolute address (not a const-
 * bank global), since the driver sets up the constant bank for the ORIGINAL's
 * launch, not the Prelude's. */
static int build_prelude(AtomCtx* c, CUdeviceptr meta_va) {
    char src[2048];
    snprintf(src, sizeof(src),
        "struct AtomMetadata{unsigned lo,hi,gen,pad;unsigned long long entry,noop;};\n"
        "extern \"C\" __global__ void lithos_noop(){}\n"
        "extern \"C\" __global__ void lithos_prelude(){\n"
        "  unsigned long long b=(unsigned long long)blockIdx.z*gridDim.y*gridDim.x"
        "+(unsigned long long)blockIdx.y*gridDim.x+blockIdx.x;\n"
        "  AtomMetadata* a=(AtomMetadata*)%lluULL;\n"
        "  unsigned long long inr=(unsigned long long)((b>=a->lo)&(b<a->hi));\n"
        "  unsigned long long mask=(unsigned long long)0-inr;\n"        /* all-ones if in range */
        "  unsigned long long e=(a->entry & mask)|(a->noop & ~mask);\n" /* branchless select    */
        "  ((void(*)())e)();\n"
        "  __builtin_unreachable();\n"
        "}\n",
        (unsigned long long)meta_va);

    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src, "prelude.cu", 0, NULL, NULL) != NVRTC_SUCCESS) return -1;
    /* Compile to the running GPU's real SASS (sm_XY, not compute_XY) so the
     * CUBIN carries patchable machine code. Arch is auto-detected so the same
     * build runs on A6000 (sm_86), A100 (sm_80), etc. */
    char arch[32] = "--gpu-architecture=sm_80";
    int major = 0, minor = 0; CUdevice dev;
    if (cuCtxGetDevice(&dev) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) == CUDA_SUCCESS)
        snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d", major, minor);
    const char* opts[] = { arch };
    nvrtcResult rc = nvrtcCompileProgram(prog, 1, opts);
    if (rc != NVRTC_SUCCESS) {
        size_t lsz; nvrtcGetProgramLogSize(prog, &lsz);
        char* log = malloc(lsz); nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "[atomizer] NVRTC failed:\n%s\n", log);
        free(log); nvrtcDestroyProgram(&prog); return -1;
    }
    /* Emit a CUBIN (not PTX) so we can patch the Prelude's tail transfer from a
     * CALL into a register-indirect JUMP (BRX). ptxas always lowers an indirect
     * call to CALL, which pushes a return PC; when the callee (original kernel)
     * ends in EXIT rather than RET, the unpopped call-return entry faults at
     * warp retirement (INVALID_PC). A jump pushes nothing, so EXIT is clean.
     * The real LithOS gets this jump for free from Rust/LLVM guaranteed tail
     * calls; ptxas needs the post-compile patch below. */
    size_t bsz; nvrtcGetCUBINSize(prog, &bsz);
    char* cubin = malloc(bsz); nvrtcGetCUBIN(prog, cubin);
    nvrtcDestroyProgram(&prog);
    c->call_off = 0;
    if (g_lithos_cfg.use_brx) {
        /* BRX mode (GA102 workaround attempt): rewrite the tail CALL into a
         * register-indirect jump so the callee's EXIT is clean. */
        c->call_off = patch_call_to_brx((uint8_t*)cubin, bsz);
        if (c->call_off < 0) { fprintf(stderr, "[atomizer] could not find CALL to patch\n"); free(cubin); return -1; }
        AZLOG("patched CALL->BRX at in-function offset %#lx", c->call_off);
    }

    CUresult r = cuModuleLoadData(&c->mod, cubin);
    free(cubin);
    if (r != CUDA_SUCCESS) { fprintf(stderr, "[atomizer] module load failed %d\n", r); return -1; }
    CUfunction pf, nf;
    if (cuModuleGetFunction(&pf, c->mod, "lithos_prelude") != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&nf, c->mod, "lithos_noop") != CUDA_SUCCESS) return -1;
    int pregs = 0; cuFuncGetAttribute(&pregs, CU_FUNC_ATTRIBUTE_NUM_REGS, pf);
    c->prelude_regs = pregs;
    AZLOG("prelude NUM_REGS=%d", pregs);
    void* none = NULL;

    /* Capture the no-op kernel's entry + QMD FIRST (it just EXITs, safe). */
    qmd_arm_capture();
    g_real.cuLaunchKernel(nf, 1,1,1, 1,1,1, 0, c->stream, &none, NULL);
    CUresult sr = g_real.cuStreamSynchronize(c->stream);
    AZLOG("noop capture sync = %d", sr);
    c->noop_entry = qmd_get_captured();
    qmd_get_captured_qmd(c->noop_qmd);

    if (!g_lithos_cfg.use_brx) {
        /* CALL mode: capture the Prelude's entry directly by launching it with
         * metadata that makes its tail CALL target the (valid) no-op. On a GPU
         * that tolerates the callee's EXIT inside a call frame this runs cleanly;
         * a non-zero sync here means this GPU needs the jump (e.g. GA102). */
        AtomMetadata seed = {0, 1, 0, 0, c->noop_entry, c->noop_entry};
        cuMemcpyHtoD(meta_va, &seed, sizeof(seed));
        qmd_arm_capture();
        g_real.cuLaunchKernel(pf, 1,1,1, 1,1,1, 0, c->stream, &none, NULL);
        sr = g_real.cuStreamSynchronize(c->stream);
        AZLOG("prelude capture sync = %d (CALL mode; 0 => this GPU tolerates CALL+EXIT)", sr);
        c->prelude_entry = qmd_get_captured();
    } else {
        /* BRX mode: the Prelude always executes the patched jump, so it can't be
         * launched safely for capture. Derive its entry from the no-op's via the
         * deterministic per-source module layout delta. */
        long delta = getenv("LITHOS_PRELUDE_DELTA") ? strtol(getenv("LITHOS_PRELUDE_DELTA"),0,0) : 0xa00;
        c->prelude_entry = c->noop_entry + delta;
    }
    AZLOG("built prelude for stream %p: entry=%#lx noop=%#lx meta=%#llx",
          (void*)c->stream, c->prelude_entry, c->noop_entry, (unsigned long long)meta_va);
    return (c->prelude_entry && c->noop_entry) ? 0 : -1;
}

static AtomCtx* get_ctx(CUstream s) {
    for (int i = 0; i < MAX_QUEUES; i++)
        if (g_ctx[i].in_use && g_ctx[i].stream == s) return &g_ctx[i];
    /* Lazily create */
    for (int i = 0; i < MAX_QUEUES; i++) {
        if (!g_ctx[i].in_use) {
            AtomCtx* c = &g_ctx[i];
            c->stream = s; c->ring_idx = 0;
            if (cuMemAlloc(&c->meta_dev, META_RING * sizeof(AtomMetadata)) != CUDA_SUCCESS) return NULL;
            if (cuMemAllocHost((void**)&c->meta_host, META_RING * sizeof(AtomMetadata)) != CUDA_SUCCESS) return NULL;
            if (build_prelude(c, c->meta_dev) != 0) return NULL;
            c->in_use = 1;
            return c;
        }
    }
    return NULL;
}

/* Launch the original kernel unmodified (single atom / fallback path). */
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
    /* First sighting of this CUfunction: run it normally while capturing its
     * entry VA. Correct results; atomization kicks in on later launches. */
    uint64_t entry = lookup_entry(k->func);
    if (!entry) {
        pthread_mutex_lock(&g_mtx);
        entry = lookup_entry(k->func);
        if (!entry) {
            qmd_arm_capture();
            launch_original(k);
            g_real.cuStreamSynchronize((CUstream)k->stream);
            entry = qmd_get_captured();
            if (entry) store_entry(k->func, entry);
        }
        pthread_mutex_unlock(&g_mtx);
        int oregs = 0; cuFuncGetAttribute(&oregs, CU_FUNC_ATTRIBUTE_NUM_REGS, (CUfunction)k->func);
        AZLOG("captured entry %#lx for func %p (blocks=%lu, orig NUM_REGS=%d)",
              entry, k->func, k->total_blocks, oregs);
        return 1;
    }

    /* Decide atom count. Duration prediction is stubbed as proportional to the
     * block count (replaced by the predictor module, Section 5.7). */
    double pred_us = (double)k->total_blocks * 0.5;  /* rough: 0.5us/block */
    int n = atomizer_num_atoms(k, pred_us);
    if (n <= 1 || !g_qmd_prog_addr_off) return launch_original(k);

    uint64_t blocks = k->total_blocks;
    uint64_t per = (blocks + n - 1) / n;   /* ceil so last atom covers remainder */

    /* The transparent redirect-to-Prelude is dispatched only when the Prelude
     * can actually transfer control into the original. That requires the callee
     * (original) not to fault on its EXIT inside the Prelude's call frame -- the
     * plain CALL path (default) on datacenter GPUs -- or the BRX jump. Gated by
     * LITHOS_ATOM_JUMP so the library stays correct where the transfer is
     * unsupported (e.g. GA102). Otherwise we launch the original once and just
     * surface the atom plan the scheduler would dispatch. */
    if (!g_lithos_cfg.enable_jump) {
        AZLOG("atom plan for func %p: %d atoms of ~%lu blocks (of %lu) [transfer gated off]",
              k->func, n, per, blocks);
        return launch_original(k);
    }

    pthread_mutex_lock(&g_mtx);
    AtomCtx* c = get_ctx((CUstream)k->stream);
    if (!c) { pthread_mutex_unlock(&g_mtx); return launch_original(k); }

    /* The Prelude's branchless select loads the target into R2. In CALL mode the
     * target is an absolute VA; in BRX mode it is a PC-relative displacement. */
    uint64_t entry_t, noop_t;
    if (g_lithos_cfg.use_brx) {
        long K = getenv("LITHOS_BRX_K") ? atoi(getenv("LITHOS_BRX_K")) : -192;
        uint64_t base = c->prelude_entry + (uint64_t)c->call_off + (uint64_t)K;
        entry_t = entry - base;
        noop_t  = c->noop_entry - base;
    } else {
        entry_t = entry;             /* absolute entry points for the CALL */
        noop_t  = c->noop_entry;
    }

    int launched = 0;
    for (int i = 0; i < n; i++) {
        uint64_t lo = (uint64_t)i * per;
        uint64_t hi = lo + per; if (hi > blocks) hi = blocks;
        if (lo >= hi) break;

        /* The Prelude reads a FIXED device address (c->meta_dev). Each atom's
         * metadata is copied to that same address, ordered on the stream ahead
         * of its launch, so every block of atom_i observes meta_i before the
         * next atom overwrites it. The host ring only keeps each async copy's
         * source buffer alive until the copy executes. */
        uint64_t slot = c->ring_idx++ % META_RING;
        AtomMetadata* m = &c->meta_host[slot];
        m->block_idx_lo = (uint32_t)lo;
        m->block_idx_hi = (uint32_t)hi;
        m->generation   = (uint32_t)c->ring_idx;
        m->_pad = 0;
        m->kernel_entrypoint = entry_t;
        m->noop_entrypoint   = noop_t;

        cuMemcpyHtoDAsync(c->meta_dev, m, sizeof(AtomMetadata), (CUstream)k->stream);
        qmd_arm_atomize(c->prelude_entry, c->prelude_regs);
        launch_original(k);
        launched++;
    }
    pthread_mutex_unlock(&g_mtx);
    AZLOG("atomized func %p into %d atoms (%lu blocks)", k->func, launched, blocks);
    return launched;
}
