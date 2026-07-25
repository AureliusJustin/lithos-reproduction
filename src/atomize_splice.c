/* atomize_splice.c — the LithOS Kernel Atomizer's core, done WITHOUT a control
 * transfer.
 *
 * Background. The paper's atomizer redirects a launch to a small "Prelude" kernel
 * that range-checks the block index and then JUMPS into the original kernel. That
 * jump can't be reproduced by byte-patching ptxas output (the original ends in
 * EXIT, and every register-indirect transfer we could inject faults on it — see
 * sass-jump/ and the technical report). So instead of a separate Prelude, we
 * PREPEND the range-check directly into each app kernel's own `.text.<fn>` section,
 * so in-range blocks simply FALL THROUGH into the original code — no CALL, no BRX,
 * no convergence barrier; the original's EXIT is its own clean top-level exit.
 *
 * The spliced-in prologue (built once by NVRTC, with the metadata VA baked in as a
 * literal) does, in SASS:
 *     b = global block index               // from blockIdx / gridDim
 *     if (b < meta->lo || b >= meta->hi) EXIT;   // predicated; no barrier
 *     <fall through into the original kernel body>
 * At launch time the atomizer writes AtomMetadata{lo,hi} into the meta buffer and
 * relaunches the (unmodified) grid once per atom.
 *
 * Prepending bytes to a `.text` section is not free: it shifts that section — and
 * every later section — down by `prolen`, so we must fix up every file offset and
 * every recorded instruction offset that the driver relies on (section/program
 * headers, the function symbol, `.nv.info` instruction-offset attributes, and
 * `-rdc` relocations). That fix-up is most of this file.
 *
 * Public entry points:
 *   int atomize_build_prologue(sm_major, sm_minor, meta_va, out, len)
 *   int atomize_splice_cubin(cubin, sz, pro, prolen, out, outsz, names, n_names)
 *   int atomize_elf_ok(buf, sz)          // bounds-validate before parsing
 *   int atomize_prologue_regs(void)      // register count the prologue needs
 */
#include <cuda.h>
#include <nvrtc.h>
#include <elf.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* ------------------------------------------------------------------------- *
 *  Named constants for the SASS / cubin details we poke at.
 * ------------------------------------------------------------------------- */

/* Ampere/Ada encode one instruction per 16 bytes (128-bit). */
#define SASS_INSN_BYTES 16

/* A single NOP instruction (Ampere/Ada encoding), used to pad the extracted
 * prologue up to a 16-byte-aligned, 128-byte-multiple length. */
static const unsigned char ATOM_NOP[SASS_INSN_BYTES] =
    {0x18,0x79,0,0, 0,0,0,0, 0,0,0,0, 0,0xc0,0x0f,0};

/* ptxas marks a `.text.<fn>` symbol as a launchable KERNEL ENTRY (as opposed to a
 * plain `__device__` function) by setting this bit in the symbol's st_other. */
#define STO_CUDA_ENTRY 0x10

/* `.nv.info` is a byte stream of attributes, each: [format:1][attr:1] <value...>.
 * The four value formats and how many total bytes each entry occupies: */
#define EIFMT_NONE 0x00   /* padding: 1 byte, no attr/value            */
#define EIFMT_NVAL 0x01   /* attr only:            2 bytes             */
#define EIFMT_BVAL 0x02   /* attr + 1-byte value:  3 bytes             */
#define EIFMT_HVAL 0x03   /* attr + 2-byte value:  4 bytes             */
#define EIFMT_SVAL 0x04   /* attr + [len:2][data]: 4+len bytes         */

/* Attribute ids we read or rewrite. */
#define EIATTR_EXIT_INSTR_OFFSETS   0x1c  /* offsets of EXIT instructions        */
#define EIATTR_CTAID_OFFSETS        0x1d  /* offsets of S2R %ctaid reads         */
#define EIATTR_COOP_INSTR_OFFSETS   0x31  /* another pure instruction-offset list */
#define EIATTR_REGCOUNT             0x2f  /* [sym_idx:4][regcount:4]             */
#define EIATTR_INDIRECT_BRANCH_TARGETS 0x34 /* [brx_off][flags][count][targets…] */

/* Section-name prefix every kernel/device-function `.text` section carries. */
#define TEXT_PREFIX     ".text."
#define TEXT_PREFIX_LEN 6

/* Register count the range-check prologue needs. Learned from the prologue cubin's
 * own REGCOUNT (see atomize_build_prologue); used to raise pathologically-small
 * kernels' allocation before splicing so the added code has enough registers. */
static int g_prologue_regs = 8;
int atomize_prologue_regs(void) { return g_prologue_regs; }

/* ------------------------------------------------------------------------- *
 *  .nv.info iterator — one decoder used by all three walk sites below.
 * ------------------------------------------------------------------------- */

typedef struct {
    unsigned char  attr;   /* attribute id (EIATTR_*)                       */
    unsigned char  fmt;    /* value format (EIFMT_*)                        */
    unsigned char* val;    /* pointer to the value bytes (NULL for NVAL)    */
    unsigned       vlen;   /* value length in bytes                         */
} NvInfoAttr;

/* Decode the attribute at byte offset *pos in [base, base+size) and advance *pos
 * past it. Returns 1 and fills `out` for each attribute, 0 at end of stream. */
static int nv_info_next(unsigned char* base, size_t size, size_t* pos, NvInfoAttr* out) {
    size_t x = *pos;
    if (x + 2 > size) return 0;
    out->fmt  = base[x];
    out->attr = base[x + 1];
    out->val  = NULL;
    out->vlen = 0;
    switch (out->fmt) {
        case EIFMT_NONE:  *pos = x + 1; return 1;                 /* padding byte  */
        case EIFMT_NVAL:  *pos = x + 2; return 1;
        case EIFMT_BVAL:  out->val = base + x + 2; out->vlen = 1; *pos = x + 3; return 1;
        case EIFMT_HVAL:  out->val = base + x + 2; out->vlen = 2; *pos = x + 4; return 1;
        case EIFMT_SVAL: {
            unsigned vlen = base[x + 2] | (base[x + 3] << 8);
            out->val = base + x + 4; out->vlen = vlen; *pos = x + 4 + vlen; return 1;
        }
        default:          *pos = x + 1; return 1;                 /* unknown: skip 1 */
    }
}

/* ------------------------------------------------------------------------- *
 *  Build the range-check prologue (SASS bytes) for a given arch + meta VA.
 * ------------------------------------------------------------------------- */
int atomize_build_prologue(int sm_major, int sm_minor, uint64_t meta_va,
                           unsigned char** out, size_t* len) {
    /* NVRTC source: compute the global block index, EXIT out-of-range blocks, and
     * then write a unique marker (0xDEADBEEF). We compile the whole kernel, then
     * keep only the bytes UP TO that marker — that prefix is the range-check we
     * splice; the marker + trailing store are discarded. `meta_va` is baked in as
     * a literal so the check reads AtomMetadata{lo,hi} from a fixed address (a
     * `__device__` global wouldn't work: the const bank belongs to the ORIGINAL
     * kernel we relaunch, so a global-address load would read garbage). */
    char arch[32];
    snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d", sm_major, sm_minor);

    char src[1024];
    snprintf(src, sizeof(src),
        "struct M{unsigned lo,hi;};\n"
        "extern \"C\" __global__ void probe(unsigned* sink){\n"
        " unsigned long long b=(unsigned long long)blockIdx.z*gridDim.y*gridDim.x"
        "+(unsigned long long)blockIdx.y*gridDim.x+blockIdx.x;\n"
        " M* a=(M*)%lluULL;\n"
        " if(b<a->lo||b>=a->hi) return;\n"
        " asm volatile(\"mov.u32 %%0, 0xDEADBEEF;\":\"=r\"(sink[0])::\"memory\");\n"
        "}\n", (unsigned long long)meta_va);

    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src, "atom_prologue.cu", 0, 0, 0)) return -1;
    const char* opts[] = { arch };
    if (nvrtcCompileProgram(prog, 1, opts)) {
        size_t log_sz; nvrtcGetProgramLogSize(prog, &log_sz);
        char* log = malloc(log_sz); nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "[atomize] NVRTC prologue compile failed:\n%s\n", log);
        free(log); return -1;
    }
    size_t cubin_sz; nvrtcGetCUBINSize(prog, &cubin_sz);
    unsigned char* cb = malloc(cubin_sz);
    nvrtcGetCUBIN(prog, (char*)cb);
    nvrtcDestroyProgram(&prog);

    /* Parse the prologue cubin: eh = ELF header, sh = section headers, ss = the
     * section-header string table (section names live here). */
    Elf64_Ehdr* eh = (void*)cb;
    Elf64_Shdr* sh = (void*)(cb + eh->e_shoff);
    const char* ss = (char*)(cb + sh[eh->e_shstrndx].sh_offset);

    /* Record how many registers the prologue itself needs, from the module-wide
     * `.nv.info`'s REGCOUNT attribute ([sym_idx:4][regcount:4]); keep the max. */
    for (int i = 0; i < eh->e_shnum; i++) {
        if (strcmp(ss + sh[i].sh_name, ".nv.info")) continue;
        unsigned char* info = cb + sh[i].sh_offset;
        size_t info_sz = sh[i].sh_size, pos = 0;
        NvInfoAttr a;
        while (nv_info_next(info, info_sz, &pos, &a)) {
            if (a.attr == EIATTR_REGCOUNT && a.vlen >= 8) {
                unsigned regcount = ((unsigned*)a.val)[1];
                if ((int)regcount > g_prologue_regs) g_prologue_regs = (int)regcount;
            }
        }
    }

    /* Extract `.text.probe` up to the 0xDEADBEEF marker instruction. The marker is
     * a `mov.u32 Rd, 0xDEADBEEF`, so the immediate's little-endian bytes
     * ef be ad de appear somewhere inside its 16-byte instruction; `cut` = the
     * start of that instruction. */
    int rc = -1;
    for (int i = 0; i < eh->e_shnum; i++) {
        if (strcmp(ss + sh[i].sh_name, ".text.probe")) continue;
        unsigned char* text = cb + sh[i].sh_offset;
        size_t text_sz = sh[i].sh_size, cut = text_sz;
        for (size_t x = 0; x + SASS_INSN_BYTES <= text_sz; x += SASS_INSN_BYTES)
            for (int k = 0; k < 13; k++)   /* marker can sit at any byte 0..12 */
                if (text[x+k]==0xef && text[x+k+1]==0xbe && text[x+k+2]==0xad && text[x+k+3]==0xde) {
                    cut = x; goto found;
                }
    found:
        if (cut == text_sz) { fprintf(stderr, "[atomize] prologue marker not found\n"); break; }

        /* Pad the range-check bytes up to a 128-byte multiple with NOPs (SASS
         * scheduling groups are 128-byte aligned), and hand them back. */
        size_t padded = (cut + 127) & ~(size_t)127;
        unsigned char* pr = malloc(padded);
        memcpy(pr, text, cut);
        for (size_t x = cut; x < padded; x += SASS_INSN_BYTES) memcpy(pr + x, ATOM_NOP, SASS_INSN_BYTES);
        *out = pr; *len = padded; rc = 0;
        break;
    }
    free(cb);
    return rc;
}

/* ------------------------------------------------------------------------- *
 *  Splice the prologue into ONE kernel's .text section.
 *
 *  cb/z     : the current cubin bytes and size (read-only input)
 *  textsec  : the ".text.<fn>" section name to splice into
 *  fname    : the kernel name "<fn>" (used to find .rel/.rela/.nv.info sections)
 *  pro/prolen: the prologue bytes to prepend
 *  Returns a freshly-malloc'd, spliced copy (size in *nzout), or NULL to skip this
 *  kernel (it will be loaded/launched verbatim).
 * ------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------- *
 *  Validate that `b` is a well-formed ELF cubin fully contained in `sz` bytes,
 *  so the splicer never reads past the end of a small/unusual image (e.g. the
 *  tiny stub cubins cuBLASLt loads, or a differently-laid-out cubin from a newer
 *  CUDA). Returns 1 if safe to parse, 0 otherwise (caller loads it verbatim).
 * ------------------------------------------------------------------------- */
int atomize_elf_ok(const unsigned char* b, size_t sz) {
    if (sz < sizeof(Elf64_Ehdr) || memcmp(b, ELFMAG, SELFMAG)) return 0;
    const Elf64_Ehdr* eh = (const void*)b;
    if (eh->e_shentsize != sizeof(Elf64_Shdr) || eh->e_shnum == 0 || eh->e_shoff == 0) return 0;
    if (eh->e_shoff > sz || (size_t)eh->e_shnum * eh->e_shentsize > sz - eh->e_shoff) return 0;
    if (eh->e_phnum && (eh->e_phoff > sz || (size_t)eh->e_phnum * eh->e_phentsize > sz - eh->e_phoff)) return 0;
    if (eh->e_shstrndx >= eh->e_shnum) return 0;

    const Elf64_Shdr* sh = (const void*)(b + eh->e_shoff);
    size_t stro = sh[eh->e_shstrndx].sh_offset, strsz = sh[eh->e_shstrndx].sh_size;
    if (stro > sz || strsz > sz - stro) return 0;               /* string table fits */
    for (int i = 0; i < eh->e_shnum; i++) {
        if (sh[i].sh_name >= strsz) return 0;                   /* name index in range */
        if (sh[i].sh_type != SHT_NOBITS &&
            (sh[i].sh_offset > sz || sh[i].sh_size > sz - sh[i].sh_offset)) return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------------- *
 *  Single-pass splice of EVERY kernel entry in the cubin.
 *
 *  A naive implementation splices one kernel at a time, re-copying the whole
 *  cubin each time — O(n^2) in kernels per module (measured: 600 kernels = 92 ms).
 *  Instead we plan every insertion first, then do ONE segmented copy and ONE
 *  header fix-up pass, which is linear.
 *
 *  The key simplification: only FILE offsets (section headers, program headers,
 *  e_shoff/e_phoff) see the cumulative effect of several insertions. Symbol
 *  st_value, `.nv.info.<fn>` instruction offsets and relocation r_offsets are all
 *  SECTION-RELATIVE, so each kernel shifts by just its own `prolen` regardless of
 *  how many other kernels were spliced.
 * ------------------------------------------------------------------------- */

/* One planned insertion: prepend `prolen` bytes at the front of section `ti`. */
typedef struct {
    int         ti;        /* section index of .text.<fn>                    */
    size_t      toff;      /* its ORIGINAL file offset (the insertion point) */
    const char* fname;     /* kernel name (points into the input's strtab)   */
    int         rel_i;     /* .rel.text.<fn>  section index, or -1           */
    int         rela_i;    /* .rela.text.<fn> section index, or -1           */
    int         fsym;      /* index of the function's own symbol, or -1      */
} Splice;

static int splice_cmp(const void* a, const void* b) {
    size_t x = ((const Splice*)a)->toff, y = ((const Splice*)b)->toff;
    return x < y ? -1 : (x > y ? 1 : 0);
}

/* Name -> splice index. Several fix-up passes need to map a section name suffix
 * (".rel.text.<fn>", ".nv.info.<fn>") back to its kernel; scanning the splice list
 * per section would be O(sections x kernels), the last quadratic term. We keep an
 * index sorted by name and binary-search it instead. */
typedef struct { const char* name; int k; } NameIdx;
static int nameidx_cmp(const void* a, const void* b) {
    return strcmp(((const NameIdx*)a)->name, ((const NameIdx*)b)->name);
}
static int nameidx_find(const NameIdx* idx, int n, const char* name) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2, c = strcmp(idx[m].name, name);
        if (c == 0) return idx[m].k;
        if (c < 0) lo = m + 1; else hi = m - 1;
    }
    return -1;
}

/* How many insertions land before a given file offset — i.e. how far that offset
 * slides. `sp` is sorted by toff, so this is a binary search (O(log n)); a linear
 * count here would make the whole fix-up pass O(sections x kernels). */
static int shift_count_before(const Splice* sp, int n, size_t o) {
    int lo = 0, hi = n;                       /* first index with toff >= o */
    while (lo < hi) { int m = (lo + hi) / 2; if (sp[m].toff < o) lo = m + 1; else hi = m; }
    return lo;
}
static int shift_count_at_or_before(const Splice* sp, int n, size_t o) {
    int lo = 0, hi = n;                       /* first index with toff > o */
    while (lo < hi) { int m = (lo + hi) / 2; if (sp[m].toff <= o) lo = m + 1; else hi = m; }
    return lo;
}

int atomize_splice_cubin(const void* cubin, size_t sz, const unsigned char* pro, size_t prolen,
                         void** out, size_t* outsz, char*** names_out, int* n_names_out) {
    *names_out = 0; *n_names_out = 0;
    if (!atomize_elf_ok(cubin, sz)) return -1;      /* malformed: load verbatim */

    const unsigned char* cb = cubin;
    const Elf64_Ehdr* eh = (const void*)cb;
    const Elf64_Shdr* sh = (const void*)(cb + eh->e_shoff);
    const char* ss = (const char*)(cb + sh[eh->e_shstrndx].sh_offset);
    int shnum = eh->e_shnum;

    /* ---- locate the symbol table ---------------------------------------- */
    const Elf64_Sym* syms = NULL; int nsyms = 0; const char* symstr = NULL;
    for (int i = 0; i < shnum; i++) {
        if (sh[i].sh_type != SHT_SYMTAB) continue;
        syms   = (const void*)(cb + sh[i].sh_offset);
        nsyms  = sh[i].sh_size / sizeof(Elf64_Sym);
        symstr = (const char*)(cb + sh[sh[i].sh_link].sh_offset);
        break;
    }

    /* Which symbol defines each section's function? Computed in ONE pass over the
     * symbol table — searching per section instead would be O(sections x symbols),
     * which is the other quadratic trap in a 600-kernel module. */
    int* sec_func_sym = malloc(sizeof(int) * (shnum ? shnum : 1));
    for (int i = 0; i < shnum; i++) sec_func_sym[i] = -1;
    for (int j = 0; j < nsyms; j++) {
        if (ELF64_ST_TYPE(syms[j].st_info) != STT_FUNC) continue;
        if (syms[j].st_shndx >= shnum || syms[j].st_value != 0) continue;
        sec_func_sym[syms[j].st_shndx] = j;      /* the function at offset 0 */
    }

    /* ---- plan: which .text.<fn> sections are kernel entries we can splice? */
    Splice* sp = calloc(shnum ? shnum : 1, sizeof(Splice));
    int nsp = 0;
    for (int i = 0; i < shnum; i++) {
        const char* nm = ss + sh[i].sh_name;
        if (strncmp(nm, TEXT_PREFIX, TEXT_PREFIX_LEN)) continue;
        const char* fn = nm + TEXT_PREFIX_LEN;

        /* Only real kernel ENTRIES: a __device__ function also gets a .text.<fn>,
         * but it is CALLed, so an EXIT-prologue would kill the thread instead of
         * returning. ptxas flags entries with st_other bit STO_CUDA_ENTRY. */
        if (syms) {
            int j = sec_func_sym[i];
            if (j < 0 || !(syms[j].st_other & STO_CUDA_ENTRY)) continue;
            if (strcmp(symstr + syms[j].st_name, fn)) continue;   /* name must match */
        }
        sp[nsp].ti = i; sp[nsp].toff = sh[i].sh_offset; sp[nsp].fname = fn;
        sp[nsp].rel_i = sp[nsp].rela_i = -1; sp[nsp].fsym = -1;
        nsp++;
    }
    free(sec_func_sym);
    if (nsp == 0) { free(sp); return -1; }

    /* Name index for the lookup passes below (built once, binary-searched). */
    NameIdx* nidx = malloc(sizeof(NameIdx) * nsp);
    for (int k = 0; k < nsp; k++) { nidx[k].name = sp[k].fname; nidx[k].k = k; }
    qsort(nidx, nsp, sizeof(NameIdx), nameidx_cmp);

    /* ---- attach each kernel's relocation sections (one pass over sections) - */
    for (int i = 0; i < shnum; i++) {
        const char* nm = ss + sh[i].sh_name;
        const char* fn = NULL; int is_rela = 0;
        if      (!strncmp(nm, ".rel.text.",  10)) { fn = nm + 10; is_rela = 0; }
        else if (!strncmp(nm, ".rela.text.", 11)) { fn = nm + 11; is_rela = 1; }
        else continue;
        int k = nameidx_find(nidx, nsp, fn);
        if (k >= 0) { if (is_rela) sp[k].rela_i = i; else sp[k].rel_i = i; }
    }

    /* ---- drop kernels whose REL relocations are self-referential ---------- *
     * A REL keeps its addend inside the instruction bytes, so shifting the code
     * would require rewriting the instruction — we can't do that safely. (Cross-
     * section RELs only need r_offset bumped, which is fine.) */
    for (int k = 0; k < nsp; ) {
        int drop = 0;
        if (sp[k].rel_i >= 0) {
            const Elf64_Sym* rs = (const void*)(cb + sh[sh[sp[k].rel_i].sh_link].sh_offset);
            const Elf64_Rel* r  = (const void*)(cb + sh[sp[k].rel_i].sh_offset);
            int nrel = sh[sp[k].rel_i].sh_size / sizeof(Elf64_Rel);
            for (int j = 0; j < nrel; j++)
                if (rs[ELF64_R_SYM(r[j].r_info)].st_shndx == sp[k].ti) { drop = 1; break; }
        }
        if (drop) { memmove(&sp[k], &sp[k+1], (nsp - k - 1) * sizeof(Splice)); nsp--; }
        else k++;
    }
    if (nsp == 0) { free(sp); return -1; }

    /* Sort by file offset so the segmented copy below is a single forward sweep. */
    qsort(sp, nsp, sizeof(Splice), splice_cmp);

    /* ---- one segmented copy --------------------------------------------- */
    size_t nz = sz + (size_t)nsp * prolen;
    unsigned char* nb = malloc(nz);
    if (!nb) { free(sp); return -1; }
    {
        size_t src = 0, dst = 0;
        for (int k = 0; k < nsp; k++) {
            size_t run = sp[k].toff - src;            /* original bytes before it */
            memcpy(nb + dst, cb + src, run);
            dst += run; src += run;
            memcpy(nb + dst, pro, prolen);            /* the prologue */
            dst += prolen;
        }
        memcpy(nb + dst, cb + src, sz - src);         /* the tail */
    }

    Elf64_Ehdr* ne = (void*)nb;

    /* ---- file-offset fix-ups (the only place insertions accumulate) ------- */
    ne->e_shoff += (size_t)shift_count_before(sp, nsp, eh->e_shoff) * prolen;
    ne->e_phoff += (size_t)shift_count_before(sp, nsp, eh->e_phoff) * prolen;
    Elf64_Shdr* nsh = (void*)(nb + ne->e_shoff);

    /* Map section index -> its entry in sp (or -1), for the per-kernel passes. */
    int* spliced_of = malloc(sizeof(int) * (shnum ? shnum : 1));
    for (int i = 0; i < shnum; i++) spliced_of[i] = -1;
    for (int k = 0; k < nsp; k++) spliced_of[sp[k].ti] = k;

    for (int i = 0; i < shnum; i++) {
        int k = spliced_of[i];
        if (k >= 0) {
            /* A spliced section now STARTS at its prologue, which sits after all
             * insertions strictly before it (k of them, since sp is sorted). */
            nsh[i].sh_offset = sp[k].toff + (size_t)k * prolen;
            nsh[i].sh_size  += prolen;
        } else {
            nsh[i].sh_offset += (size_t)shift_count_at_or_before(sp, nsp, sh[i].sh_offset) * prolen;
        }
    }

    /* Program headers (the driver loads via THESE): a segment grows by prolen for
     * every insertion inside it, and slides by prolen for every insertion before
     * its start. */
    Elf64_Phdr* ph = (void*)(nb + ne->e_phoff);
    const Elf64_Phdr* oph = (const void*)(cb + eh->e_phoff);
    for (int i = 0; i < eh->e_phnum; i++) {
        size_t p_off = oph[i].p_offset, p_end = p_off + oph[i].p_filesz;
        int inside = 0, before = 0;
        for (int k = 0; k < nsp; k++) {
            if (sp[k].toff < p_off)                              before++;
            else if (sp[k].toff >= p_off && sp[k].toff < p_end)   inside++;
        }
        ph[i].p_offset = p_off + (size_t)before * prolen;
        ph[i].p_filesz += (size_t)inside * prolen;
        ph[i].p_memsz  += (size_t)inside * prolen;
    }

    const char* nss = (const char*)(nb + nsh[ne->e_shstrndx].sh_offset);

    /* ---- section-relative fix-ups (each kernel shifts by its OWN prolen) --- */

    /* Symbols: one pass. The function symbol grows (its entry stays at offset 0,
     * which is now the prologue — exactly what the launch should hit); any other
     * symbol inside a spliced section is a local label that moves with the code. */
    for (int i = 0; i < shnum; i++) {
        if (nsh[i].sh_type != SHT_SYMTAB) continue;
        Elf64_Sym* nsy = (void*)(nb + nsh[i].sh_offset);
        int cnt = nsh[i].sh_size / sizeof(Elf64_Sym);
        const char* st = (const char*)(nb + nsh[nsh[i].sh_link].sh_offset);
        for (int j = 0; j < cnt; j++) {
            int k = (nsy[j].st_shndx < shnum) ? spliced_of[nsy[j].st_shndx] : -1;
            if (k < 0) continue;
            if (!strcmp(st + nsy[j].st_name, sp[k].fname)) { nsy[j].st_size += prolen; sp[k].fsym = j; }
            else if (nsy[j].st_value > 0) nsy[j].st_value += prolen;
        }
    }

    /* Which symbol indices are spliced kernels (for the REGCOUNT pass below). */
    unsigned char* sym_is_spliced = calloc(nsyms ? nsyms : 1, 1);
    for (int k = 0; k < nsp; k++) if (sp[k].fsym >= 0) sym_is_spliced[sp[k].fsym] = 1;

    /* Module-wide `.nv.info`: raise each spliced kernel's REGCOUNT so a tiny
     * kernel doesn't under-allocate registers for the added code. */
    for (int i = 0; i < shnum; i++) {
        if (strcmp(nss + nsh[i].sh_name, ".nv.info")) continue;
        unsigned char* info = nb + nsh[i].sh_offset;
        size_t info_sz = nsh[i].sh_size, pos = 0;
        NvInfoAttr a;
        while (nv_info_next(info, info_sz, &pos, &a)) {
            if (a.attr != EIATTR_REGCOUNT || a.vlen < 8) continue;
            unsigned* v = (unsigned*)a.val;             /* v[0]=sym_idx, v[1]=regs */
            if (v[0] < (unsigned)nsyms && sym_is_spliced[v[0]] &&
                (int)v[1] < g_prologue_regs) v[1] = g_prologue_regs;
        }
    }

    /* Per-kernel `.nv.info.<fn>`: every recorded instruction offset moves down by
     * that kernel's prolen. */
    for (int i = 0; i < shnum; i++) {
        const char* nm = nss + nsh[i].sh_name;
        if (strncmp(nm, ".nv.info.", 9)) continue;
        int k = nameidx_find(nidx, nsp, nm + 9);
        if (k < 0) continue;

        unsigned char* info = nb + nsh[i].sh_offset;
        size_t info_sz = nsh[i].sh_size, pos = 0;
        NvInfoAttr a;
        while (nv_info_next(info, info_sz, &pos, &a)) {
            if (a.fmt != EIFMT_SVAL) continue;
            unsigned* v = (unsigned*)a.val;
            unsigned  n = a.vlen / 4;
            if (a.attr == EIATTR_EXIT_INSTR_OFFSETS ||
                a.attr == EIATTR_CTAID_OFFSETS ||
                a.attr == EIATTR_COOP_INSTR_OFFSETS) {
                for (unsigned q = 0; q < n; q++) v[q] += (unsigned)prolen;
            } else if (a.attr == EIATTR_INDIRECT_BRANCH_TARGETS && n >= 3) {
                /* [brx_off][flags][count][targets...] — bump the site and targets */
                v[0] += (unsigned)prolen;
                unsigned count = v[2];
                for (unsigned q = 0; q < count && 3 + q < n; q++) v[3 + q] += (unsigned)prolen;
            }
        }
    }

    /* Relocations: every r_offset is a patch site inside the moved code. A
     * self-referential RELA whose symbol is the function base (st_value == 0)
     * carries an offset INTO that code in its addend (e.g. the return address for
     * a device-function CALL), so bump the addend too. A RELA aimed at a local
     * label (st_value > 0) was already handled by that label's st_value bump. */
    for (int k = 0; k < nsp; k++) {
        if (sp[k].rel_i >= 0) {
            Elf64_Rel* r = (void*)(nb + nsh[sp[k].rel_i].sh_offset);
            int nrel = nsh[sp[k].rel_i].sh_size / sizeof(Elf64_Rel);
            for (int j = 0; j < nrel; j++) r[j].r_offset += prolen;
        }
        if (sp[k].rela_i >= 0) {
            Elf64_Sym*  rs = (void*)(nb + nsh[nsh[sp[k].rela_i].sh_link].sh_offset);
            Elf64_Rela* r  = (void*)(nb + nsh[sp[k].rela_i].sh_offset);
            int nrela = nsh[sp[k].rela_i].sh_size / sizeof(Elf64_Rela);
            for (int j = 0; j < nrela; j++) {
                r[j].r_offset += prolen;
                Elf64_Sym* s = &rs[ELF64_R_SYM(r[j].r_info)];
                if ((int)s->st_shndx == sp[k].ti && s->st_value == 0) r[j].r_addend += prolen;
            }
        }
    }

    /* ---- report the names we spliced, so the caller gates exactly those ---- */
    char** done = calloc(nsp, sizeof(char*));
    for (int k = 0; k < nsp; k++) done[k] = strdup(sp[k].fname);

    free(sym_is_spliced); free(nidx); free(spliced_of); free(sp);
    *out = nb; *outsz = nz; *names_out = done; *n_names_out = nsp;
    return 0;
}
