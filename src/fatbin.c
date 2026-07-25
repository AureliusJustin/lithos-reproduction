/* fatbin.c -- unwrap CUDA fatbins and PTX so the atomizer can splice them.
 *
 * cuModuleLoadData accepts three image kinds; the splicer needs a raw ELF cubin:
 *   - raw ELF cubin (\x7fELF)      -> used directly
 *   - fatbin (magic 0xBA55ED50)    -> find the ELF entry matching the running SM,
 *                                     decompress it if flagged, use that cubin
 *   - PTX (text) / PTX-only fatbin -> JIT to a cubin with the driver linker
 * In every case we hand back a bare cubin; the driver loads a raw cubin fine, so
 * no fatbin *repackaging* is needed -- we just load the (spliced) inner cubin.
 *
 * Compression: NVIDIA used plain LZ4 *block* format through CUDA 12 (the payload
 * begins with an LZ4 token, not the frame magic) and switched to Zstandard in
 * CUDA 13. Both carry the uncompressed size in the entry header, so we can decode
 * straight into a correctly-sized buffer. See the layout notes below.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <elf.h>

#define FATBIN_MAGIC         0xBA55ED50u
/* The CUDA runtime hands cuLibraryLoadData/cuModuleLoadData a __fatBinC_Wrapper_t
 * { int magic=0x466243b1; int version; const void* data; const void* filename; }
 * whose `data` (offset 8) points at the real fatbin (0xBA55ED50). */
#define FATBIN_WRAPPER_MAGIC 0x466243b1u
#define FATBIN_WRAPPER_DATA_OFF 8      /* byte offset of `data` in the wrapper */

/* ---- fatbin layout (reverse-engineered; all little-endian) ----------------
 *
 * Container header:
 *   +0x00 u32  magic      = FATBIN_MAGIC
 *   +0x04 u16  version
 *   +0x06 u16  header_size          (entries start here)
 *   +0x08 u64  fat_size             (total size of all entries)
 *
 * Then a sequence of entries, each: [entry header][payload], where the header is
 * `entry_header_size` bytes and the payload `stored_size` bytes, so the next entry
 * is at (entry + entry_header_size + stored_size). Entry header fields we use:
 */
#define FE_KIND        0x00   /* u16: 1 = PTX, 2 = ELF cubin                     */
#define FE_HEADER_SIZE 0x04   /* u32: size of THIS entry header (>= 0x40)        */
#define FE_STORED_SIZE 0x08   /* u64: bytes of payload on disk (may be padded)   */
#define FE_COMP_SIZE   0x10   /* u64: exact compressed byte count (zstd needs it)*/
#define FE_ARCH        0x1c   /* u32: SM arch, e.g. 86                           */
#define FE_FLAGS       0x28   /* u64: bit 0x2000 = LZ4, bit 0x8000 = zstd        */
#define FE_UNCOMP_SIZE 0x38   /* u64: uncompressed payload size                  */
#define FE_MIN_HEADER  0x40   /* smallest entry header we know how to parse      */

#define FE_KIND_PTX 1
#define FE_KIND_ELF 2

#define FE_FLAG_LZ4  0x2000   /* CUDA <= 12 */
#define FE_FLAG_ZSTD 0x8000   /* CUDA 13+   */

/* Minimal LZ4 *block* decompressor (NVIDIA stores raw blocks, not LZ4 frames, so
 * there is no frame magic/header to skip and no checksum to verify).
 *
 * LZ4 block format: a sequence of "sequences", each
 *   [token:1] [extra literal length…] [literals…] [match offset:2] [extra match length…]
 * where token's high nibble = literal length and low nibble = match length, each
 * extended by trailing 255-chained bytes when the nibble is 15. A match copies
 * `matchlen` bytes from `offset` bytes EARLIER in the output (so it can overlap,
 * which is how runs are encoded — hence the byte-by-byte copy below).
 *
 *   src/srcSize : compressed input      s = read cursor
 *   dst/dstCap  : output buffer         d = write cursor (dstCap = uncompressed
 *                                           size, known from the entry header)
 * Returns bytes written, or -1 if any length/offset would read or write out of
 * bounds (i.e. the input isn't the LZ4 we expect). */
static int lz4_block_decompress(const uint8_t* src, int srcSize, uint8_t* dst, int dstCap) {
    int s = 0, d = 0;
    while (s < srcSize && d < dstCap) {
        int token = src[s++];

        /* --- literals: copy `litlen` bytes straight through --- */
        int litlen = token >> 4;
        if (litlen == 15) {                 /* 15 means "add the following bytes" */
            int b;
            do { if (s >= srcSize) return -1; b = src[s++]; litlen += b; } while (b == 255);
        }
        if (s + litlen > srcSize || d + litlen > dstCap) return -1;
        memcpy(dst + d, src + s, litlen);
        s += litlen; d += litlen;
        if (d >= dstCap || s >= srcSize) break;   /* last sequence is literals-only */

        /* --- match: copy `matchlen` bytes from `offset` back in the OUTPUT --- */
        if (s + 2 > srcSize) return -1;
        int offset = src[s] | (src[s + 1] << 8); s += 2;   /* little-endian back-reference */
        int matchlen = token & 0xf;
        if (matchlen == 15) {
            int b;
            do { if (s >= srcSize) return -1; b = src[s++]; matchlen += b; } while (b == 255);
        }
        matchlen += 4;                       /* LZ4 minimum match length is 4 */
        int mpos = d - offset;               /* source position within dst */
        if (offset <= 0 || mpos < 0 || d + matchlen > dstCap) return -1;
        /* byte-by-byte on purpose: source and destination ranges may overlap when
         * offset < matchlen (that's how LZ4 encodes repeated runs). */
        for (int i = 0; i < matchlen; i++) dst[d + i] = dst[mpos + i];
        d += matchlen;
    }
    return d;
}

/* CUDA 13 switched fatbin compression from LZ4 (entry flag 0x2000) to Zstandard
 * (flag 0x8000, payload magic 0x28b52ffd). zstd needs the EXACT compressed size
 * (the entry's `compressed_size` field), not the padded stored size. We link the
 * system libzstd.so.1 and declare only what we call (no dev headers needed). */
#define FAT_COMP_NONE 0
#define FAT_COMP_LZ4  1
#define FAT_COMP_ZSTD 2
extern size_t ZSTD_decompress(void*, size_t, const void*, size_t);
extern unsigned ZSTD_isError(size_t);
static int fat_decompress(int comp, const unsigned char* src, size_t stored,
                          size_t csize, unsigned char* dst, int usize) {
    if (comp == FAT_COMP_ZSTD) {
        size_t r = ZSTD_decompress(dst, (size_t)usize, src, csize ? csize : stored);
        return ZSTD_isError(r) ? -1 : (int)r;
    }
    return lz4_block_decompress(src, (int)stored, dst, usize);   /* LZ4 */
}

/* Extract the cubin for `want_sm` from a fatbin. On success returns a malloc'd
 * ELF image in *out (caller frees) and 1; 0 if no matching ELF entry (caller may
 * try PTX); -1 on parse error. Also reports the best PTX entry (for JIT). */
static int fatbin_extract_elf(const unsigned char* fb, int want_sm,
                              void** out, size_t* outsz,
                              const unsigned char** ptx, size_t* ptxsz) {
    uint32_t magic; uint16_t ver, hsz; uint64_t fatsz;
    memcpy(&magic, fb, 4); memcpy(&ver, fb + 4, 2); memcpy(&hsz, fb + 6, 2); memcpy(&fatsz, fb + 8, 8);
    if (magic != FATBIN_MAGIC) return -1;
    if (ptx) { *ptx = NULL; *ptxsz = 0; }

    size_t off = hsz, endp = (size_t)hsz + fatsz;
    const unsigned char* best_elf = NULL; size_t best_elf_stored = 0, best_elf_csize = 0;
    int best_elf_comp = 0, best_elf_usz = 0, best_sm = -1;
    const unsigned char* ptx_pl = NULL; size_t ptx_stored = 0, ptx_csize = 0; int ptx_comp = 0, ptx_usz = 0;
    while (off + FE_MIN_HEADER <= endp) {
        uint16_t kind;   uint32_t ehsz;
        uint64_t stored, csize, flags, usize;
        uint32_t arch;
        memcpy(&kind,   fb + off + FE_KIND,        2);
        memcpy(&ehsz,   fb + off + FE_HEADER_SIZE, 4);
        memcpy(&stored, fb + off + FE_STORED_SIZE, 8);
        memcpy(&csize,  fb + off + FE_COMP_SIZE,   8);
        memcpy(&arch,   fb + off + FE_ARCH,        4);
        memcpy(&flags,  fb + off + FE_FLAGS,       8);
        memcpy(&usize,  fb + off + FE_UNCOMP_SIZE, 8);
        /* Unknown/blackwell-style entries use a bigger header; bail rather than
         * mis-parse. Also guard against a truncated/lying size. */
        if (ehsz < FE_MIN_HEADER || off + ehsz + stored > endp) break;
        const unsigned char* payload = fb + off + ehsz;
        int comp = (flags & FE_FLAG_ZSTD) ? FAT_COMP_ZSTD
                 : ((flags & FE_FLAG_LZ4) ? FAT_COMP_LZ4 : FAT_COMP_NONE);

        if (kind == FE_KIND_ELF) {   /* prefer exact SM match, else highest <= want */
            int sm = (int)arch;
            if (sm == want_sm || (sm <= want_sm && sm > best_sm)) {
                best_elf = payload; best_elf_stored = stored; best_elf_csize = csize;
                best_elf_comp = comp; best_elf_usz = (int)(comp ? usize : stored); best_sm = sm;
            }
        } else if (kind == FE_KIND_PTX && !ptx_pl) {  /* first PTX entry: JIT fallback */
            ptx_pl = payload; ptx_stored = stored; ptx_csize = csize; ptx_comp = comp;
            ptx_usz = (int)(comp ? usize : stored);
        }
        off += ehsz + stored;
    }

    if (best_elf) {
        if (!best_elf_comp) {
            void* c = malloc(best_elf_stored); memcpy(c, best_elf, best_elf_stored);
            *out = c; *outsz = best_elf_stored; return 1;
        }
        unsigned char* c = malloc(best_elf_usz);
        int n = fat_decompress(best_elf_comp, best_elf, best_elf_stored, best_elf_csize, c, best_elf_usz);
        if (n != best_elf_usz || memcmp(c, ELFMAG, SELFMAG) != 0) { free(c); return -1; }
        *out = c; *outsz = best_elf_usz; return 1;
    }

    /* No usable ELF -- hand back a decompressed, NUL-terminated PTX for JIT. */
    if (ptx && ptx_pl) {
        unsigned char* p = malloc(ptx_usz + 1);
        if (ptx_comp) {
            if (fat_decompress(ptx_comp, ptx_pl, ptx_stored, ptx_csize, p, ptx_usz) != ptx_usz) { free(p); return 0; }
        } else memcpy(p, ptx_pl, ptx_usz);
        p[ptx_usz] = 0;
        *ptx = p; *ptxsz = ptx_usz;
    }
    return 0;
}

/* JIT a PTX string to a cubin via the driver linker (uses the context's device
 * arch). Returns malloc'd cubin in *out, or -1. */
static int ptx_to_cubin(const char* ptx, size_t ptxsz, void** out, size_t* outsz) {
    CUlinkState ls;
    if (cuLinkCreate(0, NULL, NULL, &ls) != CUDA_SUCCESS) return -1;
    /* cuLinkAddData wants a NUL-terminated PTX buffer. */
    char* buf = malloc(ptxsz + 1); memcpy(buf, ptx, ptxsz); buf[ptxsz] = 0;
    CUresult r = cuLinkAddData(ls, CU_JIT_INPUT_PTX, buf, ptxsz + 1, "lithos_ptx", 0, NULL, NULL);
    free(buf);
    if (r != CUDA_SUCCESS) { cuLinkDestroy(ls); return -1; }
    void* image; size_t isz;
    if (cuLinkComplete(ls, &image, &isz) != CUDA_SUCCESS) { cuLinkDestroy(ls); return -1; }
    void* c = malloc(isz); memcpy(c, image, isz);   /* owned by ls; copy out */
    cuLinkDestroy(ls);
    *out = c; *outsz = isz; return 0;
}

/* Public: turn any module image into a raw cubin the splicer can consume.
 * Returns 0 and sets *out (malloc'd, caller frees) + *outsz on success; the
 * caller must free *out iff *out != image. Returns -1 to signal "load verbatim".
 * `want_sm` is the running device SM (e.g. 86). */
int atomize_image_to_cubin(const void* image, int want_sm, void** out, size_t* outsz) {
    const unsigned char* b = image;
    uint32_t m; memcpy(&m, b, 4);

    if (getenv("LITHOS_DIAG")) {
        const void* real = (m == FATBIN_WRAPPER_MAGIC) ? *(const void* const*)(b + FATBIN_WRAPPER_DATA_OFF) : image;
        uint32_t rm; memcpy(&rm, real, 4);
        fprintf(stderr, "[diag] image magic=%08x (real=%08x %s)\n", m, rm,
                rm == FATBIN_MAGIC ? "fatbin" : (memcmp(real, "\177ELF", 4) == 0 ? "elf" : "other"));
    }

    if (m == FATBIN_WRAPPER_MAGIC) {                  /* runtime wrapper -> real fatbin */
        const void* real = *(const void* const*)(b + FATBIN_WRAPPER_DATA_OFF);
        if (real) return atomize_image_to_cubin(real, want_sm, out, outsz);
        return -1;
    }
    if (memcmp(b, ELFMAG, SELFMAG) == 0) {           /* already a cubin */
        *out = (void*)image; *outsz = 0; return 0;
    }
    if (m == FATBIN_MAGIC) {
        void* cub = NULL; size_t cubsz = 0;
        const unsigned char* ptx = NULL; size_t ptxsz = 0;
        int rc = fatbin_extract_elf(b, want_sm, &cub, &cubsz, &ptx, &ptxsz);
        if (rc == 1) { *out = cub; *outsz = cubsz; return 0; }
        if (rc == 0 && ptx) {                        /* no ELF for us: JIT the PTX */
            int jr = ptx_to_cubin((const char*)ptx, ptxsz, out, outsz);
            free((void*)ptx);
            if (jr == 0) return 0;
        }
        return -1;
    }
    /* Heuristic: raw PTX is text beginning with a comment or a directive. */
    if (b[0] == '/' || b[0] == '\n' || memmem(b, 64, ".version", 8)) {
        size_t n = strlen((const char*)b);
        if (ptx_to_cubin((const char*)b, n, out, outsz) == 0) return 0;
    }
    return -1;
}
