/*
 * QMD/TMD hook — where LithOS actually controls which SMs a kernel may run on.
 *
 * WHAT A QMD IS. Every kernel launch is described to the GPU by a "Queue Meta
 * Data" block (QMD, a.k.a. TMD on older chips): a few hundred bytes of packed
 * fields holding the program address, grid/block dims, register count, shared-mem
 * size, and — crucially for us — an **SM-disable mask**. The driver builds this
 * block and uploads it to the hardware queue. There is no public API to set the
 * mask, so we intercept the QMD *just before upload* and write it ourselves.
 *
 * HOW WE INTERCEPT. The driver exposes an internal CUPTI-style callback table via
 * cuGetExportTable; subscribing to (domain 0xb, callback 0x1) invokes us with the
 * QMD pointer immediately before it is uploaded. Both the table UUID and the
 * subscribe/enable slot indices were reverse-engineered by libsmctrl (Bakita,
 * ../libsmctrl/libsmctrl.c) and are reused here unchanged.
 *
 * THE FIELD OFFSETS below are QMD-version specific and were found empirically
 * (see tests/probe_*.c and the technical report). They are the load-bearing magic
 * of this file, so each is named and documented.
 *
 * Note (CUDA graphs): this callback fires ONCE per graph exec — at its first
 * launch — because replays re-run a pre-compiled command buffer that bypasses the
 * driver's per-node QMD path. So a graph's SM mask is baked at first launch and
 * can only be changed by re-instantiating (see graphsched.c).
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "qmd.h"

/* The internal callback table's UUID, and the (domain, callback-id) pair that
 * fires just before a QMD is uploaded — all as extracted by libsmctrl. */
static const CUuuid callback_funcs_id = {{0x2c, (char)0x8e, 0x0a, (char)0xd8, 0x07, 0x10, (char)0xab, 0x4e, (char)0x90, (char)0xdd, 0x54, 0x71, (char)0x9f, (char)0xe5, (char)0xf7, 0x4b}};
#define QMD_DOMAIN     0xb
#define QMD_PRE_UPLOAD 0x1

/* ---- QMD field offsets (bytes from the start of the QMD) -------------------
 *
 * The callback's in_params is a small struct: [size:4][...]; the QMD pointer is
 * the 5th pointer-sized slot, so we require size >= 5 pointers and read slot 4. */
#define CB_PARAMS_MIN_SLOTS 5
#define CB_PARAMS_QMD_SLOT  4

/* QMD version byte. <0x40 = Kepler2..Ampere/Ada (QMDV03_00 etc); >=0x40 = Hopper,
 * which moved the mask fields and needs an extra enable bit. */
#define QMD_VERSION_OFF 72
#define QMD_VER_HOPPER  0x40

/* SM-disable mask, Ampere/Ada (a 64-bit mask split across two 32-bit words; a SET
 * bit DISABLES that TPC). Verified on GA102/GA100. */
#define QMD_MASK_LO_OFF 84
#define QMD_MASK_HI_OFF 88

/* SM-disable mask, Hopper: moved, plus two "extended" words that must be set to
 * all-ones and a top bit in word 0 that enables the masking feature at all. */
#define QMD_H_MASK_LO_OFF 304
#define QMD_H_MASK_HI_OFF 308
#define QMD_H_EXT_LO_OFF  312
#define QMD_H_EXT_HI_OFF  316
#define QMD_H_ENABLE_BIT  0x80000000u

/* Per-thread register allocation (Ampere/Ada QMDV03_00), byte 81. The atomizer
 * raises this when a spliced kernel needs more registers than it declared. */
#define QMD_REG_COUNT_OFF 81

/* Program address (the kernel entry PC the launch jumps to). Discovered
 * empirically; 0 until qmd_probe() locates it on this driver/arch. */
int g_qmd_prog_addr_off = 0;

/* Valid TPC count (set by the scheduler once detected); only used to keep
 * LITHOS_LOG_MASK output from listing nonexistent TPCs. */
static int g_qmd_ntpc = 64;

/* ---- Per-thread arming state ----------------------------------------------
 * The callback runs on the launching thread, so all "what should the next launch
 * do" state is thread-local:
 *   t_next_mask    one-shot SM-disable mask, consumed by the next upload
 *   t_sticky_mask  mask applied to EVERY upload until cleared (used when one
 *                  subgraph launch contains several kernel nodes)
 *   t_sticky       whether t_sticky_mask is armed
 *   t_capture      arm capture of the next QMD (reverse-engineering probes)
 *   t_captured     the captured program address
 *   t_captured_qmd a raw copy of the captured QMD bytes
 *   t_atomize      legacy: redirect the program address to t_prelude
 *   t_prelude      legacy: Prelude entry VA to redirect to
 *   t_prelude_regs legacy: register count the Prelude needs
 *   t_dump         dump QMD bytes to stderr (probing) */
static __thread uint64_t t_next_mask    = 0;
static __thread uint64_t t_sticky_mask  = 0;
static __thread int      t_sticky       = 0;
static __thread int      t_capture      = 0;
static __thread uint64_t t_captured     = 0;
static __thread int      t_atomize      = 0;
static __thread uint64_t t_prelude      = 0;
static __thread int      t_prelude_regs = 0;
static __thread int      t_dump         = 0;
static __thread uint8_t  t_captured_qmd[256];

static int setup_done = 0;

/* Called by the driver immediately before it uploads a QMD to the hardware queue.
 * `in_params` is an opaque parameter block whose 5th pointer slot is the QMD. */
static void control_callback(void* ukwn, int domain, int cbid, const void* in_params) {
    (void)ukwn; (void)domain; (void)cbid;
    /* First word is the block's size; make sure the QMD slot is actually present. */
    if (*(uint32_t*)in_params < CB_PARAMS_MIN_SLOTS * sizeof(void*)) return;
    void* tmd = *((void**)in_params + CB_PARAMS_QMD_SLOT);
    if (!tmd) return;

    if (getenv("LITHOS_LOG_CB")) {   /* count EVERY callback, mask armed or not */
        static int cbn = 0;
        fprintf(stderr, "[cb] pre-upload callback #%d (tmd=%p)\n", ++cbn, tmd);
    }

    /* Pick the mask field locations for this QMD version. lower/upper point at the
     * 64-bit SM-disable mask; on Hopper we must also fill the extended words and
     * set the enable bit, or the mask is ignored. */
    uint8_t tmd_ver = *(uint8_t*)((char*)tmd + QMD_VERSION_OFF);
    uint32_t *lower_ptr = NULL, *upper_ptr = NULL, *ext_lo = NULL, *ext_hi = NULL;
    if (tmd_ver >= QMD_VER_HOPPER) {              /* Hopper (untested here) */
        lower_ptr = (uint32_t*)((char*)tmd + QMD_H_MASK_LO_OFF);
        upper_ptr = (uint32_t*)((char*)tmd + QMD_H_MASK_HI_OFF);
        ext_lo    = (uint32_t*)((char*)tmd + QMD_H_EXT_LO_OFF);
        ext_hi    = (uint32_t*)((char*)tmd + QMD_H_EXT_HI_OFF);
        *ext_lo = -1; *ext_hi = -1;
        *(uint32_t*)tmd |= QMD_H_ENABLE_BIT;
    } else if (tmd_ver >= 0x16) {                 /* Kepler2 .. Ampere/Ada */
        lower_ptr = (uint32_t*)((char*)tmd + QMD_MASK_LO_OFF);
        upper_ptr = (uint32_t*)((char*)tmd + QMD_MASK_HI_OFF);
    }

    /* --- TPC mask (scheduler / stealing / graph subgraphs) ---
       t_next_mask is one-shot (consumed per launch: normal eager scheduling).
       t_sticky is a mask that applies to EVERY upload until cleared: used when
       launching a subgraph whose (possibly several) kernel nodes must all land on
       the same scheduler-assigned TPC set. */
    if (lower_ptr) {
        uint64_t mm = 0; int have = 0;
        if (t_next_mask)   { mm = t_next_mask; t_next_mask = 0; have = 1; }
        else if (t_sticky) { mm = t_sticky_mask;                have = 1; }
        if (have) {
            *lower_ptr = (uint32_t)mm;
            *upper_ptr = (uint32_t)(mm >> 32);
            if (getenv("LITHOS_LOG_MASK")) {   /* observe the applied TPC allocation */
                char en[256]; int p = 0;
                for (int t = 0; t < g_qmd_ntpc && p < 240; t++)
                    if (!((mm >> t) & 1)) p += snprintf(en+p, sizeof(en)-p, "%d,", t);
                fprintf(stderr, "[mask]%s disable=0x%016llx enabled_TPCs=[%s]\n",
                        t_sticky ? "(sticky)" : "", (unsigned long long)mm, en);
            }
        }
    }

    /* --- Dump for reverse-engineering --- */
    if (t_dump) {
        t_dump = 0;
        uint32_t* w = (uint32_t*)tmd;
        fprintf(stderr, "[qmd] TMD ver=0x%02x dump (64 dwords):\n", tmd_ver);
        for (int i = 0; i < 64; i++) {
            if (i % 4 == 0) fprintf(stderr, "[qmd] +%3d: ", i * 4);
            fprintf(stderr, "%08x ", w[i]);
            if (i % 4 == 3) fprintf(stderr, "\n");
        }
    }

    /* --- Diagnostic: perturb a candidate program-address field on every launch
     * to identify the live program counter. LITHOS_CORRUPT_OFF=byte offset. --- */
    {
        const char* co = getenv("LITHOS_CORRUPT_OFF");
        if (co) {
            int off = atoi(co);
            *(uint32_t*)((char*)tmd + off) ^= 0x100;  /* nudge by 0x100 bytes */
        }
    }

    /* --- Program capture / patch (Atomizer) --- */
    if (g_qmd_prog_addr_off) {
        uint32_t* pa = (uint32_t*)((char*)tmd + g_qmd_prog_addr_off);
        /* On capture, save both the entry VA and the whole QMD so its code-
         * identity fields can later be transplanted into an original's QMD. */
        if (t_capture) {
            t_captured = (uint64_t)pa[0] | ((uint64_t)pa[1] << 32);
            memcpy(t_captured_qmd, tmd, 256);
            t_capture = 0;
        }
        /* On atomize, overwrite the original's program address with the Prelude's
         * (the GPU then runs the Prelude while keeping the original's launch
         * config, constant bank, and shared memory), and raise the register
         * allocation to cover the Prelude. The program address is written in
         * both encodings the driver uses: the full 64-bit pointer at +192 and
         * (address >> 8) at +32, so they stay consistent across a module. */
        if (t_atomize && t_prelude) {
            uint32_t* pa_s = (uint32_t*)((char*)tmd + 32);
            pa[0]   = (uint32_t)t_prelude;
            pa[1]   = (uint32_t)(t_prelude >> 32);
            pa_s[0] = (uint32_t)(t_prelude >> 8);
            uint8_t* regc = (uint8_t*)((char*)tmd + QMD_REG_COUNT_OFF);
            if (t_prelude_regs > *regc) *regc = (uint8_t)t_prelude_regs;
            t_atomize = 0; t_prelude = 0; t_prelude_regs = 0;
        }
    }
}

void qmd_init(void) {
    if (__atomic_test_and_set(&setup_done, __ATOMIC_SEQ_CST))
        return;
    int (*subscribe)(uint32_t*, void(*)(void*,int,int,const void*), void*);
    int (*enable)(uint32_t, uint32_t, int, int);
    uintptr_t* tbl_base;
    uint32_t hndl;
    cuGetExportTable((const void**)&tbl_base, &callback_funcs_id);
    subscribe = (typeof(subscribe))*(tbl_base + 3);
    enable    = (typeof(enable))*(tbl_base + 6);
    if (subscribe(&hndl, control_callback, NULL))
        fprintf(stderr, "[qmd] failed to subscribe callback\n");
    if (enable(1, hndl, QMD_DOMAIN, QMD_PRE_UPLOAD))
        fprintf(stderr, "[qmd] failed to enable callback\n");

    /* Program-address offset per architecture. Ampere/Ada (GA/AD) place the
     * lower/upper program address at QMD dword 10/11 (byte 40/44). Validated
     * empirically on this GPU by qmd_probe() in the atomizer. Overridable. */
    const char* off = getenv("LITHOS_QMD_PROG_OFF");
    g_qmd_prog_addr_off = off ? atoi(off) : 192;  /* Ampere/Ada QMDV03_00 */
}

void qmd_set_next_mask(uint64_t m) { t_next_mask = m; }
void qmd_set_num_tpcs(int n) { if (n > 0 && n <= 64) g_qmd_ntpc = n; }
void qmd_set_sticky_mask(uint64_t m) { t_sticky_mask = m; t_sticky = 1; }
void qmd_clear_sticky_mask(void) { t_sticky = 0; t_sticky_mask = 0; }
void qmd_arm_capture(void) { t_capture = 1; t_captured = 0; }
uint64_t qmd_get_captured(void) { return t_captured; }
void qmd_get_captured_qmd(void* out256) { memcpy(out256, t_captured_qmd, 256); }
void qmd_arm_atomize(uint64_t prelude_entry, int prelude_regs) {
    t_atomize = 1; t_prelude = prelude_entry; t_prelude_regs = prelude_regs;
}
void qmd_dump_next(void) { t_dump = 1; }
