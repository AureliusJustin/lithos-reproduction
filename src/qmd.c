/*
 * QMD/TMD hook implementation. The callback registration mirrors libsmctrl
 * (Bakita, ../libsmctrl/libsmctrl.c) -- proven on this driver -- and the body
 * is extended for the Kernel Atomizer (program-address capture/patch).
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "qmd.h"

/* Same callback identifiers libsmctrl extracted by tracing CUPTI. */
static const CUuuid callback_funcs_id = {{0x2c, (char)0x8e, 0x0a, (char)0xd8, 0x07, 0x10, (char)0xab, 0x4e, (char)0x90, (char)0xdd, 0x54, 0x71, (char)0x9f, (char)0xe5, (char)0xf7, 0x4b}};
#define QMD_DOMAIN 0xb
#define QMD_PRE_UPLOAD 0x1

/* Ampere/Ada QMD (TMD ver < 0x40): SM-disable mask is at TMD+84/+88 (libsmctrl).
 * The program-address field offset is discovered empirically (see qmd_probe). */
int g_qmd_prog_addr_off = 0;

/* valid TPC count (set by the scheduler once detected), for clean mask logging */
static int g_qmd_ntpc = 64;
/* Thread-local arming state */
static __thread uint64_t t_next_mask   = 0;
static __thread uint64_t t_sticky_mask = 0;   /* applied to every upload until cleared */
static __thread int      t_sticky      = 0;
static __thread int      t_capture     = 0;
static __thread uint64_t t_captured    = 0;
static __thread int      t_atomize     = 0;
static __thread uint64_t t_prelude     = 0;
static __thread int      t_prelude_regs = 0;
static __thread int      t_dump        = 0;
static __thread uint8_t  t_captured_qmd[256];

/* QMD byte offset of the per-thread register-count allocation (Ampere/Ada
 * QMDV03_00). Discovered empirically: register count is at byte 81. */
#define QMD_REG_COUNT_OFF 81

static int setup_done = 0;

static void control_callback(void* ukwn, int domain, int cbid, const void* in_params) {
    (void)ukwn; (void)domain; (void)cbid;
    if (*(uint32_t*)in_params < 5 * sizeof(void*))
        return;
    void* tmd = *((void**)in_params + 4);
    if (!tmd) return;
    if (getenv("LITHOS_LOG_CB")) {   /* count EVERY pre-upload callback (any mask or not) */
        static int cbn = 0;
        fprintf(stderr, "[cb] pre-upload callback #%d (tmd=%p)\n", ++cbn, tmd);
    }

    uint8_t tmd_ver = *(uint8_t*)((char*)tmd + 72);
    uint32_t *lower_ptr = NULL, *upper_ptr = NULL, *ext_lo = NULL, *ext_hi = NULL;
    if (tmd_ver >= 0x40) {              /* Hopper */
        lower_ptr = (uint32_t*)((char*)tmd + 304);
        upper_ptr = (uint32_t*)((char*)tmd + 308);
        ext_lo    = (uint32_t*)((char*)tmd + 312);
        ext_hi    = (uint32_t*)((char*)tmd + 316);
        *ext_lo = -1; *ext_hi = -1;
        *(uint32_t*)tmd |= 0x80000000;
    } else if (tmd_ver >= 0x16) {       /* Kepler2 .. Ampere/Ada */
        lower_ptr = (uint32_t*)((char*)tmd + 84);
        upper_ptr = (uint32_t*)((char*)tmd + 88);
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
