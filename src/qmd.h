/*
 * QMD/TMD hook (Section 6, "TPCs and Atomization").
 *
 * LithOS intercepts the QMD (Queue MetaData) immediately before the driver
 * uploads it to the GPU, using the same undocumented CUDA debug callback that
 * libsmctrl co-opts (domain 0xb, cbid 0x1). Inside the callback LithOS can:
 *   - apply a per-launch TPC mask (TPC Scheduler / stealing),
 *   - read the kernel's program address (its SASS entry VA),
 *   - patch the program address to the Prelude (Kernel Atomizer).
 *
 * All arming is thread-local so concurrent dispatcher threads don't interfere.
 */
#ifndef LITHOS_QMD_H
#define LITHOS_QMD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void qmd_init(void);   /* register the pre-upload callback (idempotent) */

/* Per-launch TPC mask (set bit = TPC disabled), applied to the next launch on
 * this thread. Mirrors libsmctrl_set_next_mask. */
void qmd_set_next_mask(uint64_t disable_mask);

/* Tell the QMD layer the valid TPC count (for clean LITHOS_LOG_MASK output). */
void qmd_set_num_tpcs(int n);

/* Arm capture of the next launch. After the launch call returns, read the
 * program-address VA with qmd_get_captured() and the full QMD (256 bytes) with
 * qmd_get_captured_qmd(). */
void qmd_arm_capture(void);
uint64_t qmd_get_captured(void);
void qmd_get_captured_qmd(void* out256);

/* Arm atomization of the next launch: the callback overwrites the QMD program
 * address with `prelude_entry` (a device VA) so the GPU runs the Prelude while
 * retaining the original's launch configuration, constant bank, and shared
 * memory. The register-count allocation is raised to at least `prelude_regs`.
 * The Prelude must already have its AtomMetadata (block range + relative jump
 * targets) set up. */
void qmd_arm_atomize(uint64_t prelude_entry, int prelude_regs);

/* Dump the next launch's raw QMD to stderr (for reverse-engineering). */
void qmd_dump_next(void);

/* Byte offset of the program-address field within the QMD for the running
 * architecture, discovered/validated at init. 0 if unknown. */
extern int g_qmd_prog_addr_off;

#ifdef __cplusplus
}
#endif

#endif /* LITHOS_QMD_H */
