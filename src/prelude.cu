/*
 * Prelude kernel -- Algorithm 1 in the paper (Section 5.4).
 *
 *   kernel fn prelude(*args):
 *       let atom = AtomMetadataAddr
 *       let block_idx = z*gridDim.y*gridDim.x + y*gridDim.x + x
 *       if atom->lo <= block_idx < atom->hi:
 *           atom->kernel_entrypoint(*args)
 *
 * How it runs transparently: LithOS patches the QMD program address so the GPU
 * begins executing THIS kernel instead of the original, while retaining the
 * original kernel's resources AND its parameter constant-bank. The Prelude
 * therefore takes no parameters of its own; it reads the atom range and the
 * original entry point from a fixed device address (g_atom), and if the block
 * is in range it transfers control to the original entry. Because the original
 * kernel reads its arguments from the same constant bank (unchanged) and ends
 * in EXIT, we never return -- so a tail call into the entry is correct.
 *
 * The transfer of control is emitted as an absolute indirect call/branch in
 * SASS (CALL.ABS/BRX). The original entry re-initialises its own stack pointer
 * and reads params from c[0x0], so no register setup is required from us.
 */
#include <stdint.h>

struct AtomMetadata {
    uint32_t block_idx_lo;
    uint32_t block_idx_hi;
    uint32_t generation;
    uint32_t _pad;
    uint64_t kernel_entrypoint;
};

/* Fixed device address the Prelude reads its metadata from. LithOS updates this
 * (via cuModuleGetGlobal + cuMemcpyHtoD) before each atom launch. */
extern "C" __device__ AtomMetadata g_lithos_atom = {0, 0, 0, 0, 0};

/* Signature-agnostic entry: params are already in the constant bank. */
typedef void (*kernel_fn_t)();

extern "C" __global__ void lithos_prelude() {
    uint64_t block_idx =
          (uint64_t)blockIdx.z * gridDim.y * gridDim.x
        + (uint64_t)blockIdx.y * gridDim.x
        + blockIdx.x;

    AtomMetadata a = g_lithos_atom;
    if (block_idx < a.block_idx_lo || block_idx >= a.block_idx_hi)
        return;                                   /* early-exit out-of-range blocks */

    /* Transfer control to the original kernel entry. A plain C++ call through
     * a function pointer is lowered by ptxas to a PC-RELATIVE call (CALL.REL),
     * which is wrong for a cross-module absolute address. We emit an indirect
     * PTX call through a register holding the absolute VA, which lowers to
     * CALL.ABS. The original kernel reads its args from the (unchanged) param
     * constant bank and ends in EXIT, so control never returns here. */
    uint64_t entry = a.kernel_entrypoint;
    asm volatile(
        "{\n\t"
        ".reg .b64 %%fp;\n\t"
        "mov.b64 %%fp, %0;\n\t"
        "cp%=: .callprototype _ ();\n\t"
        "call %%fp, cp%=;\n\t"
        "}\n\t"
        :: "l"(entry) : "memory");
    (void)kernel_fn_t(0);
}
