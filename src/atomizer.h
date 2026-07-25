/*
 * Kernel Atomizer (Section 5.4).
 *
 * Splits a kernel's grid into "atoms" -- disjoint contiguous ranges of thread
 * blocks -- transparently, without source or PTX. Each atom is a full-grid
 * launch of a Prelude kernel (Algorithm 1) that early-exits blocks outside the
 * atom's [lo, hi) range and otherwise branches into the original kernel.
 */
#ifndef LITHOS_ATOMIZER_H
#define LITHOS_ATOMIZER_H

#include <cuda.h>
#include "lithos.h"

void atomizer_init(void);

/* Module-load interception. Given a cubin image (as passed to cuModuleLoadData),
 * splice the range-check prologue into every kernel. On success *out is a
 * malloc'd spliced image (caller loads then frees) and *atomized=1; otherwise
 * *out==image and *atomized=0 (fatbin/PTX/failure -> load verbatim). */
/* Splice the range-check into an image's kernels. On success *out is a malloc'd
 * spliced cubin, *atomized=1, and *names is an opaque token (the spliced kernel
 * names) to hand to atomizer_register_*; otherwise *out==image, *atomized=0. */
int  atomizer_intercept_cubin(const void* image, void** out, size_t* outsz,
                              int* atomized, void** names);
/* Record a loaded container + its spliced-name set. A function/kernel resolved
 * from the container is gated only if its NAME was actually spliced (so a module
 * with one un-spliceable kernel still atomizes the rest). */
void atomizer_register_module(CUmodule m, void* names);
void atomizer_note_get_function(CUfunction f, CUmodule m, const char* name);

/* CUDA 12 Library API (the runtime's path). A CUlibrary carries the same name
 * set; modules/kernels/functions derived from it inherit gating by name. */
void atomizer_register_library(CUlibrary lib, void* names);
void atomizer_note_library_module(CUmodule m, CUlibrary lib);
void atomizer_note_get_kernel(CUkernel k, CUlibrary lib, const char* name);
void atomizer_note_kernel_function(CUfunction f, CUkernel k);

/* Ex / cooperative launch dispatch (mirror of atomizer_dispatch). Ex splits the
 * grid into atoms and replays cuLaunchKernelEx; cooperative kernels are never
 * split (grid-wide sync needs every block) but still get full-range metadata so
 * the spliced range-check passes all blocks. Return the number of launches. */
int atomizer_dispatch_ex(const CUlaunchConfig* cfg, CUfunction f, void** params, void** extra);
int atomizer_dispatch_coop(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                           unsigned bx, unsigned by, unsigned bz, unsigned shmem,
                           CUstream stream, void** params);

/* Decide how many atoms `k` splits into (given the app's TPC quota) and
 * dispatch each atom to the GPU. Returns the number of atoms launched.
 *
 * The QMD program-address patch that redirects execution to the Prelude is
 * installed via the libsmctrl-style pre-upload debug callback (qmd.c); this
 * function drives the per-atom launches and metadata updates. */
int atomizer_dispatch(LithosKernel* k, int quota_tpcs);
void atomizer_set_ex_pred(double us);   /* predicted us for the next Ex dispatch (§5.7) */
void atomizer_set_capture_hint(int capturing); /* -1 = unknown; avoids a duplicate cuStreamIsCapturing */

/* Compute the number of atoms from a predicted duration and atom_duration. */
int atomizer_num_atoms(const LithosKernel* k, double pred_us);

#endif /* LITHOS_ATOMIZER_H */
