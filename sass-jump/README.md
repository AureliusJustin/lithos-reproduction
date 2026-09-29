# SASS transfer reverse-engineering (Prelude → original)

The Kernel Atomizer redirects a launch to a **Prelude** (via the QMD program
address), which must then transfer control into the original kernel for in-range
blocks. ptxas lowers the natural `((fn)e)()` to a `CALL`, and the original ends
in `EXIT` (not `RET`), which faults. These harnesses reverse-engineer, at the
SASS level, whether any control transfer can do this cleanly. See
[../docs/TECHNICAL_REPORT.md](../docs/TECHNICAL_REPORT.md) §5.1 for the summary.

Build each as: `gcc <file>.c ../src/qmd.c -o /tmp/x -I../src -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lnvrtc && /tmp/x`

## What each harness proves

| harness | fact established |
|---------|------------------|
| `callabs.c` | `CALL.REL.NOINC R2` uses R2 as an **absolute** target; the branchless Prelude CALLs the absolute original entry — but faults on the original's `EXIT`. |
| `callspin.c` | Making the original spin instead of exit **hangs** → the CALL *lands and runs the original correctly*; the fault is purely the `EXIT`. |
| `trivial.c` | Even an **empty** original (`{}`) faults on `EXIT` after a CALL → not a target/args bug; it's the transfer's return/convergence state. |
| `xfunc.c` | A **`BRX`** (register-indirect branch) with proper metadata jumps **cross-function** and runs the target (`0xFACE`) — the register-jump mechanism works, but a `BSSY` convergence barrier makes the `EXIT` fault. |
| `inject.c` | ELF-surgery injector that adds the `EIATTR_INDIRECT_BRANCH_TARGETS` (`0x34`) attribute a bare `BRX` needs (`nvdisasm` confirms). |
| `brava.c` | A plain **`BRA`** (unconditional jump) patched over the CALL gives **`sync=0`, no fault** for several landings → a *fault-free* transfer is achievable (BRA pushes no return, sets no barrier). |
| `braexact.c` | Tries a clean transfer with exact addressing: capture the load addresses, patch `CALL`→`BRA(orig_entry)`, reload at a (hoped-deterministic) VA and launch prelude → `BRA` → original. |
| `splice_test.c`, `atom_complex_test.c` | Prove the **prologue-splice / fall-through** atomizer (see *Resolved* below). |

Two further findings were established with harnesses that were not kept: backward `BRX` jumps (negative offset) land wrong, and patching a **loaded** module's code via `cuMemcpyHtoD` fails with `ILLEGAL_ADDRESS`, so a `BRA` immediate cannot be fixed up at runtime.

## ✅ Resolved (the splice makes the transfer moot)

These harnesses proved the Prelude→original *transfer* can't be closed by
byte-patching ptxas output. The **atomizer nonetheless works end-to-end** by
avoiding the transfer entirely: instead of a separate Prelude that jumps into the
original, we prepend the range-check into the app kernel's **own** `.text`
section so in-range blocks **fall through** into the original (no `CALL`/`BRX`,
clean `EXIT`). See `splice_test.c` / `atom_complex_test.c` here and
[`../src/atomize_splice.c`](../src/atomize_splice.c). The material below is the
record of the transfer dead-end.

## The conclusion (transfer dead-end)

Every control transfer to an `EXIT`-ending `__global__` either:
* sets up return/convergence state the `EXIT` violates (`CALL`, `BRX`), or
* is immediate-encoded and so can't target a **runtime, cross-module** address
  (`BRA` — clean, but can't be pointed at the app's kernel without patching
  loaded code, which the driver blocks).

A register-indirect jump with **no** convergence barrier would solve it, but that
instruction doesn't exist in the Ampere ISA (`BRX` always needs `BSSY`). The
original LithOS (Rust) presumably emits a transfer whose target is resolved at
**link/load time** (an immediate branch relocated by the driver's linker), which
isn't reproducible by byte-patching a ptxas-compiled Prelude for a target
captured at runtime. This is the paper's deferred "separate technical report."

Everything *up to* the transfer — QMD redirect, register/metadata handling, atom
partitioning, scheduling — works and is in the main library.
