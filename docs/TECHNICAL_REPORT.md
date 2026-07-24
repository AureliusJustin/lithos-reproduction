# LithOS Reproduction — Technical Report

A from-scratch reproduction of the core mechanisms of **LithOS** (SOSP), a GPU
operating system that interposes at the CUDA Driver API to (a) schedule ML
workloads at the granularity of individual TPCs and (b) transparently split
kernels into thread-block "atoms." The original prototype is ~5000 lines of Rust;
this reproduction is ~2,000 lines of C/C++/CUDA, reusing the QMD/TPC-masking
reverse-engineering from [`../libsmctrl`](../../libsmctrl) (Bakita) and building
directly with the system `nvcc`.

**Verified hardware/software:** NVIDIA RTX A6000 (GA102, `sm_86`, QMD version
`QMDV03_00`/`0x30`), CUDA 12.8, driver 570.195.03, Ubuntu (kernel 6.8), on a
bare-metal host.

---

## Table of contents

1. [What was reproduced](#1-what-was-reproduced)
2. [System architecture](#2-system-architecture)
3. [Module reference](#3-module-reference)
4. [End-to-end control & data flow](#4-end-to-end-control--data-flow)
5. [The Kernel Atomizer in depth](#5-the-kernel-atomizer-in-depth)
6. [Framework interception (the CUDA-runtime path)](#6-framework-interception-the-cuda-runtime-path)
7. [Reverse-engineering findings](#7-reverse-engineering-findings)
8. [Validation & results](#8-validation--results)
9. [Limitations & version-sensitivity](#9-limitations--version-sensitivity)

---

## 1. What was reproduced

LithOS has three pillars; all three work end-to-end here:

| Pillar | Paper section | Status |
|--------|---------------|--------|
| **LibLithOS** — transparent CUDA Driver-API interposition | §5.2 / §6 | ✅ two deployment paths |
| **TPC Scheduler** — per-stream compute quotas, TPC stealing, outstanding-work tracking | §5.3 | ✅ verified SM confinement |
| **Kernel Atomizer** — transparently split a kernel's grid into thread-block atoms | §5.4 / §6 | ✅ end-to-end incl. CUDA graphs |

It runs unmodified **PyTorch, TensorFlow, JAX, TensorRT, cuDNN, and vLLM** with
100% kernel-atomization coverage (see §8).

The **one deviation** from the paper is *how* the atomizer transfers control into
the original kernel. The paper redirects a launch to a separate *Prelude* kernel
(Algorithm 1) that jumps into the original; that jump is not reproducible by
byte-patching `ptxas` output (see §5). We instead **splice** the range-check
directly into each kernel's own machine code so in-range blocks *fall through* —
functionally identical, no control transfer.

---

## 2. System architecture

```
   application  (PyTorch / TensorFlow / JAX / TensorRT / cuDNN / vLLM / driver-API)
        │  cuInit, cuModuleLoad*, cuLibraryLoadData, cuLaunchKernel[Ex], cuGetProcAddress, ...
        ▼
┌──────────────────────────────── LibLithOS ─────────────────────────────────┐
│ interpose.c   the interposition surface: overrides a small subset of the    │
│               Driver API, forwards the rest; two entry paths (PLT + the      │
│               runtime's cuGetProcAddress resolver)                           │
│                                                                              │
│ sched.c       TPC Scheduler: per-stream launch queues, compute quotas →      │
│               TPC masks, TPC stealing, outstanding-work tracking             │
│                                                                              │
│ atomizer.c    Kernel Atomizer: module-load splice, per-launch atom split,    │
│ atomize_splice.c   the ELF/SASS surgery that injects the range-check         │
│ fatbin.c           fatbin/PTX → cubin unwrapping (LZ4/Zstd, JIT)             │
│                                                                              │
│ graphsched.c  CUDA-graph subgraph scheduling: partition a graph into K       │
│               subgraphs, per-subgraph TPC allocation, runtime re-instantiate │
│                                                                              │
│ qmd.c         QMD pre-upload hook (libsmctrl-style): applies the TPC mask    │
│               (one-shot or sticky) to every launch, below the launch API     │
│                                                                              │
│ real.c        resolves the genuine driver entry points                       │
│ config.c      environment-variable configuration                             │
│ wrapper.c     libcuda.so.1 wrapper glue (framework path)                      │
└───────────────────────────────┬────────────────────────────────────────────┘
                                 ▼  forwards everything else
                        real libcuda.so.1  (NVIDIA driver)
```

### Two deployment paths

* **Driver-API apps** — `LD_PRELOAD=build/liblithos_full.so ./app`. The app's
  `cuLaunchKernel` etc. bind through the PLT to our exported symbols.
* **CUDA-runtime apps** (all the ML frameworks) — deploy the `libcuda.so.1`
  **wrapper** via `LD_LIBRARY_PATH=build`. The runtime `dlopen`s `libcuda.so.1`
  *by name*, so `LD_PRELOAD` is invisible to it; the wrapper's `DT_NEEDED` is
  patched to point at the real `libcuda.so` so forwarded calls reach the driver
  (§6). This is why frameworks use the wrapper, not `LD_PRELOAD`.

Both builds share the same object files; only `wrapper.c` and a linker tweak
differ.

---

## 3. Module reference

### `interpose.c` (353 LOC) — the interposition surface

Defines LithOS's version of each overridden Driver-API function and forwards the
rest. Overridden calls fall into four groups:

* **Lifecycle / streams:** `cuInit`, `cuStreamCreate[WithPriority]`,
  `cuStreamDestroy`, `cuStreamSynchronize`, `cuCtxSynchronize`. Stream creation
  registers a per-stream launch queue with the scheduler.
* **Module loading:** `cuModuleLoad`, `cuModuleLoadData`, `cuModuleLoadDataEx`,
  `cuModuleLoadFatBinary`, `cuModuleGetFunction`, and the CUDA-12 **Library API**
  `cuLibraryLoadData`, `cuLibraryLoadFromFile`, `cuLibraryGetKernel`,
  `cuLibraryGetModule`, `cuKernelGetFunction`. These drive the atomizer's splice
  and function-gating (§4, §5).
* **Launches:** `cuLaunchKernel[_ptsz]`, `cuLaunchKernelEx[_ptsz]`,
  `cuLaunchCooperativeKernel[_ptsz]`. These flow into the scheduler → atomizer.
* **Resolvers:** `cuGetProcAddress[_v2]` and `cuGetExportTable`. The
  `cuGetProcAddress` override is where the CUDA runtime discovers our wrappers;
  critically it **returns itself** for `cuGetProcAddress` (§6).

An `overrides[]` table maps symbol name → our function pointer; the
`cuGetProcAddress` override consults it so the runtime is handed our wrapper for
each interposed symbol and the real pointer for everything else.

### `real.c` / `real.h` (90 LOC) — real-driver resolution

Populates a `g_real` struct of genuine driver function pointers, resolved (in
order) via `dlsym(RTLD_NEXT)`, an explicit `dlopen` of the versioned
`libcuda.so.1`, and finally the driver's own `cuGetProcAddress`. Everything
LibLithOS forwards goes through `g_real`.

### `config.c` / `lithos.h` (44 LOC) — configuration

`g_lithos_cfg` is initialized once from environment variables (`LITHOS_*`): atom
duration, min-blocks-to-atomize, enable flags, quota, stealing, verbosity. Paper
defaults (atom duration 250–500 µs, 100 µs outstanding limit) are the defaults.

### `sched.c` (275 LOC) — TPC Scheduler (§5.3)

Per-stream `StreamState` (quota, assigned TPC range, launch/atom counters, idle
timer, outstanding count). Key pieces:

* **Launch queues** — one per stream, created on `cuStreamCreate`; the default
  stream is registered lazily on first launch so a global quota still attaches.
* **Compute quotas** — `LITHOS_QUOTA=N` guarantees a stream `N` TPCs. Enforced by
  computing a **disable mask** (`compute_disable_mask`) and handing it to the QMD
  hook (`qmd_set_next_mask`) for the upcoming launch.
* **TPC stealing** — idle streams' TPCs are lent to a stream with runnable work
  by OR-ing their ranges into the active mask.
* **Outstanding-work tracking** — per-stream in-flight counters, cleared on sync.
* **Dispatch** — `lithos_submit_launch[_ex]` does the quota/mask bookkeeping then
  calls the atomizer's dispatch. `presubmit`/`postsubmit` factor the common path.

Verified: `LITHOS_QUOTA=4` confines a kernel to 8 SMs (4 TPCs × 2 SMs/TPC),
`LITHOS_QUOTA=8` → 16 SMs, on both driver-API and runtime (framework) launches.

### `qmd.c` (146 LOC) — the QMD pre-upload hook

The scheduler needs to modify a launch's TPC mask *below* the launch API (so it
works no matter how the app launches). It uses the same mechanism as libsmctrl:

* `cuGetExportTable(&tbl, callback_funcs_id)` fetches an internal driver table
  (UUID `2c8e0ad8…`); `subscribe`/`enable` live at table offsets 3 and 6.
* We `subscribe` a `control_callback` to domain `0xb`, cbid `0x1` (QMD
  **pre-upload**). It fires for **every** QMD the driver is about to upload —
  every launch, regardless of API path (public, Ex, or the runtime's internal
  table).
* In the callback we locate the TMD/QMD (`*((void**)in_params + 4)`), read the
  TMD version at byte 72, and write the SM-disable mask at bytes **84/88**
  (Ampere/Ada; Hopper uses 304–316). This is what enforces quotas/stealing.
* The file also contains the legacy program-address **capture/patch** used by the
  original QMD-redirect atomizer and the `sass-jump/` harnesses (program address
  at byte **192**, register count at byte **81**). The current splice-based
  atomizer does **not** use these — it only uses `qmd_set_next_mask` — but they
  remain for the scheduler and the reverse-engineering record.

### `atomizer.c` (443 LOC) — Kernel Atomizer orchestration

The brain of the atomizer. Responsibilities:

* **Prologue + metadata setup** (`ensure_prologue`, lazy, needs a context):
  allocates one per-process device buffer `g_meta` (the `{lo,hi}` atom range,
  default `{0, 2³²}`), a serialization event `g_atom_ev`, and builds the
  range-check prologue for the running SM.
* **Module interception** (`atomizer_intercept_cubin`): unwrap the image to a raw
  cubin (fatbin.c), splice the prologue into every kernel (atomize_splice.c), and
  return the spliced cubin plus the set of successfully-spliced kernel *names*.
* **Function gating** (`atomizer_note_get_function` / `_get_kernel` /
  `_kernel_function` / `_library_module`): as the app resolves functions from an
  atomized container, classify each: **gated** (name was spliced → will be split)
  or **full-range** (from an atomized module but not in the spliced set → forced
  full-range so it can't read a stale range). Everything else is verbatim.
* **Dispatch** (`atomizer_dispatch`, `_ex`, `_coop`): decide the atom count, then
  for each atom write `[lo,hi]` to `g_meta` and relaunch the full grid; the range
  check skips out-of-range blocks. Cross-stream serialization and per-launch
  metadata via `cuMemsetD32Async` (§5) keep it correct under concurrency and CUDA
  graph capture.
* **Coverage stats** (`LITHOS_STATS` / `LITHOS_STATS_FILE`): counts split /
  single-atom / full-range / un-atomized launches and distinct kernels.

### `atomize_splice.c` (197 LOC) — the SASS/ELF surgery

The mechanical core. Two entry points:

* `atomize_build_prologue(sm, meta_va)`: NVRTC-compiles a tiny probe kernel
  (`if (b < meta->lo || b >= meta->hi) return; <marker>;`), extracts the
  range-check SASS up to a `0xDEADBEEF` marker, pads it to a 128-byte multiple
  with `NOP`s, and records its register requirement. `b` is the global block
  index from `blockIdx`/`gridDim`; `meta` is read from a literal absolute address.
* `atomize_splice_cubin(cubin, prologue)`: for each `.text.<fn>` section, inserts
  the prologue bytes at the front and fixes up **section headers, program headers
  (the driver loads via these), the function symbol size, the `.nv.info`
  register count, and every instruction-offset attribute** (`EXIT_INSTR_OFFSETS`
  and the special `INDIRECT_BRANCH_TARGETS`). Kernels that can't be spliced (e.g.
  with `.text` relocations) are skipped and stay runnable verbatim; only the
  spliced names are returned for gating.

### `fatbin.c` (170 LOC) — image unwrapping

`atomize_image_to_cubin(image, want_sm)` turns any module image into a raw cubin:

* **`__fatBinC_Wrapper_t`** (magic `0x466243b1`) — the struct the CUDA runtime
  passes; follow its `data` pointer (offset 8) to the real fatbin.
* **Fatbin** (magic `0xBA55ED50`) — walk the entries, pick the ELF cubin matching
  the running SM, and **LZ4-decompress** it if flagged (NVIDIA's fatbin
  compression is plain LZ4 block format). A minimal LZ4 block decoder is included.
* **PTX** (or a PTX-only fatbin) — JIT to a cubin with the driver linker
  (`cuLinkCreate`/`AddData(CU_JIT_INPUT_PTX)`/`Complete`).
* **Raw ELF cubin** — used directly.

In every case the driver is later handed a bare (spliced) cubin — no fatbin
repackaging is needed, since `cuLibraryLoadData`/`cuModuleLoadData` accept a raw
cubin.

### `wrapper.c` (33 LOC) — libcuda.so.1 wrapper glue

Only in the wrapper build. Intercepts `dlopen("libcuda.so[.1]")` to hand back the
*real* driver (avoiding infinite recursion into our wrapper), and sets
`CUDA_DEVICE_MAX_CONNECTIONS=8` so MPS exposes enough hardware channels (per the
paper / libsmctrl). The Makefile also patches the wrapper's `DT_NEEDED` from
`libcuda.so.1` → `libcuda.so` so forwarded symbols resolve to the real driver.

---

## 4. End-to-end control & data flow

### 4.1 Module load → splice

```
app: cuLibraryLoadData(code)  ─▶ interpose.c
   atomizer_intercept_cubin(code)
     ├─ fatbin.c: unwrap __fatBinC_Wrapper_t → fatbin → LZ4-decode → raw cubin
     └─ atomize_splice.c: prepend range-check into every .text.<fn>, fix ELF
   g_real.cuLibraryLoadData(spliced_cubin)  → CUlibrary
   register (CUlibrary → set of spliced kernel names)
```

### 4.2 Function resolution → gating

```
app: cuLibraryGetModule(lib) → module (inherits lib's spliced-name set)
app: cuModuleGetFunction(module, "name") → CUfunction
   if name ∈ spliced set  → gate as ATOMIZABLE   (g_atom_funcs)
   else (from atomized module, name not spliced)  → FULL-RANGE   (g_full_funcs)
```

Gating by name (not just by module) means a module with one un-spliceable kernel
still atomizes the rest, and a spliced-but-unrecognized kernel is still handled
safely (full range).

### 4.3 Launch → atom dispatch

```
app: cuLaunchKernel[Ex](f, grid, ...)  ─▶ interpose.c ─▶ lithos_submit_launch[_ex]
   presubmit: quota → TPC disable mask → qmd_set_next_mask   (scheduler)
   atomizer_dispatch[_ex]:
     if f is ATOMIZABLE:
        n = decide_atoms(grid)                # ceil(pred_us / atom_duration)
        lock; wait(g_atom_ev)                 # cross-stream serialization
        for each atom [lo,hi):
           cuMemsetD32Async(g_meta+0, lo); cuMemsetD32Async(g_meta+4, hi)
           relaunch full grid                 # range check skips out-of-range blocks
        record(g_atom_ev); unlock
     elif f is FULL-RANGE: serialized write {0, grid}; relaunch once
     else: forward verbatim
```

Cooperative / cluster launches are never split (a grid-/cluster-wide barrier
needs every block live); they get a full-range write and one launch.

### 4.4 The QMD hook fires underneath all of this

Every one of those relaunches ends in a QMD upload, at which point `qmd.c`'s
callback applies the stream's TPC mask (set by `presubmit`). This is why the
scheduler works even for the runtime's internal launch path.

---

## 5. The Kernel Atomizer in depth

### 5.1 The paper's mechanism, and why we deviate

Algorithm 1 launches the *original* kernel but patches the QMD **program address**
so the GPU runs a small **Prelude** that range-checks the block and then
*transfers control into the original kernel entry*. We reproduced the redirect
(program address at QMD byte 192, register bump at byte 81, literal-address
metadata, branchless target select) — but the final **transfer** is the crux:

* `ptxas` lowers the Prelude's indirect tail call to `CALL`, which pushes a
  return PC. The original kernel ends in `EXIT`, not `RET`, so the unpopped
  call/convergence state faults (`INVALID_PC`) at warp retirement — even for an
  *empty* original.
* The correct lowering is a **jump**, which is *not expressible in PTX at all*
  (PTX has no computed branch), so no PTX-compiled language — Rust included — can
  emit it; only SASS can. LithOS gets it from Rust/LLVM guaranteed tail calls.
* We reverse-engineered the SASS jump (`BRX`) exhaustively (`sass-jump/`): a
  forward `BRX` with injected `EIATTR_INDIRECT_BRANCH_TARGETS` metadata *does*
  jump cross-function, but every register-indirect transfer needs a `BSSY`
  convergence barrier that the callee's `EXIT` then violates; the only
  convergence-free transfer (`BRA`) is immediate-encoded and can't target a
  runtime cross-module address (and loaded code can't be patched:
  `cuMemcpyHtoD → ILLEGAL_ADDRESS`). This is the paper's deferred "separate
  technical report."

### 5.2 The insight: prologue splice / fall-through

Instead of a *separate* Prelude that must jump into the original, **prepend the
range-check directly into the app kernel's own `.text` section** so in-range
blocks *fall through* into the original code:

```
  [ range check: if (b < lo || b >= hi) EXIT; ]   ← spliced prologue (predicated EXIT, no BSSY)
  [ original kernel SASS, unchanged            ]   ← falls through; its EXIT is clean
```

No `CALL`, no `BRX`, no convergence barrier — the wall disappears. It's
functionally identical to Algorithm 1 (a per-block gate) achieved with no control
transfer.

### 5.3 Splice mechanics

The prologue is NVRTC-built once per arch and inserted at the front of each
kernel's `.text.<fn>`. Because prepending shifts the code, the splicer fixes:

* `.text.<fn>` `sh_size += prolen`; later sections' `sh_offset` and `e_shoff`.
* **Program headers** — the `LOAD` segment containing the insertion grows
  (`p_filesz`/`p_memsz`); later segments and `e_phoff` shift. *(Missing this
  caused an early driver segfault — the driver loads via program headers, not
  section headers.)*
* The function **symbol** `st_size`.
* `.nv.info` **register count** (`max(orig, prologue_regs)`), and every
  instruction-offset attribute: `EXIT_INSTR_OFFSETS` (bump each), and
  `INDIRECT_BRANCH_TARGETS` (bump the BRX-site offset and each branch target, but
  not the flags/count fields).

The prologue is padded to a 128-byte multiple with `NOP`s so alignment is
preserved.

### 5.4 Metadata: `cuMemsetD32Async`, not a copy

The atom range lives in one device buffer `g_meta`. Each atom's `[lo,hi]` is
written with **two `cuMemsetD32Async` immediate writes** (to `g_meta+0/+4`) rather
than a host→device copy, for two reasons:

1. A tiny async HtoD copy is often executed **synchronously**, which is *illegal
   during CUDA-graph capture* and invalidates it.
2. A memset **bakes its value into the recorded graph node**, so a captured atom
   subgraph *replays* the correct range — no host staging, no stale-value hazard.

This makes the eager and graph-capture paths identical and removed an entire
ring-buffer/persistent-pool design.

### 5.5 Multi-stream correctness

One `g_meta` per process → an app's *concurrent streams* could interleave their
metadata writes (there is **no cross-process race** — each process has its own
buffer). Two safeguards:

* **Cross-stream serialization** via a CUDA event `g_atom_ev`: each atomized
  launch `cuStreamWaitEvent`s before writing and `cuEventRecord`s after — a no-op
  for single-stream apps (same-stream wait), serializing multi-stream. Skipped
  during graph capture (can't wait on an event recorded outside the capture).
* **Full-range for ungated-spliced kernels** — a spliced kernel that wasn't
  individually gated is launched with a serialized full range so it can never
  read another atom's stale `[lo,hi]` and drop blocks.

Together these took JAX/XLA (heavily multi-stream) from 2/10 to **10/10** correct.

### 5.6 CUDA graphs: two modes and the replay constraint

A hardware fact shapes everything here: the QMD **pre-upload callback fires exactly
once per graph exec** — at the first `cuGraphLaunch`. Replays re-run a compiled
command buffer that bypasses the driver's per-node path, so the SM/TPC mask (which
`qmd.c` writes *in* that callback) **cannot be changed on replay**; only
**re-instantiating** the exec re-fires it. Verified with an unconditional callback
counter (`LITHOS_LOG_CB`): 1 firing across instantiate + N replays; destroy +
re-instantiate → the callback fires again. Two modes follow:

1. **Atomize-in-graph (default).** During capture the atomizer records each
   kernel's `[set-range → relaunch]` atoms as graph nodes (§5.4). Correct and
   replay-stable (ranges bake into the memset immediates), but the schedule is
   **frozen**: atom ranges are baked, and per-atom TPC masks don't apply (the
   one-shot mask can't map onto N nodes). It's also **expensive on graphs** — even
   at n=1 it adds two metadata `memset` nodes per kernel (+120 % replay; see
   [BENCHMARKS](BENCHMARKS.md)).
2. **Partition-into-subgraphs (`LITHOS_GRAPH_SUBGRAPHS=K`,
   [`src/graphsched.c`](../src/graphsched.c)) — the paper's model.** Kernels are
   *not* atomized. At `cuGraphInstantiate` the graph is partitioned along a
   **topological cut** (Kahn sort of `cuGraphGetNodes`/`GetEdges`; each subgraph
   built by `cuGraphClone` + deleting non-chunk nodes). At `cuGraphLaunch` the
   single app launch is fanned into K sequential subgraph launches on the stream
   (order ⇒ dependencies), each preceded by the scheduler's TPC allocation applied
   as a **sticky mask** (`qmd_set_sticky_mask`, so *all* kernel nodes of a subgraph
   share it — the one-shot mask would only cover the first). The subgraph is the
   scheduling unit; launch-bypass is retained (K ≪ #kernels).
   - **Runtime reallocation.** Subgraph *templates* (`CUgraph`) are kept, and per
     `cuGraphLaunch` each subgraph whose desired allocation changed is
     **re-instantiated from its template** (a fresh exec re-bakes the new mask);
     unchanged subgraphs just replay. So the same app-level graph exec runs on a
     different TPC allocation each replay — cost ~7–14 µs per *changed* subgraph
     (see BENCHMARKS), paid only on change. `cuGraphExecUpdate` can't do this (the
     SM mask isn't a graph-API param), which is why re-instantiation is the path.

---

## 6. Framework interception (the CUDA-runtime path)

ML frameworks sit on the CUDA **Runtime** (`libcudart`), which reaches the driver
in ways that bypass naive interposition. Three discoveries were needed:

1. **`libcudart` `dlopen`s `libcuda.so.1` by name** → `LD_PRELOAD` is invisible.
   Solution: the `libcuda.so.1` **wrapper** (DT_NEEDED patched to the real
   `libcuda.so`), deployed via `LD_LIBRARY_PATH`.
2. **The runtime asks the driver for `cuGetProcAddress` once, then routes every
   other lookup through what it gets back.** If we return the *real* resolver, all
   launch/load calls bypass us. Fix: our `cuGetProcAddress` override **returns
   itself** for `cuGetProcAddress[_v2]`, keeping every subsequent lookup
   interposed.
3. **Loading + launching go through the CUDA-12 Library API** —
   `cuLibraryLoadData` (given a `__fatBinC_Wrapper_t`) → `cuLibraryGetModule` →
   `cuModuleGetFunction`, and `cuLaunchKernelEx`. We intercept that whole chain,
   follow the fatbin wrapper, and gate each launch.

(The runtime also pulls the QMD **callback table** via `cuGetExportTable` — the
same UUID `2c8e0ad8…` that `qmd.c` uses — which is why TPC masking already fires
for framework launches.)

Robustness for real kernels: per-kernel splice (skip the odd one), never split
cooperative/cluster launches, and the multi-stream safeguards of §5.5.

---

## 7. Reverse-engineering findings

* **QMD/TMD layout** (Ampere/Ada `QMDV03_00`, this A6000): TMD version at byte 72;
  SM-disable mask at bytes **84/88**; live program address (64-bit) at byte
  **192** (its `>>8` mirror at byte 32); register-count allocation at byte **81**
  (min 16). Grid dims at bytes 48/52/56. Perturbing byte 192 faults; perturbing
  byte 32 does not — that's how the live PC field was located.
* **`cuGetExportTable` callback table** (UUID `2c8e0ad8-0710-ab4e-90dd-54719fe5f74b`):
  `subscribe` at table index 3, `enable` at index 6; domain `0xb`, cbid `0x1` =
  QMD pre-upload. Fires for all launches.
* **Fatbin format**: top header magic `0xBA55ED50` (`ver`, `hdrSize=16`, `fatSize`);
  each entry `[kind:u16 (1=PTX,2=ELF)][headerSize:u32@+4][storedSize:u64@+8]
  [compressedSize:u64@+0x10][arch:u32@+0x1c][flags:u64@+0x28 (bit 0x2000 =
  compressed)][uncompressedSize:u64@+0x38]`; payload at `entry+headerSize`. The
  runtime passes a `__fatBinC_Wrapper_t` (magic `0x466243b1`, real fatbin at +8).
* **Fatbin compression changed across CUDA versions**: CUDA ≤12 uses **LZ4 block
  format** (entry flag `0x2000`, payload begins with an LZ4 token; a ~45-line
  block decoder recovers it byte-exact). **CUDA 13 switched to Zstandard** (flag
  `0x8000`, payload magic `28 b5 2f fd`), decompressed with `ZSTD_decompress`
  using the **exact** compressed-size field (`+0x10`), not the padded stored size
  (`+0x08`). `fatbin.c` handles both. (Container magics `0xBA55ED50`/`0x466243b1`
  and the sm_XX entry layout are unchanged; Blackwell sm_100/120 entries use a
  larger 0x70 header and are skipped as non-matching.)
* **SASS (sm_86)**: prologue uses only predicated `@P EXIT` (no `BSSY`); `NOP` =
  `18 79 …`; `CALL.REL.NOINC` uses its register as an *absolute* target; `BRX`
  needs `EIATTR_INDIRECT_BRANCH_TARGETS`; no register-indirect + convergence-free
  instruction exists (the transfer wall).

---

## 8. Validation & results

### Built-in suite (`make run_tests`, all pass)

| test | what it exercises |
|------|-------------------|
| `test_interpose_driver` | interposition via `LD_PRELOAD` (driver API) |
| `test_interpose` | interposition via the `libcuda.so.1` wrapper (runtime) |
| `test_atomize_cubin` × 3 | atomizer on a raw **CUBIN**, compressed **FATBIN**, and **PTX** |
| `test_atomize_runtime` | the CUDA-runtime-API launch path (frameworks use this) |
| `test_atomize_graph` | CUDA-graph capture → atomized subgraph → correct replay |
| `test_scheduler` | `LITHOS_QUOTA=4` confines a kernel to 8 SMs |

### Frameworks — correctness **and** atomization coverage

Coverage measured with `LITHOS_STATS_FILE` (per-launch split / single-atom /
full-range / **un-atomized** counts).

| framework | result | coverage (0 un-atomized = all kernels atomized) |
|-----------|--------|--------------------------------------------------|
| PyTorch 2.6 (+ `torch.cuda.CUDAGraph`) | ✅ correct, 5/5 | **100%** — 58 kernels, 55/55 launches |
| TensorFlow 2.21 | ✅ correct | **100%** — 69 kernels, 215/215 |
| JAX 0.6 (XLA) | ✅ correct, 10/10 | **100%** — 63 kernels, 240/240 (202 split) |
| TensorRT 10.7 | ✅ checksum bit-identical | **100%** — 11,825 kernels (autotuning), 17464/17464 |
| cuDNN 9 (direct `cudnnConvolutionForward`) | ✅ checksum bit-identical | **100%** — 6/6 |
| vLLM 0.8.5 — eager | ✅ correct | **100%** — 422 kernels, 512/512 |
| vLLM 0.8.5 — CUDA graphs | ✅ correct, output == baseline | **100%** — 439 kernels, **1397 launches split**, 3328/3328 |

Every kernel of every framework is atomized; all load kernels through
`cuLibraryLoadData`/`cuModuleLoadData` (fatbins), which the splicer handles.

---

## 9. Limitations & version-sensitivity

* **Driver/arch-sensitive constants** live in `qmd.c` (QMD byte offsets 84/88/192/81
  and the `cuGetExportTable` table indices 3/6). The QMD *byte* offsets are tied
  to the QMD **architecture** version (`QMDV03_00` for all GA10x) more than the
  driver build; the export-table indices are the most driver-version-fragile
  piece (they come from libsmctrl and could shift). Both are overridable via
  `LITHOS_QMD_PROG_OFF`. **Cross-driver validation — done:** the same A6000 was
  upgraded in-place from **570.195.03 → 595.71.05** (a CUDA-13-era driver, five
  branches newer) and *every* LithOS component still worked unchanged — full test
  suite 8/8, scheduler quota still confines to 4 TPCs (so the QMD mask offsets and
  the callback-table indices held), PyTorch 100% atomization coverage. So the
  reverse-engineering is robust across a large driver jump on the same GPU. A
  **second run on a fresh A6000** repeated the 570→595 upgrade *and* installed the
  **latest CUDA-13-era libraries** (torch 2.11+cu130, vLLM 0.25.1, TensorRT 10.15,
  TF 2.21) — all previously blocked on the old driver. Everything still hit **100%
  atomization coverage** after **two** CUDA-13-specific fixes: (a) fatbin
  compression moved LZ4→**Zstandard** (handled in `fatbin.c`), and (b) an ELF
  bounds-check guard for the tiny stub cubins cuBLASLt loads (which otherwise
  segfaulted the splicer). The *driver*-level reverse-engineering (QMD, callback
  table) needed no change; only the *userspace* cubin/fatbin-format handling did.
* **Metadata is per-process**, so multi-tenant (multi-process) use has no race;
  within a process, concurrent streams are handled by the event serialization of
  §5.5. Concurrent replays of *different* CUDA graphs that were each atomized can
  still contend on the single `g_meta` — a fully robust alternative (per-launch
  metadata as a kernel parameter) exists but was unnecessary for every workload
  tested.
* **Special kernels** — cooperative and cluster launches are detected and never
  split; a kernel with genuine cross-block dependencies but launched normally
  (no cooperative flag) is undetectable and would be a correctness risk if split,
  as the paper notes for "cross-block synchronization or persistent kernels."
* **Fatbins with only larger-than-supported params / `.text` relocations** are
  skipped by the splicer and run verbatim (correct, just not atomized).
* **Version pinning** was needed in this CUDA-12.8 environment: TensorRT 10.15 and
  vLLM 0.25 ship libs demanding a newer driver, so TensorRT 10.7-cu12 and vLLM
  0.8.5 (torch 2.6-cu124, transformers 4.51) were used.

---

*Source: `src/` (~2,000 LOC). Reverse-engineering harnesses: `sass-jump/`.
Legacy QMD-redirect atomizer: `legacy/`. Reproduction status and hard-won facts:
project memory.*
