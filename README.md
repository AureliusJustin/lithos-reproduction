# LithOS reproduction

A from-scratch reproduction of the core mechanisms of **LithOS** (SOSP), a GPU
operating system that interposes at the CUDA Driver API to schedule ML workloads
at the granularity of individual TPCs and to transparently split kernels into
thread-block "atoms".

The original prototype is ~5000 lines of Rust. This reproduction is written in
**C/C++/CUDA** so it can reuse the QMD/TPC-masking reverse-engineering in
[`../libsmctrl`](../libsmctrl) (Bakita) and build directly with the system
`nvcc`. It is mechanism-for-mechanism faithful to Sections 5–6 of the paper.

```
make            # builds build/liblithos_full.so, build/libcuda.so.1, tests
make run_tests  # runs the full suite: interposition (both paths), atomizer
                # (cubin/fatbin/PTX/runtime/graph), scheduler, dispatcher,
                # coordinator, correctness matrix

bash bench/fw_all.sh   # end-to-end on real libraries: cuBLAS, cuFFT, PyTorch,
                       # Triton -- each x 8 LithOS configs, checking correctness
                       # AND atomization coverage
```

`run_tests` and `fw_all.sh` catch different things, so run both. Every bug in
[FINDINGS §5](docs/FINDINGS.md) passed the unit suite and was caught only by real
libraries — the failure mode is a system that looks healthy while doing nothing.

## Usage — putting LithOS in front of your workload

LithOS injects itself between your application and the NVIDIA driver. **Which
injection method you use depends on how your app reaches the driver** — this is
the single most important thing to get right, because the wrong one silently runs
your app *natively* (no interposition, 0% atomization, no TPC masking):

| Your workload | Inject with | Why |
|---|---|---|
| **Runtime / framework app** — PyTorch, TensorFlow, JAX, TensorRT, vLLM, or anything using `libcudart` / `<<<>>>` / the CUDA Runtime | `LD_LIBRARY_PATH=build ./app` — the **`libcuda.so.1` wrapper** | `libcudart` `dlopen`s `libcuda.so.1` *by soname* and `dlsym`s the driver from that handle, so it must find **our** `libcuda.so.1` first |
| **Driver-API app** — calls `cu*` directly (CUDA driver API, a custom loader, `cuLaunchKernel`) | `LD_PRELOAD=build/liblithos_full.so ./app` | your `cu*` calls resolve to our exported symbols through the PLT |

> ⚠️ **`LD_PRELOAD` does *not* work for framework/runtime apps.** On CUDA 12.8 the
> runtime resolves its launch entry points from its own `dlopen("libcuda.so.1")`
> handle and never routes them through the preloaded symbol (`cuGetProcAddress` is
> called **zero** times), so a preloaded `liblithos_full.so` is invisible to it.
> Use the wrapper for those. **Sanity check:** add `LITHOS_STATS=1` and read the
> coverage line printed at exit — `0/0 launches` means LithOS never saw your
> kernels (wrong injection method); `N/N launches = 100.0%` means it did.

```sh
# framework script (PyTorch / JAX / TF / TensorRT / vLLM)
LD_LIBRARY_PATH=build python3 train.py

# driver-API program
LD_PRELOAD=build/liblithos_full.so ./my_cuda_app

# confine a tenant to 8 TPCs and split kernels into ~250 µs atoms
LITHOS_QUOTA=8 LITHOS_ATOM_US=250 LD_LIBRARY_PATH=build python3 serve.py

# confirm LithOS is engaged (prints coverage + per-kernel split/full/un-atomized)
LITHOS_STATS=1 LD_LIBRARY_PATH=build python3 train.py
```

All behaviour is controlled through environment variables — see
[Configuration](#configuration-environment-variables) for the full list.

> 📄 **Full technical report:** [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) — every module, how they work together, the reverse-engineering findings, and the framework validation.
> 📊 **Benchmarks:** [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — per-component overhead, CUDA-graph costs, the paper's experiments reproduced, **full A100 results** (isolation restores 99 % of solo throughput vs 29 % under plain MPS; atomization cuts HP tail latency **23×**; right-sizing R²=1.0000), and a **use-case study** of when LithOS is worth enabling.
> 🧪 **Findings:** [docs/FINDINGS.md](docs/FINDINGS.md) — what the experiments established: **when LithOS is worth enabling** (the interfering kernel's duration decides it), what each mechanism is worth, which paper results reproduce, and the non-obvious things we had to discover (graphs freeze the TPC allocation; atom size *is* the tail latency).
> 🔬 **Benchmark harnesses:** [bench/README.md](bench/README.md) — `make bench`, then the exact command for every number in the docs.
> 🔍 **Fidelity vs. the paper:** [docs/FIDELITY.md](docs/FIDELITY.md) — what's faithful, what diverges (the atomizer transfer), what's simplified/stubbed, what's not implemented, and what the paper leaves unspecified.

## Hardware requirements

**An RTX A6000 (or any Ampere GPU) is sufficient — you do not need an A100/H100.**
The arch is auto-detected (`-arch=native`; the Prelude is JIT-compiled to the
running GPU's SASS), so the same build runs on either. Verified on:

| GPU | arch | QMD ver | interposition | TPC scheduler | Kernel Atomizer |
|-----|------|---------|---------------|---------------|-----------------|
| **RTX A6000** (GA102) | `sm_86` | `QMDV03_00` (`0x30`) | ✅ | ✅ | ✅ end-to-end (prologue splice) |
| **A100 80GB** (GA100) | `sm_80` | `QMDV02_04` (`0x24`) | ✅ | ✅ | ✅ arch-agnostic splice (prologue JIT'd to `sm_80`) |

CUDA 12.8; **validated across drivers 570.195.03, 595.71.05 and 610.43.02** —
the last on an **A100 (GA100, 54 TPCs)**, i.e. the paper's own GPU, where the full
suite (interposition, atomizer via cubin/fatbin/PTX/runtime/graph, quotas, TPC
stealing, correctness matrix) passes unchanged. Spanning three driver branches and
two Ampere dies confirms the reverse-engineering is version- and die-robust. The atomizer works by splicing a range-check into
each kernel's own SASS (below) — an **arch-agnostic** technique that needs no
QMD program-address redirect, so it does not depend on the per-arch QMD layout
differences. *Nothing in this project benefits from a datacenter GPU* — an earlier
hypothesis that A100/H100 would tolerate the `CALL`-based device transfer (the
dead-end the splice replaced) was tested on the A100 and **disproven** (both
arches fault).

## Architecture

```
  application (PyTorch / TensorRT / driver-API program)
        │  cuStreamCreate / cuLaunchKernel / cuStreamSynchronize / cuGetProcAddress
        ▼
  ┌─────────────────────────────  LibLithOS  ─────────────────────────────┐
  │ interpose.c     CUDA Driver API interposition (Section 5.2 / 6)        │
  │ sched*.c        TPC Scheduler: launch queues, quotas→TPC masks, TPC    │
  │ tpc_alloc.c     stealing, right-sizing, outstanding-work (Section 5.3) │
  │ dispatch.c      optional dispatcher thread (Section 5.2)               │
  │ predict.c       online latency prediction + Tracker (Section 5.7)      │
  │ atomizer.c      Kernel Atomizer: range-check prologue spliced into     │
  │ qmd.c           each kernel's SASS, per-atom metadata (Section 5.4/6)  │
  │ graphsched.c    CUDA-graph subgraph scheduling (Section 6)             │
  └───────────────────────────────┬───────────────────────────────────────┘
                                   ▼   forwards everything else
                          real libcuda.so.1 (NVIDIA driver)
```

| file | role |
|------|------|
| `src/interpose.c` | overrides `cuInit`, `cuStreamCreate[WithPriority]`, `cuStreamDestroy`, `cuLaunchKernel[_ptsz]`, `cuStreamSynchronize`, `cuCtxSynchronize`, and `cuGetProcAddress_v2`; forwards the rest |
| `src/real.c` | resolves the genuine driver entry points (`dlsym(RTLD_NEXT)` / driver `cuGetProcAddress`) |
| `src/wrapper.c` | `libcuda.so.1` glue: `dlopen` redirect + `CUDA_DEVICE_MAX_CONNECTIONS` |
| `src/sched.c` | the submit path: ties launch queues → TPC mask → prediction → right-sizing → throttle → atomizer together; sync/batch handling |
| `src/sched_stream.c` | the stream registry = per-stream **launch queues** (Fig. 9 ①): quotas, TPC ranges, idle/outstanding bookkeeping, TPC detection |
| `src/tpc_alloc.c` | turning a quota into an SM-disable mask: **compute quotas** (②), **TPC stealing**, per-atom/per-subgraph slice masks, **right-sizing** (⑥) |
| `src/dispatch.c` | the optional **dispatcher thread** (§5.2) that owns submission (`LITHOS_DISPATCH`) |
| `src/coord.c` | the **system-wide coordinator** (§5.1, Fig. 8): a shared-memory tenant table every LibLithOS maps — cross-application quotas, cross-application TPC stealing, priority safeguards |
| `src/predict.c` | **online latency prediction** (§5.7) + the **Tracker thread** that reaps completion events |
| `src/graphsched.c` | CUDA-graph **subgraph partitioning** + per-subgraph TPC allocation with runtime re-instantiation |
| `src/qmd.c` | QMD/TMD pre-upload hook (libsmctrl-style debug callback): TPC mask (scheduler), program-address capture/patch, register-count bump |
| `src/atomizer.c` | Kernel Atomizer: module-load hook, atom splitting, stream-ordered metadata (prologue-splice model) |
| `src/atomize_splice.c` | builds the NVRTC range-check prologue and splices it into each **kernel-entry** `.text` section (ELF surgery: headers, `.nv.info`, symbols, and `-rdc` relocations); skips `__device__` functions |
| `src/fatbin.c` | unwraps fatbins (LZ4 on CUDA ≤12 / **Zstandard** on CUDA 13, decoding the SM-matching cubin) and JITs PTX → cubin, so those images get atomized too |
| `src/prelude.cu` | reference Prelude (Algorithm 1); legacy QMD-redirect atomizer kept in `legacy/` |

## 1. Interposition (Section 5.2 / 6) — working

LibLithOS interposes at the CUDA Driver API — "the lowest common denominator" —
so applications interact with LithOS instead of the driver while CUDA semantics
are preserved. Only a small subset of calls is overridden; everything else is
forwarded. Two transparent paths are supported:

* **Driver-API apps** — `LD_PRELOAD=build/liblithos_full.so ./app`. Calls go
  through the PLT and hit our exported symbols.
* **CUDA-runtime apps** (PyTorch/TF/JAX/TensorRT) — deploy the `libcuda.so.1`
  wrapper via `LD_LIBRARY_PATH=build` (see [§4](#4-running-under-ml-frameworks-pytorch--jax--); `LD_PRELOAD` does **not** work here — the runtime `dlopen`s the driver by name).

```
make run_tests   # driver-API, CUDA-runtime, and framework-path atomizer all PASS
```

## 2. TPC Scheduler (Section 5.3) — working

* **Launch queues** — a launch queue is created per stream (`cuStreamCreate`);
  `cuLaunchKernel` enqueues and returns control to the app.
* **Compute quotas** — `LITHOS_QUOTA=N` guarantees each stream `N` TPCs. The
  quota is enforced by applying a per-launch **TPC mask** through the QMD hook
  (the same masking mechanism libsmctrl validates), giving dynamic, on-the-fly
  TPC allocation — no MIG-style reconfiguration.
* **TPC stealing** — idle streams' TPCs are lent to streams with runnable work
  (`compute_disable_mask`), improving work conservation.
* **Outstanding-work tracking** — per-stream in-flight counters modelling the
  sync queues / 100µs throttle; cleared on synchronize.

```
LITHOS_QUOTA=4 LITHOS_STEALING=0 LD_PRELOAD=build/liblithos_full.so build/test_scheduler
# -> kernel confined to the first 4 TPCs = 8 SMs (of 84 on A6000 / 108 on A100),
#    verified via %smid
```

### On top of MPS (concurrent multi-tenant partitioning) — verified

The TPC *mask* works with or without MPS (the QMD hook confines any launch, even
under MPS: 84→16 SMs measured). But TPC *partitioning* only pays off **on top of
MPS**, exactly as the paper states — without MPS, separate processes **time-slice**
the GPU, so masking one tenant to a few TPCs just idles the rest. The wrapper sets
`CUDA_DEVICE_MAX_CONNECTIONS=8` for MPS, and `LITHOS_TPC_BASE` lets each process
take a disjoint TPC range (the job a central LithOS scheduler does for tenants).
Measured on the A100 with two co-located tenants (aggregate throughput):

| | aggregate |
|---|---|
| 1 tenant (solo) | 7 218 it/s |
| 2 tenants, time-sliced (no MPS) | 3 559 it/s |
| 2 tenants, **MPS** | **14 410 it/s** |

MPS gives **4.05× time-slicing** (and 2.0× a single tenant, i.e. genuine
concurrency) — the property that makes spatial TPC partitioning meaningful. Full
numbers in [docs/BENCHMARKS.md §5](docs/BENCHMARKS.md). (`%smid` is client-relative
under MPS, so disjointness shows up as throughput, not raw SM ids.)

## 3. Kernel Atomizer (Section 5.4 / 6) — working end-to-end

The Atomizer splits a kernel's grid into **atoms** (disjoint contiguous block
ranges) transparently — no source, no PTX — and each atom runs only its blocks
while skipping the rest. This is now **working end-to-end on the A6000**
([`test_atomize_cubin`](tests/test_atomize_cubin.c)): a grid is split into *N*
atoms, relaunched *N* times, and the output is bit-identical to a single
full-grid launch, with zero faults.

### The idea that made it work: prologue splice / fall-through

Algorithm 1 in the paper redirects a launch to a **Prelude** that range-checks
the block index and then *transfers control into the original kernel*. Reproducing
that transfer by byte-patching ptxas output is the hard, ultimately-unsolved part
(see below): the original ends in `EXIT`, and every register-indirect transfer we
could inject (`CALL`, `BRX`) sets up return/convergence state that the callee's
`EXIT` then violates (`INVALID_PC`).

**We sidestep the transfer entirely.** Instead of a *separate* Prelude that must
jump into the original, we **prepend the range-check directly into each app
kernel's own `.text.<fn>` section** (cubin ELF surgery at module-load time), so
in-range blocks **fall through** into the original code. No `CALL`, no `BRX`, no
`BSSY` convergence barrier — the original's `EXIT` is its own clean top-level
exit. Functionally identical to Algorithm 1 (a per-block gate that runs the
original for in-range blocks), achieved with no control transfer at all.

The mechanism, all verified empirically:

* **Module hook** — `cuModuleLoad{,Data,DataEx}` are interposed
  ([`src/interpose.c`](src/interpose.c)); every kernel in the cubin is spliced
  ([`src/atomize_splice.c`](src/atomize_splice.c)) and the modified cubin is
  loaded. **All three image kinds are handled** ([`src/fatbin.c`](src/fatbin.c)):
  raw ELF cubins directly; **fatbins** (as PyTorch/TensorRT ship) by extracting
  the cubin for the running SM — decompressing it if flagged (**LZ4 block** on
  CUDA ≤ 12, **Zstandard** on CUDA 13); and **PTX** (or a PTX-only
  fatbin) by JIT-compiling it to a cubin with the driver linker (`cuLink*`). In
  every case we hand the driver a bare (spliced) cubin — no fatbin repackaging
  needed. Anything unrecognized loads verbatim (correct, just not atomized). ✅
* **Range-check prologue** — NVRTC-compiled per device arch (`sm_86`, `sm_80`, …),
  reading `AtomMetadata{lo,hi}` from a **literal absolute address** (a const-bank
  global would read the app's constant bank and get garbage; `gridDim` in
  `c[0x0]` *is* valid). It uses only predicated `@P EXIT` — **no** `BSSY`. ✅
* **ELF splice** — insert the prologue (padded to a 128-byte multiple) at the
  front of `.text.<fn>`, fixing section headers, **program headers** (the driver
  loads via these), the function symbol size, `.nv.info` register count, and all
  instruction-offset attributes (`EXIT_INSTR_OFFSETS`, and the special
  `INDIRECT_BRANCH_TARGETS` for kernels with a `switch`/`BRX`). Validated on a
  kernel with 4 params, a loop, and a `switch` — bit-identical to reference. ✅
* **Stream-ordered metadata** — before each atom's full-grid relaunch, `[lo,hi)`
  is copied to the prologue's fixed address ordered on the launch stream, so every
  block of atom *i* observes its range before atom *i+1* overwrites it — no host
  sync. The buffer defaults to `[0, 2³²)` so any un-gated launch runs all blocks. ✅
* **Separately-compiled (`-rdc=true`) kernels & device functions** — a
  relocatable-device-code cubin (`nvcc -dc` + `-dlink -cubin`) breaks two naïve
  assumptions, both handled ([`src/atomize_splice.c`](src/atomize_splice.c)):
  (1) **Splice only kernel *entries*.** Such a cubin emits a separate `.text.<fn>`
  for every `__device__` function too, but those are *called*, not launched —
  prepending an `EXIT`-prologue to one would kill the thread instead of returning.
  We splice only sections whose symbol carries the CUDA entry flag
  (`st_other & 0x10`), leaving device functions untouched. (2) **Fix up
  relocations.** The splice shifts a kernel's `.text` by the prologue length, so
  every relocation into it is patched: each `r_offset += prologue_len`; a
  self-referential `RELA` addend (e.g. the return address the compiler
  materialises for a device-function `CALL`) is bumped too; local-label `st_value`s
  are bumped. The only genuinely unfixable case — a self-referential `REL`, whose
  addend is baked inside the instruction bytes — is skipped conservatively (that
  kernel runs verbatim). Validated end-to-end: a kernel calling a `__noinline__`
  device function that reads a `__device__` global, split into ~32 atoms, runs
  **every block exactly once** with correct output (device function un-spliced). ✅

```
LITHOS_ATOM_US=8 LD_PRELOAD=build/liblithos_full.so \
    build/test_atomize_cubin 256 build/atomize_mark.cubin
# -> PASS: 256 blocks, cubin spliced+atomized into atoms, all correct across 4 reps.
#    Composes with LITHOS_QUOTA (atoms confined to the stream's TPCs).
```

### Per-atom TPC allocation (each atom on a distinct TPC set)

Because each atom is a *separate* relaunch and the QMD TPC mask is **one-shot**
(consumed per launch by the pre-upload callback), the atomizer can set a
**distinct mask before each atom** — so the atoms of one kernel can each run on a
different TPC set. This realizes the paper's statement that *"TPC allocations can
be dynamically adjusted throughout a kernel's execution"* (§5.4). It's driven by
`lithos_apply_atom_mask()` ([`src/tpc_alloc.c`](src/tpc_alloc.c)), called from the atom
loop ([`src/atomizer.c`](src/atomizer.c)):

- **`LITHOS_ATOM_TPC=W`** — atom *i* is confined to a distinct contiguous **W-TPC
  slice**, tiled (with wraparound) across the process's TPC span: `[tpc_base,
  tpc_base+quota)` when a quota is set, else all TPCs. So atoms subdivide *their
  kernel's* allocation and never exceed the process's quota.
- **`LITHOS_ATOM_TPC_LIST=1,2,3`** — **variable** widths: atom *i* gets
  `list[i % n]` TPCs, packed contiguously and cycled. So a single kernel's atoms
  can each get a *different-sized* allocation (atom 0 → 1 TPC, atom 1 → 2, atom 2
  → 3, …). Verified with the `%smid` probe: widths `1,2,3` produce disjoint masks
  `{0}`, `{1,2}`, `{3,4,5}` and physical SM counts `2, 4, 6` (= 1, 2, 3 TPCs), all
  blocks still run exactly once, and inside a quota the widths wrap to stay within
  the allotment.
- **otherwise** the same mask (the stream's quota) is re-applied to *every* atom —
  which also fixes a latent bug: without re-setting the one-shot mask, only the
  first atom of a quota'd kernel was confined.

Verified with a `%smid` probe (`LITHOS_LOG_MASK=1` shows the mask per launch). One
32-block kernel, 4 atoms, `LITHOS_ATOM_TPC=2`:

| atom | blocks | enabled TPCs | SMs used |
|---|---|---|---|
| 0 | [0,8)  | `{0,1}` | 4 (=2 TPCs) |
| 1 | [8,16) | `{2,3}` | 4 |
| 2 | [16,24)| `{4,5}` | 4 |
| 3 | [24,32)| `{6,7}` | 4 |

Each atom got its own disjoint TPC pair; the mask physically took effect (each
atom used exactly 2 TPCs' worth of SMs); and every block still ran **exactly once**
with correct results (the correctness matrix passes 7/7 under `LITHOS_ATOM_TPC`).
With a quota it stays inside the allotment — e.g. `LITHOS_QUOTA=8 LITHOS_TPC_BASE=10`
tiles the atoms across `{10,11},{12,13},{14,15},{16,17}`, never touching another
tenant's TPCs. (Note: `%smid` is renumbered relative to the *enabled* set on this
A6000, so the **mask** — the actual allocation control — is the ground truth;
`%smid` only confirms the allocation *size* took effect.)

### Correctness validation — atomization must not change results

Atomization is only sound if every block runs **exactly once** across the atoms:
a double-run corrupts atomic reductions, a dropped block undercounts.
[`tests/correctness/matrix.cu`](tests/correctness/matrix.cu) exercises a spread of
real kernel patterns, each checked against a CPU reference — elementwise
(`vecadd`, `saxpy`), a **tiled matmul** (shared memory + `__syncthreads`, 2-D
grid), a **transpose**, a **3-D grid**, and two **atomic** kernels (`histogram`,
`reduce`) that are the exactly-once guards. Run identically under baseline and
every LithOS configuration:

```sh
nvcc -arch=sm_86 -O2 tests/correctness/matrix.cu -o /tmp/matrix -lcudart
for cfg in "" "LITHOS_ATOM_US=1" "LITHOS_QUOTA=8" "LITHOS_ATOM_US=1 LITHOS_QUOTA=8"; do
  env $cfg LITHOS_STATS=1 LD_LIBRARY_PATH=build /tmp/matrix   # 7/7 PASS, 100% coverage
done
```

| configuration | correctness | coverage |
|---|---|---|
| baseline (no LithOS) | 7/7 PASS | — |
| atomizer default | 7/7 PASS | 100% (7/7) |
| forced max-split (`LITHOS_ATOM_US=1`) | 7/7 PASS | 100% |
| + TPC quota=8 | 7/7 PASS | 100% |
| split + quota (full stack) | 7/7 PASS | 100% |
| + MPS on (as deployed) | 7/7 PASS | 100% |

The atomic kernels passing under *genuinely split* launches (each grid divided
into many atoms) is the proof that partitioning is bit-exact — no block runs twice
or is skipped. Stable across repeated runs (no races in the metadata/event path).

### Appendix: the device-transfer dead-end (why the splice was necessary)

The direct reproduction of Algorithm 1's transfer — redirect the QMD **program
address** (byte 192) to a separate Prelude that jumps into the original — was
reverse-engineered exhaustively and is the **one piece that can't be closed by
byte-patching ptxas output**. ptxas lowers the Prelude's indirect tail call to
`CALL` (pushes a return PC → the original's `EXIT` faults); the correct lowering
is a **jump**, which is *not expressible in PTX at all*, so no PTX-compiled
language (Rust included, `become`/`explicit_tail_calls` verified) can emit it —
only SASS can. We proved `CALL`/`BRX` land and run the original but fault on its
`EXIT`; a plain `BRA` transfers cleanly but is immediate-encoded (can't target a
runtime cross-module address, and loaded code can't be patched:
`cuMemcpyHtoD → ILLEGAL_ADDRESS`). The QMD redirect, register/metadata handling,
branchless select, and `BRX` metadata injection all work (harnesses in
[`sass-jump/`](sass-jump/), legacy Prelude atomizer in
[`legacy/`](legacy/)); it is the transfer alone that LithOS's Rust/LLVM toolchain
resolves at link/load time in a way byte-patching can't. **The prologue splice
above makes this moot** by never transferring in the first place.

## 4. Running under ML frameworks (PyTorch / JAX / …)

Frameworks sit on the CUDA **Runtime** (`libcudart`), which reaches the driver in
ways that bypass naive public-API interposition. Making the atomizer + scheduler
work transparently under them required closing four gaps (in
[`src/interpose.c`](src/interpose.c) / [`src/fatbin.c`](src/fatbin.c) /
[`src/atomizer.c`](src/atomizer.c)):

1. **Wrapper, not `LD_PRELOAD`.** `libcudart` `dlopen`s `libcuda.so.1` *by name*
   and `dlsym`s the driver, so `LD_PRELOAD` of our `.so` is invisible. We ship a
   `libcuda.so.1` **wrapper** (its `DT_NEEDED` patched to the real `libcuda.so`)
   deployed via `LD_LIBRARY_PATH=build`.
2. **Return our own resolver.** The runtime asks the driver for `cuGetProcAddress`
   *once*, then routes every other lookup through whatever it gets back. So our
   `cuGetProcAddress` override must return **itself** for `cuGetProcAddress[_v2]`
   — otherwise all launch/load calls resolve to the real driver and bypass us.
3. **Library API + fatbin wrapper.** The runtime loads kernels via the CUDA 12
   **Library API** (`cuLibraryLoadData → cuLibraryGetModule → cuModuleGetFunction`)
   passing a `__fatBinC_Wrapper_t` (magic `0x466243b1`, real fatbin at `+8`), and
   launches via `cuLaunchKernel`/`cuLaunchKernelEx`. We intercept that whole chain,
   follow the wrapper, and gate each launch. (`cuGetExportTable`'s callback table
   `2c8e0ad8…` is the same low-level hook `libsmctrl`/`qmd.c` use for TPC masking —
   it fires for *every* launch regardless of API.)
4. **Kernel handles that arrive without a name.** Splicing a module is only half
   the job: at launch we must recognise the `CUfunction` as one whose module we
   spliced. Intercepting the APIs that hand out handles *by name*
   (`cuModuleGetFunction`, `cuLibraryGetKernel`, `cuKernelGetFunction`) is not
   enough — CUDA 12 also offers **bulk enumeration** (`cuLibraryEnumerateKernels`,
   `cuLibraryGetKernelCount`), which returns handles with **no names at all**, and
   cuFFT uses it. Its kernels were therefore treated as "not ours" and ran at
   **0 % coverage** — while still computing the right answer, so nothing looked
   broken. Chasing each new enumeration API is a losing game, so an unrecognised
   handle is now classified by asking the driver directly: `cuFuncGetModule` gives
   the module it came from and `cuFuncGetName` its name, which is exactly what the
   named path would have used. This is done **once per distinct handle** (the
   result is memoised), and it closes the whole class, including APIs that do not
   exist yet. `LITHOS_DIAG=1` shows these as `byHandle`.

Robustness for real kernels: splicing is **per-kernel** — only the kernels we
successfully splice are gated, so a module with one odd kernel still atomizes the
rest. `.text` **relocations are handled** (including `-rdc=true` kernels — see §3);
just the residually-unfixable cases (a self-referential `REL`, or a malformed/tiny
stub cubin that fails the ELF bounds check) are skipped and run verbatim.
**Device functions and cooperative/cluster launches are never split** — a device
function is called not launched, and grid-/cluster-wide sync needs every block.
`LITHOS_QUOTA` also attaches to the default stream so framework work gets TPC
confinement.

**Multi-stream correctness.** The atom range lives in one per-process device
buffer, so an app's *concurrent streams* could interleave their metadata writes
(there is no cross-*process* race — each process has its own buffer). Two
safeguards keep it correct: (1) atomized launches are **serialized across streams
with a CUDA event** — a no-op for single-stream apps, so the common inference path
pays nothing; (2) a kernel resolved from an atomized module but not individually
gated (splice-skipped, or a handle we couldn't match) is launched with a
serialized **full range** so it can never read a stale `[lo,hi]` and drop blocks.
This took JAX/XLA (heavily multi-stream) from 2/10 to **10/10** correct.

**CUDA Graphs** (§6 — "atomize graphs into subgraphs, ensuring correct execution
ordering"). There are **two modes**, because a key hardware fact constrains what's
possible: the QMD **pre-upload callback fires exactly once per graph exec** (at the
first `cuGraphLaunch`); replays re-run a compiled command buffer that bypasses the
driver's per-node path, so the SM-mask **cannot be changed on replay** — only by
**re-instantiating** (which re-fires the callback, verified).

- **(default) atomize-into-graph.** During capture, our per-atom `[set-range →
  relaunch]` sequence is recorded as nodes, so a captured kernel becomes an
  atomized **subgraph** that replays correctly (ranges bake into `cuMemsetD32Async`
  immediates). Results are correct and replay reproduces each atom's range, but the
  schedule is **frozen**: atom ranges are baked and per-atom TPC masks don't apply
  in a graph (the one-shot mask can't map onto N nodes). Good for correctness +
  launch-bypass, not for dynamic rescheduling. Verified: `build/test_atomize_graph`
  and PyTorch `torch.cuda.CUDAGraph`.
- **(`LITHOS_GRAPH_SUBGRAPHS=K`) partition-into-subgraphs — the paper's model.**
  Don't atomize kernels. At instantiate, partition the graph along a **topological
  cut** into K subgraphs ([`src/graphsched.c`](src/graphsched.c), via
  `cuGraphClone` + node deletion); at launch, fan the single `cuGraphLaunch` into K
  sequential subgraph launches on the stream (order ⇒ dependencies), applying the
  scheduler's TPC allocation as a **sticky mask** before each subgraph (covers all
  its kernel nodes). The **subgraph is the scheduling/reallocation unit**; kernels
  run whole; most launch-bypass is retained (K launches, K ≪ #kernels).
  Verified: an 8-kernel chain → 4 subgraphs, each confined to a distinct 2-TPC
  slice (`{0,1}{2,3}{4,5}{6,7}`), every kernel used exactly its 4 SMs, output
  bit-identical to baseline.
  - **Runtime reallocation.** The subgraph *templates* are kept, so on each
    `cuGraphLaunch` LithOS compares each subgraph's desired allocation to what's
    currently baked and **re-instantiates only the subgraphs whose allocation
    changed** (a fresh exec re-fires the callback, baking the new mask); unchanged
    subgraphs just replay. So the *same* app-level graph exec can run on a
    *different* TPC allocation each replay, transparently. Demonstrated with
    `LITHOS_SUBGRAPH_ROTATE=1` (rotates the allocation per replay): the 4 subgraphs
    move through `{0,1}{2,3}{4,5}{6,7}` → `{2,3}{4,5}{6,7}{0,1}` → … with output
    still correct on every replay. Cost (8-kernel chain): steady replay ≈ 16 µs
    (vs 12 µs baseline, the fan-out); a reallocating replay ≈ 49 µs (≈ 8 µs per
    re-instantiated subgraph) — paid only when the allocation actually changes, and
    only for the subgraphs that changed.

**Verified on this A6000** (`LD_LIBRARY_PATH=build python3 …`):

Every one was run under the wrapper (`LD_LIBRARY_PATH=build`) and checked for
**correctness** *and* **atomization coverage** (`LITHOS_STATS_FILE` counts, per
launch, how many kernels were split / single-atom / full-range / un-atomized):

| framework | components tested | result | atomization coverage |
|-----------|------------------|--------|----------------------|
| **PyTorch 2.6** | matmul/cuBLAS, conv/cuDNN, elementwise+reduce/ATen, MLP fwd+bwd, **`torch.cuda.CUDAGraph`** | ✅ all correct, 5/5 repeats | **100%** — 58 kernels, 0 un-atomized |
| **TensorFlow 2.21** | matmul, conv2d, softmax-reduce, elementwise | ✅ all correct | **100%** — 69 kernels, 0 un-atomized |
| **JAX 0.6** (XLA) | matmul, softmax, `@jit` fusion, cumsum/scan, elementwise | ✅ all correct, **10/10** repeats | **100%** — 63 kernels, 202 split, 0 un-atomized |
| **TensorRT 10.7** (build engine + `execute_async_v3`) | conv→relu→conv network, bit-exact vs baseline | ✅ checksum identical | **100%** — **11,825 kernels** (engine-build autotuning), 17464/17464 |
| **cuDNN 9** (direct `cudnnConvolutionForward`) | conv forward, bit-exact vs baseline | ✅ checksum identical | **100%** — 6/6 launches |
| **vLLM 0.8.5** (PagedAttention + custom kernels) | offline generation, **eager** and **CUDA-graph** decode | ✅ correct in both modes, output == baseline | **100%** — eager 422 kernels (512/512), **graph 439 kernels, 1397 launches split (3328/3328)** |
| **+ `LITHOS_QUOTA`** | any framework under a TPC quota | ✅ correct *and* confined (SM-probe: 4→8, 8→16 SMs) | — |

**Every kernel of every framework is atomized (0 un-atomized).** Each LithOS
component — interposition / atomizer / scheduler / CUDA graphs / end-to-end —
passes for all of them.

That result is *not* explained by "they all load through
`cuLibraryLoadData`/`cuModuleLoadData`, so nothing slips through" — an earlier
version of this section said so, and it is wrong. Loading was never the hard part;
**recognising the kernel handle at launch** is, and a library can obtain that
handle without ever naming it (gap 4 above). cuFFT does exactly that and ran at
0 % coverage until the by-handle fallback existed, with correct output throughout.
Coverage is what makes this visible, which is why it is reported next to every
correctness result here.

### Re-validated on the A100 (`bash bench/fw_all.sh`)

Four targets × eight LithOS configurations, **32/32 PASS at 100 % coverage**
(A100 80GB PCIe, driver 570.195.03). Each target was chosen for a *different*
route to the GPU:

| target | reaches the GPU via | launches |
|---|---|---|
| cuBLAS | fatbin kernels fetched **by name** | 3 |
| cuFFT | kernels fetched by **bulk enumeration**, unnamed | 2 |
| PyTorch 2.13.0+cu126 | cuBLAS + cuDNN + ATen + autograd + CUDA graphs | 63 |
| Triton 3.7.1 | cubins **JIT-compiled at run time** | 18 |

Configs: default, `ATOM_US=1`, `QUOTA=16`, right-sizing, dispatcher, dispatcher
with 4 workers, predictor, and all at once. Two concurrent PyTorch tenants also
receive disjoint ranges `[0,16)`/`[16,32)` from the coordinator with **no**
hand-assigned `LITHOS_TPC_BASE`. See [FINDINGS §5](docs/FINDINGS.md) for the three
bugs this harness caught that the unit suite could not.

## Configuration (environment variables)

| var | default | meaning |
|-----|---------|---------|
| `LITHOS_VERBOSE` | 0 | log interposition / scheduler / atomizer activity |
| `LITHOS_ATOMIZER` | 1 | enable the atomizer (module-load cubin splice + per-atom launch) |
| `LITHOS_ATOM_US` | 300 | target atom duration (µs); paper uses 250–500 |
| `LITHOS_SLO_US` | 0 | **latency budget** this process may impose on a co-located tenant. Caps atom duration, because a latency-critical neighbour waits behind at most **one atom** (0 = off) |
| `LITHOS_ATOM_COST_US` | 5 | measured cost of one extra atom (the early-exit full-grid relaunch) |
| `LITHOS_ATOM_MAX_OVERHEAD` | 0.10 | cap splitting overhead at this fraction of the kernel's runtime; sets a **floor** on atom duration (0 = no floor) |
| `LITHOS_MIN_BLOCKS` | 8 | skip atomization below this grid size |
| `LITHOS_FORCE_ATOMS` | 0 | force an exact atom count, overriding the duration heuristic (0 = auto; testing/policy) |
| `LITHOS_ATOM_TPC` | 0 | **per-atom TPC allocation**: give each atom of a kernel a distinct *W*-TPC slice, tiled (with wraparound) across the process's TPC span (0 = off; all atoms share the stream's mask) |
| `LITHOS_ATOM_TPC_LIST` | — | **variable** per-atom widths, e.g. `1,2,3` → atom *i* gets `list[i % n]` TPCs, packed contiguously and cycled (overrides `LITHOS_ATOM_TPC`) |
| `LITHOS_QUOTA` | −1 | per-stream TPC quota (compute quotas) |
| `LITHOS_STEALING` | 1 | enable TPC stealing from idle streams (needs disjoint ranges — see `LITHOS_PERSTREAM_QUOTA`) |
| `LITHOS_TPC_TIMERS` | 1 | **per-TPC timers** (§5.3): each launch publishes `now + predicted_duration` against the TPCs it occupies, and a steal skips any TPC whose timer has not expired. Closes the hole in idle-only detection — a stream that submits one long kernel and goes quiet reads as *idle* after 1 ms while its kernel still has milliseconds to run. Never restricts a stream's own quota. `0` = the older idle-only heuristic |
| `LITHOS_PERSTREAM_QUOTA` | 0 | give **each stream** its own disjoint quota-sized TPC slice instead of one shared application-wide range. Required for intra-process TPC stealing to have anything to lend |
| `LITHOS_PREDICT` | 1 | online latency prediction (§5.7): event-measured, operator-indexed; drives atom sizing/right-sizing/throttle |
| `LITHOS_RIGHTSIZE` | 0 | per-kernel TPC right-sizing (§5.5): occupancy filter + `l=m/t+b` scaling model |
| `LITHOS_SLIP` | 1.1 | right-sizing latency-slip factor `k` (e.g. 1.1 = tolerate 10% slowdown) |
| `LITHOS_THROTTLE` | 0 | enforce the outstanding-work throttle (§5.3): defer dispatch while in-flight µs > limit |
| `LITHOS_DISPATCH` | 0 | buffer launches in per-stream launch queues and submit them from dispatcher threads (§5.2). The application's `cuLaunchKernel` returns as soon as the launch is queued |
| `LITHOS_DISPATCH_THREADS` | 2 | dispatcher worker threads. Workers never drain the same launch queue concurrently, so this only adds parallelism across *streams* |
| `LITHOS_DISPATCH_DEPTH` | 1024 | max launches buffered per queue before the submitting thread blocks. `1` makes deferral nearly synchronous — useful for isolating whether a bug is caused by buffering |
| `LITHOS_DISPATCH_PRIO` | 1 | let stream priority choose what the dispatcher sends next. `0` makes every queue compare equal, so selection is purely by arrival order — the control condition for priority experiments. That is a *strict* global FIFO only with `LITHOS_DISPATCH_THREADS=1`; with more workers some reordering remains whatever this is set to, because a worker skips any queue another worker is already draining (measured: 0 reorders at 1 worker, ~750 at 4, with priority off in both) |
| `LITHOS_PRIO` | 0 | default scheduling priority for streams the application did not prioritise itself (lower = more important, matching CUDA). Distinct from `LITHOS_PRIORITY`, which is this *tenant's* priority in the cross-application table |
| `LITHOS_LOG_PREDICT` | 0 | log per-operator measured/EMA latencies |
| `LITHOS_PREDICT_ACC` | 0 | log predicted-vs-measured pairs per launch, for scoring predictor accuracy (`bench/predacc.py`) |
| `LITHOS_COORD` | 1 | join the **system-wide tenant table** (§5.1) so quotas and stealing span *applications*, not just streams. Assigns disjoint TPC ranges automatically — no manual `LITHOS_TPC_BASE` needed. 0 = per-process only |
| `LITHOS_PRIORITY` | 0 | this tenant's priority (larger = higher). A higher-priority tenant's TPCs are never stolen, even while it is idle (§5.3) |
| `LITHOS_LOG_COORD` | 0 | log coordinator registration and range assignment |
| `LITHOS_TPC_BASE` | 0 | first TPC of this process's range (give concurrent processes disjoint ranges under MPS) |
| `LITHOS_GRAPH_SUBGRAPHS` | 0 | **paper-model graph scheduling**: partition each instantiated CUDA graph into K subgraphs (topological cut) and give each subgraph its own TPC allocation, instead of atomizing kernels inside the graph. Kernels run whole; the subgraph is the scheduling unit. Reallocates on replay by re-instantiating only the changed subgraphs |
| `LITHOS_SUBGRAPH_ROTATE` | 0 | demo: rotate each subgraph's TPC allocation every replay (stands in for a live scheduler changing allocations), which exercises the runtime re-instantiation path |
| `LITHOS_MPS` | 1 | auto-start the MPS control daemon on init; `0` disables (see [BENCHMARKS §8](docs/BENCHMARKS.md) for the pipe/log-directory caveat) |
| `LITHOS_STATS` | 0 | print the coverage summary to stderr at exit: distinct kernels by class, launches by class, and **atomization coverage**. The first thing to check when LithOS appears to do nothing |
| `LITHOS_STATS_FILE` | — | also dump that summary periodically to `<file>.<pid>`, so coverage survives a subprocess that is *killed* rather than exited (e.g. vLLM's EngineCore). One file per pid, so a launch-free parent cannot overwrite the worker's numbers |
| `LITHOS_DIAG` | 0 | diagnose *why* coverage is what it is: image kinds and splice time at module load, the class assigned to each kernel and how it was decided (by name, or `byHandle` — see below), and the name of every kernel that ran un-atomized |
| `LITHOS_LOG_MASK` | 0 | log the TPC disable-mask (and enabled-TPC list) applied to each launch — useful to observe per-atom/per-subgraph allocation |
| `LITHOS_LOG_CB` / `LITHOS_LOG_GRAPH` | 0 | log every QMD pre-upload callback / graph-partition decision |
| `LITHOS_OUTSTANDING_US` | 100 | outstanding-work throttle threshold |

Debug / reverse-engineering knobs (not part of normal operation):
`LITHOS_TRACE_GPA` logs every `cuGetProcAddress` lookup — the quickest way to tell
whether the CUDA runtime is resolving through us at all; `LITHOS_TRACE_ET` traces
`cuGetExportTable`; `LITHOS_QMD_PROG_OFF` and `LITHOS_CORRUPT_OFF` override/probe
QMD byte offsets (used to derive the constants in `qmd.c`).

## Reverse-engineering probes

`tests/probe_*.c[u]` dump/diff the QMD to locate fields (program address,
register count, grid dims) — the tools used to derive the offsets in `src/qmd.c`.

## Relationship to libsmctrl / nvdebug

The QMD pre-upload debug callback and TPC-mask layout come from
[`../libsmctrl`](../libsmctrl); `../nvdebug` exposes the GPC/TPC topology used by
`libsmctrl_get_*_info`. LithOS (and this reproduction) reimplement TPC control
directly through the QMD and add dynamic allocation on launch.
