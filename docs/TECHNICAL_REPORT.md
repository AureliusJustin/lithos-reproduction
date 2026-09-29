# LithOS Reproduction — Technical Report

A from-scratch reproduction of the core mechanisms of **LithOS**
([*An Operating System for Efficient Machine Learning on GPUs*, SOSP '25](../LithOS%20-%20SOSP.pdf)),
a layer that interposes at the CUDA Driver API to (a) schedule ML workloads at the
granularity of individual TPCs and (b) transparently split kernels into
thread-block "atoms". The original prototype is ~5,000 lines of Rust; this
reproduction is ~6,900 lines of C/C++/CUDA (about 5,900 of C sources plus headers),
reusing the QMD/TPC-masking reverse-engineering from
[libsmctrl](https://github.com/JoshBakita/libsmctrl) (Bakita) and building directly
with the system `nvcc`.

**Verified hardware/software.** NVIDIA **A100 80GB PCIe** (GA100, `sm_80`, 54 TPCs,
QMD `QMDV02_04`/`0x24`) and **RTX A6000** (GA102, `sm_86`, 42 TPCs, QMD
`QMDV03_00`/`0x30`). CUDA 12.8 (and 12.6 for the later A100 re-measurements),
drivers **560.35.05, 570.195.03, 595.71.05 and 610.43.02**, bare metal. The A100
is the same die and SM count as the paper's testbed (an SXM4 40 GB part; ours is a
PCIe 80 GB one). Performance numbers in this report are A100 numbers; the A6000 is
validated functionally (the full suite passes) but is no longer a source of
performance data.

This report is self-contained: it describes what the code does, what was measured,
and where it departs from the paper. Line counts are for the current tree.

---

## Table of contents

1. [What was reproduced](#1-what-was-reproduced)
2. [System architecture](#2-system-architecture)
3. [Module reference](#3-module-reference)
4. [End-to-end control & data flow](#4-end-to-end-control--data-flow)
5. [The Kernel Atomizer in depth](#5-the-kernel-atomizer-in-depth)
6. [Framework interception (the CUDA-runtime path)](#6-framework-interception-the-cuda-runtime-path)
7. [Reverse-engineering findings](#7-reverse-engineering-findings)
8. [Configuration reference](#8-configuration-reference)
9. [Validation](#9-validation)
10. [Results and findings](#10-results-and-findings)
11. [Fidelity to the paper](#11-fidelity-to-the-paper)
12. [Limitations, version-sensitivity and gotchas](#12-limitations-version-sensitivity-and-gotchas)

---

## 1. What was reproduced

Every mechanism the paper specifies for the runtime (§5.1–§5.7, §6) is present in
the tree. Rows marked *opt-in* are implemented and tested but off by default.

| Paper § | Mechanism | Status | Source | Switch |
|---|---|---|---|---|
| 5.2 / 6 | **LibLithOS** — transparent Driver-API interposition | ✅ two deployment paths | `interpose.c`, `barrier.c`, `wrapper.c` | — |
| 5.2 | Launch queues, dispatcher threads (fire-and-forget enqueue) | ✅ *opt-in* | `sched_stream.c`, `dispatch.c`, `params.c` | `LITHOS_DISPATCH` |
| 5.1 | System-wide coordinator (cross-application quotas, stealing, priorities) | ✅ | `coord.c` | `LITHOS_COORD` (on) |
| 5.3 | **TPC Scheduler** — compute quotas, TPC stealing, per-TPC timers | ✅ | `tpc_alloc.c`, `sched.c`, `qmd.c` | `LITHOS_QUOTA`, `LITHOS_STEALING`, `LITHOS_TPC_TIMERS` |
| 5.3 | Outstanding-work throttle; limit on outstanding atoms | ✅ *opt-in* | `sched.c`, `atomizer.c` | `LITHOS_THROTTLE`, `LITHOS_ATOMS_INFLIGHT` |
| 5.4 / 6 | **Kernel Atomizer** — split a grid into thread-block atoms | ✅ end-to-end, incl. CUDA graphs | `atomizer.c`, `atomize_splice.c`, `fatbin.c` | `LITHOS_ATOMIZER` (on) |
| 5.5 | Hardware right-sizing (occupancy filter + `l = m/t + b`) | ✅ *opt-in* | `tpc_alloc.c`, `predict.c` | `LITHOS_RIGHTSIZE` |
| 5.6 | Transparent power management (per-operator sensitivity → DVFS) | ✅ *opt-in*, needs root | `power.c` | `LITHOS_DVFS` |
| 5.7 | Online latency prediction + Tracker thread | ✅ | `predict.c` | `LITHOS_PREDICT` (on) |
| 6 | CUDA-graph scheduling (subgraphs, runtime re-instantiation) | ✅ *opt-in* | `graphsched.c` | `LITHOS_GRAPH_SUBGRAPHS` |
| 6 | Special kernels (cooperative/cluster: no splitting, spoofed SM count) | ✅ | `sched.c`, `interpose.c` | — |

It runs unmodified **PyTorch, TensorFlow, JAX, TensorRT, cuDNN, vLLM, cuBLAS, cuFFT
and Triton** with 100 % kernel-atomization coverage (§9).

The **one deliberate deviation** from the paper is *how* the atomizer transfers
control into the original kernel. The paper redirects a launch to a separate
*Prelude* kernel (Algorithm 1) that jumps into the original; that jump is not
reproducible by byte-patching `ptxas` output (§5.1). We instead **splice** the
range-check directly into each kernel's own machine code so in-range blocks *fall
through* — functionally identical, with no control transfer. §11 lists every other
difference and everything not implemented.

---

## 2. System architecture

```
   application  (PyTorch / TensorFlow / JAX / TensorRT / cuDNN / vLLM / driver-API)
        │  cuInit, cuModuleLoad*, cuLibraryLoadData, cuLaunchKernel[Ex], cuGraph*,
        │  cuGetProcAddress, cuMemcpy*Async, ...
        ▼
┌──────────────────────────────── LibLithOS ─────────────────────────────────┐
│ interpose.c   the interposition surface: overrides a small subset of the    │
│               Driver API, forwards the rest; two entry paths (PLT + the      │
│               runtime's cuGetProcAddress resolver)                           │
│ barrier.c     ordering barriers: any other stream-ordered call drains the   │
│  (barrier.def) launch queues first, so a buffered launch is never overtaken │
│                                                                              │
│ sched.c            the submit path: quota mask → right-size → predict →      │
│                    throttle → atomize; sync/batch boundaries                 │
│ sched_stream.c     per-stream launch queues, quotas, TPC detection           │
│ tpc_alloc.c        quotas → SM-disable masks, TPC stealing, per-atom/        │
│                    per-subgraph slice masks, hardware right-sizing           │
│ dispatch.c         optional dispatcher threads owning submission             │
│ params.c           kernel-argument layout recovery so launches can be        │
│                    deep-copied and buffered (cuFuncGetParamInfo)             │
│ coord.c            system-wide coordinator: POSIX shm tenant table,          │
│                    cross-application quotas / stealing / per-TPC timers      │
│ predict.c          online latency prediction + the Tracker thread            │
│ power.c            transparent power management (DVFS through NVML)          │
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
│ config.c      environment-variable configuration (lithos.h: shared types)    │
│ wrapper.c     libcuda.so.1 wrapper glue (framework path)                      │
└───────────────────────────────┬────────────────────────────────────────────┘
                                 ▼  forwards everything else
                        real libcuda.so.1  (NVIDIA driver)  — on top of MPS
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
differ. **Picking the wrong path fails silently** — the app just runs natively —
so set `LITHOS_STATS=1` and check the coverage line printed at exit: `0/0 launches`
means LithOS never saw the kernels; `N/N launches = 100.0%` means it did.

LithOS runs on top of **MPS** (auto-started at init; `LITHOS_MPS=0` disables), as the
paper does: without MPS, separate processes time-slice the GPU and masking one
tenant to a few TPCs merely idles the rest.

---
## 3. Module reference

### `interpose.c` (510 LOC) — the interposition surface

Defines LithOS's version of each overridden Driver-API function and forwards the
rest. The `overrides[]` table holds 36 entries in these groups:

* **Lifecycle / streams:** `cuInit`, `cuStreamCreate[WithPriority]`,
  `cuStreamDestroy[_v2]`, `cuStreamSynchronize[_ptsz]`, `cuCtxSynchronize`. Stream
  creation registers a per-stream launch queue with the scheduler.
* **Module loading:** `cuModuleLoad`, `cuModuleLoadData`, `cuModuleLoadDataEx`,
  `cuModuleLoadFatBinary`, `cuModuleGetFunction`, and the CUDA-12 **Library API**
  `cuLibraryLoadData`, `cuLibraryLoadFromFile`, `cuLibraryGetKernel`,
  `cuLibraryGetModule`, `cuKernelGetFunction`. These drive the atomizer's splice
  and function-gating (§4, §5).
* **Launches:** `cuLaunchKernel[_ptsz]`, `cuLaunchKernelEx[_ptsz]`,
  `cuLaunchCooperativeKernel[_ptsz]`. These flow into the scheduler → atomizer.
* **Graphs / capture:** `cuGraphInstantiate[_v2|WithFlags]`, `cuGraphLaunch`,
  `cuGraphExecDestroy`, `cuStreamBeginCapture[_v2]`, `cuStreamEndCapture` (the last
  two fence the Tracker thread and the dispatcher while a capture is open).
* **Device queries:** `cuDeviceGetAttribute` — `MULTIPROCESSOR_COUNT` is reported
  as the tenant's *allocated* SM count (`quota × 2`), so cross-block-sync and
  persistent kernels that size themselves to the device see only their partition
  (§6, special kernels).
* **Resolvers:** `cuGetProcAddress[_v2]` and `cuGetExportTable`. The
  `cuGetProcAddress` override is where the CUDA runtime discovers our wrappers;
  critically it **returns itself** for `cuGetProcAddress` (§6).

The `cuGetProcAddress` override consults `overrides[]` first and then the ordering
barriers (below), so the runtime is handed our wrapper for each interposed symbol
and the real pointer for everything else. On init it also starts the MPS control
daemon unless `LITHOS_MPS=0`.

### `barrier.c` / `barrier.def` (106 LOC + 55 entries) — ordering barriers

Deferring a launch (dispatcher on) opens a window in which the launch exists only in
a LithOS queue and its stream looks empty to the driver. Any other stream-ordered
call issued in that window would reach the GPU *ahead* of the kernel the
application already issued — a `cuLaunchKernel` followed by `cuMemcpyDtoHAsync`
would copy back memory the kernel had not yet written. `barrier.def` lists the 55
calls that must therefore **drain first** (27 scoped to one stream — async copies
and fills, stream-ordered alloc/free, event record/wait, host callbacks,
`cuStreamQuery`; 28 that drain everything — blocking copies/fills, teardown/free,
which race any queue). An X-macro expands each into a forwarder that drains and
then calls the real function; `cuGetProcAddress` lookups of these names are answered
with the forwarder. The guard is one relaxed atomic load, so it is free when nothing
is buffered — including always, when the dispatcher is off. LithOS's own CUDA calls
opt out through a re-entrancy flag (draining on them would deadlock).

### `real.c` / `real.h` (91 LOC) — real-driver resolution

Populates a `g_real` struct of genuine driver function pointers, resolved (in
order) via `dlsym(RTLD_NEXT)`, an explicit `dlopen` of the versioned
`libcuda.so.1`, and finally the driver's own `cuGetProcAddress`. Everything
LibLithOS forwards goes through `g_real`.

### `config.c` / `lithos.h` (136 + 181 LOC) — configuration and shared types

`g_lithos_cfg` is initialized once from environment variables (`LITHOS_*`); §8 lists
all of them. Paper defaults (atom duration 250–500 µs, 100 µs outstanding limit) are
the defaults, except where measurement showed a default was wrong (`LITHOS_ATOM_COST_US`,
§5.7). `lithos.h` also holds the per-thread re-entrancy flag that marks LithOS's own
CUDA work so the ordering barriers do not drain on it.

### `sched.c` + `sched_stream.c` + `tpc_alloc.c` — TPC Scheduler (§5.2/5.3/5.5)

Per-stream `StreamState` (quota, assigned TPC range, launch/atom counters, idle
timer, outstanding count). Key pieces:

* **Launch queues** (`sched_stream.c`) — one per stream, created on
  `cuStreamCreate`; the default stream is registered lazily on first launch so a
  global quota still attaches. TPC count is detected lazily from the SM count.
* **Compute quotas** — `LITHOS_QUOTA=N` guarantees a stream `N` TPCs. Enforced by
  computing a **disable mask** (`compute_disable_mask`) and handing it to the QMD
  hook (`qmd_set_next_mask`) for the upcoming launch. With the coordinator on, the
  tenant's range is assigned automatically; otherwise `LITHOS_TPC_BASE` places it.
* **TPC stealing** — idle streams'/tenants' TPCs are lent to a stream with runnable
  work by OR-ing their ranges into the active mask (`coord_stealable_tpcs`),
  minus any TPC whose **per-TPC timer** has not expired (§ coordinator below). Never
  applied to a stream's own quota — that is a guarantee. Within one process, streams
  share one range unless `LITHOS_PERSTREAM_QUOTA=1` gives each its own slice;
  lending a range identical to your own is a provable no-op.
* **Per-atom / per-subgraph masks** (`lithos_slice_mask`, `lithos_apply_atom_mask`) —
  re-arm the one-shot mask for *every* atom, recomputed from live coordinator state,
  optionally tiled across distinct TPC slices (§5.8).
* **Right-sizing** (`rightsize_tpcs`) — occupancy filter (`cuOccupancyMaxActiveBlocksPerMultiprocessor`
  → `ceil(blocks / (blocks_per_sm × 2))`) as an upper bound, then the `l = m/t + b`
  model from `predict.c`, within the latency-slip factor `k`. Opt-in.
* **Outstanding-work throttle** (`throttle_wait`) — defers dispatch while event-reaped
  in-flight µs exceed `LITHOS_OUTSTANDING_US` (100 µs). Evaluated once per *kernel*,
  before the atom loop, so it bounds in-flight µs, not outstanding atoms.
* **Special kernels** (`cfg_is_special`) — cooperative/cluster launches are never
  split, get their exact quota, and disable stealing and right-sizing.
* **Submit path** — `lithos_submit_launch[_ex]` (and `_coop`) first offers the launch
  to the dispatcher (`dispatch_try_*`); otherwise `presubmit` does the
  quota/stealing/mask bookkeeping, then right-sizing → `predict_lookup` → throttle →
  `atomizer_dispatch`, then `postsubmit` records the completion event.

### `dispatch.c` + `params.c` (595 + 179 LOC) — dispatcher threads and argument capture (§5.2)

With `LITHOS_DISPATCH=1`, launches funnel into per-stream launch queues drained by
dispatcher threads (`LITHOS_DISPATCH_THREADS`, default 2; a queue is never drained
by two workers at once) that apply global policy and submit to the GPU. This is a
**true fire-and-forget enqueue**, matching §5.2's "LithOS enqueues the kernel and
returns control to the application". The hard part is that `cuLaunchKernel` takes
`void** kernelParams` whose sizes appear nowhere in the call, and the application
may free them the moment the call returns. `params.c` recovers each kernel's
argument layout with `cuFuncGetParamInfo` (CUDA 12.4+; works for closed-source
cuBLAS/cuDNN kernels too) and **deep-copies** the argument values into the queue
entry; kernels whose layout cannot be recovered are submitted inline after draining
the stream. The submitting thread blocks only when a queue reaches
`LITHOS_DISPATCH_DEPTH` (1024). `LITHOS_DISPATCH_PRIO=0` makes selection purely
arrival-ordered — the control condition for priority experiments. The dispatcher
is off by default (§10 explains why: it costs tens of µs per launch and buys nothing where
LithOS decides inline).

Two correctness details worth knowing: `cuCtxSetCurrent` binds *per thread*, so every
dispatcher worker must bind the context itself (a process-wide "already bound" cache
silently dropped launches on all but the first worker), and the ordering barriers
above keep a buffered launch from being overtaken.

### `coord.c` (321 LOC) — the system-wide coordinator (§5.1, Figure 8)

Everything else is per-process; this is the shared layer. It is a **POSIX
shared-memory segment** (`/lithos_coord`), not a daemon: every LibLithOS instance
maps it, registers as a tenant (up to 64), and both publishes its own liveness and
reads everyone else's — so cross-application scheduling adds no RPC hop to the launch
path. It provides:

* **Cross-application compute quotas** — disjoint contiguous TPC ranges are handed
  out automatically; no hand-assigned `LITHOS_TPC_BASE`.
* **Cross-application TPC stealing** with the paper's priority safeguard: an idle
  tenant's TPCs can be borrowed, but a *higher-priority* tenant's TPCs are never
  stolen, even while it idles, and stealing stops as soon as the lender has runnable
  work again (`LITHOS_PRIORITY`).
* **Per-TPC timers** (§5.3) — each launch publishes `now + predicted_µs` against the
  TPCs it occupies; a steal skips any TPC whose timer has not expired. The horizon
  **accumulates** (a launch starts when the TPC's existing horizon expires) with a
  200 ms cap, so a tenant that bursts 200 short kernels reads as busy for the full
  queue depth rather than as free after 0.25 ms.

The segment's creator is decided by `O_EXCL` on `shm_open`, not by "is it size zero"
(two tenants starting together both initialised it and were handed the same range).
The mask is 64-bit, so at most 64 TPCs are addressable (§12).

### `qmd.c` (239 LOC) — the QMD pre-upload hook

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
  (Ampere/Ada; Hopper, TMD version ≥ 0x40, uses 304–316 plus two "extended" words —
  present but **untested**, §12). This is what enforces quotas/stealing.
* The mask is **one-shot** (`qmd_set_next_mask`: consumed by the upload it applies
  to, so an atomized launch must re-arm it for every atom) or **sticky**
  (`qmd_set_sticky_mask`: applies to every kernel node of a graph subgraph).
* The file also contains the legacy program-address **capture/patch** used by the
  original QMD-redirect atomizer and the `sass-jump/` harnesses (program address
  at byte **192**, register count at byte **81**). The current splice-based
  atomizer does **not** use these — it only uses the mask — but they remain for the
  reverse-engineering record.

### `predict.c` (448 LOC) — online latency prediction + the Tracker thread (§5.7)

Learns each operator's execution time online, with no offline profiling. An
**operator** is identified the way the paper describes: by its **ordinal `k`** on a
launch queue — the k-th kernel since that queue's last sync — because the same
kernel function recurs with different tensor shapes, so the function alone is not a
useful key. `predict_reset_op` restarts the ordinal at every sync (a batch
boundary). Predictions are EMA-refined and TPC-scaled; when an operator has not
been measured at the requested TPC count the fallback is the conservative "optimal
linear scaling" the paper specifies (`pred = ema × ema_tpc / tpcs`), and only before
an operator's first measurement is the stub `blocks × 0.5 µs` used.

Measurement records **one completion event per launch**, plus (with
`LITHOS_PREDICT_BRACKET=1`, the default) a start event ahead of it, so a kernel's
duration is measured rather than inferred from the gap to the previous completion —
the gap method silently folds host stalls into kernel time. Bracketing doubles the
per-launch cost (§10); `LITHOS_PREDICT_BRACKET=0` gets half of it back. A **Tracker
thread** reaps those events, updates an EMA per operator, and maintains the
in-flight-µs counter the outstanding-work throttle reads. Per-task tracking is
deliberately not sampled: the same Tracker signal clears the sync queues and
updates the per-TPC timers, not just the predictor.

`predict_rightsize` additionally fits the §5.5 curve `l = m/t + b` from two probed
samples (all-TPC and 1-TPC) and returns the fewest TPCs within a latency-slip bound.

Because CUDA calls from another thread can invalidate an in-progress graph capture,
the Tracker is fenced: `predict_capture_begin` waits for it to leave its CUDA
section, and the Tracker refuses to enter one while a capture is open.

### `power.c` (427 LOC) — transparent power management (§5.6)

Reproduces the paper's DVFS model, learned online from the predictor's
measurements. Each operator (indexed as in §5.7) accumulates its latency at `f_max`
and at one probe clock; the ratio gives its **sensitivity**
`s = slowdown / (f_max/f_probe - 1)`, clamped to [0,1]. Each operator's **weight**
`w` is its share of observed runtime, so a kernel that runs rarely cannot drag the
whole workload's clock down. The **aggregate** `S = Σ w·s` yields the model target
`f_final = f_max/(1 + slip/S)`.

Learning follows §5.6's conservative order — collect at `f_max`, probe once at
`LITHOS_DVFS_PROBE × f_max`, then apply — with transitions rate-limited
(`LITHOS_DVFS_SWITCH_MS`, 500 ms) well above the ~50 ms switching cost. Four details
are load-bearing:

* **Phase gating is runtime-weighted, not per operator.** Requiring every operator
  to reach the sample threshold never becomes true: one warm-up kernel that runs
  once leaves an unsatisfied entry forever and pins the system in the learning
  phase. Advancing when the operators accounting for most of the runtime are known
  is what makes it converge.
* **The formula picks a starting point, not a destination.** It is a first-order
  extrapolation from one probe; on a real kernel mix a target derived from a 25 %
  clock drop overshot a 10 % latency budget by six times. So LithOS starts from the
  higher of the model's clock and the conservative linear-scaling clock
  `f_max/(1+slip)`, then runs a **closed loop**: compare the *observed* aggregate
  slowdown with the budget and step one supported clock up (overshoot) or down (real
  headroom). This is §5.6's operational text — "depending on the observed
  performance, either further lower the frequency or stop". Re-deriving `S` from
  samples taken at the *applied* clock feeds the output back into the input and
  walks the clock to the floor, which is why the model is used once.
* **The probe clock is a cost, not just a measurement.** Every kernel running during
  the probe is slowed and those samples become the latency tail; probing at 25 %
  below `f_max` learns the same sensitivity as a 43 % drop while leaving p99 far
  lower.
* **The device is restored on exit** via a destructor — a tenant that leaves the
  GPU pinned to its own clock silently imposes it on the next process.

NVML is `dlopen`ed rather than linked, so a machine without it still runs
everything else. Setting a clock is root-only on this driver, so run
`sudo env LD_PRELOAD=… ./app` (plain `sudo` strips `LD_*`); without privilege the
first failure disables the mechanism with an actionable message rather than
repeating per launch.

### `atomizer.c` (877 LOC) — Kernel Atomizer orchestration

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
  atomized container, classify each into one of three classes — **ATOM** (name was
  spliced → may be split), **FULL** (from an atomized module but *not* in the
  spliced set → forced to an explicit full range, since it may still carry a
  prologue and would otherwise inherit the previous atom's `[lo,hi)` and silently
  drop blocks), or **VERBATIM** (untouched). Membership is an open-addressed
  pointer hash set, so the per-launch lookup is O(1) even at TensorRT scale
  (11 k+ kernels). A handle that arrives *without a name* (bulk enumeration) is
  classified by asking the driver (`cuFuncGetModule` + `cuFuncGetName`), memoised
  once per distinct handle (§6).
* **Atom sizing** (`effective_atom_us`, `decide_atoms`) — see §5.7.
* **Dispatch** (`atomizer_dispatch`, `_ex`, `_coop`): for each atom, publish
  `[lo,hi)` to `g_meta` and relaunch the full grid; the spliced range check skips
  out-of-range blocks, and `lithos_apply_atom_mask` re-arms the (one-shot) TPC mask
  on **both** launch APIs so each atom can land on its own TPC set. Correctness under
  concurrency and graph capture comes from `cuMemsetD32Async` metadata writes (§5.4)
  plus cross-stream event serialization — the latter skipped while only one stream
  issues atomized work, which **fails safe**: a second stream latches the fast path
  off permanently. With `LITHOS_ATOMS_INFLIGHT=W`, atom *i* waits for atom *i−W*
  before submission (§5.8).
* **Coverage stats** (`LITHOS_STATS` / `LITHOS_STATS_FILE` / `LITHOS_DIAG`): counts
  split / single-atom / full-range / un-atomized launches and distinct kernels, and
  names any kernel that ran un-atomized.

### `atomize_splice.c` (549 LOC) — the SASS/ELF surgery

The mechanical core. Two entry points:

* `atomize_build_prologue(sm, meta_va)`: NVRTC-compiles a tiny probe kernel
  (`if (b < meta->lo || b >= meta->hi) return; <marker>;`), extracts the
  range-check SASS up to a `0xDEADBEEF` marker, pads it to a 128-byte multiple
  with `NOP`s, and records its register requirement. `b` is the global block
  index from `blockIdx`/`gridDim`; `meta` is read from a literal absolute address.
* `atomize_splice_cubin(cubin, prologue)`: inserts the prologue at the front of
  every **kernel-entry** `.text.<fn>` section and repairs everything that encodes
  an offset into the shifted code.

  It is a **single pass**: plan all insertions, do one segmented copy into a
  correctly-sized buffer, then fix the headers once. The reason this is possible
  is that only *file* offsets accumulate across insertions (section headers,
  program headers — the driver loads via those — and `e_shoff`/`e_phoff`); symbol
  `st_value`, `.nv.info` instruction offsets and relocation `r_offset`s are
  **section-relative**, so each kernel shifts by just its own prologue length. A
  naive one-kernel-at-a-time splice re-copies the whole cubin each time and is
  O(n²) — 92 ms for 600 kernels versus 0.25 ms now, scaling linearly with kernel count.

  What gets repaired: section + program headers, the function symbol's size, the
  `.nv.info` register count, every instruction-offset attribute
  (`EXIT_INSTR_OFFSETS`, `CTAID_OFFSETS`, and the structured
  `INDIRECT_BRANCH_TARGETS`), and — for separately compiled `-rdc=true` kernels —
  **ELF relocations**: every `r_offset` is bumped, a self-referential `RELA` addend
  (e.g. the return address the compiler materialises for a device-function `CALL`)
  is bumped, and local-label `st_value`s move with the code.

  Only **kernel entries** are spliced, identified by the `STO_CUDA_ENTRY` bit
  (`st_other & 0x10`): a `__device__` function also gets its own `.text.<fn>`, but
  it is *called*, so an `EXIT`-prologue would kill the thread instead of returning.
  The residual unfixable cases — a self-referential `REL`, whose addend lives
  inside the instruction bytes, and malformed/undersized stub cubins — are skipped
  and stay runnable verbatim; only the spliced names are returned for gating.

### `fatbin.c` (253 LOC) — image unwrapping

`atomize_image_to_cubin(image, want_sm)` turns any module image into a raw cubin:

* **`__fatBinC_Wrapper_t`** (magic `0x466243b1`) — the struct the CUDA runtime
  passes; follow its `data` pointer (offset 8) to the real fatbin.
* **Fatbin** (magic `0xBA55ED50`) — walk the entries, pick the ELF cubin matching
  the running SM, and decompress it if flagged: **LZ4 block** format on CUDA ≤ 12
  (flag `0x2000`; a minimal block decoder is included) or **Zstandard** on CUDA 13
  (flag `0x8000`; linked as `libzstd.so.1`, using the exact compressed-size field).
* **PTX** (or a PTX-only fatbin) — JIT to a cubin with the driver linker
  (`cuLinkCreate`/`AddData(CU_JIT_INPUT_PTX)`/`Complete`).
* **Raw ELF cubin** — used directly.

In every case the driver is later handed a bare (spliced) cubin — no fatbin
repackaging is needed, since `cuLibraryLoadData`/`cuModuleLoadData` accept a raw
cubin. Anything unrecognised loads verbatim (correct, just not atomized).

### `graphsched.c` (246 LOC) — CUDA-graph subgraph scheduling

Implements the paper's "atomize graphs into subgraphs, ensuring correct execution
ordering" (`LITHOS_GRAPH_SUBGRAPHS=K`). At `cuGraphInstantiate` the graph is
partitioned along a **topological cut** of its dependency DAG (Kahn sort over
`cuGraphGetNodes`/`GetEdges`; each subgraph built with `cuGraphClone` plus
`cuGraphDestroyNode` for the nodes outside its chunk). At `cuGraphLaunch` the app's
single launch is fanned into K sequential subgraph launches on the same stream — so
stream order preserves every cross-cut dependency — each preceded by that
subgraph's TPC allocation applied as a **sticky** mask, since one subgraph may hold
several kernel nodes.

The subgraph **templates** are retained, so when a subgraph's desired allocation
changes it is **re-instantiated** from its template (the only way to re-bake an SM
mask — see §7) while unchanged subgraphs simply replay.

### `wrapper.c` (33 LOC) — libcuda.so.1 wrapper glue

Only in the wrapper build. Intercepts `dlopen("libcuda.so[.1]")` to hand back the
*real* driver (avoiding infinite recursion into our wrapper), and sets
`CUDA_DEVICE_MAX_CONNECTIONS=8` so MPS exposes enough hardware channels (per the
paper / libsmctrl). The Makefile also patches the wrapper's `DT_NEEDED` from
`libcuda.so.1` → `libcuda.so` so forwarded symbols resolve to the real driver.

### `prelude.cu` and `legacy/`

`src/prelude.cu` is the reference Prelude of the paper's Algorithm 1. The QMD-redirect
atomizer that used it is superseded by the splice and kept in
`legacy/atomizer_prelude_legacy.c`, with the SASS-transfer harnesses in `sass-jump/`,
as the record of the dead end (§5.1).

---

## 4. End-to-end control & data flow

### 4.1 Module load → splice

```
app: cuLibraryLoadData(code)  ─▶ interpose.c
   atomizer_intercept_cubin(code)
     ├─ fatbin.c: unwrap __fatBinC_Wrapper_t → fatbin → LZ4/Zstd-decode → raw cubin
     └─ atomize_splice.c: single pass — prepend range-check into every kernel-entry
        .text.<fn>, then fix headers/symbols/.nv.info/relocations once
   g_real.cuLibraryLoadData(spliced_cubin)  → CUlibrary
   register (CUlibrary → set of spliced kernel names)
```

### 4.2 Function resolution → gating

```
app: cuLibraryGetModule(lib) → module (inherits lib's spliced-name set)
app: cuModuleGetFunction(module, "name") → CUfunction
   if name ∈ spliced set  → gate as ATOMIZABLE   (g_atom_funcs)
   else (from atomized module, name not spliced)  → FULL-RANGE   (g_full_funcs)
handle with no name (cuLibraryEnumerateKernels) → cuFuncGetModule + cuFuncGetName,
   then classified as above, memoised per handle
```

Gating by name (not just by module) means a module with one un-spliceable kernel
still atomizes the rest, and a spliced-but-unrecognised kernel is still handled
safely (full range).

### 4.3 Launch → atom dispatch

```
app: cuLaunchKernel[Ex](f, grid, ...)  ─▶ interpose.c ─▶ lithos_submit_launch[_ex]
   dispatch.c     : if LITHOS_DISPATCH: deep-copy args, enqueue, return to app;
                    a dispatcher thread later runs the rest of this path
   sched_stream.c : find/create this stream's launch queue          (Fig.9 (1))
   tpc_alloc.c    : quota (+ stealing, − busy TPCs) → disable mask → qmd_set_next_mask (2)
   tpc_alloc.c    : optional right-sizing may narrow the mask             (§5.5)
   predict.c      : operator = ordinal k on this queue; pred_us = lookup   (§5.7)
   sched.c        : optional outstanding-work throttle                    (§5.3)
   atomizer_dispatch[_ex]:
     if f is ATOM:
        n = decide_atoms(grid, pred_us)       # bounded by overhead floor + SLO cap
        lock; [wait(g_atom_ev) unless single-stream or capturing]
        for each atom [lo,hi):
           [wait for atom i−W if LITHOS_ATOMS_INFLIGHT=W]
           write {lo,hi} to g_meta            # 2x cuMemsetD32Async, elided if unchanged
           lithos_apply_atom_mask(...)        # one-shot mask: per-atom TPC set
           relaunch full grid                 # range check skips out-of-range blocks
        [record(g_atom_ev)]; unlock
     elif f is FULL: write {0, grid}; relaunch once
     else: forward verbatim
   predict.c      : record completion event for the Tracker; coord: publish TPC timers
```

Cooperative / cluster launches are never split (a grid-/cluster-wide barrier
needs every block live); they get a full-range write and one launch at their exact
quota.

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
  emit it; only SASS can. LithOS most likely gets it from Rust/LLVM guaranteed tail
  calls — an inference on our part: the paper states only the QMD program-address
  redirect and defers the low-level details to an unpublished technical report.
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
   at n=1 it adds two metadata `memset` nodes per kernel (+114 % replay on an
   8-kernel chain: 27.5 → 58.9 µs).
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
     different TPC allocation each replay — cost 7–14 µs per *changed* subgraph
     (7.3 µs for a 2-node subgraph, 14.1 µs for 8), paid only on change; steady replay
     costs +14 % (vs +114 % in-graph). `cuGraphExecUpdate` can't do this (the
     SM mask isn't a graph-API param), which is why re-instantiation is the path.

### 5.7 Atom sizing: derived bounds

The paper leaves `atom_duration` a tuned constant — "limits of 250–500 µs are
effective" — and only warns that a value set too low can make the kernel *slower*.
Here `n = floor(predicted_duration / atom_duration)` (so a kernel under 2×
`atom_duration` is not split), and `atom_duration` itself is **bounded on both
sides** (`effective_atom_us`):

* **Dynamic adjustment for large grids** — `blocks > 4096 → atom_duration × 2`, the
  paper's "aggressiveness control" for kernels with many thread blocks.
* **Floor** = `LITHOS_ATOM_COST_US / LITHOS_ATOM_MAX_OVERHEAD` (defaults 30 µs / 0.10
  = **300 µs**). Splitting into *n* atoms costs about `n × atom_cost` (each atom is a
  full-grid relaunch whose out-of-range blocks reach the prologue and exit), so the
  floor turns the paper's caveat into an invariant: splitting never costs more than
  the chosen fraction of the kernel. The cost is **hardware-dependent and the most
  important knob**: the original 5 µs default came from an A6000 microbenchmark and
  split real PyTorch kernels ~88 ways for a 15× throughput loss; on the A100 it
  measures 20–40 µs, and the cost is also learned per kernel.
* **Ceiling** = `LITHOS_SLO_US` (opt-in). A co-located latency-critical tenant waits
  behind at most **one atom**, so a neighbour's latency budget bounds atom size
  directly.
* Kernels below `LITHOS_MIN_BLOCKS` (8) are not split — the paper's "disable
  atomization for kernels with many short threads".

### 5.8 Per-atom TPC allocation and the in-flight bound

Because each atom is a separate relaunch and the QMD mask is one-shot, every atom
can carry its own mask — the paper's "TPC allocations can be dynamically adjusted
throughout a kernel's execution" (§5.4). By default each atom re-applies its stream's
mask (which also fixes a latent bug: without re-arming, only atom 0 was confined).
Opt-in variants: `LITHOS_ATOM_TPC=W` tiles atoms across distinct contiguous W-TPC
slices of the tenant's span; `LITHOS_ATOM_TPC_LIST=1,2,3` gives atom *i*
`list[i % n]` TPCs. Both stay inside the tenant's quota, and every block still runs
exactly once.

Recomputing each mask from live coordinator state is necessary but not sufficient
for the paper's Figure 10(c) (a kernel's allocation changing while it runs): with all
atoms submitted microseconds apart, all N decisions are made before the kernel has
begun. `LITHOS_ATOMS_INFLIGHT=W` bounds the atoms of one kernel outstanding — atom
*i* is not submitted until atom *i−W* completes — so later atoms' allocations react
to a tenant arriving mid-kernel (`tests/test_fig10c.sh`, with the unpaced run as the
control). It is also §5.3's "limits outstanding atoms", which the once-per-kernel
throttle is not. Off by default: the submitting thread waits on the GPU inside the
atom loop.

---

## 6. Framework interception (the CUDA-runtime path)

ML frameworks sit on the CUDA **Runtime** (`libcudart`), which reaches the driver
in ways that bypass naive interposition. Four discoveries were needed:

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

4. **Kernel handles can arrive without a name.** Intercepting the APIs that hand out
   handles by name (`cuModuleGetFunction`, `cuLibraryGetKernel`, `cuKernelGetFunction`)
   is not enough: CUDA 12 also offers **bulk enumeration**
   (`cuLibraryEnumerateKernels`, `cuLibraryGetKernelCount`), which returns handles
   with no names at all, and cuFFT uses it. Its kernels were treated as "not ours"
   and ran at **0 % coverage** — while still computing the right answer, so nothing
   looked broken. An unrecognised handle is now classified by asking the driver
   (`cuFuncGetModule` gives its module, `cuFuncGetName` its name), once per distinct
   handle; this closes the whole class, including APIs that do not exist yet.
   `LITHOS_DIAG=1` shows these as `byHandle`.

(The runtime also pulls the QMD **callback table** via `cuGetExportTable` — the
same UUID `2c8e0ad8…` that `qmd.c` uses — which is why TPC masking already fires
for framework launches.)

Robustness for real kernels: per-kernel splice (skip the odd one), never split
cooperative/cluster launches, `LITHOS_QUOTA` also attaches to the default stream so
framework work is confined, and the multi-stream safeguards of §5.5.

---

## 7. Reverse-engineering findings

* **QMD/TMD layout** (Ampere/Ada `QMDV03_00`, measured on the A6000; the A100's
  `QMDV02_04` uses the same mask offsets): TMD version at byte 72;
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

## 8. Configuration reference

All behaviour is controlled through environment variables, parsed in `src/config.c`
(a few per-subsystem ones in `sched_stream.c`, `dispatch.c`, `qmd.c`, `graphsched.c`).

**Diagnostics — read these first**

| variable | default | meaning |
|---|---|---|
| `LITHOS_STATS` | 0 | print the coverage summary at exit: distinct kernels by class, launches by class, and **atomization coverage**. The first thing to check when LithOS appears to do nothing |
| `LITHOS_STATS_FILE` | — | also dump that summary periodically to `<file>.<pid>`, so coverage survives a subprocess that is *killed* (e.g. vLLM's EngineCore) |
| `LITHOS_DIAG` | 0 | why coverage is what it is: image kinds and splice time, each kernel's class and how it was decided (by name / `byHandle`), and every kernel that ran un-atomized |
| `LITHOS_VERBOSE` | 0 | log interposition / scheduler / atomizer activity |
| `LITHOS_LOG_MASK` | 0 | log the TPC disable-mask applied to each launch |
| `LITHOS_LOG_PREDICT` / `LITHOS_PREDICT_ACC` | 0 | per-operator latencies / predicted-vs-measured pairs per launch |
| `LITHOS_LOG_COORD` / `LITHOS_LOG_DVFS` / `LITHOS_LOG_GRAPH` / `LITHOS_LOG_CB` | 0 | coordinator, DVFS decisions, graph partitioning, every QMD callback |

**TPC scheduling (§5.1–5.3)**

| variable | default | meaning |
|---|---|---|
| `LITHOS_QUOTA` | −1 | per-stream TPC quota (compute quotas) |
| `LITHOS_TPC_BASE` | 0 | first TPC of this process's range (only needed when the coordinator is off) |
| `LITHOS_COORD` | 1 | join the system-wide tenant table: cross-application quotas/stealing, automatic disjoint ranges. 0 = per-process only |
| `LITHOS_PRIORITY` | 0 | this tenant's priority (larger = higher); a higher-priority tenant's TPCs are never stolen |
| `LITHOS_STEALING` | 1 | enable TPC stealing from idle streams/tenants |
| `LITHOS_TPC_TIMERS` | 1 | per-TPC timers: a steal skips any TPC whose timer has not expired. 0 = idle-only heuristic |
| `LITHOS_PERSTREAM_QUOTA` | 0 | give each stream its own disjoint slice (needed for intra-process stealing) |
| `LITHOS_THROTTLE` | 0 (1 when `LITHOS_DISPATCH=1`) | defer dispatch while in-flight µs > `LITHOS_OUTSTANDING_US` |
| `LITHOS_OUTSTANDING_US` | 100 | outstanding-work throttle threshold |
| `LITHOS_DISPATCH` | 0 | buffer launches and submit from dispatcher threads (§5.2) |
| `LITHOS_DISPATCH_THREADS` / `_DEPTH` / `_PRIO` | 2 / 1024 / 1 | worker count; max buffered launches per queue; priority-ordered selection (0 = arrival order) |
| `LITHOS_PRIO` | 0 | default priority for streams the app did not prioritise (lower = more important, as in CUDA) |
| `LITHOS_MPS` | 1 | auto-start the MPS control daemon |

**Kernel Atomizer (§5.4)**

| variable | default | meaning |
|---|---|---|
| `LITHOS_ATOMIZER` | 1 | enable module-load splice + per-atom launch |
| `LITHOS_ATOM_US` | 300 | target atom duration (µs); the paper uses 250–500 |
| `LITHOS_ATOM_COST_US` | 30 | cost of one extra atom; sets the duration floor with the next knob. Hardware-dependent (§5.7) |
| `LITHOS_ATOM_MAX_OVERHEAD` | 0.10 | cap splitting overhead at this fraction of the kernel (0 = no floor) |
| `LITHOS_SLO_US` | 0 | latency budget imposed on a co-located tenant; caps atom duration (0 = off) |
| `LITHOS_MIN_BLOCKS` | 8 | skip atomization below this grid size |
| `LITHOS_FORCE_ATOMS` | 0 | force an exact atom count (testing/policy) |
| `LITHOS_ATOMS_INFLIGHT` | 0 | bound the atoms of one kernel outstanding (§5.8); 0 = submit all at once |
| `LITHOS_ATOM_TPC` / `LITHOS_ATOM_TPC_LIST` | 0 / — | per-atom TPC slices of width W / of listed widths (§5.8) |

**Prediction, right-sizing, power (§5.5–5.7)**

| variable | default | meaning |
|---|---|---|
| `LITHOS_PREDICT` | 1 | online latency prediction; drives atom sizing, right-sizing, throttle |
| `LITHOS_PREDICT_BRACKET` | 1 | bracket each launch with a start event (correct, ~2× the event cost); 0 = gap method |
| `LITHOS_RIGHTSIZE` | 0 | per-kernel TPC right-sizing |
| `LITHOS_RIGHTSIZE_OCC` | 1 | apply the occupancy filter; **0 leaves the `l = m/t + b` model alone**, which is what honours the slip budget (§10) |
| `LITHOS_SLIP` | 1.1 | right-sizing latency slip `k` |
| `LITHOS_DVFS` | 0 | transparent power management; **needs root** (implies `LITHOS_PREDICT`) |
| `LITHOS_DVFS_SLIP` | 1.1 | DVFS slowdown budget |
| `LITHOS_DVFS_PROBE` | 0.75 | probe clock as a fraction of `f_max` |
| `LITHOS_DVFS_SAMPLES` | 8 | measurements per operator before a phase advances |
| `LITHOS_DVFS_SWITCH_MS` | 500 | minimum gap between frequency transitions |

**CUDA graphs (§6)**

| variable | default | meaning |
|---|---|---|
| `LITHOS_GRAPH_SUBGRAPHS` | 0 | partition each instantiated graph into K subgraphs with per-subgraph TPC allocation (0 = atomize-in-graph) |
| `LITHOS_SUBGRAPH_ROTATE` | 0 | demo: rotate each subgraph's allocation every replay (exercises re-instantiation) |

**Debug / reverse-engineering** (not for normal operation): `LITHOS_TRACE_GPA` logs
every `cuGetProcAddress` lookup — the quickest way to tell whether the runtime is
resolving through us; `LITHOS_TRACE_ET` traces `cuGetExportTable`;
`LITHOS_QMD_PROG_OFF` and `LITHOS_CORRUPT_OFF` override/probe QMD byte offsets.

---

## 9. Validation

### Built-in suite (`make run_tests`, all pass)

| test | what it exercises |
|---|---|
| `test_interpose_driver` | interposition via `LD_PRELOAD` (driver API) |
| `test_interpose` | interposition via the `libcuda.so.1` wrapper (runtime) |
| `test_atomize_cubin` × 3 | atomizer on a raw **CUBIN**, compressed **FATBIN**, and **PTX** |
| `test_atomize_runtime` | the CUDA-runtime-API launch path (frameworks use this) |
| `test_atomize_graph` | CUDA-graph capture → atomized subgraph → correct replay |
| `test_scheduler` | `LITHOS_QUOTA=4` confines a kernel to 8 SMs |
| `test_stealing` × 2 | **TPC stealing**: with disjoint per-stream quotas an idle stream's TPCs are lent to a busy one (8 → 16 SMs); run with stealing off and on |
| `test_tpc_timers` × 2 | **per-TPC timers**: a stream that submitted a long kernel and went quiet must not have its TPCs stolen; run with timers on (must refuse) and off (the control, or the test stops discriminating) |
| `test_dispatch_order` | **dispatcher ordering**: no memcpy, event or query can overtake a launch still in a queue |
| `test_dispatch_mt` | **dispatcher submission**: every worker actually submits the launches it takes. Must be a CUDA-*runtime* app on the **default stream** — an explicit stream still succeeds without a current context, which hides the failure |
| `tests/test_atom_mask.sh` | **every atom carries the TPC mask** on both `cuLaunchKernel` and `cuLaunchKernelEx` (counts masks against QMD uploads) |
| `tests/test_fig10c.sh` | the paper's **Figure 10(c)**: a kernel's TPC allocation changes mid-execution (borrowed TPCs returned for the remaining atoms); the unpaced run is the control |
| `tests/test_dvfs.sh` | **power management**: applied clock follows `f_max/(1+slip/S)`; predicted sensitivity matches ground truth on a compute-bound kernel (needs root; without privilege it fails safe and the engaged path is skipped) |
| `tests/test_coord.sh` | **system-wide coordinator**: disjoint cross-application ranges, an idle tenant's TPCs borrowed, a higher-priority tenant never robbed, and simultaneous joins never share a range (repeated 6×, since the race is intermittent) |
| `correctness_matrix` | 7 kernel patterns (elementwise, tiled matmul, atomics, 3-D grid, transpose) vs a CPU reference under forced max-split — the atomic kernels are the exactly-once guards |

`run_tests` is not enough on its own. `bash bench/fw_all.sh` validates real
libraries end-to-end — cuBLAS, cuFFT, PyTorch and Triton × 8 configurations
(default, `ATOM_US=1`, `QUOTA=16`, right-sizing, dispatcher, dispatcher with 4
workers, predictor, all at once), **32/32 passing at 100 % coverage** on the A100
(also re-run on driver 560.35.05 / CUDA 12.6 with PyTorch 2.14 and its bundled
Triton). Each target reaches the GPU a different way: cuBLAS via fatbin kernels
fetched by name, cuFFT by bulk enumeration (unnamed), PyTorch via cuBLAS + cuDNN +
ATen + autograd + CUDA graphs, Triton via cubins JIT-compiled at run time. Every bug
in §12 marked "framework-only" passed the unit suite and was caught here — the
failure mode was a system that looked healthy while doing nothing.

### Frameworks — correctness **and** atomization coverage

Measured with `LITHOS_STATS_FILE` (per-launch split / single-atom / full-range /
**un-atomized** counts). These are *coverage and correctness* results, measured on
the RTX A6000 (drivers 570.195.03 / 595.71.05), not performance numbers.
TensorFlow, JAX, TensorRT and vLLM remain A6000-only; PyTorch, Triton, cuBLAS and
cuFFT were re-run on the A100.

| framework | result | coverage (0 un-atomized = all kernels atomized) |
|---|---|---|
| PyTorch 2.6 (+ `torch.cuda.CUDAGraph`) | ✅ correct, 5/5 | **100%** — 58 kernels, 55/55 launches |
| TensorFlow 2.21 | ✅ correct | **100%** — 69 kernels, 215/215 |
| JAX 0.6 (XLA) | ✅ correct, 10/10 | **100%** — 63 kernels, 240/240 (202 split) |
| TensorRT 10.7 | ✅ checksum bit-identical | **100%** — 11,825 kernels (autotuning), 17464/17464 |
| cuDNN 9 (direct `cudnnConvolutionForward`) | ✅ checksum bit-identical | **100%** — 6/6 |
| vLLM 0.8.5 — eager | ✅ correct | **100%** — 422 kernels, 512/512 |
| vLLM 0.8.5 — CUDA graphs | ✅ correct, output == baseline | **100%** — 439 kernels, **1397 launches split**, 3328/3328 |

Loading was never the hard part; *recognising the kernel handle at launch* is (§6,
gap 4), which is why coverage is reported next to every correctness result.

---

## 10. Results and findings

All measurements are from the current system on the **A100 80GB PCIe** (54 TPCs),
CUDA 12.6–12.8, best-of-N or median-of-N as noted; where two node re-measurements
disagreed the later one is used. Harnesses and exact commands are in
[`bench/README.md`](../bench/README.md).

### 10.1 What each mechanism is worth

| mechanism | measured effect | cost |
|---|---|---|
| **Interposition** | transparent on both paths | ≲ 0.2 µs/launch |
| **TPC quotas** (spatial isolation) | co-located HP keeps **99 %** of solo throughput and **1.04×** solo tail, vs 29 % / 3.4× under plain MPS | ~0 µs/launch |
| **Kernel atomization** | HP p99 **10 577 → 453 µs (23×)** against a 10.8 ms BE kernel; residual ≈ one atom | +0.26–0.33 µs/launch, ~⅓ of BE throughput |
| **Right-sizing** | model `l = 23.88/t + 0.0285`, R² = 1.0000 on one tiled matmul (best case); **R² = 0.9267** time-weighted across a real PyTorch model's operators; frees **28.7 %** of TPCs at the paper's 1.1 slip for **+10.7 % p99** (52.0 % at 1.5) — only with `LITHOS_RIGHTSIZE_OCC=0` | within noise |
| **Latency predictor** | **0.00 %** misprediction on a real PyTorch model, p99 error 2.8 µs (paper: 0.9 % / 0.38 %, p99 49/31 µs on variable-shape LLM inference — a harder case) | **≈ +10 µs/launch — the dominant per-launch cost**, doubled from +5 µs by bracketing each launch (the fix that made it correct) |
| **TPC stealing** | intra-process **1.92×**; cross-application **1.99×** (311 → 620 it/s). But equal-priority stealing **inverts priority**: HP p99 62.5 → 1 650 µs (26× worse); the priority safeguard fixes it (→ 55.7 µs) while keeping work conservation | ~0 |
| **Coordinator** | cross-app quotas + stealing + the priority safeguard that makes stealing safe | ~0 (shared-memory reads) |
| **Power management** (opt-in) | real PyTorch kernel mix: **−25.3 % energy** at best (slip 1.3), **−18.2 %** at the paper's slip 1.1 for +6.7 % p50 (paper: mean 26 % at +7 % p99). Two thirds of that workload's runtime is in kernels with `s < 0.5` | needs root; the probe phase slows every kernel that runs during it |
| **Graph subgraphs** | per-subgraph TPC allocation; reallocation 7–14 µs per changed subgraph | +14 % replay (vs +114 % in-graph) |
| **Module splice** | linear in kernel count | 0.25 ms LithOS-side for 600 kernels |
| **Throttle** (opt-in) | ~20 % HP tail gain against a deep queue of *short* kernels at 5× BE throughput cost; nothing against long kernels | ~0 under the limit |
| **Dispatcher** (opt-in) | deferral works (100 % buffered, ordering preserved) but does not improve tail latency here | tens of µs/launch |

The per-launch balance shifted during optimization: the atomizer used to dominate
(~5 µs) and is now nearly free, while the **predictor** — on by default — is the main
cost. `LITHOS_PREDICT=0` gets it back; `LITHOS_PREDICT_BRACKET=0` gets half back at
the cost of measuring host stalls as kernel time. (The paper reports 4 % end-to-end
overhead on real inference models; nothing here is measured on that basis.)

**Two mechanisms do not earn their keep here, for one reason.** Both act on
*submission*, and submission is not the binding constraint on this hardware. The
throttle bounds queue depth, which cannot fix head-of-line blocking from one long
kernel (no preemption); the dispatcher defers work so it can be reordered, but HP is
waiting on the kernel already executing — in a deep-backlog test HP p50 is ~5 700 µs
in *every* configuration including native CUDA. Atomization fixes exactly this by
shrinking the in-flight unit. This is not a refutation of the paper: there, deferral
keeps work uncommitted so a cross-tenant scheduler can still decide; this
reproduction applies those decisions inline at launch, so deferral has nothing left
to buy. Now that the coordinator exists, holding a low-priority launch until a
central decision arrives is the natural next test. Both stay opt-in.

### 10.2 When is LithOS worth enabling?

**The interfering kernel's duration decides it.** Sweeping a best-effort kernel's
length beside a latency-critical job (`bench/usecase.py`, `LITHOS_SLO_US=300`):

| BE kernel duration | HP p99 un-atomized | HP p99 atomized | HP gain | BE throughput kept |
|---|---|---|---|---|
| 0.04 ms | 28.6 µs | 32.0 µs | **1× (none)** | 96 % |
| 0.24 ms | 130 µs | 90 µs | 1.4× | 65 % |
| 2.2 ms | 1 869 µs | 257 µs | **7×** | 62 % |
| 10.8 ms | 10 473 µs | 360 µs | **29×** | 67 % |

* **Short kernels (< ~0.1 ms): do not atomize** — there is no head-of-line blocking to
  remove (`LITHOS_MIN_BLOCKS` and the overhead floor implement the paper's caveat).
* **Long kernels (> ~2 ms): atomize** — 7–29× tail reduction for about a third of BE
  throughput.

Where it pays: **latency-critical inference co-located with training** (training
kernels are milliseconds; the paper measures DLRM > 30 ms), **multi-tenant inference
under a tail SLO** (partitioning alone suffices), and **capacity reclamation** via
right-sizing. Where it does not: uniformly short-kernel workloads, single-tenant jobs
that own the GPU, and graph-replay-dominated workloads (use `LITHOS_GRAPH_SUBGRAPHS`).

**Whether atomization pays turns on how much of the penalty is head-of-line
blocking.** A co-located service loses latency by waiting behind an uninterruptible
kernel (blocking) and by holding fewer TPCs (sharing); atomization addresses only the
first. Measured both ways: with synthetic tenants (a few 7 ms GEMMs saturating the
device, sharing donated TPCs) it takes the tail from 6.25× to **2.01×** of ideal; with
the paper's own models (ResNet-50 served against VGG-19 training) it makes the tail
*worse* (3.33× → 4.25×), because a 15× increase in the neighbour's kernel length
(0.77 → 11.7 ms) costs the service only 1.23× — blocking was a small part of a ~2.9×
penalty. Measure the split before enabling it. Also: atomization only helps when
tenants *share* TPCs; with disjoint (fenced) ranges there is no blocking left and
splitting only adds overhead.

### 10.3 Reproducing the paper's results

Every experiment that exercises a mechanism reproduces the paper's qualitative
result, in the right ballpark. ✅ matches · ◑ same direction, different magnitude.

| paper result | paper | here |
|---|---|---|
| MPS work conservation | MPS sets the throughput bar | 4.05× over time-slicing (14 410 vs 3 559 it/s) ✅ |
| Spatial isolation beats MPS (§8.1) | MIG-like isolation at full throughput | HP 99 % of solo, 1.04× tail vs 29 % / 3.4× under MPS ✅ |
| MPS is worst for latency (§8.1) | 5.83× ideal, service throughput 60 % | **66×** ideal / 38 % throughput with the paper's models (812× / 31 % with more adversarial synthetic tenants) ◑ |
| LithOS vs MPS on tail (§8.1) | 4.7× avg, 13.54× max | **22×** with the paper's models (~401× synthetic) ✅ |
| LithOS keeps service throughput (§8.1) | within 1 % of load | 97–98 % of offered load ✅ |
| LithOS keeps the service tail (§8.1) | within 1.2× ideal | 2.01× (synthetic) – 2.99× (paper's models) ◑ |
| Atomization on the TPC scheduler (§8.4) | 1.38× → 1.19× ideal | workload-dependent: 6.25× → 2.01× (synthetic) but 3.33× → 4.25× (paper's models) ◑ |
| Atomization cuts HoL blocking (§8.4) | tail within 14 % of ideal | 23× tail reduction, residual = one atom ✅ |
| Kernel latency vs batch (Fig. 11) | grows into the multi-ms range | VGG-19 p99 0.77 → 11.7 ms across batch 16 → 256 ✅ |
| Predictor accuracy (§8.4) | 0.9 % / 0.38 % | 0.00 %, p99 error 2.8 µs ✅ (an easier, fixed-shape workload) |
| Right-sizing model accuracy (§8.2) | R² 0.92–0.99 (time-weighted, all kernels) | R² = 0.9267 on a real model ✅ (1.0000 on one matmul is best case, not like-for-like) |
| Right-sizing capacity savings (§8.2) | mean 26 %, up to 51 %, at +4 % p99 | 28.7 % at slip 1.1 (52.0 % at 1.5) ✅ savings, but for +10.7 % p99 ◑ |
| DVFS energy savings (§8.3) | mean 26 %, up to 46 %, at +7 % p99 | −25.3 % best; −18.2 % at slip 1.1 for +6.7 % p50 ✅ |

The **ordering of every system** — the paper's central claim — reproduces: MPS
maximises throughput and destroys isolation, time slicing protects throughput but not
the tail, LithOS gets both. Magnitudes differ mostly because the synthetic pair (a
0.45 ms service against 7 ms saturating GEMMs) is more adversarial than anything the
paper runs, and every ratio against a sub-millisecond baseline inflates accordingly.

**Not reproduced:** the multi-system comparison (MIG / REEF / TGS / Orion — needs
those systems) and real serving-stack SLO attainment (needs Triton Inference Server,
TensorRT-LLM and the paper's model set; synthetic and single-framework HP/BE proxies
stand in for the stacking experiments).

### 10.4 Things the paper doesn't say that we had to find out

* **CUDA graphs freeze the TPC allocation.** The QMD pre-upload callback — the only
  hook for SM masking — fires exactly once per graph exec, so a replay's allocation
  cannot change; only re-instantiation re-fires it. Hence the subgraph model: rebuilding
  one small subgraph is cheap (~8 µs), rebuilding the whole graph is not.
* **Atom size *is* the tail latency.** A co-located tenant waits behind at most one
  atom; measured tail tracked atom size almost exactly. That motivates deriving
  `atom_duration` from the neighbour's SLO and a floor from measured overhead (§5.7).
* **`%smid` is renumbered under a mask** (0…2w−1 regardless of the base TPC): the SM
  *count* is truthful, the identity is not — the **mask** is ground truth for *which*
  TPCs a launch got.
* **A transparent interposer *can* defer a launch**, via `cuFuncGetParamInfo`
  (§3, `params.c`).
* **Warm-up launches masquerade as operators.** With no synchronize between framework
  warm-up passes the ordinal never resets, so every warm-up launch is recorded as a
  distinct operator seen once (885 of 1 106 in a ResNet-50 sweep). Anything computed
  per operator must exclude them — including them dragged right-sizing R² from 0.9267
  to 0.9067.
* **A filtering heuristic that `return`s is an allocator.** Right-sizing's occupancy
  bound is an *upper bound* on useful width, not an estimate of sufficient width; taking
  it as the answer bypassed the model, so a 1.1 slip produced +121 % p99. Skipping it
  (`LITHOS_RIGHTSIZE_OCC=0`) delivers +10.7 % for a 10 % budget.
* **§5.6's formula is a starting point** (§3, `power.c`), and **a slip parameter is
  not a "more is better" knob**: DVFS energy is non-monotonic in it — on the
  compute-bound kernel slip 1.1 saves 12.9 %, 1.3 saves 21.6 %, 1.5 saves 17.1 % and
  2.0 saves only 1.7 % because the added runtime outweighs the lower power.
* **Stealing needs disjoint ranges to do anything.** With one range per application
  every stream shares it and lending your own range to yourself is a no-op; the
  coordinator hands *tenants* disjoint ranges, which is what makes cross-application
  stealing real.

---

## 11. Fidelity to the paper

Legend: ✅ faithful · ≈ divergent (same goal, different mechanism) · ◑ simplified ·
✗ not implemented · ? paper unspecified. Every "the paper does X" is a paper quote;
where the paper is silent it is marked so.

| Area | Paper | This reproduction | |
|---|---|---|---|
| Language / size | ~5000 lines Rust, macro-generated interposition of the **entire** Driver API | ~6,900 lines C/C++/CUDA, 36 hand-picked overrides + 55 ordering barriers; the rest forwarded via `cuGetProcAddress` | ≈ |
| Interposition point | Driver API, in-process | same | ✅ |
| Deployment | native + **containers** | `LD_PRELOAD` or `libcuda.so.1` wrapper; containers untested | ◑ |
| TPC masking (QMD) | Ampere, Hopper, Ada; extends libsmctrl | verified on GA102 and GA100 across four drivers. Hopper offsets present but **untested**, and the mask is 64-bit while H100 has 66 TPCs | ◑ |
| Compute quotas, stealing | guaranteed TPCs; idle **applications** lend TPCs | QMD mask; **cross-application** via the coordinator, with the priority safeguard | ✅ |
| Per-TPC timers | avoid stealing from long-running TPCs | implemented, visible across applications, horizon accumulates with a 200 ms cap | ✅ |
| Launch queues / dispatcher | per-stream queues, always-on; enqueue and return | true fire-and-forget (deep-copied args), but **opt-in** — default build submits inline, the inverse of the paper's default | ◑ |
| Outstanding-work throttle | 100 µs sync-queue throttle | enforced (`LITHOS_THROTTLE`); with the dispatcher off the calling thread waits instead of the dispatcher | ✅ |
| Limits outstanding atoms | yes | `LITHOS_ATOMS_INFLIGHT` (opt-in) | ✅ |
| **Lower hw stream priority for stolen work** | yes | **not implemented** — the priority safeguard is admission-side only; once TPCs are lent the borrower competes at full priority | ✗ |
| Duration predictor | §5.7, keyed by ordinal `k` | operator-indexed, event-measured, EMA + TPC-scaled, linear-scaling fallback | ✅ |
| Right-sizing | occupancy filter + `l = m/t + b`, slip `k` | both stages, but the filter pre-empts the model unless `LITHOS_RIGHTSIZE_OCC=0` | ◑ |
| Power management | virtualized DVFS, `f_final = f_max/(1+k/S)`, conservative learning | same model; formula picks the start point, then a closed loop on observed slowdown (§5.6's operational text) | ✅ |
| **Atomizer control transfer** | QMD program-address → Prelude that **jumps** into the original | **prologue splice / fall-through** — no Prelude, no jump, no redirect | ≈ |
| When atomization applies | at launch (patch live QMD) | at **module load** (ELF surgery on the cubin) | ≈ |
| Per-atom metadata delivery | `AtomMetadata` struct; mechanism unspecified | shared device buffer via `cuMemsetD32Async` + event serialization | ? |
| Atom sizing | tunable 250–500 µs constant, adjusted dynamically for large grids; warns too-small can be slower | ×2 for > 4096 blocks; floor `atom_cost/max_overhead` makes the warning an invariant; optional SLO ceiling | ✅+ |
| Algorithm 1 range gate | per-block range check | computes the flat block index and tests `lo ≤ b < hi` — exact | ✅ |
| CUDA graphs | "atomize graphs into subgraphs, ensuring correct execution ordering" | both readings: in-graph atomization (default) and `LITHOS_GRAPH_SUBGRAPHS=K` partitioning with runtime re-instantiation | ✅≈ |
| Special kernels | disable stealing+atomization; report allocated SM count | not split, exact quota, `MULTIPROCESSOR_COUNT` spoofed to `quota×2`; non-cooperative persistent kernels undetectable | ✅≈ |
| Hopper thread-block clusters | atoms are multiples of cluster size | cluster launches detected and not split; no cluster-multiple sizing | ◑ |
| Multi-tenant coordination | system-wide view across applications (§5.1) | shm tenant table (`coord.c`); no richer policy layer (fair-share, preemption, arbitration) | ✅ |
| MPS | built on MPS | auto-started | ✅ |
| Fault isolation / driver reinit (§4.1) | yes | absent | ✗ |

### The one load-bearing divergence

**Paper (§5.4, §6, Algorithm 1):** launch the original kernel, then "patch the QMD's
program address to point to the Prelude"; the Prelude range-checks the block and
transfers control into the original kernel's entry. **That transfer is not
reproducible from `ptxas` output** (§5.1). We splice the range-check into each
kernel's own `.text` at module load, so in-range blocks fall through — Algorithm 1's
semantics with the transfer deleted. The claim that LithOS gets its jump from
Rust/LLVM guaranteed tail calls is **our inference**: the paper never mentions tail
calls, LLVM, SASS, cubins or module loading, and defers low-level details to an
unpublished technical report.

The costs of that choice: a burden the paper's design does not carry (intercept every
image-loading path, repair every offset the shift invalidates, and — the part that
bit — recognise a `CUfunction` at launch as one whose module we spliced); the gate
stays in the kernel forever, including for launches LithOS chooses not to atomize
(hence ungated spliced kernels are forced to an explicit full range); and the prologue
needs 8 registers, so kernels declaring fewer are bumped to 8 (occupancy-neutral in
practice).

### Where the reproduction goes past the paper

* **Derived atom bounds** (§5.7) — the paper's constant with a warning becomes an
  invariant plus an SLO-driven ceiling.
* **CUDA graphs** — the paper spends one sentence on them; the once-per-exec callback
  constraint it does not mention forces either re-instantiation or a frozen schedule,
  and both are implemented and measured.
* **By-handle kernel classification** — closes a whole class of interception failure.
* **Ordering barriers** — nothing overtakes a buffered launch (the paper's
  "auto-generated rest of the API" role, made explicit).
* **Cross-driver evidence** — QMD offsets and callback-table indices hold across
  drivers 560.35.05, 570.195.03, 595.71.05 and 610.43.02 on two Ampere dies.

---

## 12. Limitations, version-sensitivity and gotchas

* **Driver/arch-sensitive constants** live in `qmd.c` (QMD byte offsets 84/88/192/81
  and the `cuGetExportTable` indices 3/6). The QMD offsets are tied to the QMD
  *architecture* version more than the driver build; the export-table indices are the
  most driver-fragile piece (they come from libsmctrl). Both are overridable
  (`LITHOS_QMD_PROG_OFF`). **Cross-driver validation — done:** an A6000 upgraded in
  place from 570.195.03 to 595.71.05 (a CUDA-13-era driver, five branches newer) kept
  every component working unchanged; a second A6000 with the newest CUDA-13 libraries
  (torch 2.11+cu130, vLLM 0.25.1, TensorRT 10.15, TF 2.21) hit 100 % coverage after
  two userspace fixes — Zstandard fatbins and an ELF bounds-check guard for the tiny
  stub cubins cuBLASLt loads. The driver-level reverse-engineering needed no change.
* **Hopper is untested**, and the 64-bit mask means TPCs 64+ (H100 has 66) cannot be
  masked without widening it; `COORD_MAX_TPCS` is likewise 64.
* **Metadata is per-process**, so multi-process tenancy has no race; within a process,
  concurrent streams are handled by the event serialization of §5.5. Concurrent
  replays of *different* atomized CUDA graphs can still contend on the single `g_meta`
  (per-launch metadata as a kernel parameter would fix it but was unnecessary for
  every workload tested).
* **Special kernels.** Cooperative and cluster launches are never split. A kernel with
  genuine cross-block dependencies launched normally (no cooperative flag) is
  undetectable and would be a correctness risk if split, as the paper notes for
  "cross-block synchronization or persistent kernels".
* **Un-splicable cubins** — a self-referential `REL` relocation, or a malformed/tiny
  stub cubin — run verbatim: correct, just not atomized.
* **Explicitly built graphs** (`cuGraphAddKernelNode`) are not atomized by the in-graph
  mode, which sees only stream capture; the subgraph mode handles them, since it
  partitions at `cuGraphInstantiate`.
* **Privilege.** Setting GPU clocks is root-only (an unprivileged process is refused
  even after `nvidia-smi -acp UNRESTRICTED`, which is inert on recent drivers), so
  `LITHOS_DVFS` needs `sudo env LD_PRELOAD=… ./app`.
* **MPS gotchas.** MPS refuses the default log directory (`/var/log/nvidia-mps`) and
  won't start — set `CUDA_MPS_PIPE_DIRECTORY` / `CUDA_MPS_LOG_DIRECTORY`. MPS wedges
  if a client is killed mid-kernel, after which every CUDA process hangs, looking
  exactly like a LithOS deadlock; recover with
  `echo quit | nvidia-cuda-mps-control; pkill -f nvidia-cuda-mps`.
* **Arch-mismatched cubins fail silently** as 0 % coverage; rebuild benchmark cubins
  (`make bench`) after changing GPU.
* **Version pinning.** In the CUDA-12.8 environment TensorRT 10.15 and vLLM 0.25 need a
  newer driver, so TensorRT 10.7-cu12 and vLLM 0.8.5 (torch 2.6-cu124) were used.

### Bugs that only a real workload exposed

Each passed the unit suite; the failure mode was silence.

* **`cuLaunchKernelEx` atoms escaped the quota.** The one-shot mask was re-armed per
  atom on `cuLaunchKernel` but not `cuLaunchKernelEx` (the runtime's path): atom 0 was
  confined and atoms 1…N−1 ran on the whole device. Guarded by `tests/test_atom_mask.sh`.
* **Per-TPC timers expired while the queue was full** — deadlines were measured from
  submission, so a tenant bursting 200 short kernels read as free almost at once.
  The horizon now accumulates.
* **cuFFT ran at 0 % coverage** (bulk-enumerated, unnamed handles) — fixed by the
  by-handle fallback (§6). *(framework-only)*
* **`cuCtxSetCurrent` binds per thread but its cache was process-wide**, so all but
  the first dispatcher worker submitted with no context and launches were dropped
  silently — surfacing in PyTorch as garbage gradients. Only legacy-default-stream
  launches fail, which is where frameworks put most work. *(framework-only)*
* **The coordinator's shm creation was a race** — "did I create it?" inferred from size
  zero let two simultaneous tenants zero each other and share a range. `O_EXCL` now
  decides. *(framework-only)*
