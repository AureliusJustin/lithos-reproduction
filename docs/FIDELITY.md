# Fidelity: this reproduction vs. the LithOS paper

An honest accounting of where this reproduction **matches** the LithOS paper
(SOSP), where it **diverges** (different mechanism, same goal), where it is
**simplified/stubbed**, where it is **not implemented**, and where the paper
**doesn't say enough to compare**. Written to avoid over-claiming — every "the
original does X" below is backed by a paper quote; anything the paper leaves
unspecified is marked as such.

Legend: ✅ faithful · ≈ divergent (same goal, different mechanism) · ◑ simplified
· ✗ not implemented · ? paper unspecified.

## Summary table

| Area | Paper | This reproduction | |
|------|-------|-------------------|--|
| Language / size | ~5000 lines **Rust**, macro-generated interposition of the **entire** Driver API | ~2000 lines **C/C++/CUDA**, hand-picked subset | ≈ |
| Interposition point | CUDA Driver API, no cross-address-space marshaling | same (in-process, driver API) | ✅ |
| API coverage | auto-generates the whole Driver API; implements a small subset | overrides ~two dozen calls by hand; forwards the rest via `cuGetProcAddress` | ◑ |
| Deployment | native + **containers** | `LD_PRELOAD` or `libcuda.so.1` wrapper; containers untested | ◑ |
| TPC masking (QMD) | reverse-engineered; Ampere, **Hopper**, Ada; extends libsmctrl | reverse-engineered; **Ampere only** tested (Hopper offsets present, untested) | ◑ |
| Compute quotas | guaranteed TPCs per tenant | `LITHOS_QUOTA` → QMD mask, verified | ✅ |
| TPC stealing | idle tenants lend TPCs | implemented (timestamp idle-detection) | ◑ |
| Launch queues / dispatcher | per-stream queues + dispatcher/tracker threads (Fig. 9) | per-stream launch queues; a **Tracker thread** reaps completions; a **Dispatcher thread** submits launches (`LITHOS_DISPATCH`, opt-in — a hand-off, since a transparent interposer can't safely snapshot `cuLaunchKernel`'s implicit-size args) | ◑→✅ |
| Outstanding-work throttle | 100 µs sync-queue throttle, event reaping | **enforced** (`LITHOS_THROTTLE`): defers dispatch until event-reaped in-flight µs < limit | ✅ |
| Duration predictor | a predictor (§5.7) | **online event-measured, operator-indexed** (ordinal k per launch queue, reset on sync), EMA-refined, TPC-scaled | ✅ |
| Hardware right-sizing | yes | **occupancy filtering heuristic + `l=m/t+b` scaling model + latency-slip `k`** (`LITHOS_RIGHTSIZE`, `LITHOS_SLIP`) | ✅ |
| Power management | yes | ✗ (out of scope) | ✗ |
| **Atomizer control transfer** | QMD **program-address → Prelude**, Prelude **jumps** into the original (Rust/LLVM tail call) | **splice the range-check into each kernel's own `.text`** so in-range blocks **fall through**; no Prelude, no jump, no QMD redirect | ≈ |
| When atomization is applied | at launch (patch the live QMD) | at **module load** (ELF surgery on the cubin) | ≈ |
| Per-atom metadata delivery | Prelude gets an `AtomMetadata` struct; **mechanism not specified** | shared device buffer written per launch (`cuMemsetD32Async`) + event serialization | ? |
| Atom sizing | target duration 250–500 µs | `LITHOS_ATOM_US`, fed by the **online predictor** (falls back to the stub only until an operator is learned) | ✅ |
| CUDA Graphs | "interpose graph creation APIs and atomize graphs into subgraphs, ensuring correct execution ordering" | **both** interpretations: (a) atomize kernels into an in-graph subgraph (default, correct but schedule frozen); (b) `LITHOS_GRAPH_SUBGRAPHS=K` partitions the graph into K subgraphs along a topological cut, each independently TPC-allocated (the paper's likely intent) | ✅≈ |
| Graph replay rescheduling | (unspecified) | TPC mask **cannot** change on replay — QMD pre-upload callback fires once/exec; reallocation needs re-instantiation (subgraph granularity makes it cheap) | ? |
| Hopper Thread Block Clusters | atoms are multiples of cluster size | cluster launches detected and **not split** (no cluster-multiple sizing) | ◑ |
| Special (cross-block-sync/persistent) kernels | disable stealing+atomization; report allocated SM count via `cuDeviceGetAttribute` | cooperative/cluster launches **not split** and given their **exact quota (no stealing/right-sizing)**; **`cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)` spoofed** to `quota×2` SMs; non-cooperative persistent kernels still undetectable | ✅≈ |
| Multi-tenant coordination | central scheduler assigns resources across tenants | **per-process**; `LITHOS_TPC_BASE` gives manual disjoint ranges (no central coordinator) | ◑ |
| MPS | "we build on top of MPS" | auto-starts MPS; concurrency + partitioning verified | ✅ |

---

## The one deliberate, load-bearing divergence: the atomizer transfer

This is the biggest difference and the reason the reproduction exists.

**Paper (§5.4, §6, Algorithm 1):** launch the original kernel, then *"patch the
QMD's program address to point to the Prelude"* so *"execution begins at the
Prelude while retaining the original kernel's resources."* The Prelude
range-checks the block and, for in-range blocks, **transfers control into the
original kernel's entry point**. In the real system that transfer is a **jump**,
which LithOS gets from Rust/LLVM **guaranteed tail calls**.

**Why we couldn't reproduce it as-is:** that jump is *not expressible in PTX*
(PTX has no computed branch), so no PTX-compiled path — Rust included — emits it;
it only exists at SASS. We reverse-engineered the SASS transfer exhaustively
(`sass-jump/`): `CALL`/`BRX` land on the original but fault on its `EXIT`
(return/convergence state the `EXIT` violates), and the only convergence-free
transfer (`BRA`) is immediate-encoded so it can't reach a runtime cross-module
address, and loaded code can't be patched. This is precisely the *"low-level
reverse-engineering ... deferred to a separate technical report"* the paper
mentions — we could not reproduce it by byte-patching `ptxas` output.

**What we do instead:** we **splice the range-check into each kernel's own
`.text` section** at module-load time, so in-range blocks **fall through** into
the original code — no separate Prelude, no control transfer, no QMD
program-address redirect. It is functionally the same per-block gate (Algorithm
1's semantics) reached by a different, transfer-free route.

**Consequences of this divergence:**
- We atomize at **module load** (cubin ELF surgery) rather than at **launch**
  (live QMD patch). This means we must intercept module loading (and unwrap
  fatbins/PTX), which the QMD-redirect approach would not need.
- Because the splice **shifts a kernel's `.text`**, we must fix up everything
  that encodes an offset into it — section/program headers, the `.nv.info.<fn>`
  instruction-offset attributes, and (for separately-compiled `-rdc=true` kernels)
  the kernel's **ELF relocations**: every `r_offset` is bumped, a self-referential
  RELA addend (e.g. the return address for a device-function `CALL`) is bumped,
  and local-label `st_value`s are bumped. We also splice **only kernel entries**
  (`st_other` bit `0x10`), never the `__device__` functions that get their own
  `.text.<fn>` — an `EXIT`-prologue on a *called* function would kill the thread
  instead of returning. Only the genuinely unfixable cases (a self-referential
  `REL`, whose addend lives inside the instruction, or malformed/tiny stubs) are
  skipped and run un-atomized; the QMD-redirect approach is oblivious to cubin
  structure and needs none of this.
- Per-atom metadata: the paper says the Prelude receives an `AtomMetadata` struct
  but **does not say how it's delivered per launch**. We use a shared device
  buffer + `cuMemsetD32Async` + cross-stream event serialization. This is *our*
  design; we cannot claim it matches the original, and it is the source of the
  ~4 µs/launch atomizer overhead measured in [BENCHMARKS.md](BENCHMARKS.md).

---

## Faithful mechanisms

- **Driver-API interposition** at the "lowest common denominator," in-process, no
  cross-address-space marshaling — same layer and philosophy as the paper.
- **QMD/TMD TPC masking** via the pre-upload debug callback (the libsmctrl
  mechanism the paper says it extends): mask at TMD+84/88 on Ampere. Verified to
  confine kernels to a TPC quota, including under MPS and on the framework path.
- **Compute quotas & TPC stealing** — the core §5.3 mechanisms, driven by the
  same QMD mask.
- **Atomization semantics** — a per-block range gate that runs the original for
  in-range blocks and skips the rest, dispatched as N per-atom full-grid launches.
- **CUDA Graphs** — two modes: atomize kernels into an in-graph subgraph on stream
  capture; or `LITHOS_GRAPH_SUBGRAPHS=K` partitions the graph into K subgraphs
  (topological cut) for per-subgraph TPC scheduling with runtime re-instantiation
  (the paper's likely intent). The subgraph mode works on any instantiated graph
  (captured or built), since it partitions at `cuGraphInstantiate`.
- **Building on MPS** for concurrent multi-tenant execution.

---

## Now implemented (§5.2–5.7 scheduler mechanisms)

The scheduler beyond quotas/stealing — previously stubbed — is now reproduced
([`src/predict.c`](../src/predict.c), [`src/sched.c`](../src/sched.c)):

- **Online latency predictor (§5.7).** Real, event-measured per-kernel latency,
  keyed by **operator ordinal `k`** on each launch queue (reset at sync/batch
  boundaries — the paper's DFG-node identification), EMA-refined, TPC-scaled. On by
  default (`LITHOS_PREDICT`); drives atom sizing, right-sizing, and the throttle.
- **Hardware right-sizing (§5.5)** (`LITHOS_RIGHTSIZE`). The **occupancy filtering
  heuristic** (blocks / occupancy-per-TPC via `cuOccupancyMaxActiveBlocksPerMultiprocessor`)
  plus the **`l = m/t + b` scaling model** fit from the all-TPC and 1-TPC samples
  (probed online), reducing a kernel to the fewest TPCs within a **latency-slip
  factor `k`** (`LITHOS_SLIP`). Verified: an 8-block kernel → 1 TPC; a large kernel
  → 4 TPCs at 1.5× slip.
- **Tracker thread + outstanding-work throttle (§5.3)** (`LITHOS_THROTTLE`). A
  Tracker thread reaps completion events, feeds the predictor, and maintains the
  in-flight-µs counter; dispatch is deferred while it exceeds the limit (paper:
  100 µs).
- **Dispatcher thread / launch queues (§5.2)** (`LITHOS_DISPATCH`, opt-in). Launches
  funnel through one dispatcher thread that applies global policy and submits to the
  GPU. It's a *hand-off* (the app thread waits until the dispatcher consumes the
  request) rather than a fire-and-forget enqueue, because a transparent interposer
  can't safely snapshot `cuLaunchKernel`'s `void**` args — their sizes are implicit
  — so we keep them alive across the hand-off instead of copying them. This gives
  the single-dispatch-authority role without the CPU-latency decoupling.
- **Special kernels (§6).** Cooperative/cluster launches are not atomized and get
  their **exact quota** (stealing + right-sizing disabled), and
  `cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)` is **spoofed** to the tenant's
  allocated SM count (`quota × 2`).

---

## Simplified or stubbed (present but not to the paper's depth)

- **Predictor sophistication.** Ours is operator-ordinal + EMA + a 2-point scaling
  curve; it does not model input-size features beyond the ordinal, and probing
  advances on async reaps (a few iterations of warm-up). The stub `blocks × 0.5 µs`
  remains only as the cold-start fallback before an operator is first measured.
- **Dispatcher decoupling.** The dispatcher is a correct hand-off, not a
  fire-and-forget async enqueue (see above) — the app thread still blocks until
  submission, so there's no CPU-launch-latency win, only the central-scheduling role.
- **API coverage.** We override only the calls the mechanisms need and forward
  the rest; we do not macro-generate the entire Driver API.
- **Hopper.** `qmd.c` carries the Hopper (TMD ≥ 0x40) mask offsets but they are
  untested here; Thread Block Cluster–aware atom sizing is not implemented.

---

## Not implemented

- **Power management.**
- **Central multi-tenant scheduler.** Our scheduler is per-process; disjoint TPC
  ranges across tenants are assigned manually via `LITHOS_TPC_BASE` rather than by
  a coordinator. Cross-process quota arbitration, fairness, and preemption
  policies are out of scope.
- **Non-cooperative persistent-kernel detection.** We detect cross-block-sync via
  the cooperative/cluster launch attributes; a persistent kernel that grid-syncs
  without those attributes is still undetectable transparently.
- **Explicit CUDA-graph construction API** interposition — the *atomize-in-graph*
  mode only sees stream capture (it intercepts `cuLaunchKernel`), so manually-built
  graphs (`cuGraphAddKernelNode`) aren't atomized. (The *subgraph* mode does handle
  them — it partitions at `cuGraphInstantiate`, independent of how the graph was
  built.)
- **Container-specific** integration (untested).

---

## Things this reproduction worked out that the paper doesn't detail

These are implementation specifics the paper abstracts over; documented here
because they were non-trivial and are load-bearing for the reproduction, not
because they represent gaps:

- The exact QMD offsets on this GPU (mask 84/88, program address 192, register
  count 81) and the `cuGetExportTable` callback-table indices (3/6).
- The CUDA-runtime interception chain that makes frameworks work: returning our
  own `cuGetProcAddress`, following the `__fatBinC_Wrapper_t`, the Library API
  (`cuLibraryLoadData`/`GetModule`/`GetKernel`), and `cuLaunchKernelEx`.
- Fatbin unwrapping incl. **LZ4** (CUDA ≤12) and **Zstandard** (CUDA 13)
  decompression, and JIT of PTX-only fatbins.
- The fall-through splice's ELF surgery (program headers, `.nv.info`
  instruction-offset attributes, per-kernel gating).

---

*Bottom line:* the **interposition + TPC-scheduling half is a faithful,
mechanism-for-mechanism reproduction** (simplified in the scheduler's threading
and policy depth). The **atomizer reproduces the *semantics* of Algorithm 1 but
via a deliberately different, transfer-free mechanism** (source-splice fall-through
instead of a QMD-redirected Prelude that jumps), because the paper's jump is not
reproducible from `ptxas` output — the very detail the paper defers. Several
production concerns (power, right-sizing, a central multi-tenant scheduler, the
duration predictor) are out of scope.*
