# LithOS reproduction — benchmarks

Benchmarks for the reproduction, all on an **RTX A6000** (GA102, sm_86), CUDA 12.8.
Sections:

1. **Latency microbenchmark** — host overhead per CUDA call (driver 595.71.05).
2. **CUDA graph overhead** — the two graph modes + subgraph reallocation (570.195.03).
3. **Scheduler-mechanism overhead** — per-launch cost of the §5.2–5.7 mechanisms (570.195.03).
4. **Reproducing the paper's performance experiments** — isolation, work conservation,
   atomization HoL, right-sizing (570.195.03).

---

## 1. Latency microbenchmark

Measures the **host-side latency overhead** LithOS adds to CUDA calls, and where
it comes from. Numbers on **driver 595.71.05**, via the driver API under
`LD_PRELOAD=build/liblithos_full.so`.

Harness: [`bench/microbench.c`](../bench/microbench.c) + a null kernel
([`bench/nullk.cu`](../bench/nullk.cu)). Three probes:
- **async-launch** — `cuLaunchKernel` in a tight loop, one `cuCtxSynchronize` at
  the end → amortized **host enqueue cost** per launch (GPU work overlaps).
- **launch+sync** — `cuLaunchKernel` + `cuCtxSynchronize` each iteration →
  **round-trip latency** of a trivial kernel.
- **module-load** — `cuModuleLoadData` + `cuModuleUnload` → per-module cost
  (includes the ELF splice under LithOS).

```
nvcc -arch=sm_86 -cubin bench/nullk.cu -o /tmp/nullk.cubin
gcc bench/microbench.c -o /tmp/microbench -I$CUDA/include -L$CUDA/lib64 -lcuda
NULLK=/tmp/nullk.cubin  [env...] LD_PRELOAD=build/liblithos_full.so /tmp/microbench <grid> [preload]
```

## Results (grid = 1 block, isolating host overhead)

| configuration | async-launch (µs) | round-trip (µs) | Δ vs baseline |
|---|---|---|---|
| **[A]** baseline (no LithOS) | **2.08** | **4.92** | — |
| **[B]** LithOS, interpose only | 2.43 | 5.39 | **+0.35 µs** enqueue |
| **[C]** LithOS + atomizer (n=1, gated) | 6.20 | 8.76 | **+4.1 µs** enqueue |
| **[D]** + TPC quota (QMD mask/launch) | 7.18 | 12.11 | +1 µs enqueue, +3.3 µs round-trip |
| **[E]** + MPS on (full deployed system) | 6.20 | 8.91 | **≈ 0** vs [C] |
| **[F]** atomizer, split into 32 atoms (grid=256) | 165 | 169 | ≈ 32 × per-launch cost |

Module load: baseline ≈ **10 µs**, LithOS + splice ≈ **20 µs** (one-time per module).

Gating lookup scaling (`is_atomized_func`, worst case — kernel gated last):
1 → 5.8 µs, 1 k → 8.5 µs, 4 k → 8.3 µs, **12 k → 7.6 µs** — a sequential pointer
scan the prefetcher hides, so **not** a linear blow-up even at TensorRT-scale
(11 k+ kernels).

## Where the overhead comes from

* **Interposition itself is nearly free: ~0.35 µs/launch** ([B]−[A]). That's the
  wrapper call-chain + per-stream scheduler bookkeeping + the QMD callback.
* **The atomizer dominates: ~4 µs/launch** ([C]−[B]), even without splitting.
  In *this reproduction* that cost is: each gated launch writes the atom range
  with **2 `cuMemsetD32Async`** calls and serializes streams with a
  **`cuStreamWaitEvent` + `cuEventRecord`** pair (≈4 extra async driver calls).
  Whether the *original* LithOS pays a comparable cost is **unknown from the
  paper**: it says the QMD is patched to redirect to the Prelude and that the
  Prelude receives an `AtomMetadata` struct, but it does **not** specify how that
  metadata reaches the device per launch (those low-level details are "deferred
  to a separate technical report"). So this ~4 µs is specific to *our* shared-
  buffer + event design; we can't claim the original avoids it. (Turn the
  atomizer off with `LITHOS_ATOMIZER=0` to get the ~0.35 µs interposition-only
  cost.)
* **Splitting is linear in atom count**: K atoms ≈ K × the per-launch cost, so
  atom granularity is a real host-cost knob. The paper's 250–500 µs
  `atom_duration` keeps atoms *large* (K small, each atom ≫ the ~5 µs host cost),
  so the overhead is amortized against real GPU work; splitting *tiny* kernels
  aggressively (as in [F], `atom_us=4`) is dominated by launch overhead.
* **TPC quota** adds ~1 µs enqueue but ~3 µs round-trip — the QMD callback does
  more work (computing + writing the SM-disable mask) on the upload path.
* **MPS adds no measurable per-launch latency** ([E]≈[C]); its cost is the
  server hand-off, not the launch critical path. Its *benefit* (concurrent
  multi-tenant execution) is a throughput win — see the README §2 MPS table
  (2 concurrent tenants: 16.8 s time-sliced → 8.6 s under MPS).

## Takeaways

- The **scheduler/interposition path is cheap** (~0.35 µs), so LithOS-style TPC
  scheduling is essentially free on the launch critical path.
- The **atomizer's per-launch cost here (~4 µs)** is an artifact of *this
  reproduction's* shared-metadata + event-serialization design; it is amortized
  for paper-sized atoms and is only significant when splitting many *tiny*
  kernels. How the original delivers per-atom metadata (and thus whether it has a
  similar cost) is not stated in the paper.
- Nothing scales badly with kernel count or MPS.

---

## 2. CUDA graph overhead

Graphs are the interesting case because the QMD **pre-upload callback fires exactly
once per graph exec** (at the first `cuGraphLaunch`); replays re-run a compiled
command buffer that bypasses the driver's per-node path, so the SM mask **cannot be
changed on replay** — only by **re-instantiating** (which re-fires the callback).
That fact drives the two graph modes and their very different costs.

Harness: a chain of compute kernels captured into a graph; steady replay is a
best-of-3 median (structural overheads); the reallocation cost is measured in
isolation (host-side, so it's independent of the GPU boost-clock variance that
otherwise swamps per-replay deltas across processes). **Every mode below produced
the correct checksum** — atomization/partitioning never changes graph results.

### Steady-state per-replay overhead

| mode | 8-kernel chain | 32-kernel chain | cost source |
|---|---|---|---|
| baseline (no LithOS) | — | — | 1 `cuGraphLaunch` |
| interpose only | **+2 %** | **+2 %** | wrapper passthrough |
| **atomize-in-graph (default, n=1)** | **+120 %** | **+194 %** | **2 memset (metadata) nodes per kernel** triple the node count, + spliced prologue per kernel |
| atomize-in-graph (4 atoms) | +651 % | +1081 % | each kernel → 4 atom-launches + 8 memset nodes; replay runs N× the nodes |
| **subgraph K=2** (paper model) | +7 % | −3 % | fan-out: 2 graph launches vs 1 |
| subgraph K=4 | +13 % | ~0 % | 4 launches |
| subgraph K=8 | +20 % | +8 % | 8 launches |

**The default "atomize-into-graph" mode is the wrong tool for graphs:** even at
n=1 (no splitting) it adds two `cuMemsetD32Async` metadata nodes per kernel — the
in-graph metadata delivery that's cheap for eager launches becomes a tripled node
count and a serialized memset before every kernel. Splitting into atoms multiplies
nodes and blows the replay up 6–11×. The **subgraph model is nearly free**
(single-digit %), and on a larger graph (32 kernels) it amortizes to ≈ baseline.

### One-time costs (paid once, not per replay)

| mode | capture | instantiate / partition |
|---|---|---|
| baseline | ~0.17 ms | 0.07 ms |
| subgraph K=4 | ~0.16 ms | 0.13 ms (**+0.06 ms**: 4× clone + delete + instantiate) |
| subgraph K=8 | ~0.16 ms | 0.15 ms (+0.08 ms) |
| atomize-in-graph (4 atoms) | ~0.10 ms | 0.32 ms (more nodes to compile) |

Partitioning adds a one-time ~0.06–0.08 ms at instantiate — negligible against a
graph that replays thousands of times.

### Reallocation cost (isolated host cost of re-instantiating one subgraph)

| subgraph size | instantiate+destroy |
|---|---|
| 2 nodes | **7.3 µs** |
| 4 nodes | 9.9 µs |
| 8 nodes | 14.1 µs |

Reallocation costs **~7–14 µs per *changed* subgraph**, and only changed subgraphs
rebuild — steady replays do zero. Rotating all 4 subgraphs every replay (worst
case) added `4 × 7.3 ≈ 29 µs`. This is exactly why partitioning matters: a
reallocation rebuilds a 2-node template, not the whole graph.

### Graph takeaways

- **Interposition on graphs is free (+2 %).**
- **Prefer `LITHOS_GRAPH_SUBGRAPHS` over in-graph atomization for graph-heavy
  workloads** — the per-kernel metadata nodes alone add +120 %+, before any split.
- **The subgraph model is cheap** (single-digit %, amortizing to ~0 on larger
  graphs) and delivers real per-subgraph TPC scheduling.
- **Dynamic reallocation is ~8 µs per changed subgraph**, paid only on change — the
  subgraph granularity is what makes rescheduling-at-replay affordable.
- **Correctness holds in every mode and size.**

---

## 3. Scheduler-mechanism overhead (§5.2–5.7)

Per-launch host cost added by the reproduced scheduler mechanisms. Null kernel,
grid = 1, async-enqueue cost, best-of-N, RTX A6000. Baseline round-trip 5.36 µs.

| configuration | async µs/launch | Δ | source |
|---|---|---|---|
| baseline (no LithOS) | 1.81 | — | — |
| interpose only | 2.16 | +0.35 | wrapper + bookkeeping |
| atomizer only (predict off) | 5.02 | +2.9 | metadata memsets + event serialize |
| **predictor only** (§5.7) | 7.39 | **+5.2** | **2 `cuEventRecord`/launch** + operator lookup |
| atomizer + predictor (**default**) | 12.5 | +10.7 | both |
| + right-sizing (§5.5) | 12.5 | **≈0** | occupancy query (driver-cached) + model FLOPs |
| + throttle (§5.3) | ~12.5 | ≈0 under limit* | one `outstanding_us` compare (*blocks by design when over) |
| + dispatcher (§5.2, opt-in) | 57 | **+45** | cross-thread condvar hand-off per launch |
| SM-count spoof (§6) | ~12.5 | ≈0 | one-time `cuDeviceGetAttribute`, not per-launch |

* **The predictor (~5 µs, on by default) is the dominant new cost** — entirely the
  two `cuEventRecord`s that measure real latency. Negligible for real ML kernels
  (ms-scale) but significant for tiny-kernel floods; a 1-in-N sampling knob would
  cut it back toward ~6 µs while still learning.
* **Right-sizing is effectively free** on the launch path (occupancy is
  driver-cached). **The throttle** adds nothing under the limit and blocks by
  design over it (deliberately bounding in-flight work). **The dispatcher** is
  expensive (~45 µs, cross-thread hand-off) — why it's opt-in; it buys the
  single-dispatch-authority role, not launch throughput. **SM-count spoofing** is
  one-time.

---

## 4. Reproducing the paper's performance experiments

Reproductions of the paper's main results on a **single RTX A6000** with
**synthetic kernels** (the paper used an A100 with real models on
Triton/TensorRT-LLM). Absolute magnitudes differ, but the **mechanisms and
qualitative results reproduce** in the right ballpark.

### MPS work conservation (the foundation)

Two under-utilizing tenants co-located (aggregate throughput):

| | it/s |
|---|---|
| 1 tenant (solo) | 9,846 |
| 2 tenants, time-slice (no MPS) | 5,815 |
| 2 tenants, MPS | **17,393** |

→ **MPS ≈ 3.0× time-slicing** — the concurrency LithOS builds on (time-slicing
even loses to solo, from context-switch overhead).

### Proportional TPC allocation (the QoS primitive, §5.3)

One saturating tenant, throughput vs its `LITHOS_QUOTA`:

`4→389, 8→777, 16→1536, 21→1983, 32→2979, 42→3521 it/s` — **linear in quota**
(~90 it/s per TPC). Compute quotas give predictable, proportional shares.

### Spatial isolation — MPS vs LithOS TPC-partition (§8.1, Fig. 14/16)

Compute-bound HP (latency-critical) + heavy BE co-located:

| config | HP throughput | HP p50 | **HP p99** |
|---|---|---|---|
| HP solo (ideal) | 9,820 it/s | 0.101 ms | 0.109 ms |
| HP + BE, **plain MPS** (shared TPCs) | 3,465 (35 %) | 0.288 ms | **0.302 ms** (2.8× ideal) |
| HP + BE, **LithOS partition** (21/21 TPC) | 9,727 (**99 %**) | 0.102 ms | **0.111 ms** (1.02× ideal) |

→ **LithOS fully isolates HP**: p99 within **2 % of ideal** under interference vs
**2.8×** inflation under naive MPS (a **2.7× tail-latency improvement**), and
throughput restored 35 %→99 %. Matches §8.1 (MPS has the worst latencies; LithOS
gives MIG-like spatial isolation at full throughput — the paper reports 13× vs MPS
on a harder inference workload). *Caveat:* the HP kernel must be compute-bound; a
launch-bound (tiny) HP is limited by MPS's launch-submission serialization, which
a central dispatcher (`LITHOS_DISPATCH`) addresses but per-process masking alone
does not.

### Kernel atomization reduces HoL blocking (§8.4, Fig. 21)

HP (6 µs latency-critical kernel) co-located under MPS with a BE process running a
**6.75 ms** kernel:

| configuration | HP p50 | HP p99 |
|---|---|---|
| HP alone (ideal) | 7.7 µs | 17 µs |
| + BE, **not atomized** (HoL) | 6,319 µs | **6,385 µs** |
| + BE, **atomized** (~13 × 0.5 ms atoms) | 324 µs | **~400 µs** |

→ Atomizing the long BE kernel cuts HP tail latency **16×** (6,385→400 µs); the
residual ≈ **one atom** (0.65 ms) — exactly the paper's mechanism (HoL blocking
bounded by atom size, not kernel size). Finer atoms reduce it further; the paper's
ms-scale HP + 250–500 µs atoms land within 14 % of ideal.

### Right-sizing scaling curves (§8.2, Fig. 12/18)

`l = m/t + b` fit across TPC counts:

| kernel | fit | R² | right-size @ slip 1.1 | capacity saved | latency |
|---|---|---|---|---|---|
| matmul (scales well) | `26.80/t + 0.010` | **1.0000** | 39 TPC | 7 % | 1.08× |
| heavy (diminishing returns) | `0.34/t + 0.015` | **0.9991** | 33 TPC | 21 % | 1.09× |

→ Model accuracy **R² 0.999–1.0** (paper 0.92–0.99); latency cost **~1.08×** at
slip 1.1 (paper ~4 % mean P99); savings scale with how poorly a kernel scales
(paper mean 26 % over a real-kernel mix).

### What we could not reproduce

- The full **multi-system comparison** (MIG / REEF / TGS / Orion / Priority) —
  needs those other systems.
- **DVFS energy savings** (§8.3) — needs power measurement; DVFS is out of scope
  here.
- **Real inference-serving SLO attainment** (Triton + real models) — needs the
  serving stack; the synthetic HP/BE proxies stand in for the mechanism behaviour.
