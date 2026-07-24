# LithOS reproduction — latency microbenchmark

Measures the **host-side latency overhead** LithOS adds to CUDA calls, and where
it comes from. All numbers on an **RTX A6000** (GA102, sm_86), **driver 595.71.05**,
CUDA 12.8, via the driver API under `LD_PRELOAD=build/liblithos_full.so`.

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

## CUDA graph overhead

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
