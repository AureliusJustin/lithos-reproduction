# Benchmarks

All numbers below are from the current version of the system on an
**NVIDIA A100 80GB PCIe** (GA100, `sm_80`, 108 SMs → **54 TPCs**), CUDA 12.8 —
the same GPU class the paper uses. The test node was re-provisioned during this
work, so results span two driver branches on identical hardware: **610.43.02** and
**570.195.03**. 

## 1. Per-launch host overhead

Null kernel, grid = 1, async-enqueue cost, **best-of-5 on an otherwise idle GPU
with MPS off** (`bench/launchbench.c`) — this isolates LithOS's host overhead from
GPU work and from the MPS server hop, which adds a roughly constant amount to
every row.

| configuration | µs/launch | Δ vs baseline |
|---|---|---|
| baseline (no LithOS) | **2.10** | — |
| interposition only | 2.13 | **+0.03 … +0.2** (see note) |
| atomizer only (`LITHOS_PREDICT=0`) | 2.43 | +0.33 |
| predictor only (`LITHOS_ATOMIZER=0`) | 6.22 | **+4.1** |
| atomizer + predictor (**default**) | 6.27 | +4.2 |
| + right-sizing | 5.34 | within noise of default |
| + TPC quota | 6.34 | +4.2 |
| + dispatcher (opt-in) | 33.3 | +31.2 |

### Scheduler profile, stage by stage (`bash bench/schedprof.sh`)

The table above turns individual features on; this one walks the **launch path in
order**, each row keeping every stage above it, so a delta is the cost of adding
that stage to the ones already running. Re-measured on a second A100 80GB PCIe node
(driver 570.195.03, CUDA 12.8), best-of-5, MPS off. The **spread** column is
max−min across the 5 reps and is the thing to read first: several rows have deltas
smaller than their own noise, and those are *not* measurable costs.

| stage | µs/launch | Δ prev | spread |
|---|---|---|---|
| baseline (no LithOS) | 2.22 | — | ±0.26 |
| interposition + launch queues (§5.2) | 2.35 | +0.13 | ±0.45 |
| + TPC quota mask (§5.2, QMD callback) | 2.32 | −0.03 | ±0.66 |
| + atomizer gating (§5.4, not split here) | 2.54 | +0.22 | ±0.50 |
| + predictor (§5.7, one event/launch) | **7.68** | **+5.14** | ±2.05 |
| + TPC stealing scan (§5.3) | 7.82 | +0.14 | ±1.92 |
| + system-wide coordinator (§5.1) | 7.46 | −0.36 | ±2.97 |

Opt-in policies, each measured against that last row (7.46 µs):

| policy | µs/launch | Δ | spread |
|---|---|---|---|
| right-sizing (§5.5) | 7.45 | −0.01 | ±1.44 |
| forced max split (§5.4, `ATOM_US=1`) | 7.39 | −0.07 | ±2.94 |
| outstanding-work throttle (§5.3) | 26.80 | **+19.34** | ±2.43 |
| dispatcher hand-off (§5.2) | 32.63 | **+25.17** | ±4.35 |

**Only three stages have a cost above their own noise**: the predictor (+5.1 µs),
the throttle (+19.3 µs), and the dispatcher (+25.2 µs). Everything else — the TPC
mask, the stealing scan, the cross-application coordinator, right-sizing — is at or
below the measurement floor on this workload.

* **The QMD mask, the stealing scan and the coordinator are free on the launch
  path.** The mask is a single 64-bit store into the pre-upload callback; the
  stealing scan walks at most `MAX_STREAMS` slots of plain 64-bit fields under no
  lock; the coordinator reads a shm array the same way. Negative deltas here are
  noise, not speedups.
* **The predictor dominates the always-on cost**, for the reason given above: one
  CUDA event per launch, deliberately unsampled.
* **The throttle's cost is the mechanism working, not overhead.** It blocks until
  in-flight work drops below `LITHOS_OUTSTANDING_US` (100 µs — the paper's figure),
  so a launch loop with no GPU work to wait on hits that gate constantly.
* **Neither the throttle nor the dispatcher is a fixed per-launch tax.** Both grow
  with the launch's own cost rather than staying constant, so the null-kernel
  numbers are a worst case and cannot be extrapolated:

  | null-kernel grid | ref | + throttle | + dispatcher |
  |---|---|---|---|
  | 1 | 7.46 | +21.1 | +24.0 |
  | 256 | 7.41 | +19.0 | +25.5 |
  | 4096 | 12.13 | +74.7 | +74.7 |

  What decides whether that matters is **kernel duration**, and the crossover is
  sharp. Sustained throughput (`build/tenant`, grid 256, 6 s):

  | kernel duration | ref | + throttle | + dispatcher |
  |---|---|---|---|
  | ≈ 0.12 ms | 3740 it/s | 2606 (**−30 %**) | 2805 (**−25 %**) |
  | ≈ 1.17 ms | 9 it/s | 9 (−0 %) | 8 (−11 %) |

  So "amortized on real kernels" is only true from roughly the millisecond scale
  — exactly the kernels §5.4 targets. On short kernels these mechanisms cost a
  quarter to a third of throughput, which is why both are off by default.
* **Caveat on the workload.** The stage table uses a null kernel at grid=1 —
  chosen to isolate host cost, but also the case where LithOS can do the least:
  below `LITHOS_MIN_BLOCKS` it is never atomized, so the atomizer and right-sizing
  rows measure only their decision logic, not their effect. For what the mechanisms
  buy on real kernels, see §5–§7.

* **Interposition is nearly free — but not literally free.** The +0.03 µs above is
  a best-of-5 minimum; an interleaved A/B on a second driver (570.195.03) put it at
  **+0.13 µs (min) / +0.23 µs (median)**, i.e. ~6–10 % of a bare 2.2 µs launch.
  Treat "interposition costs ≲0.2 µs" as the durable claim: the wrapper chain,
  launch-queue bookkeeping and QMD callback are small but measurable, and the exact
  figure moves with driver and clock state.
* **The predictor dominates** (+4.1 µs) and is **on by default**. It records one
  CUDA event per launch as a completion marker; per-task tracking is deliberately
  *not* sampled, because the same Tracker signal also clears the sync queues (the
  throttle) and updates the stealing timers. `LITHOS_PREDICT=0` gets the ~0.33 µs
  atomizer-only cost back.
* **The atomizer is cheap** (+0.33 µs) — see §7 for why it used to be ~15× that.
* **The dispatcher is expensive** (+31 µs): a cross-thread hand-off per launch.
  It buys the single-submission-authority role, not throughput, so it is off by
  default.

## 2. Module load + splice

One-time per module (`bench/modload.c`; `LITHOS_DIAG=1` separates LithOS's own
splice time from the driver's load).

| module | baseline load | with splice | of which LithOS |
|---|---|---|---|
| 1 kernel | 15.7 µs | 23.4 µs | — |
| 75 kernels | 278 µs | 490 µs | **0.03 ms** |
| 150 kernels | 365 µs | 951 µs | 0.08 ms |
| 300 kernels | 695 µs | 1 972 µs | 0.13 ms |
| 600 kernels | 1 415 µs | 4 655 µs | **0.25 ms** |

Splice time scales **linearly** (~2× per doubling). The remaining gap at 600
kernels is the *driver* loading a larger module — the spliced cubin grows
841 KB → 1149 KB from the added prologues — which is inherent to the approach,
not something the splicer can avoid.

## 3. CUDA graphs

A hardware constraint shapes this section: the QMD pre-upload callback fires
**exactly once per graph exec** (at the first `cuGraphLaunch`). Replays re-run a
compiled command buffer that bypasses the driver's per-node path, so a graph's SM
mask **cannot be changed on replay** — only **re-instantiating** re-fires the
callback (verified with an unconditional callback counter).

Replay cost, 8-kernel chain (`bench/graphprof.c`):

| mode | µs/replay | Δ |
|---|---|---|
| baseline | 27.5 | — |
| interpose only | 28.0 | **+2 %** |
| atomize-in-graph (default) | 58.9 | +114 % |
| **subgraph K=4** (`LITHOS_GRAPH_SUBGRAPHS`) | 31.3 | **+14 %** |

* **In-graph atomization roughly doubles replay cost.** Every kernel gains two
  metadata `memset` nodes, and during capture those writes cannot be elided — each
  must be *recorded*. Correct, but the schedule is frozen: atom ranges bake in and
  per-atom TPC masks do not apply.
* **The subgraph model stays cheap** and is the one that preserves scheduling: the
  graph is partitioned along a topological cut, each subgraph gets its own TPC
  allocation, and reallocation re-instantiates only the changed subgraph
  (**7.3 µs** for a 2-node subgraph, 9.9 µs for 4, 14.1 µs for 8 — paid only on
  change). Prefer it for graph-heavy workloads.

## 4. Atom sizing (derived bounds)

The paper leaves `atom_duration` a hand-tuned constant — *"limits of 250–500 µs
are effective"* — and only warns that *"if this parameter is set too low, an
atomized kernel may actually take longer to complete."* Both bounds are derived
instead (`effective_atom_us` in [`src/atomizer.c`](../src/atomizer.c)):

* **Floor** = `atom_cost_us / max_overhead`. Splitting into *n* atoms costs about
  `n × atom_cost` (each atom is a full-grid relaunch whose out-of-range blocks
  reach the prologue and exit), so this *guarantees* the overhead stays under a
  chosen fraction of the kernel — turning the paper's caveat into an invariant.
* **Ceiling** = `LITHOS_SLO_US`. A co-located latency-critical tenant waits behind
  at most one atom (§5), so a neighbour's latency budget bounds atom size directly.
  The paper's fixed constant has no relationship to any SLO.

20 000-block kernel:

| `LITHOS_SLO_US` | 0 (off) | 500 | 200 | 100 | 50 |
|---|---|---|---|---|---|
| atoms chosen | 16 | 20 | 50 | 100 | 200 |

| setting | atoms chosen |
|---|---|
| `ATOM_US=300` (default) | 16 |
| `ATOM_US=10` | **200** — clamped up by the 50 µs floor |
| `ATOM_US=10`, floor disabled | **500** — the pathological over-split the paper warns about |

## 5. Reproducing the paper's experiments

Synthetic kernels rather than the paper's real models on Triton/TensorRT-LLM, so
absolute magnitudes differ — but each mechanism's qualitative result reproduces.

### MPS work conservation (the foundation)

Two under-utilizing tenants (`bench/tenant.c`):

| | aggregate throughput |
|---|---|
| 1 tenant (solo) | 7 218 it/s |
| 2 tenants, time-sliced (no MPS) | 3 559 it/s |
| 2 tenants, **MPS** | **14 410 it/s** |

**MPS gives 4.05× time-slicing**, and 2.0× a single tenant — the two tenants run
genuinely concurrently. This is the concurrency LithOS builds on; without it,
confining a tenant to a TPC subset would simply idle the rest of the GPU.

### Spatial isolation (§8.1)

Compute-bound latency-critical (HP) job + heavy best-effort (BE) job, co-located:

| config | HP throughput | HP p99 |
|---|---|---|
| HP solo (ideal) | 7 275 it/s | 0.142 ms |
| HP + BE, **plain MPS** | 2 124 it/s (29 %) | 0.483 ms (3.4× ideal) |
| HP + BE, **LithOS partition** (27/27 TPC) | **7 179 it/s (99 %)** | **0.147 ms (1.04× ideal)** |

LithOS restores the latency-critical tenant to **99 % of solo throughput and within
4 % of its solo tail**, while MPS alone loses 71 % of throughput and inflates the
tail 3.4×. Matches §8.1's finding that MPS has the worst latencies while LithOS
provides MIG-like spatial isolation at full throughput.

### Head-of-line blocking (§8.4)

HP (9 µs kernel) co-located with a BE process running a **10.8 ms** kernel:

| configuration | HP p50 | HP p99 |
|---|---|---|
| HP alone (ideal) | 11.0 µs | 22.7 µs |
| + BE, **not atomized** | 10 419 µs | **10 577 µs** |
| + BE, **atomized** (`SLO_US=500`) | 231 µs | **453 µs** |

**23× tail-latency reduction**, and the residual is ≈ one atom — exactly the
mechanism the paper describes (HoL blocking bounded by atom size, not kernel size).

### Right-sizing scaling model (§8.2)

matmul N=1024 swept across all 54 TPCs (`bench/scalebench.c`), fitting `l = m/t + b`:

**`l = 23.88/t + 0.0285`, R² = 1.0000** (paper reports 0.92–0.99).

| latency slip | right-sized to | capacity saved | latency |
|---|---|---|---|
| 1.1 | 49 TPC | 9 % | 1.10× ideal |
| 1.5 | 36 TPC | **33 %** | 1.47× ideal |

Comparable to the paper's mean 26 % (up to 51 %) over a broader kernel mix.

### Not reproduced

The multi-system comparison (MIG / REEF / TGS / Orion — needs those systems), DVFS
energy savings (§8.3 — DVFS is out of scope here), and real serving-stack SLO
attainment (needs Triton + real models; synthetic HP/BE proxies stand in).

## 6. Use-case study: when is LithOS worth it?

Sweeping the best-effort kernel's duration while a latency-critical job runs
alongside (`bench/usecase.py`, `LITHOS_SLO_US=300`):

| BE kernel | duration | HP p99 un-atomized | HP p99 atomized | HP gain | BE throughput kept |
|---|---|---|---|---|---|
| G=512, W=4 000 | 0.04 ms | 28.6 µs | 32.0 µs | **1×** (none) | 96 % |
| G=2 048, W=8 000 | 0.24 ms | 130 µs | 90 µs | 1.4× | 65 % |
| G=8 192, W=20 000 | 2.2 ms | 1 869 µs | 257 µs | **7×** | 62 % |
| G=20 000, W=40 000 | 10.8 ms | 10 473 µs | 360 µs | **29×** | 67 % |

**The interfering kernel's duration decides everything.** Below ~0.1 ms there is no
head-of-line blocking to remove, so atomization buys nothing and only costs BE
throughput; above ~2 ms it delivers 7–29× tail reduction for roughly a third of BE
throughput. See [FINDINGS.md](FINDINGS.md) for the resulting use-case guidance.


## 7. Mechanism value: does each one earn its cost?

Overhead tables say what a mechanism *costs*; these experiments ask what it *buys*.

### Predictor accuracy (comparable to the paper's §8.4)

Scored the paper's way — compare each launch's prediction to its measured duration,
count |error| > 50 µs as a misprediction (`bench/predacc.py`):

| workload | n | misprediction | |err| p50 | |err| p99 |
|---|---|---|---|---|
| steady batched (8 ops/batch) | 376 | **0.0 %** | 0.5 µs | 2.5 µs |
| steady solitary (launch+sync) | 96 | **0.0 %** | 0.4 µs | 3.6 µs |
| long kernels (batched) | 124 | 1.6 % | 0.7 µs | 61.5 µs |

Paper: 0.9 % / 0.38 % for HP workloads (P99 error 49 µs / 31 µs), 14 % / 11 % for BE.
Ours is comparable or better — but on **much more regular** workloads: each operator
here has a fixed shape, whereas the paper's BE numbers come from real models with
varying input sizes. Treat this as "the mechanism is sound", not "more accurate
than the paper".

> Building this exposed a **real bug**. The single-event optimization (§8) derives a
> kernel's duration from the gap to the *previous* completion on its queue — but a
> workload that syncs after **every** launch has exactly one kernel per batch, so
> there is no predecessor and **nothing was ever measured**: 0 samples versus 62 for
> a batched workload. That is precisely the latency-critical inference shape LithOS
> targets, and their atom sizing was silently falling back to the stub. Fixed by
> giving a launch its own start event when it has no in-batch predecessor — one
> event in the common batched case, two only for the first kernel after a sync.
> Coverage went 0 → 19 of 20 solitary launches.

### Right-sizing: is the freed capacity usable?

The scaling model says a kernel needs only *k* of 54 TPCs. If a scheduler hands the
rest to a second tenant, does the GPU net a win? Tenant A right-sized to *k*, tenant
B given the remaining 54−*k*:

| A's TPCs | A throughput | A latency slip | B throughput | aggregate |
|---|---|---|---|---|
| 54 (solo) | 13 214 it/s | 1.00× | — | 13 214 |
| **40** | 13 187 it/s | **1.01×** | 5 413 it/s | **18 600 (1.41×)** |
| 27 | 7 960 it/s | 1.67× | 7 971 it/s | 15 931 (1.21×) |
| 18 | 7 332 it/s | 1.81× | 13 183 it/s | 20 515 (1.55×) |
| 9 | 3 884 it/s | 3.43× | 13 145 it/s | 17 029 (1.29×) |

**The k=40 row is the point:** giving up 14 TPCs costs the first tenant **1 %**
latency and yields **+41 % aggregate GPU throughput**. That is the capacity-savings
claim, measured end to end rather than inferred from the model. (The middle rows are
noisy — contention between the two tenants is not perfectly linear.)

### TPC stealing: throughput, not just masks

Two streams with disjoint quotas (9 TPCs each); A runs flat out, B is bursty and
mostly idle (`bench/stealbench.c`):

| | A throughput |
|---|---|
| stealing off | 3 963 it/s |
| **stealing on** | **7 612 it/s (1.92×)** |

The mechanism does real work-conservation, not just mask arithmetic.

### Throttle and dispatcher: honest negative results

Both are implemented, and neither earns its cost in this reproduction.

**Throttle.** A first measurement looked spectacular — co-located HP tail 548 → 22.5 µs
— until the cost side showed best-effort throughput collapsing **31×** (3 100 → 100
kernels). The gain came entirely from starving BE. The cause: gating on *total
in-flight µs* means a single ms-scale kernel already exceeds a 100 µs budget, so the
submitter waits for near-full drain and the GPU idles between kernels — the opposite
of the paper's "keep the GPU busy while maintaining scheduling flexibility". Fixed by
always permitting at least one launch in flight. After the fix:

| BE kernel | throttle | HP p99 | BE throughput |
|---|---|---|---|
| short (~0.02 ms) | off | 27.0 µs | 329 760 |
| short (~0.02 ms) | on | **21.7 µs** | 66 140 (−5×) |
| long (~2.2 ms) | off | 533 µs | 3 080 |
| long (~2.2 ms) | on | 594 µs | 2 960 |

So the throttle gives ~20 % tail improvement against a deep queue of *short* kernels,
at a 5× throughput cost, and **nothing** against long kernels — because one long
kernel in flight already dominates HP latency and there is no preemption. Bounding
queue depth cannot fix head-of-line blocking from a single kernel; that is
atomization's job, which achieves 23–29× on the same workloads (§6).

**Dispatcher.** The first attempt measured the wrong thing: that dispatcher was a
blocking hand-off, so it could not *defer* at all — the mechanism the paper's
dispatcher exists to provide. It has since been rebuilt properly (§8), and the
result below is from a dispatcher that genuinely buffers.

*The mechanism works.* Argument capture via `cuFuncGetParamInfo` makes true
fire-and-forget deferral possible for any kernel, closed-source included:
**100 % of launches buffered, 0 inline fallbacks, 105 priority reorders** in a
representative run, with `tests/test_dispatch_order.c` confirming every ordering
guarantee still holds (and failing 6/6 when the barriers are deliberately
sabotaged, so the test has teeth).

*It still does not improve tail latency here*, in either regime we can construct.
One process, HP stream + best-effort stream, atomizer off to isolate the
dispatcher:

| regime | config | HP p50 | HP p99 | BE |
|---|---|---|---|---|
| **shallow** (BE 0.24 ms kernels) | native CUDA | 99.8 µs | 218.5 µs | 4 239/s |
| | LithOS, dispatch off | 105.7 µs | 225.2 µs | 4 226/s |
| | dispatch, FIFO | 167.0 µs | 290.3 µs | 4 011/s |
| | dispatch, **priority** | 165.1 µs | 305.0 µs | 4 010/s |
| **deep** (BE 10.8 ms kernels, burst 4 ≈ 43 ms queued) | native CUDA | 5 682 µs | 5 905 µs | 99/s |
| | LithOS, dispatch off | 5 698 µs | 5 815 µs | 99/s |
| | dispatch, FIFO | 5 720 µs | 38 945 µs | 95/s |
| | dispatch, **priority** | 5 705 µs | 49 437 µs | 97/s |

**Why, precisely.** Deferral can only reorder work that has *not yet been
submitted*. Neither regime is limited by that:

* In the **deep** regime, HP p50 is ~5 700 µs in **every** configuration
  *including native* — HP is waiting on the BE kernel that is **already
  executing**. There is no preemption, so no submission-side policy can help;
  the mechanism that fixes this is **atomization**, which cuts the same tail
  23–29× by making the in-flight unit small.
* In the **shallow** regime the hardware is already doing the job — two CUDA
  streams are interleaved by the GPU itself, so there is no queue to reorder and
  buffering merely inserts a software queue in front of a hardware path that was
  keeping up (+60 µs p50).

The tail *regressions* (38–49 ms) are buffering's own artifact: with bursty BE,
an HP launch can arrive just behind a full buffered burst.

**What this does and does not say.** It is evidence that *deferral by itself*
buys nothing on this hardware, not that the paper is wrong. In LithOS, deferral is
the enabler: it keeps work uncommitted so priority, TPC allocation, atomization
and right-sizing decisions can still be applied to it, coordinated by a central
scheduler across tenants. This reproduction applies those same decisions **inline
at launch** — computing a TPC mask or an atom count needs no buffering — so the
deferral has nothing left to buy. A workload where deferral would pay is one where
the *decision itself* must wait for information that arrives after the call: for
example, holding a low-priority launch until a central scheduler confirms no
high-priority tenant needs the TPCs. That requires the cross-tenant coordinator
this reproduction does not have (§FIDELITY).

Accordingly the dispatcher remains **opt-in and off by default** — but as
"unnecessary in this configuration", not "a mechanism that does not work".

*Caveat on the throttle:* it is still simplified relative to the paper, where the
throttle lives inside an asynchronous dispatcher. Ours can run in either place.

### Framework validation on this hardware

`bash bench/fw_all.sh` — four targets, each reaching the GPU by a different route,
crossed with eight LithOS configurations (A100 80GB PCIe, driver 570.195.03).
Every run reports two things: the target's own correctness check against a CPU
reference, and atomization coverage. Coverage matters as much as correctness,
because a LithOS that quietly passes every launch through looks perfectly healthy
while doing nothing.

| target | reaches the GPU via | launches | correctness | coverage |
|---|---|---|---|---|
| cuBLAS (`fw_cublas.cu`) | fatbin kernels fetched **by name** | 3 | 8/8 configs PASS | **100 %** |
| cuFFT (`fw_cufft.cu`) | kernels fetched by **bulk enumeration**, unnamed | 2 | 8/8 configs PASS | **100 %** |
| PyTorch 2.13.0+cu126 (`fw_torch.py`) | cuBLAS + cuDNN + ATen + autograd + graphs | 63 | 8/8 configs PASS | **100 %** |
| Triton 3.7.1 (`fw_triton.py`) | cubins **JIT-compiled at run time** | 18 | 8/8 configs PASS | **100 %** |

The eight configs: default, forced max split (`ATOM_US=1`), `QUOTA=16`,
right-sizing, dispatcher, dispatcher with 4 workers, predictor, and all of them
at once. 32/32 combinations pass.

Two tenants also share the GPU correctly with **no hand-assigned `LITHOS_TPC_BASE`**:
two concurrent PyTorch processes are handed disjoint ranges `[0,16)` and `[16,32)`
by the coordinator, and both pass.

Two real bugs surfaced only here, both invisible to the unit tests at the time —
see FINDINGS §5. cuFFT is the reason the atomizer has a classify-by-handle
fallback: it obtains kernel handles through `cuLibraryEnumerateKernels`, which
carries no names, so before the fix its kernels ran at **0 %** coverage while
still producing correct output.

This refreshes the older A6000/torch-2.6 coverage result (§TECHNICAL_REPORT §8) on
newer hardware, a newer driver, and a much newer framework version.

## 8. Design history

Measurements that no longer describe the system, kept because each changed a design
decision. The tables above supersede them.

* **The atomizer used to cost ~5.0 µs/launch (default path ~12.4 µs).** Profiling
  found five driver calls per gated launch, three of them redundant: a
  `cuStreamIsCapturing` duplicated between the scheduler and the atomizer; a
  cross-stream serialization pair that is a provable no-op while only one stream
  issues atomized work; and metadata writes repeated when the range hadn't changed.
  Removing them took the atomizer to **+0.33 µs** over baseline and the default
  path to **6.27 µs** (§1). The single-stream fast path fails safe — a second stream
  latches it off permanently — verified with 4- and 8-stream concurrent atomized
  workloads.
* **The splicer was O(n²) in kernels per module** — 92 ms for 600 kernels, ~4× per
  doubling, because each kernel re-copied the whole cubin. That extrapolated to
  tens of seconds for a TensorRT-scale single module, which motivated the
  single-pass rewrite (plan every insertion, one segmented copy, one header
  fix-up): now **0.25 ms at 600 kernels** and linear (§2).
* **Halving the predictor's events saved only ~0.65 µs** (a start/stop pair per
  launch → one completion marker). That small gain is the evidence that the
  remaining predictor cost sits in event-pool management and the Tracker's
  `cuEventQuery` rather than the recording itself — and it is why *sampling* the
  measurements was rejected instead, as a fidelity regression rather than an
  optimization.
* **Gating lookups were a linear scan** and measured flat up to 12 k kernels (the
  prefetcher hid it), so this was never a live bottleneck. They were still replaced
  with an open-addressed hash set to remove the scaling risk at TensorRT scale
  (11 k+ kernels in one process).

## 9. Operational notes

MPS on the test node needed care, and both issues below present as LithOS hangs:

* **MPS refuses the default `/var/log/nvidia-mps`** when it isn't writable, and then
  won't start:

  ```sh
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/lithos-mps/pipe
  export CUDA_MPS_LOG_DIRECTORY=/tmp/lithos-mps/log
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d
  ```

* **MPS wedges if a client is killed mid-kernel** — every later CUDA process then
  hangs. Let co-located clients exit normally; recover with
  `echo quit | nvidia-cuda-mps-control; pkill -f nvidia-cuda-mps`.
* **Rebuild benchmark cubins after changing GPU** (`make bench`). They are built
  `-arch=native`, and an arch-mismatched cubin fails to load *silently*, showing up
  as 0 % atomization coverage rather than an error.

---

## 10. System-wide coordinator (§5.1, Figure 8)

Figure 8 shows several applications, each linked against LibLithOS, above **one
shared LithOS layer** that *"maintains a system-wide view of GPU state across
applications with varying priorities"*. Everything else in this reproduction is
per-process; [`src/coord.c`](../src/coord.c) is that shared layer, implemented as a
POSIX shared-memory tenant table (no daemon, so nothing is added to the launch
path). It supplies the three things a per-process scheduler structurally cannot:

**1. Compute quotas across applications (§5.2).** Two independent processes, each
`LITHOS_QUOTA=8`, with **no `LITHOS_TPC_BASE` anywhere**:

```
[coord] joined slot=0 pid=6947 quota=8 range=[0,8)  of 54 TPCs
[coord] joined slot=1 pid=6958 quota=8 range=[8,16) of 54 TPCs
```

Previously this required hand-assigning a disjoint base to every tenant.

**2. TPC Stealing across applications (§5.3)** — *"the scheduler dynamically
reassigns underutilized TPCs across applications"*. With a second tenant alive but
idle, the busy tenant's mask grows from its own range to include the idle peer's:

| stealing | busy tenant's enabled TPCs | throughput |
|---|---|---|
| off | `[8..15]` (its quota) | 311 it/s |
| **on** | `[0..15]` (+ the idle tenant's) | **620 it/s — 1.99×** |

**3. The priority safeguard (§5.3).** A higher-priority tenant's quota is a
guarantee, so its TPCs are never borrowed *even while it is idle*:

| borrower | lender | result |
|---|---|---|
| priority 0 | priority **10**, idle | stays `[8..15]` — **not stolen** ✅ |
| priority 10 | priority 0, idle | takes `[0..15]` — stolen ✅ |

This is also what makes §5.4's *"stealing is disabled for the latter's subsequent
atoms once request B is submitted"* fall out naturally: a lender publishes its
liveness on every launch, so the borrow evaporates on the lender's next kernel.

Regression: [`tests/test_coord.sh`](../tests/test_coord.sh), in `make run_tests`.
The lender must be **alive but quiet** — a process that exits releases its slot —
which is what `bench/idler.c` provides.

*Implementation notes.* Slots carry the owner's pid and are reclaimed if that
process dies; the shared mutex is `PTHREAD_MUTEX_ROBUST` so a crash mid-update
doesn't wedge the table. Range assignment happens at `cuStreamCreate` time, not
first launch — a stream created before the coordinator speaks would otherwise be
pinned to the default base, which is exactly the bug that made this look broken
during development.

### Priority inversion from stealing, and what actually fixes it

§5.3 warns that stealing can backfire: *"this may cause head-of-line (HoL)
blocking from priority inversion if a new request B is delayed by C2 occupying the
stolen TPCs."* That is reproducible here, and it is severe. A latency-critical
tenant on a 5 ms request period looks **idle** to the coordinator between requests
(see "how idle is decided" below), so a co-located batch tenant borrows its TPCs —
and the borrowed work is still running when the next request arrives:

| configuration | HP p99 |
|---|---|
| no stealing (strict partition) | 62.5 µs |
| stealing, **equal priority** | **1 650 µs — 26× worse** |
| stealing, **HP priority 10 > BE priority 0** | **55.7 µs — protected** |

The **priority safeguard is the fix**, not disabling stealing: with priorities set,
the HP tenant is never borrowed from, so it keeps its partition-quality tail
(55.7 µs vs 62.5 µs for a strict partition) *while BE still gets to steal whenever
HP is genuinely idle*. Strict partitioning gives up that work conservation for no
latency benefit.

**How "idle" is decided.** A peer is stealable when it is not higher-priority and
has not launched within `g_idle_ns` (**1 ms**). Measured directly:

| peer's launch period | borrowed? |
|---|---|
| 300 µs | **no** — peer looks busy |
| 5 000 µs | **yes** — peer looks idle |

This is a *wall-clock recency heuristic*, and that is a divergence worth stating:
the paper instead keeps **per-TPC timers informed by the latency predictor**,
"estimating kernel (and atom) durations at submission time… these timers help avoid
stealing from long-running TPCs" (§5.3). Recency cannot distinguish "idle" from
"between requests", which is exactly why the equal-priority row above collapses.
The paper's other two mitigations — limiting outstanding atoms on stolen TPCs and
running stolen-TPC work at lower hardware stream priority — are also not
implemented. The priority safeguard is doing all the work here.

### Does the dispatcher help, now that the coordinator exists?

Earlier the dispatcher could not be evaluated fairly: there was no cross-tenant
scheduler for deferral to serve. With the coordinator in place the natural
experiment is finally constructible — hold a batch launch until the system-wide
view says no higher-priority tenant needs the TPCs. Run in the inversion scenario
above (equal priority, so stealing is permitted and HP is actually being hurt):

| BE configuration | HP p50 | HP p99 |
|---|---|---|
| dispatch **off** (submit inline) | 1 609 µs | 1 632 µs |
| dispatch **on** (buffered, 1 240 launches deferred) | 1 599 µs | 1 762 µs |

**No benefit, again — and now we can say precisely why.** Deferral can only change
work that has *not yet been submitted*, and neither thing that hurts HP is in that
category:

1. What blocks HP is the batch kernel **already executing** on its TPCs. Deferral
   cannot recall it; nothing on the submission path can. (Atomization can, by
   making the in-flight unit small — 23–29× on comparable workloads.)
2. The scheduling *decisions* deferral is supposed to enable — the TPC mask, the
   atom count — are computed in `submit_launch_now`, which runs at the same moment
   in both modes. Deferring the call does not make the decision any fresher,
   because the decision was never made early.

So the dispatcher's mechanism is sound (100 % buffered, priority reordering
observed, ordering guarantees intact — §8) and it remains **opt-in and off by
default**: on this hardware the scheduling decisions it exists to protect are
already made late enough without it. Where it would matter is a design in which the
decision must *block* on information arriving after the call — for instance
admission control that holds a batch launch until a coordinator round-trip
completes. Our coordinator answers from shared memory in nanoseconds, so there is
nothing to wait for.
