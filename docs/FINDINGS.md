# Findings

What the experiments actually established — the results worth remembering,
separated from the raw numbers in [BENCHMARKS.md](BENCHMARKS.md) and the
paper-comparison in [FIDELITY.md](FIDELITY.md).

All measurements are from the **current** system on an **A100 80GB PCIe**
(GA100, `sm_80`, 54 TPCs; drivers 610.43.02 and 570.195.03 — the node was
re-provisioned mid-work, and spot-checks agreed across both). The RTX A6000 (GA102, drivers
570.195.03 / 595.71.05) is **functionally** validated — the full suite passes
there — but is no longer a source of performance numbers.
Harnesses and repro commands: [`bench/README.md`](../bench/README.md).

---

## 1. When is LithOS worth enabling?

**The interfering kernel's duration decides it — nothing else comes close.**
Sweeping a best-effort kernel's length while a latency-critical job runs alongside
(`bench/usecase.py`, `LITHOS_SLO_US=300`):

| BE kernel duration | HP p99 un-atomized | HP p99 atomized | HP gain | BE throughput kept |
|---|---|---|---|---|
| 0.04 ms | 28.6 µs | 32.0 µs | **1× (none)** | 96 % |
| 0.24 ms | 130 µs | 90 µs | 1.4× | 65 % |
| 2.2 ms | 1 869 µs | 257 µs | **7×** | 62 % |
| 10.8 ms | 10 473 µs | 360 µs | **29×** | 67 % |

* **Short kernels (< ~0.1 ms): do not atomize.** There is no head-of-line blocking
  to remove, so the split buys nothing and costs best-effort throughput. (The paper
  says the same — "LithOS may disable atomization for kernels with many short
  threads"; `LITHOS_MIN_BLOCKS` and the overhead floor implement it.)
* **Long kernels (> ~2 ms): atomize.** 7–29× tail-latency reduction for roughly a
  third of BE throughput — the trade a latency-SLO tenant wants.

**Use cases where it pays off**

1. **Latency-critical inference co-located with training.** Training kernels are
   milliseconds (the paper measures DLRM > 30 ms), so HoL blocking dominates and
   atomization removes it. LithOS's strongest case.
2. **Multi-tenant inference under a tail SLO.** Spatial TPC partitioning *alone*
   restores a co-located tenant to **99 % of solo throughput and 1.04× solo tail**,
   versus 29 % / 3.4× under plain MPS — atomization isn't even needed here.
3. **Capacity reclamation.** Right-sizing frees **33 %** of the GPU at a 1.5×
   latency slip on kernels with diminishing returns.

**Where it does not pay off:** uniformly short-kernel workloads, single-tenant jobs
that own the whole GPU, and CUDA-graph-replay-dominated workloads (in-graph
atomization roughly doubles replay cost; use `LITHOS_GRAPH_SUBGRAPHS`).

---

## 2. The mechanisms, and what each is actually worth

| mechanism | measured effect | cost |
|---|---|---|
| **Interposition** | transparent on both paths | **≲0.2 µs/launch** (+0.03 best-case; +0.13–0.23 measured interleaved on a second driver) |
| **TPC quotas** (spatial isolation) | HP keeps 99 % of solo throughput, 1.04× solo tail | ~0 µs/launch |
| **Kernel atomization** | HP p99 **10 577 → 453 µs (23×)**; residual ≈ one atom | +0.33 µs/launch, ~⅓ of BE throughput |
| **Right-sizing** | `l = 23.88/t + 0.0285`, **R² = 1.0000**; freeing 14 of 54 TPCs costs 1 % latency and yields **+41 % aggregate throughput** | within noise |
| **Latency predictor** | operator-indexed, event-measured; **0–1.6 % misprediction** (paper: 0.9 %/0.38 % HP) | **+4.1 µs/launch — the dominant cost** |
| **TPC stealing** | intra-process **1.92×**; **cross-application 1.99×** (311 → 620 it/s). But equal-priority stealing **inverts priority**: HP p99 62.5 → 1 650 µs (26× worse). The priority safeguard fixes it (→ 55.7 µs) while keeping work conservation | ~0 (scan skipped when it can't help) |
| **Coordinator** (§5.1) | cross-app quotas + stealing + the priority safeguard that makes stealing safe | ~0 (shared-memory reads) |
| **Graph subgraphs** | per-subgraph TPC allocation, reallocation 7–14 µs/subgraph | +14 % replay (vs +114 % in-graph) |
| **Throttle** (opt-in) | ~20 % HP tail gain vs a deep queue of *short* kernels, at 5× BE throughput cost; **nothing** vs long kernels | ~0 when under the limit |
| **Dispatcher** (opt-in) | true deferral **works** (100 % buffered, 0 fallback, 105 reorders, ordering preserved) but **does not improve tail latency here** — nothing left to reorder | +60 µs p50 — off by default |

The balance shifted during optimization: the atomizer used to dominate (~5 µs) and
is now nearly free (+0.33 µs), while the **predictor** — on by default — is the
main per-launch cost (+4.1 µs). `LITHOS_PREDICT=0` gets it back.

**Two mechanisms do not earn their keep here — for the same underlying reason.**
Both act on *submission*, and submission is not the binding constraint on this
hardware. The **throttle** bounds queue depth, which cannot fix head-of-line
blocking from one long kernel (no preemption). The **dispatcher** defers work so it
can be reordered — verified working (100 % buffered, 105 priority reorders,
ordering intact) — but in a deep-backlog test HP p50 is ~5 700 µs in *every*
configuration **including native CUDA**, because HP is waiting on the kernel already
executing; and in a shallow one the GPU already interleaves the two streams, so
there is no queue to reorder. Atomization fixes exactly this, 23–29×, by shrinking
the in-flight unit.

This is not a refutation of the paper. There, deferral is the *enabler*: it keeps
work uncommitted so priority, TPC allocation, atomization and right-sizing can still
be applied to it under a cross-tenant scheduler. This reproduction applies those
decisions **inline at launch** — computing a mask or an atom count needs no
buffering — so deferral has nothing left to buy. It would pay where the decision
itself must await information arriving after the call (e.g. holding a low-priority
launch until a central scheduler confirms no HP tenant needs the TPCs). The
coordinator now exists (`src/coord.c`), so that experiment is finally constructible
— it is the natural next test for the dispatcher. Both stay opt-in. See BENCHMARKS §7.

---

## 3. Reproducing the paper

Every experiment that exercises a mechanism reproduces the paper's qualitative
result, in the right quantitative ballpark:

| paper result | paper | here (A100) |
|---|---|---|
| Spatial isolation beats MPS (§8.1) | MPS worst on latency; LithOS = MIG-like isolation at full throughput | HP 99 % of solo tput, 1.04× tail vs 29 % / 3.4× under MPS ✅ |
| Atomization cuts HoL blocking (§8.4) | tail within 14 % of ideal | 23× tail reduction; residual = one atom ✅ |
| Right-sizing model accuracy (§8.2) | R² 0.92–0.99 | **R² = 1.0000** ✅ |
| Right-sizing capacity savings (§8.2) | mean 26 %, up to 51 % | 33 % @ 1.5× slip (9 % @ 1.1×) ✅ |
| MPS work conservation | MPS sets the throughput bar | **4.05×** over time-slicing ✅ |

**Not reproduced:** the multi-system comparison (MIG / REEF / TGS / Orion — needs
those systems), DVFS energy (§8.3 — out of scope), and real serving-stack SLO
attainment (needs Triton + real models; synthetic HP/BE proxies stand in).

---

## 4. Things the paper doesn't say that we had to find out

* **CUDA graphs freeze the TPC allocation.** The QMD pre-upload callback — the only
  hook for SM masking — fires **exactly once per graph exec**, at the first
  `cuGraphLaunch`. Replays re-run a compiled command buffer that bypasses the
  driver's per-node path, so a graph's allocation **cannot change on replay**;
  only **re-instantiation** re-fires the callback (verified). This is why the
  subgraph model exists: rebuilding one small subgraph is cheap (~8 µs), rebuilding
  the whole graph is not. The paper's one sentence on graphs doesn't address it.
* **Atom size *is* the tail latency.** A co-located latency-critical tenant waits
  behind at most one atom — measured tail tracked atom size almost exactly. That
  motivates deriving `atom_duration` from the neighbour's SLO
  (`LITHOS_SLO_US`) rather than the paper's fixed 250–500 µs constant, and
  deriving a floor from measured overhead so the paper's "too low may be slower"
  caveat becomes an invariant. See BENCHMARKS §4.
* **`%smid` is renumbered under a mask.** A kernel confined to TPCs `[base, base+w)`
  reports `%smid` 0…2w−1 regardless of `base`. The SM *count* is truthful but the
  identity is not — so the **mask** is the ground truth for *which* TPCs a launch
  got; `%smid` only confirms *how many*.
* **A transparent interposer cannot safely defer a launch.** `cuLaunchKernel`'s
  `void** kernelParams` carries no sizes (they live in the cubin's `.nv.info`), so
  the arguments cannot be deep-copied. The dispatcher is therefore a *hand-off*
  (caller waits) rather than fire-and-forget — a real limit on how far §5.2's
  decoupling can be reproduced transparently.
* **Stealing needs disjoint ranges to do anything.** With the paper's
  one-quota-per-application model every stream of a process shares one range, and
  lending a range identical to your own is a provable no-op. That is why the
  system-wide coordinator matters: it hands *tenants* disjoint ranges, which is
  what makes §5.3's cross-application stealing real (**1.99× throughput** for the
  borrower). Intra-process stealing between streams still needs
  `LITHOS_PERSTREAM_QUOTA=1`.

---

## 5. Operational gotchas (cost real debugging time)

* **MPS refuses the default log directory** (`/var/log/nvidia-mps`, usually not
  writable) and then won't start. Set `CUDA_MPS_PIPE_DIRECTORY` /
  `CUDA_MPS_LOG_DIRECTORY` to writable paths.
* **MPS wedges if a client is killed mid-kernel** — every subsequent CUDA process
  hangs, looking exactly like a LithOS deadlock. Let co-located clients exit
  naturally; recover with `echo quit | nvidia-cuda-mps-control; pkill -f nvidia-cuda-mps`.
* **Arch-mismatched cubins fail silently**, showing up as 0 % atomization coverage
  rather than an error. Rebuild benchmark cubins (`make bench`) after changing GPU.
* **The injection method is not interchangeable.** Framework/runtime apps need the
  `libcuda.so.1` wrapper (`LD_LIBRARY_PATH=build`); `LD_PRELOAD` is invisible to
  them because `libcudart` resolves the driver from its own `dlopen` handle.
  `LITHOS_STATS=1` reporting `0/0 launches` is the tell.

### Three bugs that only real frameworks exposed

Each passed the entire unit suite and was found by `bash bench/fw_all.sh`. What
they have in common is that the *failure mode was silence*: correct-looking
output, or a healthy-looking system doing nothing.

* **Intercepting every API that hands out a kernel handle is not enough.** cuFFT
  never calls `cuModuleGetFunction` or `cuLibraryGetKernel`; it uses
  `cuLibraryEnumerateKernels`, which returns handles in bulk with **no names**. Its
  kernels were therefore classified "not ours" and ran at **0 % coverage** — while
  still computing the right answer, so nothing looked wrong. Chasing each new
  enumeration API is a losing game, so the atomizer now falls back to asking the
  driver (`cuFuncGetModule` + `cuFuncGetName`) which module an unknown handle came
  from. That closes the whole class, including APIs that don't exist yet.
  `LITHOS_DIAG=1` now prints the classification of every kernel by name, and names
  any kernel that ran un-atomized.
* **`cuCtxSetCurrent` binds per thread; the cache for it was process-wide.** Only
  the first dispatcher worker ever called it. Every other worker skipped a bind it
  had never performed and submitted with no current context, so its launches failed
  with `CUDA_ERROR_INVALID_CONTEXT` — silently, because a deferred launch has
  already returned success to the application. In PyTorch this surfaced as
  gradients that were quietly zero or garbage. Two details made it hide well: a
  launch on an *explicit* stream still succeeds without a current context (the
  driver infers it from the stream handle), so only **legacy-default-stream**
  launches fail — and that is exactly where frameworks put most of their work.
  `tests/test_dispatch_mt.cu` reproduces it (144/256 launches dropped); it must be
  a CUDA-*runtime* app on the default stream, or the bug stays hidden.
* **Creating the coordinator's shm segment was itself a race.** "Did I create it?"
  was inferred from the segment's size being zero, so two tenants starting together
  both sized and zeroed it, and the second erased the first's registration. Each
  then saw an empty table and both were assigned the **same TPC range** — the exact
  overlap the coordinator exists to prevent. `O_EXCL` now decides the creator, which
  publishes the magic only after initialising. It collided ~2 times in 5 and never
  in 10 after the fix; `tests/test_coord.sh` repeats the simultaneous join 6 times,
  since one attempt is not enough to catch it.
