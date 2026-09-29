# Benchmarks & experiments

Every number in the benchmark write-up (kept outside the repo) is produced by a
harness here. Build them all with:

```sh
make bench          # -> build/{launchbench,modload,...} and build/kernels/*.cubin
```

## Setting up a fresh node

The C harnesses need nothing beyond the build. The framework and figure harnesses
(`kernelmix*`, `hp_infer`, `be_train`, `stacking.sh`, `batchsweep.sh`,
`figures_paper.py`, and the PyTorch/Triton rows of `fw_all.sh`) need Python packages:

```sh
sudo apt-get install -y python3-pip
python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install --user numpy matplotlib
```

Match the wheel index to the driver's CUDA version (`nvidia-smi`); cu126 is right
for driver 560.x. PyTorch brings its own Triton, so `fw_all.sh` gets both rows.

Two experiments need **root**, which they detect and skip cleanly without it:
`dvfsbench.sh` and `tests/test_dvfs.sh` set GPU clocks, and `stacking.sh` starts
and stops the MPS daemon. Under `sudo`, remember that plain `sudo` strips `LD_*` —
LithOS needs `sudo env LD_PRELOAD=… ` or `sudo env LD_LIBRARY_PATH=… `.

## Reading the paper

`LithOS - SOSP.pdf` is the reference for everything in `docs/TECHNICAL_REPORT.md`, and this node has
no `pdftotext`/poppler. [`tools/extract_paper_text.py`](../tools/extract_paper_text.py)
recovers the text with nothing but `zlib`:

```sh
python3 tools/extract_paper_text.py "LithOS - SOSP.pdf" lithos-paper.txt
```

All harnesses run **from the repo root** and find their cubin in `build/kernels/`
(override with `LITHOS_BENCH_CUBIN=/path/to.cubin`).

Injection reminder — driver-API harnesses need `LD_PRELOAD`, not the wrapper:

```sh
LD_PRELOAD=build/liblithos_full.so build/<harness> ...
```

## Harnesses

| harness | measures | usage |
|---|---|---|
| `launchbench.c` | **per-launch host overhead** — null kernel in a tight loop, periodic sync | `build/launchbench <grid> <iters>` |
| `modload.c` | **module load + splice cost** (one-time per module) | `build/modload <cubin> <iters>` |
| `graphprof.c` | **CUDA graph** capture / instantiate / replay cost | `build/graphprof <kernels> <work> <replays>` |
| `scalebench.c` | **TPC scaling curve** for right-sizing (`l = m/t + b`) | `build/scalebench <N> <iters>` |
| `heavybench.c` | median duration of one kernel — calibrates the others | `build/heavybench <grid> <work> <iters>` |
| `hpbench.c` | **latency-critical tenant**: p50/p99/p999 round-trip | `build/hpbench <grid> <work> <iters>` |
| `beload.c` | **best-effort tenant**: closed-loop load for a fixed duration | `build/beload <grid> <work> <seconds>` |
| `tenant.c` | generic tenant: throughput **and** latency percentiles | `build/tenant <tag> <grid> <work> <secs>` |
| `usecase.py` | drives `hpbench`+`beload` to sweep **when atomization pays off** | `python3 bench/usecase.py` |
| `hp_infer.py` | latency-critical inference tenant under **Poisson** load (§8.1) | `python3 bench/hp_infer.py <rate> <secs> [batch]` |
| `be_train.py` | best-effort training tenant — long GEMM kernels (§8.1) | `python3 bench/be_train.py <secs> [batch] [width]` |
| `stacking.sh` | **the paper's headline experiment**: HP+BE across time-slicing / MPS / LithOS (§8.1, §8.4) | `bash bench/stacking.sh` |
| `microbench.c` | the original A6000 launch/round-trip/module probe (§1) | see docs §1 |
| `fixedwork.c` | fixed **iteration count** (not duration) — required for fair energy comparison | `build/fixedwork <tag> <grid> <work> <iters>` |
| `dvfsbench.sh` | **energy vs latency under DVFS** (§5.6/§8.3), sweeping the latency slip | `bash bench/dvfsbench.sh [iters] [grid] [work]`, or `WORKLOAD=torch bash bench/dvfsbench.sh 6000` for the real kernel mix |
| `kernelmix.py` | a real DL kernel mix (conv/GEMM/norm/softmax/bandwidth-bound), fixed-shape so operator ordinals are stable across runs | `python3 bench/kernelmix.py [iters]` |
| `kernelmix.sh` | **right-sizing R² and DVFS sensitivity across that mix** — the studies a single hand-written kernel cannot do | `bash bench/kernelmix.sh [iters]` |
| `kernelmix_fit.py` | fits both models to the sweep output and reports the paper's metrics, including the modelled §8.2 capacity savings | called by `kernelmix.sh` |
| `rightsize.sh` | **§8.2 right-sizing measured on a running system** — time-weighted TPC utilisation before/after, plus the p99 and throughput cost. `OCC=0` skips the stage-1 occupancy bound | `bash bench/rightsize.sh [iters]` |
| `rightsize_fit.py` | computes the paper's capacity-savings metric from the predict logs | called by `rightsize.sh` |
| `schedprof.sh` | **per-stage scheduler profile** — walks the launch path, one stage at a time | `bash bench/schedprof.sh [reps] [iters]` |
| `fw_all.sh` | **end-to-end validation** across real libraries × 8 LithOS configs | `bash bench/fw_all.sh` |
| `fw_cublas.cu` | cuBLAS SGEMM vs a CPU reference | `build/fw_cublas` |
| `fw_cufft.cu` | cuFFT C2C round-trip | `build/fw_cufft` |
| `fw_torch.py` | PyTorch: cuBLAS/cuDNN/ATen/autograd/CUDA graphs | `python3 bench/fw_torch.py` |
| `fw_triton.py` | Triton: cubins JIT-compiled at run time | `python3 bench/fw_triton.py` |

The `fw_*` targets are correctness harnesses, not performance ones, and each was
chosen because it reaches the GPU by a *different* route — kernels fetched by name,
fetched by bulk enumeration with no names, or JIT-compiled at run time. Read two
numbers from every run: the PASS/FAIL and the atomization coverage. Coverage well
below 100 % is a silent failure — LithOS is passing launches through untouched, so
everything looks healthy precisely because nothing is happening. Every bug in
the framework-only bugs listed in `docs/TECHNICAL_REPORT.md` §12 were caught this way.

`kernels/` holds the device code: `work.cu` (tunable-duration compute kernel, the
workhorse), `nullk.cu` (launch-overhead probe), `probe_smid.cu` (records `%smid`
per block — proves a TPC mask took effect), `mm.cu` (tiled matmul for the scaling
curve), `detk.cu` (deterministic output — correctness under atomization),
`membound.cu` and `randacc.cu` (streaming and scattered access, meant as the
low-frequency-sensitivity end for DVFS — on this A100 hand-written
kernels of both kinds still measure `s >= 0.8` (streaming 0.81–0.87, scattered 0.93), while a real
PyTorch mix has two thirds of its runtime in kernels with `s < 0.5` — see `docs/TECHNICAL_REPORT.md` §10), and `gen_many.py` (emits an N-kernel `.cu` for
splicer-scaling tests).

## Reproducing each finding

**Per-component overhead** (docs §3, §6)

```sh
build/launchbench 1 2000                                        # baseline
LITHOS_ATOMIZER=0 LITHOS_PREDICT=0 LD_PRELOAD=build/liblithos_full.so build/launchbench 1 2000
LD_PRELOAD=build/liblithos_full.so build/launchbench 1 2000     # default (atomizer+predictor)
```

**Splicer scaling — is it linear?** (docs §6)

```sh
python3 bench/kernels/gen_many.py 600 > /tmp/m600.cu
nvcc -arch=native -cubin /tmp/m600.cu -o /tmp/m600.cubin
build/modload /tmp/m600.cubin 20                                # baseline load
LITHOS_DIAG=1 LD_PRELOAD=build/liblithos_full.so build/modload /tmp/m600.cubin 20
#   ^ LITHOS_DIAG prints "[diag] splice: N kernels, X ms" — that X is LithOS's
#     own cost, separate from the driver loading a larger module.
```

**Right-sizing scaling curve** (docs §6) — sweep the quota, fit `l = m/t + b`:

```sh
for t in 1 2 3 4 6 9 13 18 27 40 54; do
  LITHOS_ATOMIZER=0 LITHOS_PREDICT=0 LITHOS_QUOTA=$t \
    LD_PRELOAD=build/liblithos_full.so build/scalebench 1024 40
done
```

**Spatial isolation** (docs §6) — HP with and without a TPC partition:

```sh
# plain MPS (shared TPCs)
LD_PRELOAD=build/liblithos_full.so build/tenant BE 4096 8000 8 &
LD_PRELOAD=build/liblithos_full.so build/tenant HP 16 40000 3
# LithOS partition (A100: 27+27 of 54 TPCs)
LITHOS_QUOTA=27 LITHOS_TPC_BASE=27 LITHOS_STEALING=0 LD_PRELOAD=... build/tenant BE 4096 8000 8 &
LITHOS_QUOTA=27 LITHOS_TPC_BASE=0  LITHOS_STEALING=0 LD_PRELOAD=... build/tenant HP 16 40000 3
```

**Head-of-line blocking + the use-case sweep** (docs §6, §7)

```sh
python3 bench/usecase.py     # sweeps BE kernel duration, reports HP tail gain + BE cost
```

## Gotchas that will otherwise waste your time

* **Rebuild the cubins when the GPU changes.** They are built `-arch=native`; an
  `sm_86` cubin silently fails to load on an `sm_80` device and every harness then
  reports 0 % atomization coverage, which looks like a LithOS bug. `make bench`
  after moving machines.
* **MPS needs writable directories** — the default `/var/log/nvidia-mps` is often
  not writable, and MPS then refuses to start:

  ```sh
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/lithos-mps/pipe
  export CUDA_MPS_LOG_DIRECTORY=/tmp/lithos-mps/log
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d
  ```

* **Never `kill` a co-located client mid-kernel.** On the A100 node that wedges the
  MPS server, and *every* later CUDA process hangs — indistinguishable from a
  LithOS deadlock. `usecase.py` therefore lets the BE process exit on its own.
  If it does wedge: `echo quit | nvidia-cuda-mps-control; pkill -f nvidia-cuda-mps`.
* **GPU clocks drift between processes.** Compare configurations with best-of-N
  (the harnesses report medians), and prefer interleaved A/B runs over
  back-to-back batches.
* **`LITHOS_STATS=1` is the ground truth for "did LithOS see my kernels?"**
  `0/0 launches` means it never intercepted anything (usually the wrong injection
  method, or an arch-mismatched cubin).

## Harnesses not covered above

| file | purpose |
|---|---|
| `batchbench.c` | a *batched* workload (N launches then one sync) — the shape the operator-ordinal predictor is designed for; contrast with `heavybench.c`, which syncs after every launch |
| `dispatchbench.c` | what the dispatcher is for (§5.2): a latency-sensitive stream sharing one process with a flooding batch stream |
| `stealbench.c` | does TPC stealing raise throughput, not just change the mask (two streams, disjoint per-stream quotas) |
| `idler.c` | a tenant that registers with the coordinator, launches once, then idles without exiting — the lender for cross-application stealing tests |
| `mpswork.cu` | a fixed compute-bound kernel for the MPS work-conservation comparison |
| `predacc.py` | scores predictor accuracy the paper's way (§8.4: \|error\| > 50 µs is a misprediction) using `LITHOS_PREDICT_ACC` |
| `modelrun.py` | runs one torchvision model for a fixed iteration count, so operator ordinals line up across runs |
| `scalesweep.sh` | per-kernel latency across TPC counts and GPU clocks (data for the paper's Figs. 12/13); writes a data directory |
| `figures_paper.py` | redraws the paper's Figs. 12, 13, 20 and 21 from those sweeps in the paper's own style |
