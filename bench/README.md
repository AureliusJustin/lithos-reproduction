# Benchmarks & experiments

Every number in [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) is produced by a
harness here. Build them all with:

```sh
make bench          # -> build/{launchbench,modload,...} and build/kernels/*.cubin
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
| `microbench.c` | the original A6000 launch/round-trip/module probe (§1) | see docs §1 |
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
FINDINGS §5 "Three bugs that only real frameworks exposed" was caught this way.

`kernels/` holds the device code: `work.cu` (tunable-duration compute kernel, the
workhorse), `nullk.cu` (launch-overhead probe), `probe_smid.cu` (records `%smid`
per block — proves a TPC mask took effect), `mm.cu` (tiled matmul for the scaling
curve), `detk.cu` (deterministic output — correctness under atomization), and
`gen_many.py` (emits an N-kernel `.cu` for splicer-scaling tests).

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
