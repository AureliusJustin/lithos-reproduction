# LithOS reproduction

A from-scratch reproduction of the core mechanisms of **LithOS**
([*An Operating System for Efficient Machine Learning on GPUs*, SOSP '25](LithOS%20-%20SOSP.pdf)):
a layer that sits between an application and the NVIDIA driver, partitions the GPU
at **TPC** granularity, and transparently splits kernels into thread-block **atoms**
— with no changes to the application.

Written in C/C++/CUDA (the original is ~5k lines of Rust). It interposes at the CUDA
Driver API and reuses the QMD/TPC-masking work from [libsmctrl](https://github.com/JoshBakita/libsmctrl).
Verified on Ampere (A100 `sm_80`, RTX A6000 `sm_86`), CUDA 12.8.

## What's implemented

| Paper section | Mechanism | Source |
|---|---|---|
| §5.2 / §6 | CUDA Driver API interposition, launch queues, optional dispatcher | `interpose.c`, `sched*.c`, `dispatch.c` |
| §5.1 | System-wide coordinator (cross-process quotas / stealing) | `coord.c` |
| §5.3 | TPC scheduler: quotas, stealing, per-TPC timers, outstanding-work throttle | `tpc_alloc.c`, `sched.c` |
| §5.4 | Kernel Atomizer (range-check prologue spliced into each kernel's SASS) | `atomizer.c`, `atomize_splice.c`, `fatbin.c` |
| §5.5 / §5.7 | Right-sizing and online latency prediction | `predict.c` |
| §5.6 | Transparent power management (DVFS via NVML) | `power.c` |
| §6 | CUDA-graph subgraph scheduling | `graphsched.c` |

Deviations from the paper (notably the atomizer's splice-instead-of-jump design) are
described in the technical report below.

## Build and test

```sh
make            # build/liblithos_full.so, build/libcuda.so.1, and the tests
make run_tests  # interposition, atomizer, scheduler, dispatcher, coordinator, correctness
make bench      # benchmark harnesses (see bench/README.md)
bash bench/fw_all.sh   # end-to-end on cuBLAS, cuFFT, PyTorch, Triton (needs those installed)
```

Run both `run_tests` and `fw_all.sh`: they catch different classes of bug.

## Usage

Pick the injection method by how your app reaches the driver. The wrong one silently
runs the app natively (no interposition).

| Workload | Command |
|---|---|
| Framework / CUDA-runtime app (PyTorch, JAX, TF, TensorRT, vLLM) | `LD_LIBRARY_PATH=build python3 app.py` |
| Driver-API app (`cu*` calls directly) | `LD_PRELOAD=build/liblithos_full.so ./app` |

`LD_PRELOAD` does **not** work for runtime apps (`libcudart` `dlopen`s `libcuda.so.1` by
name). Add `LITHOS_STATS=1` and check the coverage line printed at exit: `0/0 launches`
means LithOS never saw your kernels.

```sh
# confine a tenant to 8 TPCs and split kernels into ~250 µs atoms
LITHOS_QUOTA=8 LITHOS_ATOM_US=250 LD_LIBRARY_PATH=build python3 serve.py
```

## Key settings

Everything is configured through environment variables (all defined in
[`src/config.c`](src/config.c)). The ones you will reach for first:

| Variable | Default | Meaning |
|---|---|---|
| `LITHOS_STATS` | 0 | print atomization coverage at exit — check this first |
| `LITHOS_QUOTA` | -1 | TPCs guaranteed per stream |
| `LITHOS_ATOMIZER` | 1 | enable kernel atomization |
| `LITHOS_ATOM_US` | 300 | target atom duration (µs) |
| `LITHOS_SLO_US` | 0 | latency budget imposed on a co-located tenant; caps atom size |
| `LITHOS_RIGHTSIZE` | 0 | per-kernel TPC right-sizing |
| `LITHOS_DISPATCH` | 0 | submit through dispatcher threads |
| `LITHOS_DVFS` | 0 | transparent power management (needs root) |
| `LITHOS_COORD` | 1 | cross-process tenant table; assigns disjoint TPC ranges |
| `LITHOS_GRAPH_SUBGRAPHS` | 0 | partition CUDA graphs into K schedulable subgraphs |
| `LITHOS_VERBOSE` / `LITHOS_DIAG` | 0 | logging / why-is-coverage-what-it-is diagnostics |

## Repository layout

| Path | Contents |
|---|---|
| `src/` | LibLithOS: interposition, scheduler, atomizer, predictor, coordinator, power, graphs |
| `tests/` | correctness suite run by `make run_tests`, plus QMD reverse-engineering probes |
| `bench/` | benchmark and experiment harnesses — see [bench/README.md](bench/README.md) |
| `docs/TECHNICAL_REPORT.md` | how the system works: modules, control flow, atomizer, framework validation |
| `sass-jump/`, `legacy/` | record of the SASS-transfer dead end and the superseded QMD-redirect atomizer |
| `tools/` | helper for extracting text from the paper PDF |
| `LithOS - SOSP.pdf` | the paper |

## Acknowledgements

TPC masking builds on [libsmctrl](https://github.com/JoshBakita/libsmctrl) (Bakita).
