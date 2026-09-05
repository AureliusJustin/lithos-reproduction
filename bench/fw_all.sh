#!/usr/bin/env bash
# fw_all.sh — end-to-end validation across real libraries and frameworks.
#
# LithOS is transparent by construction, so the thing worth testing is not a
# microbenchmark but whether closed-source, third-party GPU code still computes
# the right answer while the atomizer is splicing its kernels and the scheduler
# is masking its TPCs. Each target below reaches the GPU by a DIFFERENT route:
#
#   cuBLAS   fatbin kernels fetched by name (cuModuleGetFunction / cuLibraryGetKernel)
#   cuFFT    kernels fetched by BULK ENUMERATION, with no name -- the case that
#            needs the atomizer's classify-by-handle fallback
#   Triton   kernels JIT-compiled at run time and loaded as fresh cubins
#   PyTorch  cuBLAS + cuDNN + ATen + autograd (a second thread) + CUDA graphs
#
# Two numbers matter per run: the framework's own PASS/FAIL (results still
# correct) and atomization coverage (LithOS actually engaged rather than quietly
# passing everything through). Coverage well under 100% is a silent failure --
# the system looks healthy exactly because it is doing nothing.
#
#   bash bench/fw_all.sh
set -u
BUILD=${BUILD:-build}
CUDA=${CUDA:-/usr/local/cuda}
NVCC=${NVCC:-$CUDA/bin/nvcc}
ARCH=${ARCH:-native}
fail=0

# Every framework here uses the CUDA runtime, so all of them take the
# libcuda.so.1 wrapper path (LD_LIBRARY_PATH), never LD_PRELOAD.
CONFIGS=(
  ""
  "LITHOS_ATOM_US=1"
  "LITHOS_QUOTA=16"
  "LITHOS_RIGHTSIZE=1"
  "LITHOS_DISPATCH=1"
  "LITHOS_DISPATCH=1 LITHOS_DISPATCH_THREADS=4"
  "LITHOS_PREDICT=1"
  "LITHOS_QUOTA=16 LITHOS_ATOM_US=1 LITHOS_DISPATCH=1 LITHOS_RIGHTSIZE=1 LITHOS_PREDICT=1"
)

run_matrix() {   # run_matrix <label> <command...>
    local label=$1; shift
    echo "== $label =="
    for cfg in "${CONFIGS[@]}"; do
        local out ok cov
        out=$(env $cfg LITHOS_STATS=1 LD_LIBRARY_PATH=$BUILD timeout 900 "$@" 2>&1)
        ok=$(echo "$out"  | grep -oE "OVERALL: (PASS|FAIL)|\[(PASS|FAIL)\]" | head -1 \
                          | grep -oE "PASS|FAIL")
        cov=$(echo "$out" | grep -oE "coverage: [0-9]+/[0-9]+ launches = [0-9.]+%" \
                          | head -1 | sed 's/coverage: //')
        printf "  %-56s %-5s %s\n" "${cfg:-default}" "${ok:-NO-OUTPUT}" "$cov"
        [[ "$ok" == "PASS" ]] || fail=1
    done
}

mkdir -p "$BUILD"
for t in fw_cublas fw_cufft; do
    lib=$([[ $t == fw_cublas ]] && echo -lcublas || echo -lcufft)
    [[ -x $BUILD/$t ]] || $NVCC -arch=$ARCH -O2 bench/$t.cu -o $BUILD/$t $lib || exit 2
done

run_matrix "cuBLAS (SGEMM vs CPU reference)"        "$BUILD/fw_cublas"
run_matrix "cuFFT (C2C round-trip)"                 "$BUILD/fw_cufft"
if python3 -c "import torch" 2>/dev/null; then
    run_matrix "PyTorch (cuBLAS/cuDNN/ATen/autograd/graphs)" python3 bench/fw_torch.py
else echo "== PyTorch == SKIP (not installed)"; fi
if python3 -c "import triton" 2>/dev/null; then
    run_matrix "Triton (JIT-compiled cubins)"       python3 bench/fw_triton.py
else echo "== Triton == SKIP (not installed)"; fi

if [[ $fail -eq 0 ]]; then echo "FRAMEWORKS: PASS (0 failures)"; else echo "FRAMEWORKS: FAIL"; fi
exit $fail
