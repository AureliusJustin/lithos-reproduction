#!/usr/bin/env bash
# scalesweep.sh — per-kernel TPC and frequency scaling data, one workload per panel.
#
# Collects what the paper's Figures 12 and 13 plot: for every kernel of a workload,
# its latency at each TPC count and at each GPU clock, so a scaling curve can be
# drawn per kernel and weighted by that kernel's share of total execution time.
#
#   bash bench/scalesweep.sh <outdir>
set -u
BUILD=${BUILD:-build}
OUT=${1:-figdata}
ITERS=${ITERS:-40}
mkdir -p "$OUT"
python3 -c "import torch, torchvision" 2>/dev/null || { echo "needs torch+torchvision"; exit 0; }

# One entry per panel: label:model:mode:batch
WORKLOADS=${WORKLOADS:-"resnet50-infer:resnet50:infer:8 vgg19-train:vgg19:train:32 mobilenet_v2-train:mobilenet_v2:train:64"}
TPCS=${TPCS:-"54 40 27 18 13 9 6 4 3 2 1"}
CLOCKS=${CLOCKS:-"1410 1275 1140 1005 870 750"}

run() {  # run <logfile> <env...> -- <model> <mode> <batch>
    local log=$1; shift
    local envs=() ; while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done; shift
    env "${envs[@]}" LITHOS_LOG_PREDICT=1 LITHOS_MPS=0 LITHOS_ATOMIZER=0 \
        LD_LIBRARY_PATH=$BUILD timeout 900 \
        python3 bench/modelrun.py "$1" "$2" "$3" "$ITERS" >/dev/null 2>"$log"
}

for wl in $WORKLOADS; do
    IFS=: read -r label model mode batch <<<"$wl"
    echo "== $label =="
    for t in $TPCS; do
        run "$OUT/${label}__tpc_${t}.log" LITHOS_QUOTA=$t -- "$model" "$mode" "$batch"
        printf "   %2d TPC: %s kernels\n" "$t" "$(grep -c '^\[predict\]' "$OUT/${label}__tpc_${t}.log")"
    done
    if sudo -n true 2>/dev/null; then
        for f in $CLOCKS; do
            sudo -n nvidia-smi -lgc "$f,$f" >/dev/null 2>&1; sleep 1
            run "$OUT/${label}__clk_${f}.log" LITHOS_DUMMY=1 -- "$model" "$mode" "$batch"
            printf "   %4d MHz: %s kernels\n" "$f" "$(grep -c '^\[predict\]' "$OUT/${label}__clk_${f}.log")"
        done
        sudo -n nvidia-smi -rgc >/dev/null 2>&1
    fi
done
echo "data in $OUT"
