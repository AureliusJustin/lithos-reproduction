#!/usr/bin/env bash
# rightsize.sh — §8.2 hardware right-sizing, measured rather than modelled.
#
# BENCHMARKS §12 derives the paper's capacity-savings figure from the fitted
# l = m/t + b models: for each operator, the smallest TPC count whose PREDICTED
# latency stays inside the slip budget. That is what the allocator should choose,
# not what it does choose — the models are fitted offline over a full sweep, while
# at runtime the predictor has to learn each operator from the launches it has
# actually seen, probing as it goes.
#
# This runs the mechanism instead. LITHOS_RIGHTSIZE=1 lets src/tpc_alloc.c pick a
# per-launch allocation; LITHOS_LOG_PREDICT records the allocation each launch
# actually ran at, so the paper's metric — "the time-weighted average of TPC
# utilization before and after right-sizing" — can be computed directly, together
# with the cost it reports alongside it (+4% P99, -4% throughput at slip 1.1).
#
#   bash bench/rightsize.sh [iters]
set -u
BUILD=${BUILD:-build}
ITERS=${1:-2000}
SLIPS="${SLIPS:-1.1 1.3 1.5}"
# OCC=0 skips the stage-1 occupancy bound in src/tpc_alloc.c, leaving the learned
# scaling model to decide alone — the two stages disagree, so both are measurable.
OCC=${OCC:-1}
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT

command -v python3 >/dev/null || { echo "rightsize: no python3"; exit 1; }
python3 -c "import torch" 2>/dev/null || { echo "rightsize: PyTorch not installed — skipping"; exit 0; }

# Frameworks reach the driver through the libcuda.so.1 wrapper, never LD_PRELOAD.
run() {   # run <tag> <env...>
    local tag=$1; shift
    env "$@" LITHOS_QUOTA=54 LITHOS_ATOMIZER=0 LITHOS_LOG_PREDICT=1 LITHOS_MPS=0 \
        LD_LIBRARY_PATH=$BUILD \
        timeout 900 python3 bench/kernelmix.py "$ITERS" \
        >"$OUT/$tag.out" 2>"$OUT/$tag.log"
}

echo "== right-sizing: $(python3 -c 'import torch;print("torch "+torch.__version__)'), $ITERS iterations, stage-1 bound $OCC =="
run off LITHOS_RIGHTSIZE=0
echo "   baseline done"
for k in $SLIPS; do
    run "slip_$k" LITHOS_RIGHTSIZE=1 LITHOS_RIGHTSIZE_OCC="$OCC" LITHOS_SLIP="$k"
    echo "   slip $k done"
done

python3 bench/rightsize_fit.py "$OUT" $SLIPS
