#!/usr/bin/env bash
# kernelmix.sh — right-sizing accuracy and DVFS sensitivity across a REAL kernel mix.
#
# Both studies work the same way: run the same fixed-shape model repeatedly under
# LithOS while varying one thing (TPC quota, or GPU clock), and read the
# per-operator latencies LithOS already measures (§5.7, LITHOS_LOG_PREDICT).
# Operators are identified by their ordinal within a batch, which is stable across
# runs because the model is fixed-shape and synchronizes every iteration — so a given
# operator's measurements can be joined across runs and fitted.
#
#   study 1 (§8.2): fit l = m/t + b per operator over a TPC sweep, and report the
#                   EXECUTION-TIME-WEIGHTED mean R^2, which is the paper's metric.
#   study 2 (§5.6): compute per-operator frequency sensitivity over a clock sweep,
#                   and report its distribution — the thing a single hand-written
#                   kernel cannot show.
#
#   bash bench/kernelmix.sh [iters]
set -u
BUILD=${BUILD:-build}
ITERS=${1:-150}
# Set DATA_DIR to keep the raw sweep logs — bench/figures_paper.py reads them to redraw
# the paper's per-kernel scaling figures.
if [[ -n "${DATA_DIR:-}" ]]; then
    OUT=$DATA_DIR; mkdir -p "$OUT"
else
    OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT
fi

command -v python3 >/dev/null || { echo "kernelmix: no python3"; exit 1; }
python3 -c "import torch" 2>/dev/null || { echo "kernelmix: PyTorch not installed — skipping"; exit 0; }

# Frameworks reach the driver through the libcuda.so.1 wrapper, never LD_PRELOAD.
run() {   # run <outfile> <env...>
    local out=$1; shift
    env "$@" LITHOS_LOG_PREDICT=1 LITHOS_MPS=0 LD_LIBRARY_PATH=$BUILD \
        timeout 900 python3 bench/kernelmix.py "$ITERS" >/dev/null 2>"$out"
}

echo "== kernel mix: $(python3 -c 'import torch;print("torch "+torch.__version__)') =="

# ---- study 1: right-sizing accuracy over a TPC sweep ----------------------
TPCS="${TPCS:-54 40 27 18 13 9 6 4 3 2 1}"
echo "-- sweeping TPC quota ($TPCS) --"
for t in $TPCS; do
    run "$OUT/tpc_$t.log" LITHOS_QUOTA=$t LITHOS_ATOMIZER=0
    printf "   quota %2d: %s measurements\n" "$t" "$(grep -c '^\[predict\]' "$OUT/tpc_$t.log")"
done

# ---- study 2: DVFS sensitivity over a clock sweep -------------------------
CLOCKS="${CLOCKS:-1410 1215 1005 810}"
if sudo -n true 2>/dev/null; then
    echo "-- sweeping GPU clock ($CLOCKS MHz) --"
    for f in $CLOCKS; do
        sudo -n nvidia-smi -lgc "$f,$f" >/dev/null 2>&1; sleep 1
        run "$OUT/clk_$f.log" LITHOS_ATOMIZER=0
        printf "   %4d MHz: %s measurements\n" "$f" "$(grep -c '^\[predict\]' "$OUT/clk_$f.log")"
    done
    sudo -n nvidia-smi -rgc >/dev/null 2>&1
else
    echo "-- clock sweep skipped (needs sudo) --"
fi

python3 bench/kernelmix_fit.py "$OUT"
