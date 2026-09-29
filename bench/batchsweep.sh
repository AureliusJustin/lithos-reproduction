#!/usr/bin/env bash
# batchsweep.sh — LithOS Figure 21: head-of-line blocking against the size of the
# best-effort job's kernels.
#
# §8.4 varies the best-effort training batch size and measures the latency-critical
# job's tail. The mechanism is simple and is the whole argument for atomization:
# a bigger batch means longer kernels, a GPU cannot preempt a kernel, so the
# service waits behind whatever is running. Atomization shrinks the unit that has
# to finish before the service can be served.
#
# Run in the DONATED configuration (equal priority, so the service's idle TPCs are
# actually lent out) — the fenced configuration has no blocking to show.
#
#   bash bench/batchsweep.sh > figdata/batch.txt
set -u
BUILD=${BUILD:-build}
DUR=${DUR:-10}
RATE=${RATE:-300}
Q=${Q:-27}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
L="LITHOS_MPS=0 LD_LIBRARY_PATH=$BUILD LITHOS_QUOTA=$Q LITHOS_PRIORITY=0"

# MPS is not optional here. LithOS "builds on top of MPS" (§6) because without it
# separate processes TIME-SLICE the GPU: they never run concurrently, so a TPC
# partition has nothing to partition and the service's tail is set by the
# time-slice quantum instead. Measured without it, this sweep reports 24-68 ms
# tails that say nothing about atomization.
[[ -e /tmp/nvidia-mps/control ]] || { nvidia-cuda-mps-control -d >/dev/null 2>&1; sleep 2; }
[[ -e /tmp/nvidia-mps/control ]] || { echo "batchsweep: MPS failed to start" >&2; exit 1; }

REPS=${REPS:-3}   # co-location is noisy; a single run inverts the trend

for batch in 512 1024 2048 4096; do
    for mode in off on; do
        atom=$([[ $mode == off ]] && echo "LITHOS_ATOMIZER=0" || echo "LITHOS_ATOMIZER=1")
        p95s=(); bets=()
        for ((r = 0; r < REPS; r++)); do
            rm -f /dev/shm/lithos_coord     # fresh tenant table per measurement
            env $L $atom timeout -k 10 $((DUR + 40)) \
                python3 bench/be_train.py $((DUR + 5)) "$batch" 4096 >"$TMP/be" 2>/dev/null &
            be=$!
            sleep 3
            env $L $atom timeout -k 10 $((DUR + 40)) \
                python3 bench/hp_infer.py "$RATE" "$DUR" 8 >"$TMP/hp" 2>/dev/null
            wait $be 2>/dev/null
            p95s+=("$(sed -E 's/.*p95=([0-9.]+).*/\1/' "$TMP/hp" | tail -1)")
            bets+=("$(sed -E 's/.*tput=([0-9.]+).*/\1/' "$TMP/be" | tail -1)")
        done
        python3 - "$batch" "$mode" "${p95s[@]}" -- "${bets[@]}" <<'PY'
import sys, statistics
batch, mode = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
i = rest.index("--")
p95 = [float(x) for x in rest[:i] if x]
bet = [float(x) for x in rest[i+1:] if x]
print(f"{batch} {mode} {statistics.median(p95):.3f} {statistics.median(bet):.2f}")
PY
    done
done
