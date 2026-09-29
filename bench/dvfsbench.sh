#!/usr/bin/env bash
# dvfsbench.sh — energy and latency under transparent power management (§5.6/§8.3).
#
# The paper's metric: energy is average power times wall time, with power sampled
# by nvidia-smi every 100 ms (its smallest granularity), measured over a fixed
# amount of work so the comparison is like for like. Reported against the latency
# cost, since the whole mechanism is a controlled trade of one for the other.
#
# The knob is the latency slip: LithOS solves f_final = f_max / (1 + slip/S) for
# the aggregate sensitivity S it has learned, so a larger slip buys a lower clock.
# Sweeping it traces the trade-off curve the paper's slip parameter exposes.
#
# Setting clocks needs privilege, so this runs LithOS under sudo (plain sudo
# strips LD_*, hence `sudo env`).
#
#   bash bench/dvfsbench.sh [iterations] [grid] [work]
set -u
BUILD=${BUILD:-build}
LIB=$BUILD/liblithos_full.so
ITERS=${1:-200000}
GRID=${2:-256}
WORK=${3:-20000}
CUBIN=${LITHOS_BENCH_CUBIN:-$BUILD/kernels/work.cubin}
# WORKLOAD=torch measures the real kernel mix (bench/kernelmix.py) instead of the
# synthetic loop. It is the more informative of the two: a real mix contains the
# frequency-insensitive kernels that DVFS exists for, which a hand-written loop
# does not (BENCHMARKS §11).
WORKLOAD=${WORKLOAD:-synthetic}
USER_HOME=${USER_HOME:-$HOME}
PW=$(mktemp)

if ! sudo -n true 2>/dev/null; then
    echo "dvfsbench: needs passwordless sudo to set GPU clocks — skipping"; exit 0
fi

run() {   # run <label> <extra env...>
    sudo -n nvidia-smi -rgc >/dev/null 2>&1; sleep 1
    ( nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms 100 >"$PW" 2>/dev/null ) &
    local smp=$! t0 t1 out
    t0=$(date +%s.%N)
    if [[ "$WORKLOAD" == torch ]]; then
        # A framework reaches the driver through the libcuda.so.1 wrapper, and sudo
        # resets the environment, so HOME must be carried through too or the
        # user-site PyTorch is invisible to root.
        out=$(sudo -n env HOME="$USER_HOME" LITHOS_MPS=0 "${@:2}" \
              LD_LIBRARY_PATH=$BUILD timeout 1800 python3 bench/kernelmix.py "$ITERS" 2>/dev/null | tail -1)
    else
        out=$(sudo -n env LITHOS_MPS=0 LITHOS_BENCH_CUBIN=$CUBIN "${@:2}" \
              LD_PRELOAD=$LIB timeout 1800 $BUILD/fixedwork DV "$GRID" "$WORK" "$ITERS" 2>/dev/null | tail -1)
    fi
    t1=$(date +%s.%N)
    kill $smp 2>/dev/null; wait $smp 2>/dev/null
    sudo -n nvidia-smi -rgc >/dev/null 2>&1
    python3 - "$1" "$t0" "$t1" "$PW" "$out" <<'PY'
import sys
label, t0, t1, pwf, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
p = [float(x) for x in open(pwf) if x.strip()]
t = t1 - t0
avg = sum(p)/len(p) if p else 0.0
# fixedwork prints: [tag] N iters in T s  p50=X p99=Y ms
# kernelmix prints: kernelmix: N iters in T s (X ms/iter) checksum=...
p50 = p99 = 0.0
for tok in out.split():
    if tok.startswith('p50='): p50 = float(tok[4:])
    elif tok.startswith('p99='): p99 = float(tok[4:])
print(f"{label:<22} {t:7.2f} s  p50 {p50:7.3f} p99 {p99:7.3f} ms  {avg:6.1f} W  {avg*t:8.1f} J")
PY
}

echo "energy = avg power x wall time, power sampled at 100 ms (paper's method)"
if [[ "$WORKLOAD" == torch ]]; then
    echo "workload: bench/kernelmix.py (real DL kernel mix)  $ITERS iterations per run (FIXED work)"
else
    echo "kernel: $(basename "$CUBIN")  grid=$GRID work=$WORK  $ITERS iterations per run (FIXED work)"
fi
printf "%-22s %9s %25s %9s %10s\n" "configuration" "time" "latency" "power" "energy"
run "baseline (no DVFS)"  LITHOS_DVFS=0
for slip in 1.1 1.3 1.5 2.0; do
    run "DVFS slip=$slip"  LITHOS_DVFS=1 LITHOS_DVFS_SLIP=$slip
done
rm -f "$PW"
