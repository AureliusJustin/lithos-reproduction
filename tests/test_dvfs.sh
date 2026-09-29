#!/usr/bin/env bash
# test_dvfs.sh — transparent power management (LithOS §5.6).
#
# Four properties, in the order they matter:
#
#   1. Without privilege it fails SAFE. Setting a clock is root-only on this
#      driver, and a mechanism that cannot engage must disable itself and leave
#      the workload correct rather than erroring out or half-applying.
#   2. It follows the paper's phases: learn at f_max, probe one lower clock to
#      derive sensitivities, then apply the model.
#   3. The frequency it picks is the one the model specifies. §5.6 gives
#      f_final = f_max / (1 + slip/S), so the logged S, slip and target must be
#      arithmetically consistent — not merely "some clock was set".
#   4. It restores the device on exit. A tenant that leaves the GPU pinned to its
#      own chosen clock would silently impose it on the next process.
set -u
BUILD=${BUILD:-build}
LIB=$BUILD/liblithos_full.so
export LITHOS_BENCH_CUBIN=${LITHOS_BENCH_CUBIN:-$BUILD/kernels/work.cubin}

# Isolate from any MPS daemon an earlier test in the suite started. LITHOS_MPS=0
# only stops LithOS from STARTING a server; a client still attaches to one already
# listening on the default pipe, and a server left in a bad state fails context
# creation outright — this test passed standalone and failed inside `make
# run_tests` for exactly that reason. A private (empty) pipe directory makes the
# run independent of the rest of the suite. Same fix as tests/test_fig10c.sh.
MPS_PIPE=$(mktemp -d)
export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE
trap 'rm -rf "$MPS_PIPE"' EXIT

fail=0

echo "test_dvfs: transparent power management (§5.6)"

# ---- 1. unprivileged: disable cleanly, workload unaffected ----------------
out=$(LITHOS_MPS=0 LITHOS_DVFS=1 LD_PRELOAD=$LIB timeout 120 $BUILD/test_interpose_driver 2>&1)
if grep -q "PASS" <<<"$out" && grep -q "power management disabled" <<<"$out"; then
    echo "  PASS: without privilege it disables itself and the workload still runs"
elif grep -q "PASS" <<<"$out"; then
    echo "  PASS: privileged environment (no disable path to exercise)"
else
    echo "  FAIL: workload did not complete with DVFS requested"; sed 's/^/    /' <<<"$out" | head -5; fail=1
fi

if ! sudo -n true 2>/dev/null; then
    echo "  SKIP: no passwordless sudo, cannot exercise the engaged path"
    [[ $fail -eq 0 ]] && echo "DVFS: PASS (0 failures, engaged path skipped)" || echo "DVFS: FAIL"
    exit $fail
fi

# ---- 2/3. engaged: phases, and the frequency the model specifies ----------
SLIP=${SLIP:-1.3}
# CUDA_MPS_PIPE_DIRECTORY has to be listed explicitly here. `sudo` resets the
# environment, so exporting it above covers the unprivileged run and nothing else
# — and without it the privileged run falls back to the default pipe and fails at
# context creation. It passed last time only because no daemon happened to exist.
log=$(sudo -n env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
      LITHOS_MPS=0 LITHOS_DVFS=1 LITHOS_DVFS_SLIP=$SLIP LITHOS_LOG_DVFS=1 \
      LITHOS_BENCH_CUBIN=$LITHOS_BENCH_CUBIN LD_PRELOAD=$LIB \
      timeout 180 $BUILD/tenant DVFS 256 20000 8 2>&1)

grep -q "learned at f_max" <<<"$log" && grep -q "probing at" <<<"$log" \
    && echo "  PASS: learn-at-f_max then probe, as §5.6 specifies" \
    || { echo "  FAIL: did not complete the learn/probe phases"; sed 's/^/    /' <<<"$log" | head -6; fail=1; }

# "S=0.944 (coverage 100%) slip=10% -> model 1275 MHz, linear-scaling 1281 MHz -> starting at 1260 MHz"
if [[ "$log" =~ S=([0-9.]+).*slip=([0-9]+)%\ -\>\ model\ ([0-9]+)\ MHz,\ linear-scaling\ ([0-9]+)\ MHz\ -\>\ starting\ at\ ([0-9]+)\ MHz ]]; then
    S=${BASH_REMATCH[1]}; slip=${BASH_REMATCH[2]}; ffin=${BASH_REMATCH[3]}
    flin=${BASH_REMATCH[4]}; applied=${BASH_REMATCH[5]}
    fmax=$(nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader | awk '{print $1}')
    python3 - "$S" "$slip" "$ffin" "$flin" "$applied" "$fmax" <<'PY'
import sys
S, slip = float(sys.argv[1]), float(sys.argv[2])/100
ffin, flin, applied, fmax = float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
want_model  = fmax / (1 + slip/S)      # §5.6's formula
want_linear = fmax / (1 + slip)        # its conservative s=1 special case
ok_model  = abs(want_model  - ffin) <= max(2.0, 0.01*want_model)
ok_linear = abs(want_linear - flin) <= max(2.0, 0.01*want_linear)
# The first application takes the HIGHER of the two and never jumps below it:
# an unvalidated extrapolation is not a safe starting point (see power.c).
start = max(want_model, want_linear)
ok_applied = applied <= start and applied < fmax
print(f"    S={S:.3f} slip={slip:.2f} f_max={fmax} -> model {want_model:.0f}, "
      f"linear {want_linear:.0f}, logged {ffin:.0f}/{flin:.0f}, started at {applied}")
if ok_model and ok_linear and ok_applied:
    print("  PASS: model, linear-scaling bound, and applied clock are all consistent")
else:
    print("  FAIL: applied clock does not follow the model"); sys.exit(1)
PY
    [[ $? -eq 0 ]] || fail=1
else
    echo "  FAIL: no frequency decision was logged"; sed 's/^/    /' <<<"$log" | head -6; fail=1
fi

# ---- 4. the device is restored -------------------------------------------
sleep 1
cur_max=$(nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader | awk '{print $1}')
if nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep -qi "SW Thermal Slowdown\|Not Active" || true; then :; fi
applied_now=$(nvidia-smi --query-gpu=clocks.applications.gr --format=csv,noheader | awk '{print $1}')
if [[ "$applied_now" == "$cur_max" ]]; then
    echo "  PASS: GPU clocks restored on exit ($applied_now MHz)"
else
    echo "  FAIL: GPU left pinned ($applied_now MHz, max $cur_max) — run: sudo nvidia-smi -rgc"; fail=1
fi

if [[ $fail -eq 0 ]]; then echo "DVFS: PASS (0 failures)"; else echo "DVFS: FAIL"; fi
exit $fail
