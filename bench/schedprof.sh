#!/usr/bin/env bash
# schedprof.sh — per-stage profile of the TPC Scheduler (§5.2/5.3/5.7).
#
# The workload is a NULL kernel at grid=1: no GPU work, and below
# LITHOS_MIN_BLOCKS so it is never split. That is deliberate — what is measured
# here is the scheduler's HOST-side cost per launch, not the GPU effect of any
# policy. MPS is off (LITHOS_MPS=0), since the MPS server hop adds a roughly
# constant amount to every row and would swamp the differences.
#
# Two tables, because the stages are not all of one kind:
#
#   A. THE PIPELINE — stages that genuinely compose, each row keeping everything
#      above it. Only here is a "Δ prev" the cost of that stage.
#   B. ALTERNATIVES — stages that are opt-in policies rather than pipeline steps.
#      Each is measured against the SAME reference config (the last pipeline row
#      before the throttle), so their costs are comparable to each other and not
#      to a moving baseline.
#
# Reported best-of-N: the minimum is the right statistic for a host-side cost,
# being the run least perturbed by the OS scheduler. The spread across reps is
# printed too, so a row whose delta is smaller than its own noise is visible as
# such rather than being read as a real effect.
#
#   bash bench/schedprof.sh [reps] [launches]
set -u
BUILD=${BUILD:-build}
REPS=${1:-5}
ITERS=${2:-2000}
export LITHOS_MPS=0
P="LD_PRELOAD=$BUILD/liblithos_full.so"

# Everything the scheduler can do, off. Each pipeline row switches exactly one on.
OFF="LITHOS_ATOMIZER=0 LITHOS_PREDICT=0 LITHOS_STEALING=0 LITHOS_COORD=0"

run_one() { env "$@" $BUILD/launchbench 1 "$ITERS" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+'; }

# Echoes "<min> <spread>" over REPS runs.
best() {
    local vals=()
    for _ in $(seq "$REPS"); do
        local v; v=$(run_one "$@")
        [[ -n "$v" ]] && vals+=("$v")
    done
    [[ ${#vals[@]} -gt 0 ]] || { echo "NA NA"; return; }
    python3 -c "

v=[float(x) for x in '${vals[*]}'.split()]
print(f'{min(v):.2f} {max(v)-min(v):.2f}')"
}

PREV=""
row() {    # row <label> <min> <spread>
    local label=$1 val=$2 spread=$3 d="—"
    [[ -n "$PREV" && "$val" != "NA" ]] && d=$(python3 -c "print(f'{$val-$PREV:+.2f}')")
    printf "  %-44s %9s %9s %9s\n" "$label" "$val" "$d" "±$spread"
    PREV=$val
}
hdr() { printf "  %-44s %9s %9s %9s\n" "$1" "µs/launch" "Δ prev" "spread"; }

echo "TPC Scheduler per-launch profile (null kernel, grid=1, best-of-$REPS, MPS off)"
echo
echo "A. Pipeline — each row keeps every stage above it"
hdr "stage"

read -r v s <<<"$(best env -u LD_PRELOAD)"; row "baseline (no LithOS)" "$v" "$s"

# Stream registration is unconditional (ensure_stream allocates a slot for every
# stream), so it is already inside "interposition only" — there is no config that
# separates them, and pretending otherwise would just report noise.
C="$P $OFF"
read -r v s <<<"$(best $C)";                                   row "interposition + launch queues (§5.2)" "$v" "$s"
C="$C LITHOS_QUOTA=16"
read -r v s <<<"$(best $C)";                                   row "+ TPC quota mask (§5.2, QMD callback)" "$v" "$s"
C="$P LITHOS_PREDICT=0 LITHOS_STEALING=0 LITHOS_COORD=0 LITHOS_QUOTA=16"
read -r v s <<<"$(best $C)";                                   row "+ atomizer gating (§5.4, no split here)" "$v" "$s"
C="$P LITHOS_STEALING=0 LITHOS_COORD=0 LITHOS_QUOTA=16"
read -r v s <<<"$(best $C)";                                   row "+ predictor (§5.7, 1 event/launch)" "$v" "$s"
C="$P LITHOS_COORD=0 LITHOS_QUOTA=16 LITHOS_PERSTREAM_QUOTA=1 LITHOS_STEALING=1"
read -r v s <<<"$(best $C)";                                   row "+ TPC stealing scan (§5.3)" "$v" "$s"
REF="$P LITHOS_QUOTA=16 LITHOS_PERSTREAM_QUOTA=1 LITHOS_STEALING=1 LITHOS_COORD=1"
read -r v s <<<"$(best $REF)";                                 row "+ system-wide coordinator (§5.1)" "$v" "$s"
REFV=$PREV

echo
echo "B. Opt-in policies — each measured against the row above (${REFV} µs)"
hdr "policy"
PREV=$REFV
read -r v s <<<"$(best $REF LITHOS_THROTTLE=1)"; row "outstanding-work throttle (§5.3)" "$v" "$s"; PREV=$REFV
read -r v s <<<"$(best $REF LITHOS_RIGHTSIZE=1)"; row "right-sizing (§5.5)"             "$v" "$s"; PREV=$REFV
read -r v s <<<"$(best $REF LITHOS_DISPATCH=1)";  row "dispatcher hand-off (§5.2)"      "$v" "$s"; PREV=$REFV
read -r v s <<<"$(best $REF LITHOS_ATOM_US=1)";   row "forced max split (§5.4)"         "$v" "$s"
