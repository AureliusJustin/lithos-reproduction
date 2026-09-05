#!/usr/bin/env bash
# test_coord.sh — the system-wide coordinator (LithOS §5.1, Figure 8).
#
# Figure 8 shows several applications, each linked against LibLithOS, above ONE
# shared LithOS layer that "maintains a system-wide view of GPU state across
# applications with varying priorities". These are the three properties that layer
# must provide, none of which a per-process scheduler can:
#
#   1. Compute quotas ACROSS APPLICATIONS (§5.2) — two independent processes get
#      disjoint TPC ranges automatically, with no hand-assigned LITHOS_TPC_BASE.
#   2. TPC Stealing ACROSS APPLICATIONS (§5.3) — "the scheduler dynamically
#      reassigns underutilized TPCs across applications".
#   3. The priority safeguard (§5.3) — a higher-priority tenant's quota is a
#      guarantee, so its TPCs are never stolen even while it is idle.
#
# The lender must be alive-but-quiet: a process that exits releases its slot, so
# `idler` registers, launches once, then sleeps.
set -u
BUILD=${BUILD:-build}
LIB=$BUILD/liblithos_full.so
export LITHOS_BENCH_CUBIN=${LITHOS_BENCH_CUBIN:-$BUILD/kernels/work.cubin}
fail=0

mask_of() {   # mask_of <steal 0|1> <priority> <lender-priority>
    rm -f /dev/shm/lithos_coord; sleep 0.3
    LITHOS_QUOTA=8 LITHOS_PRIORITY=$3 LD_PRELOAD=$LIB $BUILD/idler 10 >/dev/null 2>&1 &
    local lender=$!
    sleep 2.5
    LITHOS_QUOTA=8 LITHOS_PRIORITY=$2 LITHOS_STEALING=$1 LITHOS_LOG_MASK=1 \
        LD_PRELOAD=$LIB timeout 30 $BUILD/tenant T 128 8000 1 2>&1 \
        | grep -oE 'enabled_TPCs=\[[0-9,]*\]' | sort -u | head -1
    kill $lender 2>/dev/null; wait $lender 2>/dev/null; sleep 0.3
}

check() {  # check <label> <got> <expect-substring>
    if [[ "$2" == *"$3"* ]]; then echo "  PASS: $1"
    else echo "  FAIL: $1 -> got '$2', expected to contain '$3'"; fail=1; fi
}

echo "test_coord: system-wide coordinator (§5.1)"

# 1. Two processes, no TPC_BASE anywhere: the second must be given a DISJOINT range.
got=$(mask_of 0 0 0)
check "cross-app quotas give disjoint ranges (expect [8,16))" "$got" "8,9,10,11,12,13,14,15,"

# 2. With stealing on, the idle peer's range is borrowed too.
got=$(mask_of 1 0 0)
check "cross-app stealing borrows an idle tenant's TPCs" "$got" "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,"

# 3. A higher-priority tenant is never robbed, even while idle.
got=$(mask_of 1 0 10)
check "higher-priority tenant's quota is never stolen" "$got" "8,9,10,11,12,13,14,15,"
if [[ "$got" == *"0,1,2,3"* ]]; then echo "  FAIL: stole from a higher-priority tenant"; fail=1; fi

# 4. Two tenants that join AT THE SAME INSTANT still get disjoint ranges.
#
# The cases above start the lender 2.5s ahead, so they only ever exercise a join
# against an already-initialised segment. Creating the shm segment is itself a
# race: when both tenants find it missing, both may size and zero it, and the
# second wipes the first's registration -- after which each sees an empty table
# and both are handed the SAME range, defeating the whole point of cross-app
# quotas. It is intermittent, so it needs repetition rather than one attempt.
collisions=0
for trial in 1 2 3 4 5 6; do
    rm -f /dev/shm/lithos_coord; sleep 0.2
    LITHOS_QUOTA=8 LITHOS_VERBOSE=1 LD_PRELOAD=$LIB $BUILD/idler 6 >/tmp/coord_a.$$ 2>&1 &
    a=$!
    LITHOS_QUOTA=8 LITHOS_VERBOSE=1 LD_PRELOAD=$LIB $BUILD/idler 6 >/tmp/coord_b.$$ 2>&1 &
    b=$!
    sleep 3.5
    ra=$(grep -oE 'TPC range \[[0-9]+,[0-9]+\)' /tmp/coord_a.$$ | head -1)
    rb=$(grep -oE 'TPC range \[[0-9]+,[0-9]+\)' /tmp/coord_b.$$ | head -1)
    [[ -n "$ra" && "$ra" == "$rb" ]] && collisions=$((collisions+1))
    kill $a $b 2>/dev/null; wait $a $b 2>/dev/null
done
rm -f /tmp/coord_a.$$ /tmp/coord_b.$$
if [[ $collisions -eq 0 ]]; then echo "  PASS: simultaneous joins never share a range (6 trials)"
else echo "  FAIL: $collisions/6 simultaneous joins collided on the same range"; fail=1; fi

rm -f /dev/shm/lithos_coord
if [[ $fail -eq 0 ]]; then echo "COORD: PASS (0 failures)"; else echo "COORD: FAIL"; fi
exit $fail
