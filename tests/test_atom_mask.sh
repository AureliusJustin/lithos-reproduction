#!/usr/bin/env bash
# test_atom_mask.sh — every atom must carry the scheduler's TPC allocation.
#
# §5.2: a compute quota "guarantees each application a specified number of TPCs".
# That guarantee is enforced by the mask written in the QMD pre-upload callback,
# and that mask is ONE-SHOT: it is consumed by the upload it applies to. A launch
# path that arms it once and then relaunches the grid N times for N atoms leaves
# atoms 1..N-1 unmasked, so they run on the WHOLE device — the quota silently
# stops applying part-way through every atomized kernel.
#
# This is invisible in output correctness (the atoms still compute the right
# answer) and invisible to a %smid probe of the first atom, which is exactly why
# it survived on the cuLaunchKernelEx path — the one the CUDA runtime uses for
# framework launches — while the cuLaunchKernel path was correct.
#
# The check: with LITHOS_LOG_CB counting every QMD upload and LITHOS_LOG_MASK
# counting every mask actually applied, the two must be equal on BOTH paths.
set -u
BUILD=${BUILD:-build}
LIB=$BUILD/liblithos_full.so
ATOMS=${ATOMS:-4}
fail=0

echo "test_atom_mask: every atom carries the TPC allocation (§5.2)"

for mode in std ex; do
    out=$(LITHOS_MPS=0 LITHOS_QUOTA=4 LITHOS_FORCE_ATOMS=$ATOMS \
          LITHOS_LOG_CB=1 LITHOS_LOG_MASK=1 MASK_CUBIN=$BUILD/atomize_mark.cubin \
          LD_PRELOAD=$LIB timeout 60 $BUILD/atom_mask_probe $mode 256 2>&1)
    cb=$(grep -c '^\[cb\]'   <<<"$out")
    mk=$(grep -c '^\[mask\]' <<<"$out")
    ok=$(grep -c '0 wrong'   <<<"$out")

    if [[ $cb -lt $ATOMS ]]; then
        echo "  FAIL ($mode): expected >= $ATOMS QMD uploads, saw $cb — did it atomize?"; fail=1
    elif [[ $cb -ne $mk ]]; then
        echo "  FAIL ($mode): $cb atom launches but only $mk carried a mask" \
             "— atoms $((cb-mk)) ran unconfined, escaping the quota"; fail=1
    elif [[ $ok -ne 1 ]]; then
        echo "  FAIL ($mode): blocks did not all run exactly once"; fail=1
    else
        echo "  PASS ($mode): $mk/$cb atom launches masked, all blocks correct"
    fi

    # The mask must also be the quota we asked for, not some wider set.
    if grep -q 'enabled_TPCs=\[0,1,2,3,\]' <<<"$out"; then :; else
        echo "  FAIL ($mode): quota=4 did not produce TPCs [0,1,2,3]"; fail=1
    fi
done

if [[ $fail -eq 0 ]]; then echo "ATOM_MASK: PASS (0 failures)"; else echo "ATOM_MASK: FAIL"; fi
exit $fail
