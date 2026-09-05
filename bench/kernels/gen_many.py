#!/usr/bin/env python3
"""Emit a .cu with N trivial kernels — used to measure how the splicer scales
with kernels-per-module (docs/BENCHMARKS.md: single-pass splice is linear)."""
import sys
n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
print("\n".join(
    'extern "C" __global__ void k%d(int*p){if(p&&threadIdx.x==0)p[blockIdx.x]=%d;}' % (i, i)
    for i in range(n)))
