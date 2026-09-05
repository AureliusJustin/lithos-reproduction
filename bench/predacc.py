#!/usr/bin/env python3
"""Predictor accuracy, scored the way the paper does (§8.4): compare each launch's
prediction against its measured duration; |error| > 50 us counts as a
misprediction. Paper reports 0.9% / 0.38% for HP workloads and 14% / 11% for BE,
with P99 absolute errors of 49 us / 31 us."""
import os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB  = os.path.join(REPO, "build", "liblithos_full.so")
THRESH_US = 50.0

def run(cmd):
    e = dict(os.environ); e["LD_PRELOAD"] = LIB; e["LITHOS_PREDICT_ACC"] = "1"
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=e, timeout=300)
    errs = [float(m.group(1)) for m in re.finditer(r"err=(-?[0-9.]+)", p.stderr)]
    return errs

def report(name, errs, warmup=0.2):
    if not errs:
        print(f"  {name:<34} no measurements"); return
    # drop a warm-up prefix: the first samples of an operator have no history
    n0 = int(len(errs) * warmup)
    e = sorted(abs(x) for x in errs[n0:])
    if not e:
        print(f"  {name:<34} no post-warmup samples"); return
    mis = sum(1 for x in e if x > THRESH_US) / len(e) * 100
    p50, p99 = e[len(e)//2], e[min(len(e)-1, int(len(e)*0.99))]
    print(f"  {name:<34} n={len(e):<6} mispred={mis:5.1f}%  |err| p50={p50:6.1f}us p99={p99:7.1f}us")

if __name__ == "__main__":
    B = os.path.join(REPO, "build")
    print(f"### Predictor accuracy (misprediction = |pred-actual| > {THRESH_US:.0f} us) ###")
    report("steady batched (8 op/batch)",  run([f"{B}/batchbench", "8", "60"]))
    report("steady solitary (launch+sync)", run([f"{B}/heavybench", "2048", "8000", "120"]))
    report("long kernels (batched)",        run([f"{B}/batchbench", "4", "40", "4096"]))
