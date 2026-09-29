"""kernelmix_fit.py — fit the two models to a real kernel mix (see kernelmix.sh).

Reads the per-operator latencies LithOS logs under LITHOS_LOG_PREDICT and reports
the paper's own metrics:

  §8.2  right-sizing: per-operator fit of l = m/t + b over a TPC sweep, summarised
        as the EXECUTION-TIME-WEIGHTED mean R^2. Weighting matters — an unweighted
        mean lets a kernel that runs for a microsecond count as much as one that
        dominates the step, which is why the paper weights by execution time.

  §5.6  DVFS: per-operator sensitivity s = (l(f)/l(f_max) - 1) / (f_max/f - 1),
        averaged over the clock sweep, reported as a distribution. A single
        aggregate hides exactly what matters: whether the mix contains kernels the
        clock does not govern.
"""
import os, re, sys, glob, statistics

LINE = re.compile(r"\[predict\] op\(slot=(\d+),k=(\d+)\) @(\d+)TPC "
                  r"measured=([\d.]+)us ema=([\d.]+)us seen=(\d+)")


# An operator seen this many times or fewer is discarded before anything is fitted.
#
# A workload's warm-up passes run with no synchronize between them, so the operator
# ordinal never resets across them and every warm-up launch is recorded as its own
# operator, seen exactly once. There is no averaging behind those entries, and
# fitting a curve through eleven of them fits noise: including them dragged this
# figure's weighted R^2 from 0.93 down to 0.91 and produced per-kernel speedups
# above the TPC count, which is impossible. bench/figures_paper.py applies the same
# rule, so the plots and this summary agree.
MIN_SEEN = 5


def load(path):
    """Last EMA per operator in one run — the settled estimate, not the first sample."""
    ops = {}
    for line in open(path, errors="replace"):
        m = LINE.match(line)
        if not m:
            continue
        slot, k, _tpc, _meas, ema, seen = m.groups()
        if int(seen) < MIN_SEEN:
            continue
        ops[(int(slot), int(k))] = (float(ema), int(seen))
    return ops


def fit_inverse(points):
    """Least squares l = m*(1/t) + b; returns (m, b, R^2)."""
    xs = [1.0 / t for t, _ in points]
    ys = [l for _, l in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    m = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / sxx
    b = my - m * mx
    pred = [m * x + b for x in xs]
    ss_res = sum((ys[i] - pred[i]) ** 2 for i in range(n))
    ss_tot = sum((y - my) ** 2 for y in ys)
    if ss_tot <= 0:
        return None
    return m, b, 1 - ss_res / ss_tot


def rightsizing(d):
    runs = {}
    for f in glob.glob(os.path.join(d, "tpc_*.log")):
        t = int(os.path.basename(f)[4:-4])
        runs[t] = load(f)
    if len(runs) < 3:
        print("\nright-sizing: not enough TPC points"); return

    tpcs = sorted(runs)
    common = set(runs[tpcs[0]])
    for t in tpcs[1:]:
        common &= set(runs[t])

    rows, wsum, wr2 = [], 0.0, 0.0
    for op in sorted(common):
        pts = [(t, runs[t][op][0]) for t in tpcs]
        if any(l <= 0 for _, l in pts):
            continue
        fit = fit_inverse(pts)
        if not fit:
            continue
        m, b, r2 = fit
        # Weight by time spent: latency at full TPC count x times it was measured.
        full = max(tpcs)
        w = runs[full][op][0] * runs[full][op][1]
        rows.append((op, m, b, r2, w, runs[full][op][0]))
        wsum += w
        wr2 += w * r2

    if not rows:
        print("\nright-sizing: no operators measured at every TPC count"); return
    rows.sort(key=lambda r: -r[4])
    print(f"\n=== §8.2 right-sizing: l = m/t + b across {len(rows)} operators, "
          f"{len(tpcs)} TPC points ===")
    print(f"  execution-time-weighted mean R^2 = {wr2/wsum:.4f}    "
          f"(paper: 0.92-0.99)")
    r2s = sorted(r[3] for r in rows)
    print(f"  unweighted: min {r2s[0]:.4f}  median {statistics.median(r2s):.4f}  "
          f"max {r2s[-1]:.4f}")
    print("  heaviest operators:")
    print(f"    {'op':>10} {'l@maxTPC':>10} {'share':>7} {'m':>10} {'b':>9} {'R^2':>8}")
    for op, m, b, r2, w, lfull in rows[:8]:
        print(f"    {str(op):>10} {lfull:9.1f}us {w/wsum*100:6.1f}% "
              f"{m:10.1f} {b:9.2f} {r2:8.4f}")
    poor = [r for r in rows if r[3] < 0.9]
    if poor:
        share = sum(r[4] for r in poor) / wsum * 100
        print(f"  {len(poor)} operators fit poorly (R^2 < 0.9), {share:.1f}% of runtime "
              f"— the outliers §5.5's occupancy filter exists to catch")

    savings(rows, wsum, max(tpcs))


def savings(rows, wsum, full):
    """§8.2 capacity savings: "the time-weighted average of TPC utilization before
    and after right-sizing".

    Before right-sizing every kernel gets all `full` TPCs. After it, each kernel gets
    the smallest t whose *modelled* latency stays within the slip budget of its
    all-TPC latency — which is the rule src/tpc_alloc.c applies at runtime. The
    saving is one minus the runtime-weighted mean of t/full.

    This is derived from the fitted models, not from a right-sized run: it is what
    the models imply the allocator will choose, and so an upper bound on what a run
    would achieve. The paper's figure is measured. See BENCHMARKS §12.
    """
    print(f"\n=== §8.2 capacity savings: time-weighted TPC utilisation, "
          f"before vs after right-sizing ===")
    for slip in (1.1, 1.3, 1.5):
        saved = 0.0
        for _op, m, b, _r2, w, _lfull in rows:
            lfull = m / full + b
            budget = slip * lfull
            t = full
            for cand in range(1, full + 1):
                if m / cand + b <= budget:
                    t = cand
                    break
            saved += w * (1.0 - t / full)
        print(f"  slip {slip}: {saved / wsum * 100:5.1f}% of GPU capacity saved")
    print("  (paper: mean 26%, up to 51%, at slip 1.1 — averaged across its workloads)")


def dvfs(d):
    runs = {}
    for f in glob.glob(os.path.join(d, "clk_*.log")):
        mhz = int(os.path.basename(f)[4:-4])
        runs[mhz] = load(f)
    if len(runs) < 2:
        print("\nDVFS: clock sweep not available"); return

    fmax = max(runs)
    others = sorted((f for f in runs if f != fmax), reverse=True)
    common = set(runs[fmax])
    for f in others:
        common &= set(runs[f])

    rows, wsum, wS = [], 0.0, 0.0
    for op in sorted(common):
        l0 = runs[fmax][op][0]
        if l0 <= 0:
            continue
        ss = []
        for f in others:
            lf = runs[f][op][0]
            drop = fmax / f - 1.0
            if lf > 0 and drop > 1e-9:
                ss.append((lf / l0 - 1.0) / drop)
        if not ss:
            continue
        s = max(0.0, min(1.0, sum(ss) / len(ss)))
        w = l0 * runs[fmax][op][1]
        rows.append((op, s, w, l0))
        wsum += w
        wS += w * s

    if not rows:
        print("\nDVFS: no operators common to every clock"); return
    rows.sort(key=lambda r: r[1])
    S = wS / wsum
    print(f"\n=== §5.6 frequency sensitivity across {len(rows)} operators, "
          f"clocks {[fmax]+others} MHz ===")
    print(f"  runtime-weighted aggregate S = {S:.3f}  "
          f"-> f_final = f_max/(1+slip/S); at slip 0.1 that is "
          f"{fmax/(1+0.1/S):.0f} MHz of {fmax}")
    svals = [r[1] for r in rows]
    print(f"  per-operator s: min {svals[0]:.3f}  median {statistics.median(svals):.3f}  "
          f"max {svals[-1]:.3f}")
    low = [r for r in rows if r[1] < 0.5]
    print(f"  {len(low)} of {len(rows)} operators have s < 0.5 "
          f"({sum(r[2] for r in low)/wsum*100:.1f}% of runtime) "
          f"— the memory-bound kernels DVFS exists for")
    print("  least frequency-sensitive:")
    print(f"    {'op':>10} {'l@fmax':>10} {'share':>7} {'s':>7}")
    for op, s, w, l0 in rows[:6]:
        print(f"    {str(op):>10} {l0:9.1f}us {w/wsum*100:6.1f}% {s:7.3f}")
    print("  most frequency-sensitive:")
    for op, s, w, l0 in rows[-4:]:
        print(f"    {str(op):>10} {l0:9.1f}us {w/wsum*100:6.1f}% {s:7.3f}")


if __name__ == "__main__":
    d = sys.argv[1]
    rightsizing(d)
    dvfs(d)
