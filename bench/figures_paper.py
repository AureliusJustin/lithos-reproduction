"""figures_paper.py — redraw the LithOS paper's figures in the paper's own style.

This follows the paper's conventions (linear axes, every kernel drawn, colour by
share of execution time) so the figures compare side by side with the paper:

  Fig 12/13 — EVERY kernel of a workload drawn as one thin line, coloured by that
              kernel's share of total execution time against a colour bar, on
              linear axes, with a red dashed line marking ideal scaling.
  Fig 20    — grouped bars, MPS / +TPC Scheduling / +Kernel Atomization, as a
              multiple of the service's solo tail, with a dashed line at 1.0.
  Fig 21    — P95 of the latency-critical job against the best-effort job's batch
              size, with and without atomization, against a dashed ideal.

Reads only files under the data directory — no GPU, no CUDA.

    python3 bench/figures_paper.py <data-dir> [more-dirs...] -o figures/
"""
import json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})

LINE = re.compile(r"\[predict\] op\(slot=(\d+),k=(\d+)\) @(\d+)TPC "
                  r"measured=([\d.]+)us ema=([\d.]+)us seen=(\d+)")
CMAP = cm.get_cmap("jet") if hasattr(cm, "get_cmap") else plt.get_cmap("jet")
IDEAL = "#e8000b"
MIN_SEEN = 5          # samples an operator needs before its curve is drawn


def load(path):
    """Last EMA and sample count per operator in one run."""
    ops = {}
    if not os.path.exists(path):
        return ops
    for line in open(path, errors="replace"):
        m = LINE.match(line)
        if m:
            slot, k, _t, _meas, ema, seen = m.groups()
            ops[(int(slot), int(k))] = (float(ema), int(seen))
    return ops


def sweep(d, prefix, kind):
    """{point: {op: (ema, seen)}} for tpc_*/clk_* logs, optionally prefixed."""
    out = {}
    for f in os.listdir(d):
        if not f.endswith(".log"):
            continue
        base = f[:-4]
        if prefix:
            if not base.startswith(prefix + "__"):
                continue
            base = base[len(prefix) + 2:]
        elif "__" in base:
            continue
        if base.startswith(kind + "_"):
            out[int(base[len(kind) + 1:])] = load(os.path.join(d, f))
    return out


def usable(runs, cover=0.99):
    """The kernels the paper's Figure 12 plots: those measured at every point that
    together account for `cover` of total execution time.

    §5.5: "The selected kernels collectively account for 99% of total execution
    time, with color gradients indicating each kernel's relative contribution."
    The rule is not cosmetic. A workload has a long tail of sub-100us kernels whose
    measured latency at one TPC is mostly noise, and dividing two noisy numbers
    produces speedups above the TPC count — physically impossible, and enough of
    them to bury the kernels that actually matter. Here they are 121 of 1106
    operators and 0.4% of the runtime.

    Returns (points, {op: latencies}, {op: share of total time}).
    """
    # Drop sweep points whose run did not finish. Every speedup here is a ratio
    # against the slowest point, so a truncated run there poisons the whole panel:
    # ResNet-50 at 1 TPC is ~50x slower and hit its timeout with a third of the
    # samples, leaving operators whose baseline came from warm-up alone — which is
    # where speedups above the TPC count (up to 780x, physically impossible) came
    # from.
    counts = {p: sum(v[1] for v in runs[p].values()) for p in runs}
    full = max(counts.values()) if counts else 0
    runs = {p: r for p, r in runs.items() if counts[p] >= 0.8 * full}

    pts = sorted(runs)
    common = set(runs[pts[0]])
    for p in pts[1:]:
        common &= set(runs[p])
    hi = max(pts)
    # Keep only operators the predictor actually measured repeatedly.
    #
    # A workload's warm-up passes run with no synchronize between them, so the
    # operator ordinal does not reset across them and every warm-up launch lands on
    # an ordinal of its own, seen exactly once. Those single-sample entries are pure
    # noise — no averaging at all — and dividing two of them produces the speedups
    # above the TPC count that made the first version of this figure unreadable.
    # For ResNet-50 they are 885 of 1106 operators but only 8.5% of runtime;
    # dropping them removes every physically impossible curve.
    common = {o for o in common
              if all(runs[p][o][0] > 0 and runs[p][o][1] >= MIN_SEEN for p in pts)}
    weight = {o: runs[hi][o][0] * runs[hi][o][1] for o in common}
    tot = sum(weight.values()) or 1.0
    keep, acc = {}, 0.0
    for op in sorted(common, key=lambda o: -weight[o]):
        if acc / tot >= cover:
            break
        keep[op] = [runs[p][op][0] for p in pts]
        acc += weight[op]
    # Finally, drop curves that break physics. Confining a kernel to t TPCs cannot
    # make it more than t/lo times faster than it was at lo TPCs, so a curve above
    # that line is a bad measurement, not a scaling behaviour — usually an operator
    # whose baseline run at the smallest TPC count differed (the operator set is not
    # always identical there). Two of 21 kernels in the mixed workload, 1.7% of its
    # runtime; none at all in ResNet-50.
    lo = min(pts)
    keep = {o: v for o, v in keep.items()
            if all(v[0] / l <= (p / lo) * 1.1 for p, l in zip(pts, v))}
    share = {o: weight[o] / tot for o in keep}
    return pts, keep, share


# ------------------------------------------------------------------ Fig 12
def fig12(panels, out):
    """Per-kernel TPC scaling. One panel per workload, one thin line per kernel,
    coloured by that kernel's share of total execution time."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n + 1.1, 2.5), sharey=True)
    axes = [axes] if n == 1 else list(axes)
    sm = None
    for ax, (title, runs) in zip(axes, panels):
        pts, keep, share = usable(runs)
        lo, hi = min(pts), max(pts)
        for op in sorted(keep, key=lambda o: share[o]):       # heaviest drawn last
            base = runs[lo][op][0]
            ys = [base / runs[p][op][0] for p in pts]
            ax.plot(pts, ys, lw=0.7, alpha=0.9, color=CMAP(share[op]))
        ax.plot([lo, hi], [1, hi / lo], "--", color=IDEAL, lw=1.1)
        ax.set_title(title)
        ax.set_xlim(lo, hi); ax.set_ylim(1, hi)
        ax.set_xticks([t for t in (1, 18, 36, 54) if t >= lo])
        ax.set_yticks([1, 18, 36, 54])
        ax.set_xlabel("# of TPCs")
        ax.tick_params(direction="in", top=True, right=True)
        if lo > 1:
            # This panel's slowest point is not 1 TPC (that run was truncated), so
            # say what the ideal line is drawn against rather than let it look short.
            # top-left: the curves crowd the bottom-right, where this was unreadable
            ax.text(0.03, 0.97, f"baseline {lo} TPC", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5, color="0.35")
    axes[0].set_ylabel("Speedup")
    sm = cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02,
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    cb.set_label("% of Total Time", fontsize=9)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ Fig 13
def fig13(panels, out):
    """Per-kernel frequency scaling, speedup relative to the lowest clock measured."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n + 1.1, 2.5), sharey=True)
    axes = [axes] if n == 1 else list(axes)
    for ax, (title, runs) in zip(axes, panels):
        pts, keep, share = usable(runs)
        lo, hi = min(pts), max(pts)
        for op in sorted(keep, key=lambda o: share[o]):
            base = runs[lo][op][0]
            ys = [base / runs[p][op][0] for p in pts]
            ax.plot(pts, ys, lw=0.7, alpha=0.9, color=CMAP(share[op]))
        ax.plot([lo, hi], [1.0, hi / lo], "--", color=IDEAL, lw=1.1)
        ax.set_title(title)
        ax.set_xlim(lo - 40, hi + 40)
        ax.set_ylim(0.9, hi / lo * 1.05)
        ax.set_xlabel("Frequency (MHz)")
        ax.tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel("Speedup vs lowest clock")
    sm = cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02,
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    cb.set_label("% of Total Time", fontsize=9)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ Fig 20
def fig20(groups, out):
    """Grouped bars: what each mechanism buys, as a multiple of the solo tail."""
    labels = list(groups)
    series = ["MPS", "+ TPC Scheduling", "+ Kernel Atomization"]
    cols = ["#bcbd22", "#1f77b4", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    # Clip only the MPS bars, which are two orders of magnitude out; everything else
    # must read off the axis rather than sit at the ceiling with a number over it.
    w = 0.26
    ymax = max(4.0, 1.15 * max(v for g in groups.values() for v in g[1:]))
    for i, (name, c) in enumerate(zip(series, cols)):
        xs = [j + (i - 1) * w for j in range(len(labels))]
        ys = [groups[l][i] for l in labels]
        ax.bar(xs, [min(y, ymax) for y in ys], width=w, label=name, color=c,
               edgecolor="black", linewidth=0.4)
        for x, y in zip(xs, ys):
            if y > ymax:                       # annotate the ones that run off, as the paper does
                ax.text(x, ymax + 0.08, f"{y:.0f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="black", ls="--", lw=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, ymax + 0.55)
    ax.set_yticks([t for t in range(0, int(ymax) + 2) if t % max(1, int(ymax) // 6) == 0])
    ax.set_ylabel(r"$P_{99}$ Latency ($\times$)")
    ax.set_xlabel("High-priority inference, co-located with training")
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, 1.28))
    ax.tick_params(direction="in")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------ Fig 21
def fig21(batch_file, ideal, out):
    """P95 of the latency-critical job against the best-effort batch size."""
    rows = {}
    for line in open(batch_file):
        b, mode, p95, _bet = line.split()
        rows.setdefault(int(b), {})[mode] = float(p95)
    xs = sorted(rows)
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.plot(xs, [rows[x]["off"] for x in xs], "-o", color="#1f77b4", ms=4.5, lw=1.4,
            label="LithOS (w/o Kernel Atomization)")
    ax.plot(xs, [rows[x]["on"] for x in xs], "-*", color="#2ca02c", ms=9, lw=1.4,
            label="LithOS")
    ax.axhline(ideal, color="black", ls="--", lw=0.9)
    ax.set_xlabel("BE Training Batch Size")
    ax.set_ylabel(r"$P_{95}$ Latency (ms)")
    ax.set_xlim(0, max(xs) * 1.05)
    ax.set_ylim(0, max(max(r.values()) for r in rows.values()) * 1.25)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.tick_params(direction="in", top=True, right=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    outdir = "figures"
    if "-o" in args:
        i = args.index("-o"); outdir = args[i + 1]; args = args[:i] + args[i + 2:]
    os.makedirs(outdir, exist_ok=True)
    d1 = args[0] if args else "figdata"
    # One data directory holds both the prefixed per-model logs
    # (resnet50-infer__tpc_*.log) and the unprefixed mix logs (tpc_*.log), which is
    # how figdata/ is laid out. A second directory is still accepted.
    d2 = args[1] if len(args) > 1 else d1

    p12 = []
    r = sweep(d2, "resnet50-infer", "tpc")
    if len(r) >= 4:
        p12.append(("ResNet-50 Inference", r))
    r = sweep(d1, None, "tpc")
    if len(r) >= 4:
        p12.append(("Mixed conv/GEMM/norm", r))
    if p12:
        fig12(p12, os.path.join(outdir, "fig12_tpc_scaling.png"))

    p13 = []
    r = sweep(d2, "resnet50-infer", "clk")
    if len(r) >= 3:
        p13.append(("ResNet-50 Inference", r))
    r = sweep(d1, None, "clk")
    if len(r) >= 3:
        p13.append(("Mixed conv/GEMM/norm", r))
    if p13:
        fig13(p13, os.path.join(outdir, "fig13_freq_scaling.png"))

    # Measured on this node, 2026-09-09, both in the DONATED configuration (the
    # paper's: the service's idle TPCs are lent to the best-effort job, so there is
    # head-of-line blocking for atomization to remove). Medians of 3 and 2 reps.
    fig20({"ResNet-50 vs VGG-19": (66.24, 3.33, 4.25),
           "synthetic pair":      (812.01, 6.25, 2.01)},
          os.path.join(outdir, "fig20_breakdown.png"))

    bf = os.path.join(d1, "batch.txt")
    if os.path.exists(bf):
        fig21(bf, 1.115, os.path.join(outdir, "fig21_batch.png"))
