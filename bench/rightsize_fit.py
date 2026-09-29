"""rightsize_fit.py — read a bench/rightsize.sh run and report the paper's §8.2 metric.

The paper: "We compute savings by comparing the time-weighted average of TPC
utilization before and after right-sizing."

Each [predict] line records one launch: the allocation it ran at (@NN TPC) and the
GPU time it took. The time-weighted mean allocation is therefore

    sum(duration * tpcs) / sum(duration)

over every launch in the run, and the saving is one minus the ratio of that mean
after right-sizing to the same mean before it. Unlike the model-derived figure in
kernelmix_fit.py, nothing here is fitted: these are the allocations the running
system chose, including the probe launches it needed to learn them.
"""
import os, re, sys

LINE = re.compile(r"\[predict\] op\(slot=\d+,k=\d+\) @(\d+)TPC measured=([\d.]+)us")
RESULT = re.compile(r"(\d+) iters in ([\d.]+) s p50=([\d.]+) p99=([\d.]+) ms "
                    r"checksum=([-\d.]+)")


def utilisation(path):
    """(time-weighted mean TPCs, launches, total GPU us) for one run."""
    num = den = 0.0
    n = 0
    for line in open(path, errors="replace"):
        m = LINE.match(line)
        if not m:
            continue
        tpc, us = int(m.group(1)), float(m.group(2))
        num += us * tpc
        den += us
        n += 1
    return (num / den if den else 0.0), n, den


def result(path):
    """(iters, seconds, p50 ms, p99 ms, checksum) from the harness's own line."""
    for line in open(path, errors="replace"):
        m = RESULT.search(line)
        if m:
            return (int(m.group(1)), float(m.group(2)), float(m.group(3)),
                    float(m.group(4)), m.group(5))
    return None


def main(d, slips):
    base_u, base_n, _ = utilisation(os.path.join(d, "off.log"))
    base_r = result(os.path.join(d, "off.out"))
    if not base_r or base_u <= 0:
        print("rightsize: baseline run produced no usable output"); return 1

    _, base_s, base_p50, base_p99, base_sum = base_r
    base_tput = base_r[0] / base_s

    print("\n=== §8.2 right-sizing, measured (not modelled) ===")
    print(f"  baseline: {base_u:.1f} TPCs time-weighted over {base_n} launches, "
          f"{base_tput:.1f} it/s, p50 {base_p50:.3f} ms, p99 {base_p99:.3f} ms")
    print(f"\n  {'slip':>6} {'TPCs used':>10} {'saved':>7} "
          f"{'p50':>9} {'p99':>9} {'throughput':>11}  checksum")
    print(f"  {'off':>6} {base_u:9.1f} {'—':>7} "
          f"{base_p50:8.3f}m {base_p99:8.3f}m {base_tput:10.1f}/s  {base_sum}")
    for k in slips:
        tag = f"slip_{k}"
        u, n, _ = utilisation(os.path.join(d, tag + ".log"))
        r = result(os.path.join(d, tag + ".out"))
        if not r or u <= 0:
            print(f"  {k:>6}  (no usable output)"); continue
        iters, sec, p50, p99, csum = r
        tput = iters / sec
        print(f"  {k:>6} {u:9.1f} {(1 - u / base_u) * 100:6.1f}% "
              f"{p50:8.3f}m {p99:8.3f}m {tput:10.1f}/s  {csum}"
              f"{'' if csum == base_sum else '  <-- CHECKSUM DIFFERS'}")
        print(f"         {'':>10} {'':>7} "
              f"{(p50 / base_p50 - 1) * 100:+7.1f}% {(p99 / base_p99 - 1) * 100:+7.1f}% "
              f"{(tput / base_tput - 1) * 100:+9.1f}%")
    print("\n  paper: mean 26% capacity saved (up to 51%) at slip 1.1, "
          "for +4% P99 and -4% throughput")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:] or ["1.1", "1.3", "1.5"]))
