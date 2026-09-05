#!/usr/bin/env python3
"""LithOS use-case study: when does atomization actually pay off?

Co-locates a latency-critical HP job (tiny kernel) with a best-effort BE job and
sweeps the BE kernel's duration. Reports HP tail latency with and without
atomization, plus what BE gives up in throughput.
"""
import os, re, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, "build")          # harnesses live in build/ after `make bench`
LIB = os.path.join(REPO, "build", "liblithos_full.so")
BASE_ENV = dict(os.environ)
BASE_ENV.update(CUDA_MPS_PIPE_DIRECTORY="/tmp/lithos-mps/pipe",
                CUDA_MPS_LOG_DIRECTORY="/tmp/lithos-mps/log")

def env(**kw):
    e = dict(BASE_ENV); e["LD_PRELOAD"] = LIB
    for k, v in kw.items(): e[k] = str(v)
    return e

def kernel_ms(G, W):
    out = subprocess.run([f"{D}/heavybench", str(G), str(W), "8"],
                         capture_output=True, text=True, cwd=REPO, env=BASE_ENV, timeout=90).stdout
    return float(out.strip() or 0)

def hp_p99(extra_env=None, iters=150):
    e = env(LITHOS_ATOMIZER="0", LITHOS_PREDICT="0")
    if extra_env: e.update({k: str(v) for k, v in extra_env.items()})
    out = subprocess.run([f"{D}/hpbench", "16", "200", str(iters)],
                         capture_output=True, text=True, cwd=REPO, env=e, timeout=120).stdout
    m = re.search(r"p99=([0-9.]+)", out)
    return float(m.group(1)) if m else None

def with_be(G, W, be_env, secs=6):
    """Run BE in the background, measure HP p99 inside that window, then let BE
    finish NATURALLY. Killing a client mid-flight wedges the MPS server on this
    node, so we never terminate it."""
    be = subprocess.Popen([f"{D}/beload", str(G), str(W), str(secs)],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, cwd=REPO, env=env(**be_env))
    time.sleep(1.0)
    p = hp_p99(iters=80)
    out, _ = be.communicate(timeout=180)      # wait it out, do not kill
    m = re.search(r"BE (\d+)", out or "")
    return p, (int(m.group(1)) if m else None)

def main():
    print("### HP tail latency vs BE kernel duration (A100) ###")
    print(f"  {'BE kernel':<16}{'dur(ms)':>9}{'HP p99 no-atom':>16}{'HP p99 atom':>14}{'gain':>8}{'BE tput':>9}")
    for G, W in [(512, 4000), (2048, 8000), (8192, 20000), (20000, 40000)]:
        dur = kernel_ms(G, W)
        na, be_na = with_be(G, W, {"LITHOS_ATOMIZER": 0, "LITHOS_PREDICT": 0})
        sys.stdout.flush()
        time.sleep(0.5)
        at, be_at = with_be(G, W, {"LITHOS_PREDICT": 0, "LITHOS_SLO_US": 300})
        time.sleep(0.5)
        gain = f"{na/at:.0f}x" if (na and at) else "-"
        tput = f"{be_at/be_na*100:.0f}%" if (be_na and be_at) else "n/a"
        print(f"  G={G} W={W:<8}{dur:>8.3f}{na or 0:>14.1f}us{at or 0:>12.1f}us{gain:>8}{tput:>9}")

if __name__ == "__main__":
    main()
