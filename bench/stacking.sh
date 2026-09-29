#!/usr/bin/env bash
# stacking.sh — LithOS §8.1 hybrid inference/training multitenancy, and the §8.4
# mechanism breakdown.
#
# The experiment the paper's headline numbers come from: a latency-critical
# inference service (HP) is co-located with a best-effort training job (BE), and
# the question is how much of the service's solo latency survives, and how much
# throughput the pair achieves together.
#
# Metrics follow §8.1:
#   HP tail   normalized to the HP running ALONE on the device (1.00 = ideal)
#   BE tput   normalized to the BE running ALONE (0 = starved, 1 = unimpeded)
#   aggregate HP-throughput-vs-offered-load plus normalized BE throughput
#
# Configurations, in the order the paper compares them:
#   solo            each tenant alone — the two normalizers
#   time slicing    both processes, MPS off: NVIDIA's default, no concurrency
#   MPS             both processes under MPS: maximum throughput, worst isolation
#   LithOS sched    MPS + TPC quotas (§5.3), atomizer off
#   LithOS full     the above + Kernel Atomization (§5.4)
#   LithOS + SLO    atom size tied to the neighbour's latency budget
#
# ---------------------------------------------------------------------------
# PROCESS HANDLING, and why it is this careful
#
# This harness runs pairs of GPU processes under MPS, which makes three ordinary
# shell habits actively dangerous:
#
#   1. A CUDA process stuck in the driver does not die on SIGTERM. Plain
#      `timeout N` then leaves it running forever, and the run never ends. Every
#      tenant is therefore launched with `timeout -k`, so SIGTERM is escalated to
#      SIGKILL after a grace period.
#   2. SIGKILLing a CUDA client mid-kernel WEDGES MPS — every subsequent CUDA
#      process on the machine hangs, which looks exactly like a LithOS deadlock
#      (docs/TECHNICAL_REPORT.md). So a force-kill is never silent: it sets a flag, and
#      the harness then tears MPS down and rebuilds it before continuing.
#   3. Killing the harness does not kill its children. Worse, the harness would
#      notice a child had gone, move on, and spawn the next pair — so an
#      interrupted run kept starting new GPU processes. It now traps INT/TERM/EXIT
#      and takes its children down with it, and it holds a lock so a stale
#      instance can never overlap a new one.
#
# The rule underneath all of it: let tenants exit on their own wherever possible.
# Both take a duration argument and stop by themselves; force is the last resort,
# and it is always followed by an MPS reset.
#
#   bash bench/stacking.sh
# ---------------------------------------------------------------------------
set -u

BUILD=${BUILD:-build}
DUR=${DUR:-10}
RATE=${RATE:-300}
HPBATCH=${HPBATCH:-8}
BEBATCH=${BEBATCH:-4096}
BEWIDTH=${BEWIDTH:-4096}
HPQ=${HPQ:-27}
BEQ=${BEQ:-27}
# HP_MODEL / BE_MODEL choose the tenants. The defaults are the synthetic pair the
# numbers in docs/TECHNICAL_REPORT.md were measured with; set them to the paper's
# own models (Table 1/2) for a like-for-like workload, e.g.
#   HP_MODEL=resnet50 BE_MODEL=vgg19 RATE=60 HPBATCH=8 BEBATCH=64 bash bench/stacking.sh
HP_MODEL=${HP_MODEL:-synthetic}
BE_MODEL=${BE_MODEL:-synthetic}
MODE=${MODE:-all}           # paper | ablation | all
REPS=${REPS:-3}             # repetitions per configuration (co-location is noisy)
GRACE=${GRACE:-20}          # seconds a tenant gets to finish before SIGTERM
LOCK=${LOCK:-/tmp/lithos-stacking.lock}

TMP=$(mktemp -d)
CHILDREN=()                 # background tenant pids we are responsible for
FORCED=0                    # set when we had to SIGKILL a CUDA client
ABORT=0

python3 -c "import torch" 2>/dev/null || { echo "stacking: PyTorch not installed"; exit 0; }

# ---- lock: never let two instances drive the GPU at once -------------------
exec {lockfd}>"$LOCK" || { echo "stacking: cannot open $LOCK"; exit 1; }
if ! flock -n "$lockfd"; then
    echo "stacking: another instance is running (lock $LOCK) — refusing to start"
    exit 1
fi

# ---- child lifecycle -------------------------------------------------------

# Wait for a pid, but never forever. Returns 0 if it exited on its own.
wait_for() {   # wait_for <pid> <seconds>
    local pid=$1 secs=$2 i=0
    while kill -0 "$pid" 2>/dev/null; do
        (( i++ >= secs * 10 )) && return 1
        sleep 0.1
    done
    return 0
}

# Stop one tenant as gently as possible. SIGKILL is reported, because it means
# MPS may now be wedged and the caller has to rebuild it.
stop_child() {   # stop_child <pid>
    local pid=$1
    kill -0 "$pid" 2>/dev/null || return 0
    kill -TERM "$pid" 2>/dev/null
    wait_for "$pid" 10 && return 0
    kill -KILL "$pid" 2>/dev/null
    FORCED=1
    wait_for "$pid" 5
}

stop_all_children() {
    local pid
    for pid in "${CHILDREN[@]:-}"; do
        [[ -n "$pid" ]] && stop_child "$pid"
    done
    CHILDREN=()
}

# The harness must not outlive its children, and must not spawn more once it has
# been asked to stop.
on_signal() {
    ABORT=1
    echo
    echo "stacking: interrupted — stopping tenants before exit"
    stop_all_children
    exit 130
}
on_exit() {
    stop_all_children
    (( FORCED )) && mps_reset       # a force-kill may have wedged the server
    rm -rf "$TMP"
}
trap on_signal INT TERM
trap on_exit EXIT

# ---- MPS -------------------------------------------------------------------

# Compute processes on the GPU, EXCLUDING the MPS server.
#
# The exclusion is essential, not cosmetic: while MPS is up, nvidia-smi reports
# the server itself as a compute application, and it never exits while the daemon
# lives. Counting it makes the GPU look permanently busy, which deadlocks every
# wait built on this. (It also means that under MPS the individual clients are
# proxied and may not appear here at all — so this is a check for FOREIGN work,
# while our own tenants are tracked by pid.)
gpu_clients() {
    local n=0 p comm
    while read -r p; do
        [[ -z "$p" ]] && continue
        comm=$(ps -o comm= -p "$p" 2>/dev/null)
        [[ "$comm" == nvidia-cuda-mps* ]] && continue
        (( n++ ))
    done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    echo "$n"
}

mps_running() { [[ -e /tmp/nvidia-mps/control ]]; }

mps_stop() {
    mps_running || return 0
    # Shutting the server down under a live client is itself a way to wedge it.
    local waited=0
    while (( $(gpu_clients) > 0 )) && (( waited < 15 )); do sleep 1; ((waited++)); done
    echo quit | timeout 20 nvidia-cuda-mps-control >/dev/null 2>&1
    wait_for_daemon_gone 10 || {
        pkill -f nvidia-cuda-mps-server >/dev/null 2>&1
        pkill -f nvidia-cuda-mps-control >/dev/null 2>&1
        sleep 1
    }
    rm -rf /tmp/nvidia-mps
}

wait_for_daemon_gone() {   # <seconds>
    local i=0
    while pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1; do
        (( i++ >= $1 * 10 )) && return 1
        sleep 0.1
    done
    return 0
}

mps_start() {
    mps_running && return 0
    nvidia-cuda-mps-control -d >/dev/null 2>&1
    local i=0
    while ! mps_running; do
        (( i++ >= 50 )) && { echo "stacking: MPS did not start"; return 1; }
        sleep 0.1
    done
}

# Full teardown and rebuild, used after a force-kill.
mps_reset() {
    echo "stacking: a tenant had to be force-killed — resetting MPS"
    mps_stop
    FORCED=0
}

# Nothing from a previous configuration may still be on the GPU when the next
# one starts, or the measurement is silently wrong.
require_idle_gpu() {
    local i=0
    while (( $(gpu_clients) > 0 )); do
        if (( i++ >= 30 )); then
            echo "stacking: GPU still busy after 30s — another workload is running."
            echo "          Results would be meaningless; stopping. Offending pids:"
            nvidia-smi --query-compute-apps=pid,used_memory --format=csv | sed 's/^/          /'
            exit 1
        fi
        sleep 1
    done
}

# ---- tenants ---------------------------------------------------------------
# `timeout -k` is the point: SIGTERM first, SIGKILL if the process is wedged in
# the driver and ignores it. Without the -k a stuck tenant hangs the whole run.

hp_run() {   # hp_run <outfile> <env...>
    local out=$1; shift
    env HP_MODEL="$HP_MODEL" "$@" timeout -k 10 $((DUR + GRACE)) \
        python3 bench/hp_infer.py "$RATE" "$DUR" "$HPBATCH" >"$out" 2>/dev/null
}

be_start() { # be_start <outfile> <seconds> <env...>  -> echoes pid
    local out=$1 secs=$2; shift 2
    env BE_MODEL="$BE_MODEL" "$@" timeout -k 10 $((secs + GRACE)) \
        python3 bench/be_train.py "$secs" "$BEBATCH" "$BEWIDTH" >"$out" 2>/dev/null &
    local pid=$!
    CHILDREN+=("$pid")
    echo "$pid"
}

# One repetition. Echoes "p95 p99 tput be_tput" or nothing on failure.
once() {   # once <hp-env-string> <be-env-string>
    local hpenv=$1 beenv=$2
    require_idle_gpu
    rm -f /dev/shm/lithos_coord

    # BE outlives HP so the service is measured against a continuously loaded GPU.
    local be; be=$(be_start "$TMP/be.txt" $((DUR + 6)) $beenv)
    sleep 3                                    # let BE warm up and load the GPU
    if ! kill -0 "$be" 2>/dev/null; then
        echo "  $label: BE exited during warm-up — skipping"; return
    fi

    hp_run "$TMP/hp.txt" $hpenv

    # BE stops on its own; only intervene if it does not.
    if ! wait_for "$be" $((GRACE + 15)); then
        echo "  $label: BE overran its deadline"
        stop_child "$be"
    fi
    CHILDREN=()

    local hp="$(tail -1 "$TMP/hp.txt" 2>/dev/null)" bo="$(tail -1 "$TMP/be.txt" 2>/dev/null)"
    # A force-kill during this repetition may have left MPS wedged; rebuild it
    # before the next one rather than letting the next result be garbage.
    if (( FORCED )); then mps_reset; mps_start; fi
    [[ -z "$hp" || -z "$bo" ]] && return 1
    python3 - "$hp" "$bo" <<'PY'
import sys, re
def g(s, k):
    m = re.search(rf"{k}=([\d.]+)", s)
    return float(m.group(1)) if m else 0.0
hp, be = sys.argv[1], sys.argv[2]
print(f"{g(hp,'p95')} {g(hp,'p99')} {g(hp,'tput')} {g(be,'tput')}")
PY
}

# Run a configuration REPS times and report the median, with the spread, so a
# single lucky or unlucky co-location cannot be mistaken for a result.
pair() {   # pair <label> <hp-env-string> <be-env-string>
    (( ABORT )) && return 0
    local label=$1 hpenv=$2 beenv=$3 r out
    local rows=()
    for ((r = 0; r < REPS; r++)); do
        (( ABORT )) && break
        out=$(once "$hpenv" "$beenv") || continue
        rows+=("$out")
    done
    if (( ${#rows[@]} == 0 )); then
        echo "  $label: no repetition produced a result"
        return 0
    fi
    report "$label" "${rows[@]}"
    return 0
}

report() {   # report <label> <row>...
    local label=$1; shift
    # Rows go in argv, not on stdin: a heredoc supplies the program on stdin, so a
    # pipe into it would be swallowed.
    python3 - "$label" "$HP_SOLO" "$BE_SOLO" "$RATE" "$@" <<'PY'
import sys, re, statistics
label, hp_solo, be_solo, rate = sys.argv[1:5]
def g(s, k, d=0.0):
    m = re.search(rf"{k}=([\d.]+)", s)
    return float(m.group(1)) if m else d
rows = [list(map(float, a.split())) for a in sys.argv[5:] if a.strip()]
p95  = statistics.median(r[0] for r in rows)
p99s = sorted(r[1] for r in rows)
p99  = statistics.median(p99s)
tput = statistics.median(r[2] for r in rows)
be_t = statistics.median(r[3] for r in rows)
p99_solo, be_solo_t = g(hp_solo, "p99"), g(be_solo, "tput")
norm  = p99 / p99_solo if p99_solo else 0
nbe   = be_t / be_solo_t if be_solo_t else 0
served= tput / float(rate) if float(rate) else 0
spread = f"{p99s[0]:.2f}-{p99s[-1]:.2f}" if len(p99s) > 1 else "-"
print(f"{label:<24} {p95:7.2f} {p99:8.2f} {norm:8.2f}x {spread:>13} {served*100:6.0f}% "
      f"{be_t:8.2f} {nbe:8.2f}")
PY
}

# ---- run -------------------------------------------------------------------
echo "LithOS §8.1 hybrid inference/training — HP inference + BE training"
echo "  HP: $HP_MODEL inference, Poisson $RATE req/s, batch $HPBATCH"
echo "  BE: $BE_MODEL training, batch $BEBATCH   |   ${DUR}s per run, median of $REPS"
echo

mps_stop
require_idle_gpu
HP_SOLO=$(hp_run "$TMP/hps.txt" LITHOS_MPS=0; tail -1 "$TMP/hps.txt")
echo "  solo HP: $HP_SOLO"
require_idle_gpu
be=$(be_start "$TMP/bes.txt" "$DUR" LITHOS_MPS=0); wait_for "$be" $((DUR + GRACE + 15)) || stop_child "$be"
CHILDREN=()
BE_SOLO=$(tail -1 "$TMP/bes.txt")
echo "  solo BE: $BE_SOLO"
echo

echo "  median of $REPS repetitions per configuration; p99 range shown"
printf "%-24s %7s %8s %9s %13s %7s %8s %8s\n" \
       configuration p95 p99 "tail/solo" "p99 range" "served" "BE it/s" "BE/solo"

LITH="LITHOS_MPS=0 LD_LIBRARY_PATH=$BUILD"

if [[ "$MODE" == paper || "$MODE" == all ]]; then
    pair "time slicing (no MPS)" "LITHOS_MPS=0" "LITHOS_MPS=0"
    mps_start
    pair "MPS" "LITHOS_MPS=0" "LITHOS_MPS=0"

    # Fenced off: disjoint quotas, and HP's higher priority means its TPCs are
    # never lent away at all (§5.3's priority safeguard).
    pair "LithOS fenced"        "$LITH LITHOS_QUOTA=$HPQ LITHOS_PRIORITY=10 LITHOS_ATOMIZER=0" \
                                "$LITH LITHOS_QUOTA=$BEQ LITHOS_PRIORITY=0  LITHOS_ATOMIZER=0"
    pair "LithOS fenced + atoms" "$LITH LITHOS_QUOTA=$HPQ LITHOS_PRIORITY=10" \
                                 "$LITH LITHOS_QUOTA=$BEQ LITHOS_PRIORITY=0"
fi

if [[ "$MODE" == ablation || "$MODE" == all ]]; then
    # ---- §8.4, in the paper's configuration --------------------------------
    #
    # The fenced rows above cannot show what atomization is for. With the priority
    # safeguard on, the best-effort job may never touch the service's TPCs, so
    # there is no head-of-line blocking to remove and splitting its kernels only
    # adds overhead.
    #
    # The paper's setting is the opposite: "resources unused by the sensitive
    # inference app should be donated to the best-effort training job" (§8.1). A
    # Poisson service is idle between requests, so its TPCs ARE lent out — and when
    # the next request arrives they are still occupied by a multi-millisecond
    # training kernel. That is Figure 10(b), and atomization is what makes them
    # reclaimable part-way through: Figure 10(c).
    #
    # Reproducing it needs EQUAL priority, so the borrow can happen at all. HP
    # still holds a quota; the difference is that the quota is lent while idle
    # rather than fenced off.
    mps_start
    ABL_HP="$LITH LITHOS_QUOTA=$HPQ LITHOS_PRIORITY=0"
    ABL_BE="$LITH LITHOS_QUOTA=$BEQ LITHOS_PRIORITY=0"
    pair "donated: sched only"  "$ABL_HP LITHOS_ATOMIZER=0" "$ABL_BE LITHOS_ATOMIZER=0"
    pair "donated: + atomizer"  "$ABL_HP" "$ABL_BE"
    pair "donated: + SLO-sized" "$ABL_HP" "$ABL_BE LITHOS_SLO_US=300"
fi

mps_stop
