#define _GNU_SOURCE
/*
 * tpc_alloc.c — deciding WHICH TPCs a launch may use, and expressing that as an
 * SM-disable mask for the QMD hook.
 *
 * Everything here produces a 64-bit "disable mask": bit t set == TPC t is
 * DISABLED for the next launch. qmd.c writes it into the QMD just before upload.
 * A mask of 0 means unrestricted (all TPCs available).
 *
 * Three policies live here:
 *
 *   Compute quotas (§5.3, Fig. 9 step 2)  — a stream is guaranteed its own
 *       contiguous TPC range [tpc_lo, tpc_hi); everything else is masked off.
 *   TPC stealing (§5.3)                   — idle streams' TPCs are lent to a
 *       stream with runnable work, so allocations stay work-conserving.
 *   Right-sizing (§5.5, Fig. 9 step 6)    — shrink a *single kernel's* allocation
 *       to the fewest TPCs it can actually use, within a latency-slip budget.
 *
 * Plus the per-slice masks that let each ATOM of a kernel (or each SUBGRAPH of a
 * CUDA graph) land on a different TPC set.
 */
#include "sched_internal.h"
#include "lithos_sched.h"
#include "qmd.h"
#include "predict.h"
#include "coord.h"
#include <stdio.h>

/* Mask of every TPC that exists on this device (bit t set == TPC t is real).
 *
 * LIMIT: the mask is 64 bits, so TPCs 64 and above cannot be addressed. That
 * covers every GPU we test on (A6000 = 42 TPCs, A100 = 54) but NOT Hopper
 * (H100 = 66 TPCs): qmd.c already writes Hopper's two "extended" mask words as
 * all-ones, so TPCs 64+ stay permanently ENABLED and simply cannot be masked
 * off. Supporting them means widening this mask (and the qmd_set_*_mask API)
 * past 64 bits. Flagged in docs/TECHNICAL_REPORT.md rather than silently mis-scheduling. */
static uint64_t all_tpcs_bits(void) {
    return (g_num_tpcs >= 64) ? ~0ull : ((1ull << g_num_tpcs) - 1);
}

/* ---- Compute quotas + TPC stealing (§5.3) --------------------------------- */

/* Build the disable-mask for `st`: enable its own quota range, plus — when
 * `allow_steal` and stealing is on — the ranges of any streams that look idle.
 *
 * A stream counts as idle if it has never launched, hasn't launched within
 * g_idle_ns, or has no outstanding work. Special (cross-block-sync) kernels pass
 * allow_steal = 0 so they get exactly their quota: they sized themselves to the
 * SM count we reported, and handing them extra TPCs mid-flight would break that
 * assumption. Caller holds g_lock. */
uint64_t compute_disable_mask_ex(StreamState* st, uint64_t now, int allow_steal) {
    if (!st || st->quota_tpcs <= 0 || g_num_tpcs == 0) return 0;   /* unrestricted */

    uint64_t enable = 0;
    for (int t = st->tpc_lo; t < st->tpc_hi && t < 64; t++) enable |= (1ull << t);

    /* Everything we might BORROW is accumulated separately from our own quota,
     * because the per-TPC timers below may veto a borrow but must never touch the
     * range this stream is guaranteed (§5.2: a quota is a guarantee). */
    uint64_t steal = 0;

    /* Stealing can only add TPCs if some other stream owns a DIFFERENT range.
     * When every stream shares one application-wide range (the default), lending
     * is a no-op — so skip the scan rather than walk all slots on every launch. */
    if (allow_steal && g_lithos_cfg.enable_stealing && g_ranges_disjoint) {
        for (int i = 0; i < MAX_STREAMS; i++) {
            StreamState* other = &g_streams[i];
            if (!other->in_use || other == st || other->quota_tpcs <= 0) continue;
            int idle = (other->last_launch_ns == 0) ||
                       (now - other->last_launch_ns > g_idle_ns) ||
                       (other->outstanding == 0);
            if (!idle) continue;
            for (int t = other->tpc_lo; t < other->tpc_hi && t < 64; t++) steal |= (1ull << t);
        }
    }

    /* TPC Stealing ACROSS APPLICATIONS (§5.3): "the scheduler dynamically
     * reassigns underutilized TPCs across applications". The coordinator reports
     * which other tenants are idle right now and lends us their TPCs — never a
     * higher-priority tenant's, and only until that tenant has work again. */
    if (allow_steal && g_lithos_cfg.enable_stealing)
        steal |= coord_stealable_tpcs(now, g_idle_ns);

    /* Per-TPC timers (§5.3): "these timers help avoid stealing from long-running
     * TPCs". The idle checks above are a binary busy/idle test, and they are blind
     * to a TPC that was handed a multi-millisecond kernel a moment ago: its owner
     * has outstanding work but has not launched again, so it can read as idle.
     * Borrowing there puts our work behind that kernel — Figure 10(b)'s head-of-
     * line blocking. The timer says how much longer each TPC is expected to be
     * busy, so a borrow that would land behind unfinished work is dropped. */
    if (steal && g_lithos_cfg.tpc_timers) steal &= ~coord_busy_tpcs(now);

    return (~(enable | steal)) & all_tpcs_bits();  /* disable everything not enabled */
}

uint64_t compute_disable_mask(StreamState* st, uint64_t now) {
    return compute_disable_mask_ex(st, now, 1);
}

/* Which TPCs a disable-mask leaves ENABLED. The QMD wants the disable form, but
 * the per-TPC timers are published over the TPCs a launch will actually run on,
 * so the launch path needs to convert back. A zero disable-mask means
 * unrestricted, i.e. every TPC on the device. */
uint64_t tpc_enabled_of(uint64_t disable_mask) {
    return disable_mask ? (~disable_mask) & all_tpcs_bits() : all_tpcs_bits();
}

/* Record this launch's expected completion against every TPC it will occupy
 * (§5.3). `pred_us <= 0` means the predictor has no estimate yet — publishing a
 * guess would either under-protect the TPC or pin it for far too long, so we
 * publish nothing and stealing falls back to the idle checks for that launch. */
void tpc_mark_busy(uint64_t disable_mask, double pred_us, uint64_t now_ns) {
    if (!g_lithos_cfg.tpc_timers || pred_us <= 0) return;
    coord_mark_tpcs_busy(tpc_enabled_of(disable_mask), now_ns,
                         (uint64_t)(pred_us * 1000.0));
}

/* Disable-mask that enables exactly `w` TPCs starting at `base` (right-sizing). */
uint64_t mask_first_tpcs(int base, int w) {
    uint64_t enable = 0;
    for (int t = base; t < base + w && t < 64; t++) enable |= (1ull << t);
    return (~enable) & all_tpcs_bits();
}

/* ---- Per-slice masks: distinct TPCs per atom / per subgraph ---------------- *
 *
 * Each atom of an atomized kernel is a separate launch, and the QMD next-mask is
 * consumed per launch, so setting a *different* mask before each atom confines
 * each atom to its own TPC set — the paper's "TPC allocations can be dynamically
 * adjusted throughout a kernel's execution" (§5.4). The same function serves
 * graph subgraphs (graphsched.c), where "slice" means subgraph rather than atom.
 *
 * Slices are laid out inside the stream's own span — [tpc_lo, tpc_hi) if it has a
 * quota, else all TPCs — and wrap within it, so a slice can never escape the
 * tenant's allocation.
 *
 *   LITHOS_ATOM_TPC_LIST=a,b,c   slice i gets list[i % n] TPCs, packed one after
 *                                another (variable widths)
 *   LITHOS_ATOM_TPC=W            every slice gets its own W-TPC block
 *   neither                      fall back to the stream's quota mask, which still
 *                                matters: it re-applies the quota to EVERY atom
 *                                (the one-shot mask would only cover the first)
 */
uint64_t lithos_slice_mask(void* stream, int idx, int n) {
    (void)n;
    if (g_num_tpcs == 0) return 0;

    int width     = g_lithos_cfg.atom_tpc_width;
    int list_len  = g_lithos_cfg.atom_tpc_list_n;
    uint64_t dmask = 0;

    pthread_mutex_lock(&g_lock);
    StreamState* st = find_stream((CUstream)stream);

    /* The span this stream may use: its quota range, or the whole device. */
    int span_lo = 0, span = (int)g_num_tpcs;
    if (st && st->quota_tpcs > 0) { span_lo = st->tpc_lo; span = st->tpc_hi - st->tpc_lo; }
    if (span < 1) span = 1;

    if (list_len > 0) {
        /* Variable widths: slice idx is `w` TPCs wide and starts after the sum of
         * all preceding slices' widths (so slices pack contiguously), wrapping. */
        int w = g_lithos_cfg.atom_tpc_list[idx % list_len];
        if (w < 1) w = 1;
        long start = 0;
        for (int j = 0; j < idx; j++) start += g_lithos_cfg.atom_tpc_list[j % list_len];
        uint64_t enable = 0;
        for (int t = 0; t < w; t++) {
            int tpc = span_lo + (int)((start + t) % span);
            if (tpc < 64) enable |= (1ull << tpc);
        }
        dmask = (~enable) & all_tpcs_bits();
    } else if (width > 0) {
        /* Uniform widths: slice idx occupies the idx-th W-TPC block, wrapping. */
        uint64_t enable = 0;
        for (int t = 0; t < width; t++) {
            int tpc = span_lo + (((idx * width) + t) % span);
            if (tpc < 64) enable |= (1ull << tpc);
        }
        dmask = (~enable) & all_tpcs_bits();
    } else if (st) {
        dmask = compute_disable_mask(st, lithos_now_ns());
    }

    pthread_mutex_unlock(&g_lock);
    return dmask;
}

/* Arm the mask for the next atom launch (one-shot; consumed by the QMD hook). */
void lithos_apply_atom_mask(void* stream, int atom_idx, int n_atoms) {
    uint64_t dmask = lithos_slice_mask(stream, atom_idx, n_atoms);
    if (dmask) qmd_set_next_mask(dmask);
}

/* ---- Hardware right-sizing (§5.5) ----------------------------------------- *
 *
 * Give a kernel the FEWEST TPCs it can use without slowing down more than the
 * operator's tolerance. Two stages, exactly as the paper describes:
 *
 *  1. Filtering heuristic — a kernel can't use more TPCs than it has blocks to
 *     fill them with. useful_TPCs = ceil(blocks / blocks-resident-per-TPC), where
 *     blocks-per-TPC comes from the occupancy API. If that bound is already below
 *     the current allocation, take it; no model needed.
 *  2. Scaling model — otherwise consult the learned curve l = m/t + b (predict.c)
 *     for the smallest t whose predicted latency stays within `latency_slip` of
 *     the all-TPC latency. The model needs two samples (at all-TPC and at 1 TPC);
 *     until it has them it asks us to PROBE, i.e. force this launch to a specific
 *     TPC count so the measurement lands where the model needs it.
 *
 * Returns the TPC count to use (<= cur_tpc); sets *probe to a forced count, or 0.
 */
int rightsize_tpcs(CUfunction f, int block_threads, unsigned shmem, uint64_t blocks,
                   int slot, int op, int cur_tpc, int* probe) {
    *probe = 0;
    if (cur_tpc < 2 || block_threads < 1) return cur_tpc;   /* nothing to shrink */

    /* Stage 1: occupancy-based upper bound on useful TPCs.
     *
     * LITHOS_RIGHTSIZE_OCC=0 skips it, leaving stage 2 to decide alone. The two
     * stages disagree sharply on real kernels (BENCHMARKS §18), so they have to be
     * separately measurable. */
    int blocks_per_sm = 0;
    if (g_lithos_cfg.rightsize_occ &&
        cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, f, block_threads, shmem)
            == CUDA_SUCCESS && blocks_per_sm > 0) {
        int blocks_per_tpc = blocks_per_sm * 2;             /* 2 SMs per TPC */
        int useful = (int)((blocks + blocks_per_tpc - 1) / blocks_per_tpc);
        if (useful >= 1 && useful < cur_tpc) return useful;
    }

    /* Stage 2: the learned scaling model (may request a probe first). */
    int want_probe = 0;
    int t_min = predict_rightsize(slot, op, cur_tpc, g_lithos_cfg.latency_slip, &want_probe);
    if (want_probe > 0) {
        *probe = want_probe;
        return want_probe < cur_tpc ? want_probe : cur_tpc;
    }
    if (t_min > 0 && t_min < cur_tpc) return t_min;
    return cur_tpc;
}
