/* graphsched.c -- paper-model CUDA-graph scheduling.
 *
 * The Kernel Atomizer's graph path (atomizer.c) bakes each kernel's atoms into
 * the graph as nodes; that freezes the schedule and (as we measured) the QMD SM
 * mask, because a graph's pre-upload callback fires exactly ONCE at first launch
 * and replays re-run a compiled command buffer that bypasses the driver's per-node
 * path. This module implements the alternative the paper describes -- "atomize
 * graphs into subgraphs, ensuring correct execution ordering":
 *
 *   - Do NOT split kernels into atoms.
 *   - At cuGraphInstantiate, partition the graph into K subgraphs along a
 *     topological cut of its dependency DAG.
 *   - At cuGraphLaunch, launch the K subgraph-execs sequentially on the stream
 *     (stream order preserves all cross-cut dependencies), applying the
 *     scheduler's TPC allocation (a sticky mask) before each subgraph.
 *
 * The scheduling/reallocation unit is the SUBGRAPH (coarser than an atom); the
 * CPU-launch-bypass benefit is retained (K launches, K << #kernels). Because the
 * SM mask bakes at each sub-exec's first launch, changing a subgraph's allocation
 * means re-instantiating that ONE subgraph -- cheap at subgraph granularity.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include "lithos.h"
#include "qmd.h"
#include "real.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

#define GS_MAXSUB 64

static CUresult (*r_Instantiate)(CUgraphExec*, CUgraph, unsigned long long);
static CUresult (*r_Launch)(CUgraphExec, CUstream);
static CUresult (*r_ExecDestroy)(CUgraphExec);
static CUresult (*r_Clone)(CUgraph*, CUgraph);
static CUresult (*r_GetNodes)(CUgraph, CUgraphNode*, size_t*);
static CUresult (*r_GetEdges)(CUgraph, CUgraphNode*, CUgraphNode*, size_t*);
static CUresult (*r_FindInClone)(CUgraphNode*, CUgraphNode, CUgraph);
static CUresult (*r_DestroyNode)(CUgraphNode);
static CUresult (*r_GraphDestroy)(CUgraph);

static int gs_ready = 0;
static void gs_resolve(void) {
    if (gs_ready) return;
    r_Instantiate = (typeof(r_Instantiate))lithos_real_sym("cuGraphInstantiateWithFlags");
    r_Launch      = (typeof(r_Launch))lithos_real_sym("cuGraphLaunch");
    r_ExecDestroy = (typeof(r_ExecDestroy))lithos_real_sym("cuGraphExecDestroy");
    r_Clone       = (typeof(r_Clone))lithos_real_sym("cuGraphClone");
    r_GetNodes    = (typeof(r_GetNodes))lithos_real_sym("cuGraphGetNodes");
    r_GetEdges    = (typeof(r_GetEdges))lithos_real_sym("cuGraphGetEdges");
    r_FindInClone = (typeof(r_FindInClone))lithos_real_sym("cuGraphNodeFindInClone");
    r_DestroyNode = (typeof(r_DestroyNode))lithos_real_sym("cuGraphDestroyNode");
    r_GraphDestroy= (typeof(r_GraphDestroy))lithos_real_sym("cuGraphDestroy");
    gs_ready = 1;
}
static int gs_have_all(void) {
    return r_Instantiate && r_Launch && r_ExecDestroy && r_Clone && r_GetNodes &&
           r_GetEdges && r_FindInClone && r_DestroyNode && r_GraphDestroy;
}

/* one instantiated, partitioned graph */
typedef struct {
    CUgraphExec app;                 /* handle handed back to the app (== sub[0])  */
    int         n;                   /* number of subgraphs                        */
    CUgraphExec sub[GS_MAXSUB];      /* current executable per subgraph            */
    CUgraph     tmpl[GS_MAXSUB];     /* KEPT subgraph templates (for re-instantiate)*/
    uint64_t    cur_mask[GS_MAXSUB]; /* TPC allocation currently baked into sub[i] */
    int         baked[GS_MAXSUB];    /* whether sub[i] has been launched (mask baked)*/
    int         epoch;               /* launch counter (drives rotation demo)      */
    int         used;
} GsEntry;
static GsEntry g_tab[256];
static pthread_mutex_t g_lk = PTHREAD_MUTEX_INITIALIZER;

#define GSLOG(...) do{ if(getenv("LITHOS_LOG_GRAPH")) fprintf(stderr, __VA_ARGS__); }while(0)

/* Partition `graph` into K subgraphs along a topological cut and instantiate each.
 * Fills out[0..K-1] (execs) and outg[0..K-1] (KEPT templates, for later
 * re-instantiation). Returns the count (>=1), or 0 on failure. */
static int build_subgraphs(CUgraph graph, int K, CUgraphExec* out, CUgraph* outg) {
    size_t nn = 0;
    if (r_GetNodes(graph, NULL, &nn) != CUDA_SUCCESS || nn == 0) return 0;
    if (K > (int)nn) K = (int)nn;
    if (K < 1) K = 1;

    CUgraphNode* nodes = calloc(nn, sizeof(*nodes));
    r_GetNodes(graph, nodes, &nn);

    size_t ne = 0; r_GetEdges(graph, NULL, NULL, &ne);
    CUgraphNode* efrom = calloc(ne ? ne : 1, sizeof(*efrom));
    CUgraphNode* eto   = calloc(ne ? ne : 1, sizeof(*eto));
    if (ne) r_GetEdges(graph, efrom, eto, &ne);

    /* index lookup */
    int* indeg = calloc(nn, sizeof(int));
    /* successor lists */
    int* succ_cnt = calloc(nn, sizeof(int));
    for (size_t e = 0; e < ne; e++) {
        int ti = -1;
        for (size_t i = 0; i < nn; i++) if (nodes[i] == eto[e]) { ti = (int)i; break; }
        if (ti >= 0) indeg[ti]++;
    }
    /* Kahn's topological order (stable: scan node-array order) */
    int* order = calloc(nn, sizeof(int));
    int* deg = calloc(nn, sizeof(int));
    memcpy(deg, indeg, nn * sizeof(int));
    int on = 0;
    int progress = 1;
    while (on < (int)nn && progress) {
        progress = 0;
        for (size_t i = 0; i < nn; i++) {
            if (deg[i] == 0) {
                deg[i] = -1; order[on++] = (int)i; progress = 1;
                /* decrement successors of node i */
                for (size_t e = 0; e < ne; e++) if (efrom[e] == nodes[i]) {
                    for (size_t j = 0; j < nn; j++) if (nodes[j] == eto[e]) { if (deg[j] > 0) deg[j]--; break; }
                }
            }
        }
    }
    if (on != (int)nn) {   /* cycle or unmapped edge: bail, caller falls back */
        GSLOG("[graph] topo sort incomplete (%d/%zu) -- falling back\n", on, nn);
        free(nodes); free(efrom); free(eto); free(indeg); free(succ_cnt); free(order); free(deg);
        return 0;
    }
    (void)succ_cnt;
    /* chunk-of-node from topological position */
    int chunk_sz = ((int)nn + K - 1) / K;
    int* chunk_of = calloc(nn, sizeof(int));
    for (int pos = 0; pos < (int)nn; pos++) chunk_of[order[pos]] = pos / chunk_sz;
    int Keff = (((int)nn - 1) / chunk_sz) + 1;   /* actual non-empty chunks */

    int made = 0;
    for (int c = 0; c < Keff; c++) {
        CUgraph gc;
        if (r_Clone(&gc, graph) != CUDA_SUCCESS) break;
        /* delete every node NOT in chunk c from the clone */
        int ok = 1;
        for (size_t i = 0; i < nn; i++) {
            if (chunk_of[i] == c) continue;
            CUgraphNode cn;
            if (r_FindInClone(&cn, nodes[i], gc) != CUDA_SUCCESS) { ok = 0; break; }
            if (r_DestroyNode(cn) != CUDA_SUCCESS) { ok = 0; break; }
        }
        CUgraphExec ex = NULL;
        if (ok && r_Instantiate(&ex, gc, 0) == CUDA_SUCCESS) {
            out[made] = ex; outg[made] = gc;   /* KEEP the template for re-instantiate */
            made++;
        } else {
            r_GraphDestroy(gc);                /* only destroy on failure */
            ok = 0;
        }
        if (!ok) break;
    }
    int rc = (made == Keff) ? made : 0;
    if (!rc) for (int i = 0; i < made; i++) { r_ExecDestroy(out[i]); r_GraphDestroy(outg[i]); }
    GSLOG("[graph] %zu nodes, %zu edges -> %d subgraphs (chunk_sz=%d)\n", nn, ne, rc, chunk_sz);
    free(nodes); free(efrom); free(eto); free(indeg); free(succ_cnt);
    free(order); free(deg); free(chunk_of);
    return rc;
}

int lithos_graph_instantiate(void* pExec, void* graph, unsigned long long flags) {
    if (g_lithos_cfg.graph_subgraphs <= 1) return 0;   /* disabled: real call */
    gs_resolve();
    if (!gs_have_all()) return 0;                       /* missing API: fall back */

    CUgraphExec sub[GS_MAXSUB]; CUgraph tmpl[GS_MAXSUB];
    int n = build_subgraphs((CUgraph)graph, g_lithos_cfg.graph_subgraphs, sub, tmpl);
    if (n <= 0) return 0;                               /* fall back to real path */
    if (n == 1) {                                       /* nothing to partition */
        r_ExecDestroy(sub[0]); r_GraphDestroy(tmpl[0]);
        return 0;
    }
    pthread_mutex_lock(&g_lk);
    GsEntry* slot = NULL;
    for (int i = 0; i < 256; i++) if (!g_tab[i].used) { slot = &g_tab[i]; break; }
    if (!slot) { pthread_mutex_unlock(&g_lk);
        for (int i = 0; i < n; i++) { r_ExecDestroy(sub[i]); r_GraphDestroy(tmpl[i]); } return 0; }
    memset(slot, 0, sizeof(*slot));
    slot->used = 1; slot->n = n; slot->app = sub[0];
    for (int i = 0; i < n; i++) { slot->sub[i] = sub[i]; slot->tmpl[i] = tmpl[i]; slot->baked[i] = 0; }
    pthread_mutex_unlock(&g_lk);
    *(CUgraphExec*)pExec = sub[0];                       /* app sees sub[0] as its exec */
    (void)flags;
    GSLOG("[graph] instantiated as %d subgraphs, app exec=%p\n", n, (void*)sub[0]);
    return 1;
}

int lithos_graph_launch(void* exec, void* stream) {
    pthread_mutex_lock(&g_lk);
    GsEntry* slot = NULL;
    for (int i = 0; i < 256; i++) if (g_tab[i].used && g_tab[i].app == (CUgraphExec)exec) { slot = &g_tab[i]; break; }
    pthread_mutex_unlock(&g_lk);
    if (!slot) return 0;                                 /* not ours: real launch */

    /* Rotation demo: when LITHOS_SUBGRAPH_ROTATE is set, the scheduler's desired
       allocation for subgraph i rotates each replay (i -> (i+epoch) mod n). This
       stands in for a live scheduler changing allocations between replays. */
    int rotate = getenv("LITHOS_SUBGRAPH_ROTATE") ? slot->epoch : 0;

    /* Launch subgraphs in topological order on the same stream (ordering =
       dependencies). For each, get the scheduler's DESIRED TPC allocation; if it
       differs from what's currently baked into this subgraph's exec, RE-INSTANTIATE
       that one subgraph from its kept template so the new mask bakes on its (fresh)
       first launch. Unchanged subgraphs just replay -- no rebuild. */
    int reinst = 0;
    for (int i = 0; i < slot->n; i++) {
        int eff = slot->n ? ((i + rotate) % slot->n) : i;
        uint64_t want = lithos_slice_mask(stream, eff, slot->n);
        if (slot->baked[i] && slot->cur_mask[i] == want) {
            r_Launch(slot->sub[i], (CUstream)stream);              /* reuse: replay */
        } else {
            if (slot->baked[i]) {                                  /* reallocate: rebuild exec */
                r_ExecDestroy(slot->sub[i]);
                if (r_Instantiate(&slot->sub[i], slot->tmpl[i], 0) != CUDA_SUCCESS) {
                    GSLOG("[graph] re-instantiate sub %d failed\n", i); return 1; }
                reinst++;
            }
            if (want) qmd_set_sticky_mask(want);
            r_Launch(slot->sub[i], (CUstream)stream);              /* first launch bakes mask */
            qmd_clear_sticky_mask();
            slot->cur_mask[i] = want; slot->baked[i] = 1;
        }
    }
    if (reinst) GSLOG("[graph] epoch %d: re-instantiated %d subgraph(s) for reallocation\n", slot->epoch, reinst);
    slot->epoch++;
    return 1;
}

int lithos_graph_exec_destroy(void* exec) {
    pthread_mutex_lock(&g_lk);
    GsEntry* slot = NULL;
    for (int i = 0; i < 256; i++) if (g_tab[i].used && g_tab[i].app == (CUgraphExec)exec) { slot = &g_tab[i]; break; }
    if (!slot) { pthread_mutex_unlock(&g_lk); return 0; }
    int n = slot->n; CUgraphExec sub[GS_MAXSUB]; CUgraph tmpl[GS_MAXSUB];
    memcpy(sub, slot->sub, n * sizeof(CUgraphExec));
    memcpy(tmpl, slot->tmpl, n * sizeof(CUgraph));
    slot->used = 0;
    pthread_mutex_unlock(&g_lk);
    gs_resolve();
    for (int i = 0; i < n; i++) { r_ExecDestroy(sub[i]); r_GraphDestroy(tmpl[i]); }
    return 1;
}
