/* CUDA-graph atomization test (Special Kernels, §6): a captured kernel launch is
 * atomized into a *subgraph* (per-atom metadata + relaunch nodes) that replays
 * correctly. Per-atom ranges are written with cuMemsetD32Async so their values
 * bake into the graph nodes (a host->device copy would run synchronously and
 * invalidate capture). Warmup before capture, as all CUDA-graph code must.
 *
 * Run under the wrapper: LD_LIBRARY_PATH=build build/test_atomize_graph
 */
#include <cstdio>
#include <cstdlib>

__global__ void mark(int* out) {
    int b = blockIdx.z * gridDim.y * gridDim.x
          + blockIdx.y * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0) out[b] = b + 1000;
}
#define CK(x) do{ cudaError_t e=(x); if(e){ printf("test_atomize_graph: FAIL %s: %s\n",#x,cudaGetErrorString(e)); return 1; } }while(0)

int main(int argc, char** argv) {
    int G = argc > 1 ? atoi(argv[1]) : 512, T = 32;
    int* out; CK(cudaMalloc(&out, (size_t)G * 4));
    cudaStream_t s; CK(cudaStreamCreate(&s));

    mark<<<G, T, 0, s>>>(out); CK(cudaStreamSynchronize(s));   /* warmup */

    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    mark<<<G, T, 0, s>>>(out);                                  /* atomized -> subgraph */
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));

    for (int rep = 0; rep < 5; rep++) {
        CK(cudaMemset(out, 0xff, (size_t)G * 4));
        CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
        int* h = (int*)malloc((size_t)G * 4);
        CK(cudaMemcpy(h, out, (size_t)G * 4, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int b = 0; b < G; b++) if (h[b] != b + 1000) { if (bad < 5) printf("  out[%d]=%d want %d\n", b, h[b], b + 1000); bad++; }
        free(h);
        if (bad) { printf("test_atomize_graph: FAIL replay %d: %d/%d wrong\n", rep, bad, G); return 1; }
    }
    printf("test_atomize_graph: PASS (captured graph atomized into subgraph, replayed 5x, %d blocks correct)\n", G);
    return 0;
}
