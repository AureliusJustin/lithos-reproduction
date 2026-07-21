/* End-to-end atomizer test through the CUDA *Runtime* API -- the exact path the
 * ML frameworks (PyTorch / TensorFlow / JAX / TensorRT) use. Unlike the driver
 * API, the runtime resolves everything through cuGetProcAddress + the CUDA 12
 * Library API (cuLibraryLoadData / cuLibraryGetModule / cuModuleGetFunction) and
 * launches via cuLaunchKernel[Ex]. LibLithOS must intercept that whole chain
 * (see src/interpose.c) for a `<<<>>>` launch to be atomized.
 *
 * Run under the libcuda.so.1 wrapper: LD_LIBRARY_PATH=build build/test_atomize_runtime
 */
#include <cstdio>
#include <cstdlib>

__global__ void mark(int* out) {
    int b = blockIdx.z * gridDim.y * gridDim.x
          + blockIdx.y * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0) out[b] = b + 1000;
}

int main(int argc, char** argv) {
    int G = argc > 1 ? atoi(argv[1]) : 256, T = 32;
    int* out; cudaMalloc(&out, (size_t)G * 4);
    for (int rep = 0; rep < 4; rep++) {
        cudaMemset(out, 0xff, (size_t)G * 4);
        mark<<<G, T>>>(out);                 /* runtime API -> Library API + Ex launch */
        cudaError_t e = cudaDeviceSynchronize();
        if (e) { printf("test_atomize_runtime: FAIL launch %s\n", cudaGetErrorString(e)); return 1; }
        int* h = (int*)malloc((size_t)G * 4);
        cudaMemcpy(h, out, (size_t)G * 4, cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int b = 0; b < G; b++) if (h[b] != b + 1000) { if (bad < 5) printf("  out[%d]=%d want %d\n", b, h[b], b + 1000); bad++; }
        free(h);
        if (bad) { printf("test_atomize_runtime: FAIL rep %d: %d/%d wrong\n", rep, bad, G); return 1; }
    }
    printf("test_atomize_runtime: PASS (%d blocks via CUDA runtime API, atomized+correct x4)\n", G);
    return 0;
}
