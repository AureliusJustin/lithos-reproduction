// Validates that LibLithOS transparently interposes the CUDA Driver API:
// a normal CUDA-runtime program (streams + kernel launch) must produce correct
// results while every launch flows through LithOS. Run under LD_PRELOAD.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

#define CK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

int main() {
    const int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    float *x, *y, *dx, *dy;
    x = (float*)malloc(bytes); y = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { x[i] = 1.0f; y[i] = 2.0f; }

    CK(cudaMalloc(&dx, bytes)); CK(cudaMalloc(&dy, bytes));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    CK(cudaMemcpyAsync(dx, x, bytes, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(dy, y, bytes, cudaMemcpyHostToDevice, s));

    int threads = 256, blocks = (n + threads - 1) / threads;
    for (int rep = 0; rep < 4; rep++)
        saxpy<<<blocks, threads, 0, s>>>(n, 2.0f, dx, dy);
    CK(cudaMemcpyAsync(y, dy, bytes, cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));

    // y = 2 + 4*(2*1) = 2 + 8 = 10 after 4 reps
    float expect = 2.0f + 4 * (2.0f * 1.0f);
    int bad = 0;
    for (int i = 0; i < n; i++) if (y[i] != expect) { bad++; }
    if (bad) { printf("FAIL: %d/%d mismatches (y[0]=%f expect %f)\n", bad, n, y[0], expect); return 1; }
    printf("test_interpose: PASS (y[i]==%.1f for all %d elems)\n", expect, n);
    CK(cudaStreamDestroy(s));
    return 0;
}
