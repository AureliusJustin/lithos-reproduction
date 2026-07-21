/* Test kernel compiled to a real CUBIN so the Kernel Atomizer's cubin-splice
 * path runs (cuModuleLoadData with a fatbin/PTX would bypass it). Each block
 * writes its own linear index, so a dropped or double-run block is detectable. */
extern "C" __global__ void mark(int* out) {
    int b = blockIdx.z * gridDim.y * gridDim.x
          + blockIdx.y * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0) out[b] = b + 1000;
}
