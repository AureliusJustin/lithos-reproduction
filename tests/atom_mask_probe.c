/* atom_mask_probe.c — launch one atomized kernel through a chosen launch API.
 *
 * Used by tests/test_atom_mask.sh, which counts QMD pre-upload callbacks against
 * applied TPC masks. The invariant it guards: EVERY atom of a launch carries the
 * scheduler's allocation. The QMD next-mask is one-shot (consumed by the upload
 * it applies to), so a path that forgets to re-arm it leaves atoms 1..n-1
 * unmasked — they run on the whole device and silently escape the tenant's quota
 * (§5.2 makes a quota a guarantee). That regression existed on the
 * cuLaunchKernelEx path, which is one of the two the CUDA runtime uses.
 *
 *   atom_mask_probe std|ex <blocks>
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do{ CUresult r=(x); if(r){ const char* s=0; cuGetErrorName(r,&s); \
    fprintf(stderr,"atom_mask_probe: %s @%d\n", s?s:"?", __LINE__); return 2; } }while(0)

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "std";
    int G = argc > 2 ? atoi(argv[2]) : 256;
    int use_ex = strcmp(mode, "ex") == 0;

    CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d, 0));
    CUcontext ctx; CK(cuCtxCreate(&ctx, 0, d));

    const char* path = getenv("MASK_CUBIN");
    if (!path) path = "build/atomize_mark.cubin";
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "atom_mask_probe: cannot open %s\n", path); return 2; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* b = malloc(n);
    if (!b || fread(b, 1, n, f) != (size_t)n) return 2;
    fclose(f);

    CUmodule m; CK(cuModuleLoadData(&m, b));
    CUfunction fn; CK(cuModuleGetFunction(&fn, m, "mark"));
    CUdeviceptr out; CK(cuMemAlloc(&out, (size_t)G * 4)); CK(cuMemsetD32(out, 0, G));
    CUstream s; CK(cuStreamCreate(&s, 0));
    void* args[] = { &out };

    if (use_ex) {
        CUlaunchConfig cfg; memset(&cfg, 0, sizeof cfg);
        cfg.gridDimX = G; cfg.gridDimY = 1; cfg.gridDimZ = 1;
        cfg.blockDimX = 32; cfg.blockDimY = 1; cfg.blockDimZ = 1;
        cfg.hStream = s;
        CK(cuLaunchKernelEx(&cfg, fn, args, NULL));
    } else {
        CK(cuLaunchKernel(fn, G,1,1, 32,1,1, 0, s, args, NULL));
    }
    CK(cuStreamSynchronize(s));

    /* Every block must still have run exactly once, whatever the masking did. */
    int* h = malloc((size_t)G * 4);
    CK(cuMemcpyDtoH(h, out, (size_t)G * 4));
    int bad = 0;
    for (int i = 0; i < G; i++) if (h[i] != i + 1000) bad++;
    printf("atom_mask_probe(%s): %d blocks, %d wrong\n", mode, G, bad);
    return bad ? 1 : 0;
}
