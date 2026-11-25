/*
 * test_gpa_resolver: the resolver handed out for "cuGetProcAddress" must have the
 * ABI the caller asked for. The driver returns the v2 ABI (with the symbolStatus
 * out-parameter) to callers requesting CUDA >= 12.0, and runtimes built against
 * newer toolkits read that status after every lookup. LithOS used to return its
 * v1 wrapper here, so the status was never written and those runtimes failed
 * (the CuTe-DSL runtime that FlashInfer uses inside SGLang: cudaErrorUnknown).
 *
 * Run through the libcuda.so.1 wrapper (LD_LIBRARY_PATH=build), the path the
 * CUDA runtime takes, and through LD_PRELOAD.
 */
#include <cuda.h>
#include <stdio.h>
#include <string.h>

#undef cuGetProcAddress
typedef CUresult (*gpa_v2_t)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);

int main(void) {
    int fails = 0;
    if (cuInit(0) != CUDA_SUCCESS) { printf("test_gpa_resolver: cuInit failed\n"); return 1; }

    /* ask the (interposed) driver for the resolver itself, as a CUDA 12 runtime does */
    void* p = NULL;
    CUdriverProcAddressQueryResult st = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    CUresult r = cuGetProcAddress_v2("cuGetProcAddress", &p, 12000, CU_GET_PROC_ADDRESS_DEFAULT, &st);
    if (r != CUDA_SUCCESS || !p || st != CU_GET_PROC_ADDRESS_SUCCESS) {
        printf("  FAIL: resolver lookup r=%d p=%p status=%d\n", (int)r, p, (int)st); return 1;
    }

    /* call it the v2 way: the status must be written (a v1 wrapper would leave the sentinel) */
    gpa_v2_t gpa = (gpa_v2_t)p;
    const char* syms[] = { "cuInit", "cuLaunchKernel", "cuMemAlloc", "cuStreamSynchronize" };
    for (unsigned i = 0; i < sizeof(syms) / sizeof(syms[0]); i++) {
        void* f = NULL;
        CUdriverProcAddressQueryResult s = (CUdriverProcAddressQueryResult)0x5a5a5a5a;
        CUresult rr = gpa(syms[i], &f, 12000, CU_GET_PROC_ADDRESS_DEFAULT, &s);
        if (rr != CUDA_SUCCESS || !f || s != CU_GET_PROC_ADDRESS_SUCCESS) {
            printf("  FAIL: %s via the returned resolver: r=%d f=%p status=%d\n", syms[i], (int)rr, f, (int)s);
            fails++;
        }
    }
    if (!fails) printf("test_gpa_resolver: PASS (the resolver for CUDA >= 12 callers fills symbolStatus)\n");
    else printf("test_gpa_resolver: FAIL (%d)\n", fails);
    return fails ? 1 : 0;
}
