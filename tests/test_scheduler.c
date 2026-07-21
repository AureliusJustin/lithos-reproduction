/* Verifies the TPC Scheduler's compute-quota enforcement (Section 5.3 Step 2).
 * With LITHOS_QUOTA=N, a kernel launched through LithOS must run on at most N
 * TPCs (2 SMs each on Ampere), and specifically the first N TPCs. Records the
 * %smid of every block and checks the confinement. Run under LD_PRELOAD. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define CK(x) do{CUresult r=(x); if(r){const char*s;cuGetErrorName(r,&s); \
    fprintf(stderr,"%s: %s\n",#x,s); exit(1);} }while(0)

/* Each block records its SM id. */
static const char* ptx =
".version 6.0\n.target sm_50\n.address_size 64\n"
".visible .entry rec(.param .u64 po){\n"
"  .reg .b32 %r<4>; .reg .b64 %rd<5>;\n"
"  ld.param.u64 %rd1,[po];\n"
"  mov.u32 %r1,%ctaid.x;\n"
"  mov.u32 %r2,%smid;\n"
"  mul.wide.u32 %rd2,%r1,4;\n"
"  cvta.to.global.u64 %rd3,%rd1; add.s64 %rd4,%rd3,%rd2;\n"
"  st.global.u32 [%rd4],%r2;\n"
"  ret;\n}\n";

int main() {
    int quota = getenv("LITHOS_QUOTA") ? atoi(getenv("LITHOS_QUOTA")) : 0;
    const int NB = 512;
    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev,0));
    CUcontext ctx; CK(cuCtxCreate(&ctx,0,dev));
    int sms=0; cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    CUmodule mod; CK(cuModuleLoadData(&mod,ptx));
    CUfunction fn; CK(cuModuleGetFunction(&fn,mod,"rec"));
    CUdeviceptr d; CK(cuMemAlloc(&d, NB*sizeof(int)));
    int* h = malloc(NB*sizeof(int));
    CUstream s; CK(cuStreamCreate(&s,0));
    void* args[]={&d};
    CK(cuLaunchKernel(fn, NB,1,1, 256,1,1, 0, s, args, NULL));
    CK(cuStreamSynchronize(s));
    CK(cuMemcpyDtoH(h, d, NB*sizeof(int)));

    int seen[256]={0}, uniq=0, maxsm=0;
    for (int i=0;i<NB;i++){ int sm=h[i]&0xff; if(!seen[sm]){seen[sm]=1;uniq++;} if(sm>maxsm)maxsm=sm; }
    printf("test_scheduler: %d blocks used %d distinct SMs (max SM id %d) of %d total; quota=%d TPCs\n",
           NB, uniq, maxsm, sms, quota);
    if (quota > 0) {
        int allowed_sms = quota * 2;   /* 2 SMs per TPC on Ampere */
        if (uniq > allowed_sms || maxsm >= allowed_sms) {
            printf("  FAIL: expected <= %d SMs (ids < %d) with quota %d TPCs\n",
                   allowed_sms, allowed_sms, quota);
            return 1;
        }
        printf("  PASS: kernel confined to the first %d TPCs (%d SMs)\n", quota, allowed_sms);
    } else {
        printf("  (no quota set; baseline used %d SMs)\n", uniq);
    }
    return 0;
}
