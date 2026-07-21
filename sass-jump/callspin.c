#include <cuda.h>
#include <nvrtc.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
extern void qmd_init(); extern void qmd_arm_capture(); extern unsigned long long qmd_get_captured();
int main(int argc,char**argv){
 int rel=argc>1&&!strcmp(argv[1],"rel");
 cuInit(0); CUdevice d;cuDeviceGet(&d,0);CUcontext c;cuCtxCreate(&c,0,d);
 CUdeviceptr out,meta; cuMemAlloc(&out,256*4); cuMemAlloc(&meta,16); cuMemsetD8(out,0,256*4);
 char src[1024]; snprintf(src,sizeof(src),
  "struct M{unsigned long long off;};\n"
  "extern \"C\" __global__ void orig(){ ((volatile unsigned*)%lluULL)[0]=0xABCD; while(((volatile unsigned*)%lluULL)[1]!=0xDEAD){} }\n"
  "extern \"C\" __global__ void prelude(){ M* a=(M*)%lluULL; unsigned long long e=a->off; ((void(*)())e)(); }\n",
  (unsigned long long)out,(unsigned long long)out,(unsigned long long)meta);
 nvrtcProgram p;nvrtcCreateProgram(&p,src,"p.cu",0,0,0);const char*o[]={"--gpu-architecture=sm_86"};
 if(nvrtcCompileProgram(p,1,o))return 2;
 size_t z;nvrtcGetCUBINSize(p,&z);char*cb=malloc(z);nvrtcGetCUBIN(p,cb);
 CUmodule m; cuModuleLoadData(&m,cb); CUfunction of,pf; cuModuleGetFunction(&of,m,"orig");cuModuleGetFunction(&pf,m,"prelude");
 qmd_init();
 qmd_arm_capture(); cuLaunchKernel(of,1,1,1,1,1,1,0,0,0,0); // orig would spin; but capture is pre-exec
 // don't sync orig (it spins). Reset it by... actually launch orig on its own stream won't work. Skip orig direct run.
 unsigned long long oe=qmd_get_captured();
 printf("orig entry=%#llx\n",oe);
 unsigned long long tgt = oe; cuMemcpyHtoD(meta,&tgt,8);
 printf("launching prelude --CALL(abs)--> orig(spins). hang=landed, fast-fault=missed\n");
 cuLaunchKernel(pf,1,1,1,1,1,1,0,0,0,0); CUresult sr=cuCtxSynchronize();
 printf("sync=%d\n",sr);
 return 0;}
