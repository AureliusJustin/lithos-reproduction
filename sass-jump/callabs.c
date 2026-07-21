#include <cuda.h>
#include <nvrtc.h>
#include <elf.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
extern void qmd_init(); extern void qmd_arm_capture(); extern unsigned long long qmd_get_captured();
static const unsigned char CS[4]={0x44,0x73,0x00,0x02};
static long build(CUdeviceptr out,CUdeviceptr meta,CUmodule*m){
 char src[1024]; snprintf(src,sizeof(src),
  "struct M{unsigned long long off;};\n"
  "extern \"C\" __global__ void orig(){ if(threadIdx.x==0) ((unsigned*)%lluULL)[blockIdx.x]=blockIdx.x+100; }\n"
  "extern \"C\" __global__ void prelude(){ M* a=(M*)%lluULL; unsigned long long e=a->off; ((void(*)())e)(); }\n",
  (unsigned long long)out,(unsigned long long)meta);
 nvrtcProgram p;nvrtcCreateProgram(&p,src,"p.cu",0,0,0);const char*o[]={"--gpu-architecture=sm_86"};
 if(nvrtcCompileProgram(p,1,o))exit(2);
 size_t z;nvrtcGetCUBINSize(p,&z);unsigned char*cb=malloc(z);nvrtcGetCUBIN(p,(char*)cb);
 if(cuModuleLoadData(m,cb))exit(3); return 0;
}
int main(){
 cuInit(0); CUdevice d;cuDeviceGet(&d,0);CUcontext c;cuCtxCreate(&c,0,d);
 CUdeviceptr out,meta; cuMemAlloc(&out,256*4); cuMemAlloc(&meta,16); cuMemsetD8(out,0xff,256*4);
 CUmodule m; build(out,meta,&m);
 CUfunction of,pf; cuModuleGetFunction(&of,m,"orig"); cuModuleGetFunction(&pf,m,"prelude");
 qmd_init();
 qmd_arm_capture(); cuLaunchKernel(of,4,1,1,1,1,1,0,0,0,0); cuCtxSynchronize(); unsigned long long oe=qmd_get_captured();
 unsigned h0[4]; cuMemcpyDtoH(h0,out,16); printf("orig direct: out=%d,%d,%d,%d entry=%#llx\n",h0[0],h0[1],h0[2],h0[3],oe);
 cuMemsetD8(out,0xff,256*4);
 unsigned long long abs=oe; cuMemcpyHtoD(meta,&abs,8);   // ABSOLUTE target
 cuLaunchKernel(pf,8,1,1,1,1,1,0,0,0,0); CUresult sr=cuCtxSynchronize();
 unsigned h[8]; cuMemcpyDtoH(h,out,32);
 int ok=1; for(int i=0;i<8;i++) if(h[i]!=(unsigned)(i+100)) ok=0;
 printf("prelude --CALL(abs)--> orig: sync=%d out=%d,%d,..,%d %s\n",sr,h[0],h[1],h[7], ok?"<<<<<<<<<< WORKS! CALL+EXIT IS FINE":"");
 return 0;}
