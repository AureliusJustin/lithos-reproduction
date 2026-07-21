#include <cuda.h>
#include <nvrtc.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
extern void qmd_init(); extern void qmd_arm_capture(); extern unsigned long long qmd_get_captured();
int main(int argc,char**argv){
 const char* origbody = argc>1 ? argv[1] : "";  // orig body
 cuInit(0); CUdevice d;cuDeviceGet(&d,0);CUcontext c;cuCtxCreate(&c,0,d);
 CUdeviceptr meta,out; cuMemAlloc(&meta,16); cuMemAlloc(&out,64); cuMemsetD8(out,0,64);
 char src[1024]; snprintf(src,sizeof(src),
  "struct M{unsigned long long off;};\n"
  "extern \"C\" __global__ void orig(){ %s }\n"
  "extern \"C\" __global__ void prelude(){ M* a=(M*)%lluULL; unsigned long long e=a->off; ((void(*)())e)(); }\n",
  origbody,(unsigned long long)meta);
 nvrtcProgram p;nvrtcCreateProgram(&p,src,"p.cu",0,0,0);const char*o[]={"--gpu-architecture=sm_86"};
 if(nvrtcCompileProgram(p,1,o)){size_t z;nvrtcGetProgramLogSize(p,&z);char*l=malloc(z);nvrtcGetProgramLog(p,l);printf("%s\n",l);return 2;}
 size_t z;nvrtcGetCUBINSize(p,&z);char*cb=malloc(z);nvrtcGetCUBIN(p,cb);
 CUmodule m; cuModuleLoadData(&m,cb); CUfunction of,pf; cuModuleGetFunction(&of,m,"orig");cuModuleGetFunction(&pf,m,"prelude");
 qmd_init();
 qmd_arm_capture(); cuLaunchKernel(of,1,1,1,1,1,1,0,0,0,0); cuCtxSynchronize(); unsigned long long oe=qmd_get_captured();
 cuMemcpyHtoD(meta,&oe,8);
 cuLaunchKernel(pf,1,1,1,1,1,1,0,0,0,0); CUresult sr=cuCtxSynchronize();
 printf("orig body=\"%s\": prelude CALL(abs)->orig sync=%d %s\n",origbody,sr,sr==0?"<<< CLEAN EXIT":"");
 return 0;}
