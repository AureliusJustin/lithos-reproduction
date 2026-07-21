#include <cuda.h>
#include <nvrtc.h>
#include <elf.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
extern void qmd_init(); extern void qmd_arm_capture(); extern unsigned long long qmd_get_captured();
// k() does a switch (BRX+metadata). far() writes 0xFACE. We force k's BRX offset (R4) to jump into far().
static char src[2048];
static const char* tmpl=
"extern \"C\" __global__ void kk(int* o,int s){ switch(s){case 0:o[0]=10;break;case 1:o[0]=11;break;case 2:o[0]=12;break;case 3:o[0]=13;break;case 4:o[0]=14;break;case 5:o[0]=15;break;default:o[0]=99;} }\n"
"extern \"C\" __global__ void faar(){ ((unsigned*)%lluULL)[0]=0xFACE; }\n";
#define CK(x) do{CUresult r=(x); if(r){const char*s;cuGetErrorName(r,&s);printf("%s:%s\n",#x,s);}}while(0)
int main(){
 CK(cuInit(0)); CUdevice d;cuDeviceGet(&d,0);CUcontext c;cuCtxCreate(&c,0,d);
 CUdeviceptr o; cuMemAlloc(&o,16); cuMemsetD32(o,7,4);
 snprintf(src,sizeof(src),tmpl,(unsigned long long)o);
 nvrtcProgram p;nvrtcCreateProgram(&p,src,"p.cu",0,0,0);const char*opt[]={"--gpu-architecture=sm_80"};
 if(nvrtcCompileProgram(p,1,opt))return 2;
 size_t z;nvrtcGetCUBINSize(p,&z);unsigned char*cb=malloc(z);nvrtcGetCUBIN(p,(char*)cb);
 // patch kk's LDC R4 -> MOV R4, PLACEHOLDER (we'll compute after capturing addrs; so patch at runtime via 2nd load)
 // First load unpatched to capture entries
 CUmodule m; CK(cuModuleLoadData(&m,cb)); CUfunction kf,ff; cuModuleGetFunction(&kf,m,"kk");cuModuleGetFunction(&ff,m,"faar");
 qmd_init(); void*none=0;
 int s0=0; void* ka[]={&o,&s0};
 qmd_arm_capture(); cuLaunchKernel(kf,1,1,1,1,1,1,0,0,ka,0); cuCtxSynchronize(); unsigned long long ke=qmd_get_captured();
 qmd_arm_capture(); cuLaunchKernel(ff,1,1,1,1,1,1,0,0,&none,0); cuCtxSynchronize(); unsigned long long fe=qmd_get_captured();
 unsigned hh[4]; cuMemcpyDtoH(hh,o,16); printf("kk_entry=%#llx faar_entry=%#llx  faar direct wrote o[0]=%#x\n",ke,fe,hh[0]);
 // compute R4 to make kk's BRX (at ke+0xb0, imm -0xc0) jump to faar_entry
 long long imm=-0xc0; unsigned long long r4 = fe - (ke+0xb0) - imm;
 printf("need R4=%#llx (fe-kk_BRX-imm) to jump kk->faar\n",r4);
 if(r4 >> 32){ printf("R4 needs >32 bits (%#llx) - far functions need 64-bit offset via R4:R5\n",r4); }
 // patch: LDC R4 -> two MOVs won't fit; instead patch LDC to load a fixed 32-bit and rely on SHF for R5.
 // Only works if r4 fits in signed 32-bit:
 long long r4s=(long long)r4;
 if(r4s < -2147483648LL || r4s > 2147483647LL){ printf(">>> offset exceeds signed 32-bit; would need R5 too\n"); }
 unsigned char ldc[8]={0x82,0x7b,0x04,0x00,0x00,0x00,0x80,0x00};
 unsigned mov_imm=(unsigned)r4;
 unsigned char mov0[8]={0x24,0x74,0x04,0xff,mov_imm&0xff,(mov_imm>>8)&0xff,(mov_imm>>16)&0xff,(mov_imm>>24)&0xff};
 unsigned char mov1[8]={0xff,0x00,0x8e,0x07,0x00,0xca,0x0f,0x00};
 Elf64_Ehdr*eh=(void*)cb;Elf64_Shdr*sh=(void*)(cb+eh->e_shoff);
 for(int i=0;i<eh->e_shnum;i++){if(sh[i].sh_type!=SHT_PROGBITS)continue;unsigned char*b=cb+sh[i].sh_offset;
   for(size_t x=0;x+16<=sh[i].sh_size;x+=16)if(!memcmp(b+x,ldc,8)){memcpy(b+x,mov0,8);memcpy(b+x+8,mov1,8);goto done;}}
 done:;
 CUmodule m2; CK(cuModuleLoadData(&m2,cb)); CUfunction kf2; cuModuleGetFunction(&kf2,m2,"kk");
 cuMemsetD32(o,7,4);
 cuLaunchKernel(kf2,1,1,1,1,1,1,0,0,ka,0); CUresult sr=cuCtxSynchronize();
 cuMemcpyDtoH(hh,o,16);
 printf("kk --BRX--> faar: sync=%d o[0]=%#x %s\n",sr,hh[0], hh[0]==0xFACE?"<<<<<< CROSS-FUNCTION BRX WORKS!":"(0xface=works)");
 return 0;}
