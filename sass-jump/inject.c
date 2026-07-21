#include <cuda.h>
#include <nvrtc.h>
#include <elf.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
extern void qmd_init(); extern void qmd_arm_capture(); extern unsigned long long qmd_get_captured();
static const unsigned char CS[4]={0x44,0x73,0x00,0x02};
static const unsigned char BRX[16]={0x49,0x79,0x00,0x02,0x40,0xff,0xff,0xff,0xff,0xff,0x83,0x03,0x00,0xea,0x0f,0x00};
#define CK(x) do{CUresult r=(x); if(r){const char*s;cuGetErrorName(r,&s);printf("%s:%s\n",#x,s);}}while(0)

// Insert `ins` bytes (len) at file offset `at`, fixing section offsets and e_shoff.
static unsigned char* elf_insert(unsigned char* cb, size_t* n, size_t at, const unsigned char* ins, size_t len, int sec_to_grow){
 unsigned char* nb=malloc(*n+len);
 memcpy(nb,cb,at); memcpy(nb+at,ins,len); memcpy(nb+at+len,cb+at,*n-at);
 Elf64_Ehdr*eh=(void*)nb; 
 if(eh->e_shoff>=at) eh->e_shoff+=len;
 Elf64_Shdr*sh=(void*)(nb+eh->e_shoff);
 for(int i=0;i<eh->e_shnum;i++){
   if(i==sec_to_grow) sh[i].sh_size+=len;
   if(sh[i].sh_offset>=at && i!=sec_to_grow) sh[i].sh_offset+=len;
   else if(sh[i].sh_offset>at && i==sec_to_grow) sh[i].sh_offset+=0; // grown section starts before `at`
 }
 *n+=len; free(cb); return nb;
}

int main(){
 CK(cuInit(0)); CUdevice d;cuDeviceGet(&d,0);CUcontext c;cuCtxCreate(&c,0,d);
 CUdeviceptr meta,out; cuMemAlloc(&meta,8); cuMemAlloc(&out,256*4);
 char src[1024]; snprintf(src,sizeof(src),
   "extern \"C\" __global__ void orig(){ if(threadIdx.x==0) ((unsigned*)%lluULL)[blockIdx.x]=blockIdx.x+100; }\n"
   "extern \"C\" __global__ void prelude(){ unsigned long long off=*(unsigned long long*)%lluULL; ((void(*)())off)(); }\n",
   (unsigned long long)out,(unsigned long long)meta);
 nvrtcProgram p;nvrtcCreateProgram(&p,src,"p.cu",0,0,0);const char*opt[]={"--gpu-architecture=sm_80"};
 if(nvrtcCompileProgram(p,1,opt))return 2;
 size_t z;nvrtcGetCUBINSize(p,&z);unsigned char*cb=malloc(z);nvrtcGetCUBIN(p,(char*)cb); size_t n=z;
 // 1) patch transfer CALL->BRX in .text.prelude; get in-function offset
 Elf64_Ehdr*eh=(void*)cb;Elf64_Shdr*sh=(void*)(cb+eh->e_shoff);
 const char* shstr=(char*)(cb+sh[eh->e_shstrndx].sh_offset);
 long coff=-1; int info_prelude=-1;
 for(int i=0;i<eh->e_shnum;i++){
   const char* nm=shstr+sh[i].sh_name;
   if(!strcmp(nm,".text.prelude")){ unsigned char*b=cb+sh[i].sh_offset;
     for(size_t x=0;x+16<=sh[i].sh_size;x+=16) if(!memcmp(b+x,CS,4)){memcpy(b+x,BRX,16);coff=x;break;} }
   if(!strcmp(nm,".nv.info.prelude")) info_prelude=i;
 }
 printf("BRX at in-func off %#lx; .nv.info.prelude sec=%d\n",coff,info_prelude);
 // 2) build EIATTR_INDIRECT_BRANCH_TARGETS: 04 34 <size2> [brx_off][flags=0][count=1][target]
 unsigned brx=(unsigned)coff, tgt=(unsigned)coff+0x10;
 unsigned char attr[4+16]; attr[0]=0x04; attr[1]=0x34; attr[2]=0x10; attr[3]=0x00;
 memcpy(attr+4,&brx,4); unsigned zero=0,one=1; memcpy(attr+8,&zero,4); memcpy(attr+12,&one,4); memcpy(attr+16,&tgt,4);
 // 3) insert at end of .nv.info.prelude
 sh=(void*)(cb+eh->e_shoff);
 size_t at=sh[info_prelude].sh_offset+sh[info_prelude].sh_size; printf("nv.info.prelude off=%#lx size=%#lx at=%#lx\n",(long)sh[info_prelude].sh_offset,(long)sh[info_prelude].sh_size,(long)at);
 cb=elf_insert(cb,&n,at,attr,sizeof(attr),info_prelude);
 // 4) load & test
 {FILE*wf=fopen("/tmp/brxstudy/injected.cubin","wb");fwrite(cb,1,n,wf);fclose(wf);}
 CUmodule m; CUresult lr=cuModuleLoadData(&m,cb); if(lr){const char*s;cuGetErrorName(lr,&s);printf("module load: %s\n",s);return 3;}
 CUfunction of,pf; cuModuleGetFunction(&of,m,"orig");cuModuleGetFunction(&pf,m,"prelude");
 qmd_init();
 qmd_arm_capture(); cuLaunchKernel(of,4,1,1,1,1,1,0,0,0,0); cuCtxSynchronize(); unsigned long long oe=qmd_get_captured();
 unsigned long long safe=0xd0; cuMemcpyHtoD(meta,&safe,8);   // off=0xd0 self-jump for capture
 qmd_arm_capture(); cuLaunchKernel(pf,1,1,1,1,1,1,0,0,0,0); CUresult sc=cuCtxSynchronize(); unsigned long long pe=qmd_get_captured();
 printf("prelude self-jump(off=0xd0 clean-cubin+metadata): sync=%d pe=%#llx\n",sc,pe);
 long long imm=-0xc0; unsigned long long off=oe-(pe+coff)-imm;
 cuMemsetD8(out,0xff,256*4); cuMemcpyHtoD(meta,&off,8);
 cuLaunchKernel(pf,4,1,1,1,1,1,0,0,0,0); CUresult sr=cuCtxSynchronize();
 unsigned h[4]; cuMemcpyDtoH(h,out,16);
 printf("CLEAN PRELUDE --BRX--> ORIG: sync=%d out=%d %d %d %d %s\n",sr,h[0],h[1],h[2],h[3],(h[0]==100&&h[3]==103)?"<<<<<<<<<< ATOMIZER TRANSFER WORKS!!!":"");
 return 0;}
