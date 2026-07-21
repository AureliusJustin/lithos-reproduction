/* End-to-end clean transfer via exact addressing + deterministic reload:
 *  1. patch CALL->EXIT, load, capture prelude_entry + orig_entry cleanly, unload
 *  2. patch CALL->BRA(orig_entry) using the EXACT captured addresses
 *  3. reload (same deterministic VA) and launch prelude -> BRA -> orig
 * If module load addresses are deterministic across reloads of a same-size cubin,
 * the BRA hits orig exactly and the transfer is clean (no return, no barrier). */
#include <cuda.h>
#include <nvrtc.h>
#include <elf.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
extern void qmd_init(); extern void qmd_arm_capture(); extern unsigned long long qmd_get_captured();
static const unsigned char CS[4]={0x44,0x73,0x00,0x02};
/* EXIT and BRA (unconditional) 16-byte encodings for sm_86 */
static const unsigned char EXITI[16]={0x4d,0x79,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x03,0x00,0xea,0x0f,0x00};
static long find_call(unsigned char*cb,const char*sec){
 Elf64_Ehdr*eh=(void*)cb;Elf64_Shdr*sh=(void*)(cb+eh->e_shoff);const char*ss=(char*)(cb+sh[eh->e_shstrndx].sh_offset);
 for(int i=0;i<eh->e_shnum;i++) if(!strcmp(ss+sh[i].sh_name,sec)){unsigned char*b=cb+sh[i].sh_offset;
   for(size_t x=0;x+16<=sh[i].sh_size;x+=16) if(!memcmp(b+x,CS,4)) return (long)(b+x-cb);}
 return -1;}
int main(){
 cuInit(0);CUdevice d;cuDeviceGet(&d,0);CUcontext c;cuCtxCreate(&c,0,d);
 CUdeviceptr out,meta;cuMemAlloc(&out,256*4);cuMemAlloc(&meta,16);cuMemsetD8(out,0xff,256*4);
 char src[1024];snprintf(src,sizeof(src),
  "struct M{unsigned long long off;};\n"
  "extern \"C\" __global__ void orig(){ if(threadIdx.x==0) ((unsigned*)%lluULL)[blockIdx.x]=blockIdx.x+100; }\n"
  "extern \"C\" __global__ void prelude(){ M* a=(M*)%lluULL; unsigned long long e=a->off; ((void(*)())e)(); }\n",
  (unsigned long long)out,(unsigned long long)meta);
 nvrtcProgram p;nvrtcCreateProgram(&p,src,"p.cu",0,0,0);const char*o[]={"--gpu-architecture=sm_86"};nvrtcCompileProgram(p,1,o);
 size_t z;nvrtcGetCUBINSize(p,&z);unsigned char*cb=malloc(z);nvrtcGetCUBIN(p,(char*)cb);
 long fileoff=find_call(cb,".text.prelude");
 /* in-function offset of the CALL within .text.prelude (for BRA PC) */
 Elf64_Ehdr*eh=(void*)cb;Elf64_Shdr*sh=(void*)(cb+eh->e_shoff);const char*ss=(char*)(cb+sh[eh->e_shstrndx].sh_offset);
 long secoff=0; for(int i=0;i<eh->e_shnum;i++) if(!strcmp(ss+sh[i].sh_name,".text.prelude")) secoff=sh[i].sh_offset;
 long call_off = fileoff - secoff;
 printf("CALL at .text.prelude+%#lx (file %#lx)\n",call_off,fileoff);

 /* Phase 1: CALL->EXIT copy, capture entries cleanly */
 unsigned char*cbe=malloc(z);memcpy(cbe,cb,z);memcpy(cbe+fileoff,EXITI,16);
 CUmodule m1;cuModuleLoadData(&m1,cbe);CUfunction of1,pf1;cuModuleGetFunction(&of1,m1,"orig");cuModuleGetFunction(&pf1,m1,"prelude");
 qmd_init();
 qmd_arm_capture();cuLaunchKernel(of1,4,1,1,1,1,1,0,0,0,0);cuCtxSynchronize();unsigned long long oe=qmd_get_captured();
 qmd_arm_capture();cuLaunchKernel(pf1,1,1,1,1,1,1,0,0,0,0);CUresult s1=cuCtxSynchronize();unsigned long long pe=qmd_get_captured();
 printf("captured: orig=%#llx prelude=%#llx (prelude-exit sync=%d)\n",oe,pe,s1);
 cuModuleUnload(m1);

 /* Phase 2: CALL->BRA(orig) with EXACT addresses. imm=orig-(prelude+call_off)-0x10 */
 long long imm = (long long)(oe - (pe + call_off) - 0x10);
 unsigned iw=(unsigned)imm;
 unsigned char bra[16]={0x47,0x79,0x00,0x00, iw&0xff,(iw>>8)&0xff,(iw>>16)&0xff,(iw>>24)&0xff,
                        0xff,0xff,0x83,0x03,0x00,0xc0,0x0f,0x00};
 memcpy(cb+fileoff,bra,16);
 CUmodule m2;cuModuleLoadData(&m2,cb);CUfunction of2,pf2;cuModuleGetFunction(&of2,m2,"orig");cuModuleGetFunction(&pf2,m2,"prelude");
 /* verify reload determinism */
 qmd_arm_capture();cuLaunchKernel(of2,1,1,1,1,1,1,0,0,0,0);cuCtxSynchronize();unsigned long long oe2=qmd_get_captured();
 printf("reload orig=%#llx (matches=%d) imm=%#x\n",oe2,oe2==oe,iw);
 cuMemsetD8(out,0xff,256*4);
 cuLaunchKernel(pf2,8,1,1,1,1,1,0,0,0,0);CUresult sr=cuCtxSynchronize();
 unsigned h[8];cuMemcpyDtoH(h,out,32);
 int ok=1;for(int i=0;i<8;i++) if(h[i]!=(unsigned)(i+100)) ok=0;
 printf("prelude --BRA--> orig: sync=%d out=%d,%d,..,%d %s\n",sr,h[0],h[1],h[7],(ok&&sr==0)?"<<<<<<<<<< CLEAN TRANSFER WORKS!!!":"");
 return 0;}
