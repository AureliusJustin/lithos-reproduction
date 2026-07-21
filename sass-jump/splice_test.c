/* PROVE the "prologue splice / fall-through" atomizer:
 * inject a range-check prologue at the FRONT of an app kernel's .text section so
 * in-range blocks FALL THROUGH into the original (no call, no jump, no BSSY).
 * The app kernel is treated as opaque (only its cubin, from interception). */
#include <cuda.h>
#include <nvrtc.h>
#include <elf.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static unsigned char* load_file(const char*p,size_t*n){FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);*n=ftell(f);fseek(f,0,SEEK_SET);unsigned char*b=malloc(*n);fread(b,1,*n,f);fclose(f);return b;}

/* Build the range-check prologue SASS via NVRTC with meta VA baked in.
 * Returns malloc'd bytes and length (cut right before the 0xDEADBEEF marker). */
static unsigned char* build_prologue(unsigned long long meta_va,size_t*plen){
  char src[1024];snprintf(src,sizeof(src),
   "struct M{unsigned lo,hi;};\n"
   "extern \"C\" __global__ void probe(unsigned* sink){\n"
   " unsigned long long b=(unsigned long long)blockIdx.z*gridDim.y*gridDim.x+(unsigned long long)blockIdx.y*gridDim.x+blockIdx.x;\n"
   " M* a=(M*)%lluULL;\n"
   " if(b<a->lo||b>=a->hi) return;\n"
   " asm volatile(\"mov.u32 %%0, 0xDEADBEEF;\":\"=r\"(sink[0])::\"memory\");\n"
   "}\n", meta_va);
  nvrtcProgram p;nvrtcCreateProgram(&p,src,"p.cu",0,0,0);
  const char*o[]={"--gpu-architecture=sm_86"};
  if(nvrtcCompileProgram(p,1,o)){size_t L;nvrtcGetProgramLogSize(p,&L);char*lg=malloc(L);nvrtcGetProgramLog(p,lg);printf("NVRTC:%s\n",lg);return 0;}
  size_t z;nvrtcGetCUBINSize(p,&z);unsigned char*cb=malloc(z);nvrtcGetCUBIN(p,(char*)cb);
  Elf64_Ehdr*eh=(void*)cb;Elf64_Shdr*sh=(void*)(cb+eh->e_shoff);const char*ss=(char*)(cb+sh[eh->e_shstrndx].sh_offset);
  for(int i=0;i<eh->e_shnum;i++) if(!strcmp(ss+sh[i].sh_name,".text.probe")){
    unsigned char*b=cb+sh[i].sh_offset;size_t sz=sh[i].sh_size;
    /* cut at the instruction containing immediate 0xDEADBEEF (LE bytes ef be ad de) */
    size_t cut=sz;
    for(size_t x=0;x+16<=sz;x+=16) for(int k=0;k<13;k++) if(b[x+k]==0xef&&b[x+k+1]==0xbe&&b[x+k+2]==0xad&&b[x+k+3]==0xde){cut=x;goto done;}
    done:;
    /* pad to a multiple of 128 with NOPs so inserting preserves section alignment */
    static const unsigned char NOP[16]={0x18,0x79,0,0,0,0,0,0,0,0,0,0,0,0xc0,0x0f,0};
    size_t padded=(cut+127)&~(size_t)127;
    unsigned char*out=malloc(padded);memcpy(out,b,cut);
    for(size_t x=cut;x<padded;x+=16) memcpy(out+x,NOP,16);
    *plen=padded;
    printf("prologue: %zu range-check bytes + %zu NOP pad = %zu bytes (%zu instrs)\n",cut,padded-cut,padded,padded/16);
    return out;
  }
  return 0;
}

int main(){
 setbuf(stdout,0); cuInit(0);CUdevice d;cuDeviceGet(&d,0);CUcontext c;cuCtxCreate(&c,0,d);
 CUdeviceptr out,meta;cuMemAlloc(&out,256*4);cuMemAlloc(&meta,16);

 size_t plen; unsigned char*pro=build_prologue((unsigned long long)meta,&plen);
 if(!pro||plen%16){printf("bad prologue\n");return 1;}

 /* app cubin (opaque) */
 size_t z; unsigned char*cb=load_file("origk.cubin",&z);
 Elf64_Ehdr*eh=(void*)cb;Elf64_Shdr*sh=(void*)(cb+eh->e_shoff);const char*ss=(char*)(cb+sh[eh->e_shstrndx].sh_offset);
 int ti=-1; for(int i=0;i<eh->e_shnum;i++) if(!strcmp(ss+sh[i].sh_name,".text.orig")) ti=i;
 if(ti<0){printf("no .text.orig\n");return 1;}
 size_t toff=sh[ti].sh_offset;

 /* Build spliced cubin: insert plen bytes at start of .text.orig data */
 size_t nz=z+plen; unsigned char*nb=malloc(nz);
 memcpy(nb,cb,toff);                     /* everything before .text.orig */
 memcpy(nb+toff,pro,plen);               /* prologue */
 memcpy(nb+toff+plen,cb+toff,z-toff);    /* original code + rest */
 /* fix headers in the NEW buffer */
 Elf64_Ehdr*ne=(void*)nb;
 if(ne->e_shoff>toff) ne->e_shoff+=plen;
 Elf64_Shdr*nsh=(void*)(nb+ne->e_shoff);
 for(int i=0;i<ne->e_shnum;i++){
   if(i==ti) nsh[i].sh_size+=plen;
   else if(nsh[i].sh_offset>toff) nsh[i].sh_offset+=plen;
 }
 /* fix program headers: the LOAD segment containing .text.orig grows; segments
    (and e_phoff) after the insertion point shift. The driver loads via Phdrs. */
 { unsigned long ins=toff;
   if(ne->e_phoff>ins) ne->e_phoff+=plen;
   Elf64_Phdr*ph=(void*)(nb+ne->e_phoff);
   for(int i=0;i<ne->e_phnum;i++){
     if(ph[i].p_offset<=ins && ins < ph[i].p_offset+ph[i].p_filesz){ ph[i].p_filesz+=plen; ph[i].p_memsz+=plen; printf("ph%d grows: filesz=%#lx\n",i,(long)ph[i].p_filesz); }
     else if(ph[i].p_offset>ins){ ph[i].p_offset+=plen; }
   }
 }
 /* fix symtab: orig st_size += plen (entry stays at 0) */
 const char*nss=(char*)(nb+nsh[ne->e_shstrndx].sh_offset);
 for(int i=0;i<ne->e_shnum;i++) if(nsh[i].sh_type==SHT_SYMTAB){
   Elf64_Sym*sy=(void*)(nb+nsh[i].sh_offset);int cnt=nsh[i].sh_size/sizeof(Elf64_Sym);
   const char*st=(char*)(nb+nsh[nsh[i].sh_link].sh_offset);
   for(int j=0;j<cnt;j++) if(!strcmp(st+sy[j].st_name,"orig")){sy[j].st_size+=plen;printf("orig sym: value=%#lx size=%#lx\n",(long)sy[j].st_value,(long)sy[j].st_size);}
 }
 /* fix .nv.info.orig: EIATTR_EXIT_INSTR_OFFSETS entries += plen.
    format: entries are attrs: byte0=format(0x04=EIFMT_HVAL? ...), we scan for attr id 0x1c (EXIT_INSTR_OFFSETS). */
 for(int i=0;i<ne->e_shnum;i++) if(!strcmp(nss+nsh[i].sh_name,".nv.info.orig")){
   unsigned char*b=nb+nsh[i].sh_offset;size_t sz=nsh[i].sh_size,x=0;
   while(x+2<=sz){ unsigned char fmt=b[x],attr=b[x+1];
     if(fmt==0x04){ unsigned short vlen=b[x+2]|(b[x+3]<<8); /* EIFMT_SVAL */
       if(attr==0x1c){ for(unsigned k=0;k<vlen;k+=4){unsigned*e=(unsigned*)(b+x+4+k);printf("  EXIT off %#x -> %#x\n",*e,*e+(unsigned)plen);*e+=plen;} }
       x+=4+vlen;
     } else if(fmt==0x03){ x+=4; } /* EIFMT_HVAL (id+2-byte val, 4 total) */
     else if(fmt==0x01){ x+=2; }   /* EIFMT_NVAL */
     else x+=1;
   }
 }
 FILE*fo=fopen("spliced.cubin","wb");fwrite(nb,1,nz,fo);fclose(fo);

 CUmodule m; CUresult lr=cuModuleLoadData(&m,nb);
 if(lr){const char*e;cuGetErrorName(lr,&e);printf("load fail %s\n",e);return 1;}
 CUfunction f;cuModuleGetFunction(&f,m,"orig");

 /* launch grid 8, atom = blocks [2,6) */
 unsigned lohi[2]={2,6}; cuMemcpyHtoD(meta,lohi,8);
 cuMemsetD32(out,0xffffffff,8);
 void*args[]={&out};
 CUresult sr=cuLaunchKernel(f,8,1,1,1,1,1,0,0,args,0);
 CUresult syr=cuCtxSynchronize();
 unsigned h[8];cuMemcpyDtoH(h,out,32);
 printf("launch=%d sync=%d\nout:",sr,syr);
 for(int i=0;i<8;i++) printf(" [%d]=%d",i,(int)h[i]);
 printf("\n");
 int ok = syr==0 && sr==0;
 for(int i=0;i<8;i++){ unsigned exp = (i>=2&&i<6)?(unsigned)(i+100):0xffffffff; if(h[i]!=exp) ok=0; }
 printf(ok?">>>>>>>>>> ATOMIZER (prologue splice) WORKS! in-range ran, out-of-range skipped, clean\n":"xx mismatch\n");
 return 0;
}
