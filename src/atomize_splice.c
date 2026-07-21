/* atomize_splice.c — the LithOS Kernel Atomizer's core, done WITHOUT a control
 * transfer. Instead of a separate Prelude that must jump into the original
 * kernel (the unsolved CALL/BRX/BRA blocker), we PREPEND a range-check prologue
 * directly into each app kernel's own .text.<fn> section, so in-range blocks
 * FALL THROUGH into the original code. No CALL, no BRX, no BSSY convergence
 * barrier — the original's EXIT is its own clean top-level exit.
 *
 * The prologue (built by NVRTC, meta VA baked as a literal):
 *     b = global block index (from blockIdx / gridDim)
 *     if (b < meta->lo || b >= meta->hi) EXIT;      // predicated, no barrier
 *     <fall through into the original kernel>
 * At launch time the atomizer writes AtomMetadata{lo,hi} to the meta buffer and
 * relaunches the (unchanged) grid once per atom.
 *
 * This file provides:
 *   int  atomize_build_prologue(uint64_t meta_va, unsigned char**out, size_t*len)
 *   int  atomize_splice_cubin(const void*cubin, size_t sz, const unsigned char*pro,
 *                             size_t prolen, void**out, size_t*outsz)
 */
#include <cuda.h>
#include <nvrtc.h>
#include <elf.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

static const unsigned char ATOM_NOP[16]={0x18,0x79,0,0,0,0,0,0,0,0,0,0,0,0xc0,0x0f,0};

/* register count the range-check prologue needs (read from the probe cubin;
 * used to raise pathologically-small kernels' allocation before splicing). */
static int g_prologue_regs = 8;
int atomize_prologue_regs(void){ return g_prologue_regs; }

/* Build the range-check prologue for a given device SM arch and meta VA.
 * Returns malloc'd SASS bytes (padded to a 128-byte multiple) in *out. */
int atomize_build_prologue(int sm_major,int sm_minor,uint64_t meta_va,
                           unsigned char**out,size_t*len){
  char arch[32]; snprintf(arch,sizeof(arch),"--gpu-architecture=sm_%d%d",sm_major,sm_minor);
  char src[1024]; snprintf(src,sizeof(src),
   "struct M{unsigned lo,hi;};\n"
   "extern \"C\" __global__ void probe(unsigned* sink){\n"
   " unsigned long long b=(unsigned long long)blockIdx.z*gridDim.y*gridDim.x"
   "+(unsigned long long)blockIdx.y*gridDim.x+blockIdx.x;\n"
   " M* a=(M*)%lluULL;\n"
   " if(b<a->lo||b>=a->hi) return;\n"
   " asm volatile(\"mov.u32 %%0, 0xDEADBEEF;\":\"=r\"(sink[0])::\"memory\");\n"
   "}\n", (unsigned long long)meta_va);
  nvrtcProgram p; if(nvrtcCreateProgram(&p,src,"atom_prologue.cu",0,0,0))return -1;
  const char*o[]={arch};
  if(nvrtcCompileProgram(p,1,o)){
    size_t L;nvrtcGetProgramLogSize(p,&L);char*lg=malloc(L);nvrtcGetProgramLog(p,lg);
    fprintf(stderr,"[atomize] NVRTC prologue compile failed:\n%s\n",lg);free(lg);return -1;}
  size_t z;nvrtcGetCUBINSize(p,&z);unsigned char*cb=malloc(z);nvrtcGetCUBIN(p,(char*)cb);
  nvrtcDestroyProgram(&p);
  Elf64_Ehdr*eh=(void*)cb;Elf64_Shdr*sh=(void*)(cb+eh->e_shoff);
  const char*ss=(char*)(cb+sh[eh->e_shstrndx].sh_offset);
  /* record the prologue's register requirement from .nv.info REGCOUNT (0x2f) */
  for(int i=0;i<eh->e_shnum;i++) if(!strcmp(ss+sh[i].sh_name,".nv.info")){
    unsigned char*b=cb+sh[i].sh_offset;size_t sz=sh[i].sh_size,x=0;
    while(x+2<=sz){ unsigned char fmt=b[x],attr=b[x+1];
      if(fmt==0x00){x++;continue;}
      if(fmt==0x04){ unsigned short vl=b[x+2]|(b[x+3]<<8);
        if(attr==0x2f && vl>=8){ unsigned rc2=*(unsigned*)(b+x+8); if((int)rc2>g_prologue_regs) g_prologue_regs=(int)rc2; }
        x+=4+vl; } else if(fmt==0x03) x+=4; else if(fmt==0x02) x+=3; else if(fmt==0x01) x+=2; else x++;
    }
  }
  int rc=-1;
  for(int i=0;i<eh->e_shnum;i++) if(!strcmp(ss+sh[i].sh_name,".text.probe")){
    unsigned char*b=cb+sh[i].sh_offset;size_t sz=sh[i].sh_size,cut=sz;
    for(size_t x=0;x+16<=sz;x+=16) for(int k=0;k<13;k++)
      if(b[x+k]==0xef&&b[x+k+1]==0xbe&&b[x+k+2]==0xad&&b[x+k+3]==0xde){cut=x;goto done;}
    done:;
    if(cut==sz){fprintf(stderr,"[atomize] prologue marker not found\n");break;}
    size_t padded=(cut+127)&~(size_t)127;
    unsigned char*pr=malloc(padded);memcpy(pr,b,cut);
    for(size_t x=cut;x<padded;x+=16) memcpy(pr+x,ATOM_NOP,16);
    *out=pr;*len=padded;rc=0;break;
  }
  free(cb);return rc;
}

/* Splice `pro` (prolen bytes) at the FRONT of one kernel's .text section, fixing
 * section headers, program headers, the function symbol, and .nv.info.<fn>
 * instruction-offset attributes. Operates on a freshly-malloc'd copy. */
static unsigned char* splice_one(const unsigned char*cb,size_t z,const char*textsec,
                                 const char*fname,const unsigned char*pro,size_t prolen,size_t*nzout){
  Elf64_Ehdr*eh=(void*)cb;Elf64_Shdr*sh=(void*)(cb+eh->e_shoff);
  const char*ss=(char*)(cb+sh[eh->e_shstrndx].sh_offset);
  int ti=-1; for(int i=0;i<eh->e_shnum;i++) if(!strcmp(ss+sh[i].sh_name,textsec)) ti=i;
  if(ti<0) return 0;
  /* Prepending the prologue shifts this .text section down by prolen, so any
     relocation into it must move too: r_offset (the patch site) always; a
     self-referential RELA addend (an offset into the moved code, e.g. the return
     address for a device-function CALL) as well; and a local label's st_value.
     Locate this kernel's REL/RELA sections. */
  char relname[1160]; snprintf(relname,sizeof(relname),".rel.text.%s",fname);
  char relaname[1160]; snprintf(relaname,sizeof(relaname),".rela.text.%s",fname);
  int rel_i=-1, rela_i=-1;
  for(int i=0;i<eh->e_shnum;i++){ const char*nm=ss+sh[i].sh_name;
    if(!strcmp(nm,relname)) rel_i=i; else if(!strcmp(nm,relaname)) rela_i=i; }
  /* A REL relocation stores its addend implicitly inside the instruction bytes.
     If it is self-referential (its symbol is defined in THIS .text section) we'd
     have to rewrite the instruction to shift the encoded target — we can't do
     that safely, so skip the kernel (loaded/launched verbatim). Cross-section
     RELs (a CALL to another .text.<fn>, a global/constant load) only need
     r_offset bumped, which is handled below. */
  if(rel_i>=0){
    Elf64_Sym*sy=(void*)(cb+sh[sh[rel_i].sh_link].sh_offset);
    Elf64_Rel*r=(void*)(cb+sh[rel_i].sh_offset); int cnt=sh[rel_i].sh_size/sizeof(Elf64_Rel);
    for(int j=0;j<cnt;j++) if(sy[ELF64_R_SYM(r[j].r_info)].st_shndx==ti) return 0;
  }
  size_t toff=sh[ti].sh_offset;

  size_t nz=z+prolen; unsigned char*nb=malloc(nz);
  memcpy(nb,cb,toff);
  memcpy(nb+toff,pro,prolen);
  memcpy(nb+toff+prolen,cb+toff,z-toff);

  Elf64_Ehdr*ne=(void*)nb;
  if(ne->e_shoff>toff) ne->e_shoff+=prolen;
  Elf64_Shdr*nsh=(void*)(nb+ne->e_shoff);
  for(int i=0;i<ne->e_shnum;i++){
    if(i==ti) nsh[i].sh_size+=prolen;
    else if(nsh[i].sh_offset>toff) nsh[i].sh_offset+=prolen;
  }
  /* program headers: the LOAD segment containing toff grows; later segments shift */
  if(ne->e_phoff>toff) ne->e_phoff+=prolen;
  Elf64_Phdr*ph=(void*)(nb+ne->e_phoff);
  for(int i=0;i<ne->e_phnum;i++){
    if(ph[i].p_offset<=toff && toff<ph[i].p_offset+ph[i].p_filesz){ ph[i].p_filesz+=prolen; ph[i].p_memsz+=prolen; }
    else if(ph[i].p_offset>toff) ph[i].p_offset+=prolen;
  }
  const char*nss=(char*)(nb+nsh[ne->e_shstrndx].sh_offset);
  /* function symbol grows (entry stays at section start); capture its index */
  int fsym=-1;
  for(int i=0;i<ne->e_shnum;i++) if(nsh[i].sh_type==SHT_SYMTAB){
    Elf64_Sym*sy=(void*)(nb+nsh[i].sh_offset);int cnt=nsh[i].sh_size/sizeof(Elf64_Sym);
    const char*st=(char*)(nb+nsh[nsh[i].sh_link].sh_offset);
    for(int j=0;j<cnt;j++){
      if(!strcmp(st+sy[j].st_name,fname)){ sy[j].st_size+=prolen; fsym=j; }
      /* a local label defined inside the shifted section moves with the code; the
         function entry itself keeps st_value==0 (= the new prologue start) */
      else if((int)sy[j].st_shndx==ti && sy[j].st_value>0) sy[j].st_value+=prolen;
    }
  }
  /* raise this function's REGCOUNT (.nv.info attr 0x2f: [sym_idx][regcount]) to
     at least the prologue's requirement, so tiny kernels don't under-allocate. */
  for(int i=0;i<ne->e_shnum;i++) if(!strcmp(nss+nsh[i].sh_name,".nv.info")){
    unsigned char*b=nb+nsh[i].sh_offset;size_t sz=nsh[i].sh_size,x=0;
    while(x+2<=sz){ unsigned char fmt=b[x],attr=b[x+1];
      if(fmt==0x00){x++;continue;}
      if(fmt==0x04){ unsigned short vl=b[x+2]|(b[x+3]<<8);
        if(attr==0x2f && vl>=8){ unsigned*v=(unsigned*)(b+x+4);
          if((fsym<0 || v[0]==(unsigned)fsym) && (int)v[1]<g_prologue_regs) v[1]=g_prologue_regs; }
        x+=4+vl; } else if(fmt==0x03) x+=4; else if(fmt==0x02) x+=3; else if(fmt==0x01) x+=2; else x++;
    }
  }
  /* .nv.info.<fn>: bump instruction-offset attributes by prolen */
  char infoname[1152]; snprintf(infoname,sizeof(infoname),".nv.info.%s",fname);
  for(int i=0;i<ne->e_shnum;i++) if(!strcmp(nss+nsh[i].sh_name,infoname)){
    unsigned char*b=nb+nsh[i].sh_offset;size_t sz=nsh[i].sh_size,x=0;
    while(x+2<=sz){ unsigned char fmt=b[x],attr=b[x+1];
      if(fmt==0x00){x++;continue;}
      if(fmt==0x04){ unsigned short vlen=b[x+2]|(b[x+3]<<8); unsigned*v=(unsigned*)(b+x+4);
        if(attr==0x1c||attr==0x1d||attr==0x31){            /* pure instruction-offset lists */
          for(unsigned k=0;k<vlen/4;k++) v[k]+=(unsigned)prolen;
        } else if(attr==0x34){                              /* INDIRECT_BRANCH_TARGETS */
          /* [brx_off][flags][count][targets...] : bump brx_off and each target */
          unsigned n=vlen/4; if(n>=3){ v[0]+=(unsigned)prolen; unsigned cnt=v[2];
            for(unsigned k=0;k<cnt && 3+k<n;k++) v[3+k]+=(unsigned)prolen; }
        }
        x+=4+vlen;
      } else if(fmt==0x03){ x+=4; }
      else if(fmt==0x02){ x+=3; }
      else if(fmt==0x01){ x+=2; }
      else x++;
    }
  }
  /* relocation fixups for the shifted .text.<fn> (indices are stable across the
     splice — we add no sections). Every r_offset is a patch site inside the moved
     code. For a self-referential RELA whose symbol is the function base
     (st_value==0), the addend is an offset into that moved code, so bump it too;
     a RELA targeting a local label (st_value>0) is already handled by the label's
     st_value bump above, so its addend must stay put. */
  if(rel_i>=0){
    Elf64_Rel*r=(void*)(nb+nsh[rel_i].sh_offset); int cnt=nsh[rel_i].sh_size/sizeof(Elf64_Rel);
    for(int j=0;j<cnt;j++) r[j].r_offset+=prolen;
  }
  if(rela_i>=0){
    Elf64_Sym*sy=(void*)(nb+nsh[nsh[rela_i].sh_link].sh_offset);
    Elf64_Rela*r=(void*)(nb+nsh[rela_i].sh_offset); int cnt=nsh[rela_i].sh_size/sizeof(Elf64_Rela);
    for(int j=0;j<cnt;j++){ r[j].r_offset+=prolen;
      Elf64_Sym*s=&sy[ELF64_R_SYM(r[j].r_info)];
      if((int)s->st_shndx==ti && s->st_value==0) r[j].r_addend+=prolen;
    }
  }
  *nzout=nz; return nb;
}

/* Validate that `b` is a well-formed ELF cubin fully contained in `sz` bytes, so
 * the splicer never reads past the end of a small/unusual image (e.g. the tiny
 * stub cubins cuBLASLt loads, or a differently-laid-out cubin from a newer CUDA).
 * Returns 1 if safe to parse, 0 otherwise (caller then loads it verbatim). */
int atomize_elf_ok(const unsigned char* b, size_t sz){
  if(sz < sizeof(Elf64_Ehdr) || memcmp(b,ELFMAG,SELFMAG)) return 0;
  const Elf64_Ehdr* eh=(const void*)b;
  if(eh->e_shentsize!=sizeof(Elf64_Shdr) || eh->e_shnum==0 || eh->e_shoff==0) return 0;
  if(eh->e_shoff > sz || (size_t)eh->e_shnum*eh->e_shentsize > sz - eh->e_shoff) return 0;
  if(eh->e_phnum && (eh->e_phoff > sz || (size_t)eh->e_phnum*eh->e_phentsize > sz - eh->e_phoff)) return 0;
  if(eh->e_shstrndx >= eh->e_shnum) return 0;
  const Elf64_Shdr* sh=(const void*)(b+eh->e_shoff);
  size_t stro=sh[eh->e_shstrndx].sh_offset, strsz=sh[eh->e_shstrndx].sh_size;
  if(stro > sz || strsz > sz - stro) return 0;
  for(int i=0;i<eh->e_shnum;i++){
    if(sh[i].sh_name >= strsz) return 0;
    if(sh[i].sh_type!=SHT_NOBITS && (sh[i].sh_offset > sz || sh[i].sh_size > sz - sh[i].sh_offset)) return 0;
  }
  return 1;
}

/* Splice the prologue into every kernel ENTRY (.text.<fn> whose symbol carries
 * the st_other 0x10 entry flag) in the cubin; __device__ functions with their own
 * .text.<fn> are left alone (they are CALLed, not launched). .text relocations are
 * fixed up in splice_one (r_offset/RELA addend/label st_value); only the unfixable
 * cases (self-referential REL, malformed stubs) are skipped and left runnable
 * verbatim. Only the successfully-spliced names are returned in *names_out so the
 * caller gates exactly those at launch. Returns 0 if >=1 kernel was spliced,
 * -1 if none (caller then loads the image unmodified).
 *
 * Splicing shifts later sections, so we re-parse `cur` on every iteration. */
int atomize_splice_cubin(const void*cubin,size_t sz,const unsigned char*pro,size_t prolen,
                         void**out,size_t*outsz,char*** names_out,int* n_names_out){
  *names_out=0; *n_names_out=0;
  if(!atomize_elf_ok(cubin,sz)) return -1;   /* malformed/too-small: load verbatim */
  unsigned char*cur=malloc(sz); memcpy(cur,cubin,sz); size_t curz=sz;
  Elf64_Ehdr*eh=(void*)cur;Elf64_Shdr*sh=(void*)(cur+eh->e_shoff);
  const char*ss=(char*)(cur+sh[eh->e_shstrndx].sh_offset);

  /* Locate the symtab so we can tell __global__ kernel ENTRIES from __device__
     functions: both get their own .text.<fn> section, but a device function is
     CALLed, not launched — prepending the range-check prologue (which EXITs
     out-of-range threads) to one would terminate the thread instead of returning
     to the caller. ptxas marks kernel entries with st_other bit 0x10. */
  Elf64_Sym*syms=0; int nsyms=0; const char*symstr=0;
  for(int i=0;i<eh->e_shnum;i++) if(sh[i].sh_type==SHT_SYMTAB){
    syms=(void*)(cur+sh[i].sh_offset); nsyms=sh[i].sh_size/sizeof(Elf64_Sym);
    symstr=(char*)(cur+sh[sh[i].sh_link].sh_offset); break; }

  /* gather kernel names (heap; C++ mangled names can be hundreds of chars) */
  int cap=0; for(int i=0;i<eh->e_shnum;i++) if(!strncmp(ss+sh[i].sh_name,".text.",6)) cap++;
  char** all=calloc(cap>0?cap:1,sizeof(char*)); int na=0;
  for(int i=0;i<eh->e_shnum;i++){ const char*nm=ss+sh[i].sh_name;
    if(strncmp(nm,".text.",6)) continue;
    const char*fn=nm+6;
    if(syms){                                    /* require the kernel-entry flag */
      int entry=0;
      for(int j=0;j<nsyms;j++)
        if((int)syms[j].st_shndx==i && !strcmp(symstr+syms[j].st_name,fn)){
          entry=(syms[j].st_other & 0x10)!=0; break; }
      if(!entry) continue;                       /* device function: don't splice */
    }
    all[na++]=strdup(fn); }

  char** done=calloc(na>0?na:1,sizeof(char*)); int nd=0;
  for(int k=0;k<na;k++){
    size_t need=strlen(all[k])+8; char* tsec=malloc(need);
    snprintf(tsec,need,".text.%s",all[k]);
    size_t nz; unsigned char*nb=splice_one(cur,curz,tsec,all[k],pro,prolen,&nz);
    free(tsec);
    if(nb){ free(cur); cur=nb; curz=nz; done[nd++]=strdup(all[k]); }
    /* else: skip this kernel (unsupported); it stays runnable verbatim */
    free(all[k]);
  }
  free(all);

  if(nd==0){ free(cur); free(done); *names_out=0; *n_names_out=0; return -1; }
  *out=cur; *outsz=curz; *names_out=done; *n_names_out=nd;
  return 0;
}
