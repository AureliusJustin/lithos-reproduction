#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
int atomize_build_prologue(int,int,unsigned long long,unsigned char**,size_t*);
int atomize_splice_cubin(const void*,size_t,const unsigned char*,size_t,void**,size_t*);
static unsigned char* rd(const char*p,size_t*n){FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);*n=ftell(f);fseek(f,0,SEEK_SET);unsigned char*b=malloc(*n);fread(b,1,*n,f);fclose(f);return b;}
#define CK(x) do{CUresult r=(x); if(r){const char*e;cuGetErrorName(r,&e);printf("ERR %s @ %s\n",e,#x);return 1;}}while(0)
int main(){
 setbuf(stdout,0); CK(cuInit(0)); CUdevice d;CK(cuDeviceGet(&d,0));CUcontext c;CK(cuCtxCreate(&c,0,d));
 int maj,min; cuDeviceGetAttribute(&maj,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,d);
 cuDeviceGetAttribute(&min,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,d);

 int N=256, threads=64, blocks=(N+threads-1)/threads; // 4 blocks; grid.x=blocks... use 1024 elems
 N=1024; blocks=1024/threads; // 16 blocks
 CUdeviceptr out,in,meta; CK(cuMemAlloc(&out,N*4)); CK(cuMemAlloc(&in,1024*4)); CK(cuMemAlloc(&meta,16));
 float* hin=malloc(1024*4); for(int i=0;i<1024;i++) hin[i]=(float)((i%7)-3)*0.5f; CK(cuMemcpyHtoD(in,hin,1024*4));
 int nn=5, mode=1;

 /* reference: unmodified kernel, full grid */
 size_t z0; unsigned char*ref=rd("complex.cubin",&z0);
 CUmodule mr; CK(cuModuleLoadData(&mr,ref)); CUfunction fr; CK(cuModuleGetFunction(&fr,mr,"ck"));
 CK(cuMemsetD32(out,0,N)); void*ra[]={&out,&in,&nn,&mode};
 CK(cuLaunchKernel(fr,blocks,1,1,threads,1,1,0,0,ra,0)); CK(cuCtxSynchronize());
 float* href=malloc(N*4); CK(cuMemcpyDtoH(href,out,N*4));

 /* atomize: splice prologue into complex.cubin, load modified */
 unsigned char*pro; size_t prolen;
 if(atomize_build_prologue(maj,min,(unsigned long long)meta,&pro,&prolen)){printf("prologue fail\n");return 1;}
 printf("prologue %zu bytes (sm_%d%d)\n",prolen,maj,min);
 size_t z1; unsigned char*app=rd("complex.cubin",&z1);
 void*spl; size_t splz;
 if(atomize_splice_cubin(app,z1,pro,prolen,&spl,&splz)){printf("splice fail\n");return 1;}
 CUmodule ma; CUresult lr=cuModuleLoadData(&ma,spl);
 if(lr){const char*e;cuGetErrorName(lr,&e);printf("atomized load fail: %s\n",e);return 1;}
 CUfunction fa; CK(cuModuleGetFunction(&fa,ma,"ck"));

 /* run atomized: split 16 blocks into atoms of 4 blocks each, full grid each launch */
 CK(cuMemsetD32(out,0,N)); void*aa[]={&out,&in,&nn,&mode};
 int atom=4, faults=0;
 for(int lo=0; lo<blocks; lo+=atom){ int hi=lo+atom<blocks?lo+atom:blocks;
   unsigned lohi[2]={lo,hi}; CK(cuMemcpyHtoD(meta,lohi,8));
   CUresult sr=cuLaunchKernel(fa,blocks,1,1,threads,1,1,0,0,aa,0); CUresult syr=cuCtxSynchronize();
   if(sr||syr){const char*e;cuGetErrorName(syr?syr:sr,&e);printf("atom[%d,%d) FAULT %s\n",lo,hi,e);faults++;}
 }
 float* hat=malloc(N*4); CK(cuMemcpyDtoH(hat,out,N*4));
 int mism=0; for(int i=0;i<N;i++) if(fabsf(href[i]-hat[i])>1e-3f){ if(mism<5)printf(" mism[%d] ref=%f atom=%f\n",i,href[i],hat[i]); mism++; }
 printf("faults=%d mismatches=%d/%d  %s\n",faults,mism,N,(!faults&&!mism)?">>>>>>>>>> ATOMIZED == REFERENCE (switch+params+loop, transparent) <<<<<<<<<<":"xx FAIL");
 return 0;
}
