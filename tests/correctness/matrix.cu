// Comprehensive correctness matrix: a variety of kernel patterns, each verified
// against a CPU reference. Runs identically with/without LithOS; the atomic
// kernels (histogram/reduce) also catch any double- or missing-block execution.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(2);}}while(0)

__global__ void vecadd(const float*a,const float*b,float*c,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i];
}
__global__ void saxpy(float alpha,const float*x,float*y,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=alpha*x[i]+y[i];
}
// tiled matmul: shared memory + __syncthreads, 2D grid (blockIdx.x/.y)
#define TS 16
__global__ void matmul(const float*A,const float*B,float*C,int N){
  __shared__ float As[TS][TS], Bs[TS][TS];
  int r=blockIdx.y*TS+threadIdx.y, c=blockIdx.x*TS+threadIdx.x;
  float acc=0;
  for(int t=0;t<N;t+=TS){
    As[threadIdx.y][threadIdx.x]=A[r*N+(t+threadIdx.x)];
    Bs[threadIdx.y][threadIdx.x]=B[(t+threadIdx.y)*N+c];
    __syncthreads();
    for(int k=0;k<TS;k++) acc+=As[threadIdx.y][k]*Bs[k][threadIdx.x];
    __syncthreads();
  }
  C[r*N+c]=acc;
}
// atomic histogram: every element does exactly one atomicAdd -> double/missing block detectable
__global__ void histo(const int*data,int n,int*bins,int nbins){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) atomicAdd(&bins[data[i]%nbins],1);
}
// atomic reduce to a single accumulator
__global__ void reduce_sum(const int*data,int n,unsigned long long*acc){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) atomicAdd(acc,(unsigned long long)data[i]);
}
// 3D grid: exercises blockIdx.y and blockIdx.z in the prologue's block-index calc
__global__ void fill3d(int*out,int gx,int gy){
  int b=blockIdx.z*gy*gx + blockIdx.y*gx + blockIdx.x;
  if(threadIdx.x==0) out[b]=b*7+3;
}
// transpose, 2D grid
__global__ void transpose(const float*in,float*out,int N){
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x<N&&y<N) out[x*N+y]=in[y*N+x];
}

static int fails=0;
static void report(const char*name,int ok){ printf("  %-12s %s\n",name,ok?"PASS":"FAIL"); if(!ok) fails++; }

int main(){
  // ---- vecadd ----
  {
    int n=1<<20; size_t sz=n*sizeof(float);
    float*a=(float*)malloc(sz),*b=(float*)malloc(sz),*c=(float*)malloc(sz);
    for(int i=0;i<n;i++){a[i]=i*0.5f;b[i]=i*0.25f;}
    float*da,*db,*dc; CK(cudaMalloc(&da,sz));CK(cudaMalloc(&db,sz));CK(cudaMalloc(&dc,sz));
    CK(cudaMemcpy(da,a,sz,cudaMemcpyHostToDevice));CK(cudaMemcpy(db,b,sz,cudaMemcpyHostToDevice));
    vecadd<<<(n+255)/256,256>>>(da,db,dc,n); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(c,dc,sz,cudaMemcpyDeviceToHost));
    int ok=1; for(int i=0;i<n;i++) if(fabsf(c[i]-(a[i]+b[i]))>1e-3f){ok=0;break;}
    report("vecadd",ok); cudaFree(da);cudaFree(db);cudaFree(dc);free(a);free(b);free(c);
  }
  // ---- saxpy ----
  {
    int n=1<<20; size_t sz=n*sizeof(float);
    float*x=(float*)malloc(sz),*y=(float*)malloc(sz),*y0=(float*)malloc(sz);
    for(int i=0;i<n;i++){x[i]=i*0.01f;y[i]=y0[i]=100.0f-i*0.001f;}
    float*dx,*dy; CK(cudaMalloc(&dx,sz));CK(cudaMalloc(&dy,sz));
    CK(cudaMemcpy(dx,x,sz,cudaMemcpyHostToDevice));CK(cudaMemcpy(dy,y,sz,cudaMemcpyHostToDevice));
    saxpy<<<(n+255)/256,256>>>(2.5f,dx,dy,n); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(y,dy,sz,cudaMemcpyDeviceToHost));
    int ok=1; for(int i=0;i<n;i++) if(fabsf(y[i]-(2.5f*x[i]+y0[i]))>1e-2f){ok=0;break;}
    report("saxpy",ok); cudaFree(dx);cudaFree(dy);free(x);free(y);free(y0);
  }
  // ---- matmul (shared mem) ----
  {
    int N=256; size_t sz=N*N*sizeof(float);
    float*A=(float*)malloc(sz),*B=(float*)malloc(sz),*C=(float*)malloc(sz);
    for(int i=0;i<N*N;i++){A[i]=(i%7)*0.1f;B[i]=(i%5)*0.2f;}
    float*dA,*dB,*dC;CK(cudaMalloc(&dA,sz));CK(cudaMalloc(&dB,sz));CK(cudaMalloc(&dC,sz));
    CK(cudaMemcpy(dA,A,sz,cudaMemcpyHostToDevice));CK(cudaMemcpy(dB,B,sz,cudaMemcpyHostToDevice));
    dim3 blk(TS,TS),grd(N/TS,N/TS); matmul<<<grd,blk>>>(dA,dB,dC,N); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(C,dC,sz,cudaMemcpyDeviceToHost));
    // spot-check a few entries against CPU
    int ok=1; for(int t=0;t<20&&ok;t++){int r=(t*13)%N,c=(t*29)%N; double acc=0; for(int k=0;k<N;k++) acc+=(double)A[r*N+k]*B[k*N+c];
      if(fabs(C[r*N+c]-acc)>1e-1){ok=0;}}
    report("matmul",ok); cudaFree(dA);cudaFree(dB);cudaFree(dC);free(A);free(B);free(C);
  }
  // ---- histogram (atomic; exactly-once guard) ----
  {
    int n=1<<20,nb=256; int*data=(int*)malloc(n*sizeof(int));
    long ref[256]={0}; for(int i=0;i<n;i++){data[i]=(i*2654435761u)%1000; ref[data[i]%nb]++;}
    int*dd,*db;CK(cudaMalloc(&dd,n*sizeof(int)));CK(cudaMalloc(&db,nb*sizeof(int)));
    CK(cudaMemcpy(dd,data,n*sizeof(int),cudaMemcpyHostToDevice));CK(cudaMemset(db,0,nb*sizeof(int)));
    histo<<<(n+255)/256,256>>>(dd,n,db,nb); CK(cudaDeviceSynchronize());
    int*hb=(int*)malloc(nb*sizeof(int)); CK(cudaMemcpy(hb,db,nb*sizeof(int),cudaMemcpyDeviceToHost));
    int ok=1; for(int i=0;i<nb;i++) if(hb[i]!=ref[i]){ok=0;break;}
    report("histogram",ok); cudaFree(dd);cudaFree(db);free(data);free(hb);
  }
  // ---- reduce (atomic; exactly-once guard) ----
  {
    int n=1<<20; int*data=(int*)malloc(n*sizeof(int)); unsigned long long ref=0;
    for(int i=0;i<n;i++){data[i]=(i%17)-8; ref+=(unsigned long long)data[i];}
    int*dd; unsigned long long*da; CK(cudaMalloc(&dd,n*sizeof(int)));CK(cudaMalloc(&da,8));
    CK(cudaMemcpy(dd,data,n*sizeof(int),cudaMemcpyHostToDevice));CK(cudaMemset(da,0,8));
    reduce_sum<<<(n+255)/256,256>>>(dd,n,da); CK(cudaDeviceSynchronize());
    unsigned long long h; CK(cudaMemcpy(&h,da,8,cudaMemcpyDeviceToHost));
    report("reduce",h==ref); cudaFree(dd);cudaFree(da);free(data);
  }
  // ---- 3D grid ----
  {
    int gx=8,gy=6,gz=4,G=gx*gy*gz; int*out=(int*)malloc(G*sizeof(int));
    int*d;CK(cudaMalloc(&d,G*sizeof(int)));CK(cudaMemset(d,0,G*sizeof(int)));
    dim3 grd(gx,gy,gz); fill3d<<<grd,32>>>(d,gx,gy); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out,d,G*sizeof(int),cudaMemcpyDeviceToHost));
    int ok=1; for(int b=0;b<G;b++) if(out[b]!=b*7+3){ok=0;break;}
    report("grid3d",ok); cudaFree(d);free(out);
  }
  // ---- transpose ----
  {
    int N=512; size_t sz=N*N*sizeof(float);
    float*in=(float*)malloc(sz),*out=(float*)malloc(sz);
    for(int i=0;i<N*N;i++) in[i]=i*0.001f;
    float*di,*doo;CK(cudaMalloc(&di,sz));CK(cudaMalloc(&doo,sz));
    CK(cudaMemcpy(di,in,sz,cudaMemcpyHostToDevice));
    dim3 blk(16,16),grd(N/16,N/16); transpose<<<grd,blk>>>(di,doo,N); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out,doo,sz,cudaMemcpyDeviceToHost));
    int ok=1; for(int y=0;y<N&&ok;y++)for(int x=0;x<N;x++) if(out[x*N+y]!=in[y*N+x]){ok=0;break;}
    report("transpose",ok); cudaFree(di);cudaFree(doo);free(in);free(out);
  }
  printf("OVERALL: %s (%d failures)\n", fails?"FAIL":"PASS", fails);
  return fails?1:0;
}
