#define TS 16
extern "C" __global__ void mm(const float*A,const float*B,float*C,int N){
  __shared__ float As[TS][TS],Bs[TS][TS];
  int r=blockIdx.y*TS+threadIdx.y,c=blockIdx.x*TS+threadIdx.x;float acc=0.f;
  for(int t=0;t<N;t+=TS){As[threadIdx.y][threadIdx.x]=A[r*N+(t+threadIdx.x)];Bs[threadIdx.y][threadIdx.x]=B[(t+threadIdx.y)*N+c];__syncthreads();
   #pragma unroll
   for(int k=0;k<TS;k++)acc+=As[threadIdx.y][k]*Bs[k][threadIdx.x];__syncthreads();}
  C[r*N+c]=acc;}
