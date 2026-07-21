extern "C" __global__ void ck(float* out, const float* in, int n, int mode){
  int b = blockIdx.x, t = threadIdx.x, i = b*blockDim.x + t;
  float acc = 0;
  for(int k=0;k<n;k++) acc += in[(i+k)%1024] * (k+1);   // loop, cachemod loads
  switch(mode){ case 0: acc*=2; break; case 1: acc+=1; break;   // switch -> branch targets
    case 2: acc=__sinf(acc); break; default: acc=-acc; }
  if(i<1024) out[i]=acc;
}
