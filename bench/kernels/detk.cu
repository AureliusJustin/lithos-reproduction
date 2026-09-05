extern "C" __global__ void detk(int* out,int kid){int b=blockIdx.x;if(threadIdx.x==0)out[b]=kid*100000+b*7+3;}
