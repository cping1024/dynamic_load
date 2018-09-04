#include <iostream>

struct SharedMemory
{
    __device__ inline operator       float *()
    {
        extern __shared__ int __smem[];
        return (float *)__smem;
    }

    __device__ inline operator const float *() const
    {
        extern __shared__ int __smem[];
        return (float *)__smem;
    }
};

__global__ void
reduce3(float *g_idata, float *g_odata, unsigned int n)
{
    float *sdata = SharedMemory();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    float mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockDim.x < n)
        mySum += g_idata[i+blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {   
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }
        
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}


int main(int argc, char* argv[]) {
	

	
	return 0;
}
