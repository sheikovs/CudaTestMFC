extern "C" __global__ void saxpy_if (float a, float *x, float *y, float *out, size_t n)
{ 
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x; 
	if (tid < n) 
	{ 
		out[tid] = a * x[tid] + y[tid]; 
	} 
}

extern "C" __global__ void saxpy_while (float a, float *x, float *y, float *out, size_t n)
{ 
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x; 
	while (tid < n) 
	{ 
		out[tid] = a * x[tid] + y[tid]; 
		tid += blockDim.x * gridDim.x;
	} 
}

extern "C" __global__ void reduceNeighbored (float *InDataArg, float* OutDataArg, unsigned int SizeArg)
{
   // set thread ID
   unsigned int tid = threadIdx.x;
   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

   // convert global data pointer to the local pointer of this block
   float*   BlockData   = InDataArg + blockIdx.x * blockDim.x;

   // boundary check
   if (idx >= SizeArg) return;

   // in-place reduction in global memory
   for (int Stride = 1; Stride < blockDim.x; Stride *= 2)
   {
      if ((tid % (2 * Stride)) == 0)
      {
         BlockData[tid] += BlockData[tid + Stride];
      }

      // synchronize within threadblock
      __syncthreads();
   }

   // write result for this block to global mem
   if (tid == 0) OutDataArg [blockIdx.x] = BlockData[0];
}

extern "C" __global__ void reduceNeighboredLess (float *InDataArg, float* OutDataArg, unsigned int SizeArg)
{
   // set thread ID
   unsigned int tid = threadIdx.x;
   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

   // convert global data pointer to the local pointer of this block
   float*   BlockData   = InDataArg + blockIdx.x * blockDim.x;

   // boundary check
   if (idx >= SizeArg) return;

   // in-place reduction in global memory
   for (int Stride = 1; Stride < blockDim.x; Stride *= 2)
   {
      // convert tid into local array index
	  int index = 2 * Stride * tid;
	  if (index < blockDim.x) 
	  {
		BlockData[index] += BlockData[index + Stride];
	  }

      // synchronize within threadblock
      __syncthreads();
   }

   // write result for this block to global mem
   if (tid == 0) OutDataArg [blockIdx.x] = BlockData[0];
}

extern "C" __global__ void reduceInterleaved (float *g_idata, float *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    float *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}