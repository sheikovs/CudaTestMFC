#ifndef __FUNCTEST__
#define __FUNCTEST__
#include "cuda_runtime.h"

extern __device__ float __d_add  (float LhsArg, float RhsArg);
extern __device__ float __d_mult (float LhsArg, float RhsArg);

using DFunc_t = float (*)(float, float);

void  GetBinFunc (DFunc_t& FuncArg);

#endif // !__FUNCTEST__
