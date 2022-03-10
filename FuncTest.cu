#include "pch.h"
#include "Common.h"
#include "FuncTest.cuh"

__device__ float __d_add  (float LhsArg, float RhsArg)
{
   return LhsArg + RhsArg;
}

__device__ float __d_mult (float LhsArg, float RhsArg)
{
   return LhsArg * RhsArg;
}


__device__ DFunc_t  MultFunc   = __d_mult;


void  GetBinFunc (DFunc_t& FuncArg)
{
   DFunc_t  HFunc = nullptr;

   __CC(::cudaMemcpyFromSymbol(&HFunc, MultFunc, sizeof(DFunc_t)));

   FuncArg  = HFunc;
}