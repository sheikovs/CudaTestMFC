#include "pch.h"
#include "Common.h"
#include "Gpu.h"
#include "CudaHelpers.h"
#include "Tests.h"
#include <fstream>

namespace BasicRuns
{
   __global__ void k_if (
      unsigned int*  TArg
   ,  unsigned int*  BArg
   ,  unsigned int*  CArg
   ,  const size_t   SizeArg
   )
   { 
      size_t tid = blockIdx.x * blockDim.x + threadIdx.x; 

      if (tid < SizeArg) 
      { 
         TArg[tid] = threadIdx.x; 
         BArg[tid] = blockIdx.x; 
         CArg[tid] += 1; 
      } 
   }
   __global__ void k_while (
      unsigned int*  TArg
   ,  unsigned int*  BArg
   ,  unsigned int*  CArg
   ,  const size_t   SizeArg
   )
   { 
      size_t tid = blockIdx.x * blockDim.x + threadIdx.x; 

      while (tid < SizeArg) 
      { 
         TArg[tid] = threadIdx.x; 
         BArg[tid] = blockIdx.x; 
         CArg[tid] += 1;

         tid += blockDim.x * gridDim.x;
      } 
   }


   ///////////////////////////////////////////
   //
   // struct SaxpyRun::_Test
   //
   ///////////////////////////////////////////

   struct _Test
   {
      using value_type  = unsigned int;
      using Mem_t       = THAAllocator <value_type>;
      using ItemsVc_t   = _ITestReport::ItemsVc_t;

      int64_t     _tm {};
      size_t      _size {};
      _ITest&     _test;
      float       _mult = 1.25f;

      float       _result {};

      _Test (_ITest& TestImplArg, size_t SizeArg)
      :  _size (SizeArg)
      ,  _test (TestImplArg)
      {
      }

      void  RunIf ()
      {
         Mem_t   t_mem (_size);
         Mem_t   b_mem (_size);
         Mem_t   c_mem (_size);

         ::memset (t_mem.get (), 0, t_mem.size_of ());
         ::memset (b_mem.get (), 0, b_mem.size_of ());
         ::memset (c_mem.get (), 0, c_mem.size_of ());

         dim3     Grid  (_test.GetGridSize  (), 1, 1);
         dim3     Block (_test.GetBlockSize (), 1, 1);
         k_if <<<Grid, Block>>> (t_mem.get_dptr (), b_mem.get_dptr (), c_mem.get_dptr (), _size);

         __CC (::cudaDeviceSynchronize ());

         __Report ("k_if", t_mem.get (), b_mem.get (), c_mem.get ());
       }

      void  RunWhile ()
      {
         Mem_t   t_mem (_size);
         Mem_t   b_mem (_size);
         Mem_t   c_mem (_size);

         ::memset (t_mem.get (), 0, t_mem.size_of ());
         ::memset (b_mem.get (), 0, b_mem.size_of ());
         ::memset (c_mem.get (), 0, c_mem.size_of ());

         dim3     Grid  (_test.GetGridSize  (), 1, 1);
         dim3     Block (_test.GetBlockSize (), 1, 1);
         k_while <<<Grid, Block>>> (t_mem.get_dptr (), b_mem.get_dptr (), c_mem.get_dptr (), _size);

         __CC (::cudaDeviceSynchronize ());

         __Report ("k_while", t_mem.get (), b_mem.get (), c_mem.get ());
      }

      void  __Report (
         LPCTSTR           NameArg
      ,  const value_type* TArg
      ,  const value_type* BArg
      ,  const value_type* CArg
      )
      {
         auto const  FilePath (::_F("C:\\Temp\\%s_%i_%i.txt", NameArg, _test.GetGridSize(), _test.GetBlockSize ()));

         std::ofstream  File;

         File.exceptions (std::ofstream::failbit | std::ifstream::badbit);
         File.open ((LPCTSTR)FilePath, std::ios::out);

         for (size_t i = 0; i < _size; ++i)
         {
            File << '[' << i << "]\t" << TArg [i] << '\t' << BArg [i] << '\t' << CArg [i] << std::endl;
         }
      }
   };

   void  Start (_ITest& TestImplArg)
   {
      auto const LOnError  = [&TestImplArg] (CString const& MsgArg)
      {
         TestImplArg.AppendLog ("Error: %s", MsgArg);
      };

      try
      {
         _Test Test (TestImplArg, TestImplArg.GetCalcSize ());

         Test.RunIf    ();
         Test.RunWhile ();
      }
      __CATCH(LOnError);

   }

}  // namespace BasicRuns


BasicsTest::BasicsTest ()
{
}

void  BasicsTest::OnInit ()
{
   ColumnsVc_t Cols {
         {"Test Name", LVCFMT_LEFT,  200}
      ,  {"Result",    LVCFMT_RIGHT, 200}
      ,  {"Time (ms)", LVCFMT_RIGHT, 200}
   };

   MakeHeader (Cols);
}

void  BasicsTest::OnRun  ()
{
   BasicRuns::Start (*this);
}

cudaDeviceProp BasicsTest::GetCudaProperties ()
{
   return Gpu::GetProperties ();
}

void  BasicsTest_entry ()
{
   Gpu         g (true);
   BasicsTest  Test;
   CTestDialog dlg (&Test);

   dlg.DoModal ();
}