#include "pch.h"
#include "NvRtcHelpers.h"
#include "MemHelper.cuh"
#include "Tests.h"
#include "Timer.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>

namespace SaxpyRun
{
   ///////////////////////////////////////////
   //
   // struct SaxpyRun::_Test
   //
   ///////////////////////////////////////////

   struct _Test
   {
      using HostMem_t   = THostMem     <float>;
      using DMem_t      = TCuDeviceMem <float>;
      using PMem_t      = TPinnedMem   <float>;

      using Device_t    = NVRTCH::Device;
      using Program_t   = NVRTCH::Program;
      using Kernel_t    = NVRTCH::Kernel;
      using Stream_t    = NVRTCH::Stream;

      using ItemsVc_t   = _ITestReport::ItemsVc_t;

      static constexpr int  BLOCK_SIZE  = 1024;
      static constexpr int  GRID_SIZE   = 32;

      int64_t     _tm   {};
      size_t      _size {};
      _ITest&     _test;
      Device_t&   _device;
      float       _mult = 1.25f;

      float       _result {};

      HostMem_t   _x_h_mem;
      HostMem_t   _y_h_mem;

      _Test (_ITest& TestArg, Device_t& DeviceArg, size_t SizeArg)
      :  _size    (SizeArg)
      ,  _test    (TestArg)
      ,  _device  (DeviceArg)
      ,  _x_h_mem (_size)
      ,  _y_h_mem (_size)
      {
         __Init ();
      }

      void  Serial ()
      {
         Timer       Tm;
         HostMem_t   XMem (_x_h_mem);
         HostMem_t   YMem (_y_h_mem);
         HostMem_t   Out  (_size);

         for (size_t i = 0; i < _size; ++i)
         {
            Out [i]   = XMem [i] * _mult + YMem [i];
         }

         _result  = std::accumulate (Out.begin (), Out.end (), float (), std::plus <float> ());

         _tm      = Tm.get ();

         __Report ("Serial", true);
      }

      void  GPURun (CString const& FilePathArg, CString const& FuncArg)
      {
         _test.AppendLog ("Source [%s] Function [%s]", FilePathArg, FuncArg);

         auto  LOnError = [this](CString const& MsgArg)
         {
            _test.AppendLog ("GPURun Error:\r\n %s", MsgArg);
         };

         try
         {
            __GPURun (FilePathArg, FuncArg);
         }
         __CATCH (LOnError);
      }

      void  __GPURun (CString const& FilePathArg, CString const& FuncArg)
      {
         Program_t   Program (_device, FilePathArg);

         Program.AddOption ("--extra-device-vectorization");

         if (Program.Compile ("saxpy"))
         {
            if (auto const& Log   = Program.GetLog (); !Log.IsEmpty ())
            {
               _test.AppendLog ("Compile log:\r\n%s\r\n", Log);
            }

            _device.Load (Program);

            Kernel_t KSaxpy  (_device, FuncArg);
            Kernel_t KReduce (_device, "reduceNeighbored");

            __GPURunCopy   (KSaxpy, KReduce);
            __GPURunPinned (KSaxpy, KReduce);

            if (_size >= 100000)
            {
               __GPURunStreams (KSaxpy, KReduce);
            }
         }
         else
         {
            auto const& Log   = Program.GetLog ();
            _test.AppendLog ("Compile log:\r\n%s\r\n", Log);
         }
      }


      void  __GPURunCopy (Kernel_t& KSAXPYArg, Kernel_t& KReduceArg)
      {
         Timer    Tm;

         dim3     Grid  (_test.GetGridSize  (), 1, 1);
         dim3     Block (_test.GetBlockSize (), 1, 1);

         DMem_t   XDev (_x_h_mem);
         DMem_t   YDev (_y_h_mem);
         DMem_t   DOut (_size);
         size_t   Offset (0);

         void*    SAXPYArgs[] = { &_mult, &XDev._ptr, &YDev._ptr, &DOut._ptr, &_size, &Offset };

         KSAXPYArg.Execute (Grid, Block, SAXPYArgs);

         Grid.x      = _test.GetMaxBlocks (_size, Block.x);

         DMem_t      DPartSum (Grid.x);

         void*       ReduceArgs[] = { &DOut._ptr, &DPartSum._ptr, &_size };

         Device_t::Synchronize ();

         KReduceArg.Execute (Grid, Block, ReduceArgs);

         Device_t::Synchronize ();

         HostMem_t   HPartSum (DPartSum);

         _result     = std::accumulate (HPartSum.begin (), HPartSum.end (), float (), std::plus <float> ());

         _tm   = Tm.get ();

         __Report ("CUDA Copy");
      }

      void  __GPURunPinned (Kernel_t& KSAXPYArg, Kernel_t& KReduceArg)
      {
         Timer    Tm;

         dim3     Grid  (_test.GetGridSize  (), 1, 1);
         dim3     Block (_test.GetBlockSize (), 1, 1);

         PMem_t   XDev (_x_h_mem);
         PMem_t   YDev (_y_h_mem);
         PMem_t   DOut (_size);
         size_t   Offset (0);

         void*    SAXPYArgs[] = { &_mult, &XDev._ptr, &YDev._ptr, &DOut._ptr, &_size, &Offset };

         KSAXPYArg.Execute (Grid, Block, SAXPYArgs);

         Grid.x      = _test.GetMaxBlocks (_size, Block.x);

         PMem_t   DPartSum (Grid.x);

         void*    ReduceArgs[] = { &DOut._ptr, &DPartSum._ptr, &_size };

         Device_t::Synchronize ();

         KReduceArg.Execute (Grid, Block, ReduceArgs);

         Device_t::Synchronize ();

         _result  = std::accumulate (DPartSum.begin (), DPartSum.end (), float (), std::plus <float> ());

         _tm   = Tm.get ();

         __Report ("CUDA Pinned");
      }

      void  __GPURunStreams (Kernel_t& KSAXPYArg, Kernel_t& KReduceArg)
      {
         using StreamPx_t = std::unique_ptr <Stream_t []>;
         constexpr int STREAMS   = 2;

         Timer       Tm;

         auto        Size_2   (_size / 2);
         auto        Size_1   (_size - Size_2);

         auto        GridSize (_test.GetGridSize  ());
         dim3        Block    (_test.GetBlockSize (), 1, 1);
         dim3        Grid_2   (_test.GetMaxBlocks (Size_2, Block.x), 1, 1);
         dim3        Grid_1   (_test.GetMaxBlocks (Size_1, Block.x), 1, 1);

         DMem_t      XDev (_x_h_mem);
         DMem_t      YDev (_y_h_mem);
         DMem_t      DOut (_size);
         {
            StreamPx_t  Streams (new Stream_t [STREAMS]);
            Streams [0].Create ();
            Streams [1].Create ();
            size_t   Offset (0);

            void*    SAXPY_1[] = { &_mult, &XDev._ptr, &YDev._ptr, &DOut._ptr, &Size_1, &Offset };
            void*    SAXPY_2[] = { &_mult, &XDev._ptr, &YDev._ptr, &DOut._ptr, &Size_2, &Size_1 };

            KSAXPYArg.Execute (Grid_1, Block, SAXPY_1, Streams [0]);
            KSAXPYArg.Execute (Grid_2, Block, SAXPY_2, Streams [1]);
         }

         Grid_1.x    = _test.GetMaxBlocks (_size, Block.x);

         Device_t::Synchronize ();

         DMem_t   DPartSum (Grid_1.x);

         void*    ReduceArgs[] = { &DOut._ptr, &DPartSum._ptr, &_size };

         KReduceArg.Execute (Grid_1, Block, ReduceArgs);

         Device_t::Synchronize ();

         HostMem_t   HPartSum (DPartSum);

         _result  = std::accumulate (HPartSum.begin (), HPartSum.end (), float (), std::plus <float> ());

         _tm   = Tm.get ();

         __Report ("CUDA Streams");

      }

      void  __Init ()
      {
         float p (0.125f);
         int   s (8);

         auto LGen   = [&p, &s]()
         {
            float const d  = static_cast <float> (::rand () % 11);
            float const f  = static_cast <float> (((::rand () % s) + 1.0f) * p);
            return (d + f) * (::rand () % 2 ? 1.0f : -1.0f);
         };

         std::generate (_x_h_mem.begin (), _x_h_mem.end (), LGen);

         p  = 0.25;
         s  = 4;

         std::generate (_y_h_mem.begin (), _y_h_mem.end (), LGen);    
      }

      void  __Report (LPCTSTR NameArg, bool IsSerialArg = false)
      {
         ItemsVc_t   Items;

         Items.emplace_back (NameArg);
         Items.emplace_back (::TGetCString (_size));
         if (IsSerialArg)
         {
            Items.emplace_back ("");
         }
         else
         {
            Items.emplace_back (::_F("%s x %s", ::TGetCString (_test.GetGridSize ()), ::TGetCString (_test.GetBlockSize ())));
         }
         Items.emplace_back (::TGetCString (_result));
         Items.emplace_back (::TGetCString (_tm));
         _test.AddRow (Items);
      }
   };

   void  Start (_ITest& TestArg, NVRTCH::Device& DeviceArg)
   {
      CWaitCursor wait;
      LPCTSTR     FP    = "C:\\Temp\\saxpy.cu";

      _Test Test (TestArg, DeviceArg, TestArg.GetCalcSize ());

      Test.Serial ();
      //Test.GPURun (FP, "saxpy_if");
      Test.GPURun (FP, "saxpy_offset");
   }

}  // namespace SaxpyRun

SAXPYTest::SAXPYTest ()
:  _device (0)
{
}

void  SAXPYTest::OnInit ()
{
   using Columns_t   = CTestDialog::ColumnsVc_t;

   Columns_t Cols {
         {"Test Name",     LVCFMT_LEFT,  150}
      ,  {"Size",          LVCFMT_RIGHT, 150}
      ,  {"Grid x Block",  LVCFMT_RIGHT, 150}
      ,  {"Result",        LVCFMT_RIGHT, 150}
      ,  {"Time (ms)",     LVCFMT_RIGHT, 150}
   };

   MakeHeader (Cols);
}

void  SAXPYTest::OnRun  ()
{
   SaxpyRun::Start (*this, _device);
}

cudaDeviceProp SAXPYTest::GetCudaProperties ()
{
   return _device.GetProperties ();
}

void  SaxpyRun_entry ()
{
   SAXPYTest   Test;
   CTestDialog dlg (&Test);

   dlg.DoModal ();

}

