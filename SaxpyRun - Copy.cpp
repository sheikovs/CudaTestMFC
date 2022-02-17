#include "pch.h"
#include "MemHelper.cuh"
#include "Gpu.h"
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>

namespace SaxpyRun
{
   using BytePx_t    = std::unique_ptr <char []>;
   using HostMem_t   = THostMem <float>;

   ///////////////////////////////////////////
   //
   // struct NVTCProgParams
   //
   ///////////////////////////////////////////

   struct NVTCProgParams
   {
      using Params_t = std::vector <CString>;
      
      using ParamPtr_t  = LPCTSTR;
      using ParamsPtr_t = const char * const *;
      using ParamsPx    = std::unique_ptr <ParamPtr_t []>;

      Params_t    _params;
      ParamsPx    _px;

      ~NVTCProgParams ()
      {
         reset ();
      }

      void  AddParam (CString ParamArg)
      {
         if (!ParamArg.Trim ().IsEmpty ())
         {
            _params.push_back (ParamArg);
         }
      }

      int   size () const noexcept
      {
         return static_cast <int>(_params.size ());
      }

      void  reset ()
      {
         if (_px)
         {
            _px.reset ();
         }
         _params.clear ();
      }

      ParamsPtr_t get ()
      {
         if (!_px && !_params.empty ())
         {
            auto const  Size (_params.size ());
            _px.reset (new ParamPtr_t [Size]);

            for (size_t i = 0; i < Size; ++i)
            {
               _px [i] = (LPCTSTR)_params [i];
            }
         }

         return _px.get ();
      }
   };

   ///////////////////////////////////////////
   //
   // struct NVTCProg
   //
   ///////////////////////////////////////////

   struct NVTCProg
   {
      nvrtcProgram   _prog = nullptr;
      BytePx_t       _cubin;

      nvrtcProgram&  get () noexcept
      {
         return _prog;
      }

      nvrtcProgram*  get_ptr () noexcept
      {
         return &_prog;
      }

      operator nvrtcProgram* () noexcept
      {
         return &_prog;
      }

      ~NVTCProg ()
      {
         if (_prog)
         {
            nvrtcDestroyProgram (&_prog);
         }
      }
   };

   ///////////////////////////////////////////
   //
   // struct CUDADevice
   //
   ///////////////////////////////////////////

   struct CUDADevice
   {
      inline
      static constexpr int  NAME_LEN         = 100;
      inline
      static constexpr int  INVALID_DEVICE   = -1;

      CUcontext   _context = nullptr;
      CUmodule    _module  = nullptr;
      int         _id      = INVALID_DEVICE;
      CUdevice    _device  = INVALID_DEVICE;
      char        _name [NAME_LEN];

      CUDADevice (int IdArg)
      :  _id (IdArg)
      {
      }

      ~CUDADevice ()
      {
         Reset ();
      }

      void  Init ()
      {
         if (_device == INVALID_DEVICE && _id != INVALID_DEVICE)
         {
            __CDC (::cuDeviceGet(&_device, _id));
            __CDC (::cuDeviceGetName(_name, NAME_LEN, _device));

            int ComputeMode {};
            __CDC (::cuDeviceGetAttribute(&ComputeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, _id));

            if (ComputeMode == CU_COMPUTEMODE_PROHIBITED)
            {
               throw std::runtime_error (
                  "Error: device is running in <CU_COMPUTEMODE_PROHIBITED>, no "
                  "threads can use this CUDA Device"
               );
            }

            __CDC (::cuInit(0));
            __CDC (cuCtxCreate(&_context, 0, _device));
         }
      }

      void  Load (NVTCProg& ProgArg)
      {
         __CDC (::cuModuleLoadData(&_module, ProgArg._cubin.get ()));
      }

      void  Reset ()
      {
         if (_module)
         {
            ::cuModuleUnload(_module);
            _module  = nullptr;
         }

         if (_context)
         {
            cuCtxDestroy(_context);
            _context = nullptr;
         }
      }

      void  Synchronize() const
      {
         __CDC(::cuCtxSynchronize());
      }

   };

   ///////////////////////////////////////////
   //
   // struct Kernel
   //
   ///////////////////////////////////////////

   struct Kernel
   {
      CUfunction  _ptr   = nullptr;
      CString     _name;

      Kernel (CUDADevice const& DeviceArg, CString NameArg)
      :  _name (NameArg.Trim ())
      {
         __CDC (::cuModuleGetFunction(&_ptr, DeviceArg._module, _name));
      }

      void  Execute (dim3 const& GridArg, dim3 const& BlockArg, void** Args, CUstream StreamArg = nullptr)
      {
         __CDC (::cuLaunchKernel(
               _ptr
            ,  GridArg.x, GridArg.y, GridArg.z
            ,  BlockArg.x, BlockArg.y, BlockArg.z
            ,  0           // Shared Mem Size
            ,  StreamArg 
            ,  Args
            ,  nullptr     // extra
         ));
      }
   };

   void  LoadSrcFile (LPCTSTR FilePathArg, BytePx_t& MemPxPxArg)
   {
      std::ifstream  SrcFile (FilePathArg, std::ios::in | std::ios::binary | std::ios::ate);

      if (SrcFile.is_open())
      {
         auto const     Pos   (SrcFile.tellg ());
         size_t const   Size  (static_cast<size_t>(Pos));

         MemPxPxArg.reset (new char [Size + 1]);

         SrcFile.seekg (0, std::ios::beg);
         SrcFile.read  (MemPxPxArg.get (), Size);

         MemPxPxArg.get ()[Size] = '\x0';

         SrcFile.close();
      }
      else
      {
         throw std::runtime_error (::_F("Failed to open file [%s]", FilePathArg));
      }
   }
 
   bool  Compile (
      NVTCProg&   ProgArg
   ,  LPCTSTR     SrcCodeArg
   ,  LPCTSTR     FileNameArg
   )
   {
      auto const&    Prop   (Gpu::GetProperties ());
      CString        Option (::_F("--gpu-architecture=sm_%d%d", Prop.major, Prop.minor));

      NVTCProgParams Params;

      Params.AddParam (Option);

      __CNC(::nvrtcCreateProgram (ProgArg, SrcCodeArg, FileNameArg, 0, nullptr, nullptr));
      auto const  Rc (::nvrtcCompileProgram (ProgArg.get (), Params.size (), Params.get ()));

      size_t   LogSize {};
      __CNC (::nvrtcGetProgramLogSize (ProgArg.get (), &LogSize));
      BytePx_t Log (new char [LogSize + 1]);

      __CNC (::nvrtcGetProgramLog (ProgArg.get (), Log.get ()));
      Log.get ()[LogSize]   = '\x0';

      ::__AddLog ("Compile result:\r\n%s", Log.get ());

      const bool  Result (Rc == NVRTC_SUCCESS);

      if (Result)
      {
         size_t   CodeSize {};
         __CNC (::nvrtcGetCUBINSize(ProgArg.get (), &CodeSize));

         ProgArg._cubin.reset (new char [CodeSize]);
         __CNC (::nvrtcGetCUBIN (ProgArg.get (), ProgArg._cubin.get ()));
      }
      else
      {
         __CNC (Rc);
      }

      return Result; 
   }

   void  __GPURun (
      HostMem_t const&  XHostArg
   ,  HostMem_t const&  YHostArg
   ,  float             MultArg
   ,  HostMem_t&        OutHostArg
   )
   {
      using DMem_t = TCuDeviceMem <float>;

      constexpr int  BLOCK_SIZE  = 32;
      constexpr int  GRID_SIZE   = 128;

      try
      {
         CUDADevice  Device (Gpu::GetId ());

         Device.Init ();

         BytePx_t    SrcPx, CubinPx;

         LoadSrcFile ("C:\\tmp\\saxpy.cu", SrcPx);

         NVTCProg    Prog;

         if (Compile (Prog, SrcPx.get (), "saxpy.cu"))
         {
            Device.Load (Prog);
            Kernel   Func (Device, "saxpy");

            dim3     Grid  (GRID_SIZE,  1, 1);
            dim3     Block (BLOCK_SIZE, 1, 1);

            size_t   Size (XHostArg.size ());
            DMem_t   XDev (XHostArg);
            DMem_t   YDev (YHostArg);
            DMem_t   DOut (Size);

            void* Args[] = { &MultArg, &XDev._ptr, &YDev._ptr, &DOut._ptr, &Size };

            Func.Execute (Grid, Block, Args);

            Device.Synchronize ();

            OutHostArg  = DOut;
         }

      }
      __CATCH(__OnError);
   }

   void  __Populate (HostMem_t& XHostArg, HostMem_t& YHostArg)
   {
      float p (0.125f);
      int   s (8);

      auto LGen   = [&p, &s]()
      {
         float const d  = static_cast <float> (::rand () % 11);
         float const f  = static_cast <float> (((::rand () % s) + 1.0f) * p);
         return (d + f) * (::rand () % 2 ? 1.0f : -1.0f);
      };

      std::generate (XHostArg.begin (), XHostArg.end (), LGen);

      p  = 0.25;
      s  = 4;

      std::generate (YHostArg.begin (), YHostArg.end (), LGen);    
   }

   void  Serial (HostMem_t const& XHostArg, HostMem_t const& YHostArg, const float MultArg)
   {
      auto const  Size (XHostArg.size ());
      HostMem_t   Out  (Size);

      for (size_t i = 0; i < Size; ++i)
      {
         Out [i]   = XHostArg [i] * MultArg + YHostArg [i];
      }

      auto const  Result (std::accumulate (Out.begin (), Out.end (), float (), std::plus <float> ()));

      ::__AddLog ("Serial: %.4lf\r\n", Result);
   }

   void  GPURun (
         HostMem_t const&  XHostArg
      ,  HostMem_t const&  YHostArg
      ,  float             MultArg   
   )
   {
      HostMem_t   Out;

      __GPURun (XHostArg, YHostArg, MultArg, Out);

      auto const  Result (std::accumulate (Out.begin (), Out.end (), float (), std::plus <float> ()));

      ::__AddLog ("CUDA: %.4lf\r\n", Result);

   }

   void  Start ()
   {
      using HostMem_t   = THostMem <float>;

      constexpr size_t  SIZE  = 1000;

      HostMem_t   XHost (SIZE);
      HostMem_t   YHost (SIZE);

      float       Mult (1.25f);

      __Populate (XHost, YHost);

      Serial (XHost, YHost, Mult);

      GPURun (XHost, YHost, Mult);
   }

}  // namespace SaxpyRun

void  SaxpyRun_entry ()
{
   SaxpyRun::Start ();
}

