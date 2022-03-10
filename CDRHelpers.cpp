#include "pch.h"
#include "CDRHelpers.h"
#include <fstream>
#include <set>
#include <map>

namespace CDRH
{
   ///////////////////////////////////////////
   //
   // struct CDRH::ModuleImpl
   //
   ///////////////////////////////////////////

   struct ModuleImpl
   {
      #pragma region Data
      //----------------------------------------------

      CUmodule    Module   = nullptr;
      CUlinkState Link     = nullptr;

      //----------------------------------------------
      #pragma endregion

      #pragma region Ctor/Dtor
      //----------------------------------------------

      ModuleImpl ()  = default;

      ModuleImpl (const void* ImageArg)
      {
         CreateFromImage (ImageArg);
      }

      ~ModuleImpl ()
      {
         Reset ();
      }

      //----------------------------------------------
      #pragma endregion

      #pragma region Interface
      //----------------------------------------------

      void  CreateFromImage (const void* ImageArg)
      {
         Reset ();

         __CDC(::cuModuleLoadData(&Module, ImageArg));
      }

      #pragma region Add/Load Data
      //..........................................

      void  AddLibrary (CString const& PathArg)
      {
         __CDC (cuLinkAddFile (
               __GetLink ()
            ,  CU_JIT_INPUT_FATBINARY //CU_JIT_INPUT_LIBRARY
            ,  PathArg
            ,  0, nullptr, nullptr
         ));
      }

      void  AddData (
            void*          DataArg
         ,  size_t         DataSizeArg
         ,  CString const& NameArg
         ,  CUjitInputType TypeArg            
      )
      {
         __CDC (cuLinkAddData (
               __GetLink ()
            ,  TypeArg
            ,  DataArg
            ,  DataSizeArg
            ,  NameArg
            ,  0
            ,  nullptr
            ,  nullptr
         ));
      }

      void  LoadData (void*& CubinArg, size_t& CubinSizeArg)
      {
         __CDC (::cuLinkComplete   (__GetLink (), &CubinArg, &CubinSizeArg));
         __CDC (::cuModuleLoadData (&Module,  CubinArg));
      }

      bool  GetGlobal (CString const& NameArg, CUdeviceptr& PtrArg, size_t& SizeArg) const
      {
         auto const Rc = cuModuleGetGlobal (&PtrArg, &SizeArg, Module, NameArg);
         return Rc == CUDA_SUCCESS;
      }

      //..........................................
      #pragma endregion

      #pragma region Reset
      //..........................................

      void  Reset ()
      {
         Unload ();
         Unlink ();
      }

      void  Unload ()
      {
         if (Module)
         {
            __CDC (::cuModuleUnload (Module));
            Module   = nullptr;
         }
      }

      void  Unlink ()
      {
         if (Link)
         {
            __CDC (::cuLinkDestroy (Link));
            Link  = nullptr;
         }
      }

      //..........................................
      #pragma endregion

      //----------------------------------------------
      #pragma endregion

   private:

      #pragma region Interface
      //----------------------------------------------

      CUlinkState __GetLink ()
      {
         if (!Link)
         {
            __CDC (cuLinkCreate (0, nullptr, nullptr, &Link));
         }

         return Link;
      }

      //----------------------------------------------
      #pragma endregion
   };

   ///////////////////////////////////////////
   //
   // class CDRH::Module
   //
   ///////////////////////////////////////////

   Module::Module (const void* ImageArg)
   :  ImplPx (std::make_shared <Impl_t>(ImageArg))
   {
   }

   Module::~Module ()
   {
   }

   CUmodule Module::get () noexcept
   {
      return ImplPx ? ImplPx->Module : nullptr;
   }

   void  Module::CreateFromImage (const void* ImageArg)
   {
      if (!ImplPx) ImplPx = std::make_shared <Impl_t>(ImageArg);
      ImplPx->CreateFromImage (ImageArg);
   }

   void  Module::AddLibrary (CString const& PathArg)
   {
      ImplPx->AddLibrary (PathArg);
   }

   void  Module::AddPtx (char* DataArg, size_t DataSizeArg, CString const& NameArg)
   {
      ImplPx->AddData (DataArg, DataSizeArg, NameArg, CU_JIT_INPUT_PTX);
   }

   void  Module::AddCubin (char* DataArg, size_t DataSizeArg, CString const& NameArg)
   {
      ImplPx->AddData (DataArg, DataSizeArg, NameArg, CU_JIT_INPUT_CUBIN);
   }

   void  Module::LoadData (void*& CubinArg, size_t& CubinSizeArg)
   {
      ImplPx->LoadData (CubinArg, CubinSizeArg);
   }

   bool  Module::GetGlobal (
         CString const& NameArg
      ,  CUdeviceptr&   PtrArg
      ,  size_t&        SizeArg
   ) const
   {
      return ImplPx->GetGlobal (NameArg, PtrArg, SizeArg);
   }

   void  Module::Link ()
   {
   }

   void  Module::Reset ()
   {
      ImplPx->Reset ();
   }

   ///////////////////////////////////////////
   //
   // struct CDRH::DeviceImpl
   //
   ///////////////////////////////////////////

   struct DeviceImpl
   {
      #pragma region Types
      //----------------------------------------------

      using Props_t     = cudaDeviceProp;
      using PropsMap_t  = std::map <int, Props_t>;

      //----------------------------------------------
      #pragma endregion

      #pragma region Data
      //----------------------------------------------

      inline static PropsMap_t   PropsMap;

      int      Id       = INVALID;
      CUdevice Device   = INVALID;

      //----------------------------------------------
      #pragma endregion

      #pragma region Ctors/Dtors
      //----------------------------------------------

      DeviceImpl (int IdArg)
      :  Id (__GetDevice (IdArg))
      {
         __Init ();
      }

      ~DeviceImpl ()
      {
         if (Id >= 0)
         {
            __CC (::cudaDeviceReset ());
         }
      }

      //----------------------------------------------
      #pragma endregion

      #pragma region Statics
      //----------------------------------------------

      static size_t  GetCount () noexcept
      {
         if (PropsMap.empty ())
         {
            int   Count {};

            __CC (::cudaGetDeviceCount (&Count));

            if (Count > 0)
            {
               Props_t  Props;

               for (int i = 0; i < Count; ++i)
               {
                  __CC (::cudaGetDeviceProperties (&Props, i));

                  PropsMap.emplace (i, Props);
               }            
            }
         }

         return PropsMap.size ();
      }

      //----------------------------------------------
      #pragma endregion

   private:

      #pragma region Private Helpers
      //----------------------------------------------

      static int  __GetDevice (int IdArg)
      {
         if (GetCount ())
         {
            if (auto const Itr = PropsMap.find (IdArg); Itr != PropsMap.cend ())
            {
               __CC(::cudaSetDevice (Itr->first));

               if (Itr->second.canMapHostMemory)
               {
                  __CC(::cudaSetDeviceFlags (cudaDeviceMapHost));
               }
            }
         }

         return IdArg;
      }

      void  __Init ()
      {
         if (Id != INVALID)
         {
            __CDC (::cuDeviceGet (&Device, Id));

            int ComputeMode {};

            __CDC (::cuDeviceGetAttribute (&ComputeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, Id));

            if (ComputeMode == CU_COMPUTEMODE_PROHIBITED)
            {
               throw std::runtime_error (
                  "Error: device is running in <CU_COMPUTEMODE_PROHIBITED>, no "
                  "threads can use this CUDA Device"
               );
            }
         }
      }

      //----------------------------------------------
      #pragma endregion

   };

   ///////////////////////////////////////////
   //
   // class CDRH::Device
   //
   ///////////////////////////////////////////

   Device::Device (int IdArg)
   :  ImplPx (std::make_shared <Impl_t>(IdArg))
   {
   }

   int   Device::GetId () const noexcept
   {
      return ImplPx->Id;
   }

   CUdevice Device::GetDevice () const noexcept
   {
      return ImplPx->Device;
   }

   Device::Props_t const&   Device::GetProperties () const noexcept
   {
      return GetProperties (GetId ());
   }

   size_t  Device::GetCount () noexcept
   {
      return DeviceImpl::GetCount ();
   }

   bool    Device::Exists (int IdArg) noexcept
   {
      return IdArg >= 0 && GetCount () > IdArg;
   }

   Device   Device::SetDevice (int IdArg /*= 0*/) noexcept
   {
      return Exists (IdArg) ? Device (IdArg) : Device ();
   }

   Device::Props_t const&   Device::GetProperties (int IdArg)
   {
      return DeviceImpl::PropsMap [IdArg];
   }

   void  Device::Synchronize ()
   {
      __CC (::cudaDeviceSynchronize ());
   }

   ///////////////////////////////////////////
   //
   // class CDRH::Stream
   //
   ///////////////////////////////////////////

   void  Stream::__Create (unsigned int FlagsArg)
   {
      if (!_stream)
      {
         __CDC (::cuStreamCreate (&_stream, FlagsArg));
      }
   }

   void  Stream::__Wait (bool DestroyArg)
   {
      if (_stream)
      {
         __CDC (::cuStreamSynchronize (_stream));

         if (DestroyArg)
         {
            __CDC(cuStreamDestroy(_stream));
            _stream  = nullptr;
         }
      }
   }

}  // namespace CDRH