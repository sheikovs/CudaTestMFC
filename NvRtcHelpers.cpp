#include "pch.h"
#include "Common.h"
#include <vector>
#include <map>
#include <memory>
#include <set>
#include <fstream>
#include "NvRtcHelpers.h"
#include "Gpu.h"

namespace NVRTCH
{

   ///////////////////////////////////////////
   //
   // struct DeviceImpl
   //
   ///////////////////////////////////////////

   struct DeviceImpl
   :  public DeviceInfo
   {
      #pragma region Types
      //----------------------------------------------

      using SelfPx_t = std::shared_ptr <DeviceImpl>;

      ///////////////////////////////////////////
      //
      // struct DeviceCache
      //
      ///////////////////////////////////////////

      struct DeviceCache
      {
         using Devices_t      = std::map <int, SelfPx_t>;
         using DevicesInfo_t  = Device::DevicesInfo_t;

         int           _count   = INVALID;
         Devices_t     _devices;
         DevicesInfo_t _di;

         int  GetDeviceCount ()
         {
            if (_count < 0)
            {
               __CDC(::cuInit(0));
               __CDC(::cuDeviceGetCount(&_count));
            }

            return _count;
         }

         DevicesInfo_t const&  GetDevicesInfo ()
         {
            if (GetDeviceCount () > 0 && _devices.empty ())
            {
               _di.reserve (_count);

               DeviceInfo  DInfo;

               for (int i = 0; i < _count; ++i)
               {
                  __GetDeviceInfo (i, DInfo);
                  _di.push_back (DInfo);
               }
            }

            return _di;
         }

         DeviceInfo const& GetDeviceInfo (const int IdArg)
         {
            if (IdArg < 0 && GetDeviceCount () < IdArg)
            {
               throw std::runtime_error (::_F("Invalid Device Id [%i]", IdArg));
            }

            return GetDevicesInfo () [IdArg];
         }

      private:

         void  __GetDeviceInfo (const int IdArg, DeviceInfo& DInfoArg)
         {
            ::memset (&DInfoArg._props, 0, sizeof (DInfoArg._props));

            __CC(::cudaGetDeviceProperties(&DInfoArg._props, IdArg));

            DInfoArg._id   = IdArg;
         }

      };

      using Devices_t      = DeviceCache::Devices_t;
      using DevicesInfo_t  = DeviceCache::DevicesInfo_t;
      using Base_t         = DeviceInfo;
      using CachePx_t      = std::unique_ptr <DeviceCache>;

      //----------------------------------------------
      #pragma endregion

      #pragma region Data
      //----------------------------------------------

      #pragma region Static
      //..............................................

      inline static CachePx_t _cache_px;

      //..............................................
      #pragma endregion

      #pragma region Instance
      //..............................................

      CUcontext   _context = nullptr;
      CUmodule    _module  = nullptr;
      CUdevice    _device  = INVALID;
      CUlinkState _link    = nullptr;

      //..............................................
      #pragma endregion

      //----------------------------------------------
      #pragma endregion

      #pragma region Construction/Destruction
      //----------------------------------------------

      DeviceImpl ()  = default;

      DeviceImpl (Base_t const& BaseArg)
      :  Base_t (BaseArg)
      {
      }

      ~DeviceImpl ()
      {
         Reset ();
      }

      //----------------------------------------------
      #pragma endregion

      #pragma region Inrerface
      //----------------------------------------------

      void  Init ()
      {
         if (_device == INVALID && _id != INVALID)
         {
            __CDC (::cuDeviceGet(&_device, _id));

            int ComputeMode {};

            __CDC (::cuDeviceGetAttribute(&ComputeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, _id));

            if (ComputeMode == CU_COMPUTEMODE_PROHIBITED)
            {
               throw std::runtime_error (
                  "Error: device is running in <CU_COMPUTEMODE_PROHIBITED>, no "
                  "threads can use this CUDA Device"
               );
            }

            __CDC (cuCtxCreate(&_context, 0, _device));
         }
      }

      void  Reset ()
      {
         Unload ();

         if (_context)
         {
            cuCtxDestroy(_context);
            _context = nullptr;
         }
      }

      void  Load (Program& ProgArg)
      {
          (::cuModuleLoadData(&_module, ProgArg.GetImage ()));
      }

      void  Unload ()
      {
         if (_module)
         {
            __CDC (::cuModuleUnload(_module));
            _module  = nullptr;
         }

         if (_link)
         {
            __CDC (::cuLinkDestroy (_link));
            _link = nullptr;
         }
      }

      CUlinkState GetLink ()
      {
         if (!_link)
         {
            __CDC (cuLinkCreate (0, nullptr, nullptr, &_link));
         }
         return _link;
      }

      void  AddLibrary (CString const& PathArg)
      {
         __CDC (cuLinkAddFile (
               GetLink ()
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
               GetLink ()
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
         __CDC (::cuLinkComplete   (GetLink (), &CubinArg, &CubinSizeArg));
         __CDC (::cuModuleLoadData (&_module,  CubinArg));
      }

      bool  GetGlobal (CUdeviceptr& PtrArg, size_t& SizeArg, CString const& NameArg) const
      {
         auto const Rc = cuModuleGetGlobal (&PtrArg, &SizeArg, _module, NameArg);
         return Rc == CUDA_SUCCESS;
      }

      //----------------------------------------------
      #pragma endregion

      #pragma region Statics
      //----------------------------------------------

      static int  GetDeviceCount ()
      {
         return Cache ().GetDeviceCount ();
      }

      static DevicesInfo_t const&  GetDevicesInfo ()
      {
         return Cache ().GetDevicesInfo ();
      }

      static DeviceImpl*   GetDevice (const int IdArg)
      {
         SelfPx_t Px;

         if (IdArg >= 0 && IdArg < GetDeviceCount ())
         {
            Px = __GetDevice (IdArg);
         }

         return Px.get ();
      }

      //----------------------------------------------
      #pragma endregion

      #pragma region Helpers
      //----------------------------------------------

      static DeviceCache&   Cache ()
      {
         if (!_cache_px) _cache_px.reset (new DeviceCache);
         return *_cache_px;
      }

      static SelfPx_t   __GetDevice (const int IdArg)
      {
         SelfPx_t Px;
         auto&    DMap  = Cache ()._devices;
         auto     Itr   = DMap.find (IdArg);

         if (Itr != DMap.end ())
         {
            Px = Itr->second;
         }
         else
         {
            Px = __CreateDevice (IdArg);
            DMap.insert (std::make_pair (IdArg, Px));
         }

         return Px;
      }

      static SelfPx_t   __CreateDevice (const int IdArg)
      {
         auto const& Info  = Cache ().GetDeviceInfo (IdArg);

         SelfPx_t Px = std::make_shared <DeviceImpl> (Info);

         Px->Init ();

         return Px;
      }

      //----------------------------------------------
      #pragma endregion

   };

   void  ResetDeviceCache ()
   {
      DeviceImpl::_cache_px.reset ();
   }

   ///////////////////////////////////////////
   //
   // struct ProgOptions
   //
   ///////////////////////////////////////////

   ProgOptions::~ProgOptions ()
   {
      reset ();
   }

   void  ProgOptions::AddOption (CString OptionArg)
   {
      if (!OptionArg.Trim ().IsEmpty () && !Exists (OptionArg))
      {
         _options.push_back (OptionArg);
      }
   }

   bool  ProgOptions::Exists (CString OptionArg) const noexcept
   {
      if (!OptionArg.Trim ().IsEmpty ())
      {
         for (auto const& Opt : _options)
         {
            if (Opt.CompareNoCase (OptionArg) == 0)
            {
               return true;
            }
         }      
      }

      return false;
   }

   void  ProgOptions::reset ()
   {
      if (_px)
      {
         _px.reset ();
      }
      _options.clear ();
   }

   ProgOptions::OptionsPtr_t ProgOptions::get ()
   {
      if (!_px && !_options.empty ())
      {
         auto const  Size (_options.size ());
         _px.reset (new OptionPtr_t [Size]);

         for (size_t i = 0; i < Size; ++i)
         {
            _px [i] = (LPCTSTR)_options [i];
         }
      }

      return _px.get ();
   }

   ///////////////////////////////////////////
   //
   // struct ProgHeaders
   //
   ///////////////////////////////////////////

   struct ProgHeaders
   {
      using Set_t          = std::set <CString, CmpCStrings>;
      using ItemPtr_t      = LPCTSTR;
      using ItemsPtr_t     = const char * const *;
      using ItemsPx_t      = std::unique_ptr <ItemPtr_t []>;

      ItemsPx_t   _headers_px;
      ItemsPx_t   _paths_px;
      Set_t       _headers;
      Set_t       _paths;

      void  AddHeader (CString const& HeaderArg)
      {
         if (_headers.insert (HeaderArg).second)
         {
            _headers_px.reset ();
         }
      }

      void  AddPath (CString const& PathArg)
      {
         if (_paths.insert (PathArg).second)
         {
            _paths_px.reset ();
         }
      }

      ItemsPtr_t  GetHeaders ()
      {
         return __Get (_headers, _headers_px);
      }

      ItemsPtr_t  GetPaths ()
      {
         return __Get (_paths, _paths_px);
      }

      int   Count () const noexcept
      {
         return __STC(int, _headers.size ());
      }

      ItemsPtr_t  __Get (Set_t const& SetArg, ItemsPx_t& PxArg)
      {
         if (!SetArg.empty () && !PxArg)
         {
            PxArg.reset (new ItemPtr_t [SetArg.size ()]);
            int   Idx {};
            for (auto& Item : SetArg)
            {
               PxArg [Idx++] = (LPCTSTR)Item;
            }
         }

         return PxArg.get ();
      }
   };

   ///////////////////////////////////////////
   //
   // struct ProgramImpl
   //
   ///////////////////////////////////////////

   struct ProgramImpl
   {
      #pragma region Data
      //----------------------------------------------

      nvrtcProgram   _prog = nullptr;
      BytePx_t       _source;
      BytePx_t       _image;           // Cubin data
      size_t         _image_size {};   // Cubin size
      CString        _path;
      ProgOptions    _options;
      ProgHeaders    _headers;
      Device         _device;
      CString        _log;

      //----------------------------------------------
      #pragma endregion

      #pragma region Construction/Destruction
      //----------------------------------------------

      ProgramImpl (Device const& DeviceArg)
      :  _device (DeviceArg)
      {
      }

      ProgramImpl (Device const& DeviceArg, CString FilePathArg, bool IsFilePathArg)
      :  _path   (FilePathArg.Trim ())
      ,  _device (DeviceArg)
      {
      }

      ProgramImpl (Device const& DeviceArg, CString const& SourceArg)
      :  _device (DeviceArg)
      {
         __CopySource (SourceArg);
      }

      ~ProgramImpl ()
      {
         if (_prog)
         {
            __CNC (::nvrtcDestroyProgram (&_prog));
         }
      }

      //----------------------------------------------
      #pragma endregion

      #pragma region Interface
      //----------------------------------------------

      void  LoadSourceFile (CString FilePathArg)
      {
         _path = FilePathArg.Trim ();
         __LoadSourceFile ();
      }

      void  AddOption      (CString const& OptionArg)
      {
         _options.AddOption (OptionArg);
      }

      void  AddHeader (CString HeaderArg)
      {
         if (!HeaderArg.Trim ().IsEmpty ())
         {
            _headers.AddHeader (HeaderArg);
         }
      }

      void  AddHeaderPath (CString PathArg)
      {
         if (!PathArg.Trim ().IsEmpty ())
         {
            _headers.AddPath (PathArg);
         }
      }

      bool  Compile (CString const& SourceArg, LPCTSTR FileNameArg)
      {
         __CopySource (SourceArg);

         return Compile (FileNameArg);
      }

      bool  Compile (LPCTSTR FileNameArg)
      {
         _log.Empty ();

         bool  Result   = false;

         auto  LOnError = [this] (CString const& MsgArg)
         {
            _log.AppendFormat ("%sError compiling [%s]: %s"
               ,  _log.IsEmpty () ? "" : "\r\n"
               ,  _path
               ,  MsgArg
               );

            return false;
         };

         try
         {
            __Create (FileNameArg);

            auto const  Rc (::nvrtcCompileProgram (_prog, _options.size (), _options.get ()));

            __GetLog ();

            if (Result = (Rc == NVRTC_SUCCESS); Result)
            {
               __CNC (::nvrtcGetCUBINSize(_prog, &_image_size));

               _image.reset (new char [_image_size]);
               __CNC (::nvrtcGetCUBIN (_prog, _image.get ()));
            }
            else
            {
               __CNC (Rc);
            }
         }
         __CATCH_RET(LOnError);

         return Result; 
      }

      //----------------------------------------------
      #pragma endregion

   private:

      #pragma region Helpers
      //----------------------------------------------

      void  __Create (LPCTSTR FileNameArg)
      {
         if (!_source)
         {
            __LoadSourceFile ();
         }

         auto const&    Prop   (_device.GetProperties ());
         CString        Option (::_F("--gpu-architecture=sm_%d%d", Prop.major, Prop.minor));

         AddOption (Option);

         __CNC(::nvrtcCreateProgram (
               &_prog
            ,  _source.get ()
            ,  FileNameArg
            ,  _headers.Count ()
            ,  _headers.GetPaths ()
            ,  _headers.GetHeaders ()
            ));
      }

      void  __GetLog ()
      {
         size_t   LogSize {};

         __CNC (::nvrtcGetProgramLogSize (_prog, &LogSize));

         BytePx_t Log (new char [LogSize + 1]);

         __CNC (::nvrtcGetProgramLog (_prog, Log.get ()));

         Log.get ()[LogSize]   = '\x0';

         _log  = Log.get ();
      }

      void  __CopySource (CString const& SourceArg)
      {
         auto const  Size  = SourceArg.GetLength ();
         _source.reset (new char [Size + 1]);
         ::memcpy (_source.get (), (LPCTSTR)SourceArg, Size);
         _source.get ()[Size] = '\x0';

      }

      void  __LoadSourceFile ()
      {
         std::ifstream  SrcFile ((LPCTSTR)_path, std::ios::in | std::ios::binary | std::ios::ate);

         if (SrcFile.is_open())
         {
            auto const     Pos   (SrcFile.tellg ());
            size_t const   Size  (static_cast<size_t>(Pos));

            _source.reset (new char [Size + 1]);

            SrcFile.seekg (0, std::ios::beg);
            SrcFile.read  (_source.get (), Size);

            _source.get ()[Size] = '\x0';

            SrcFile.close();
         }
         else
         {
            throw std::runtime_error (::_F("Failed to open file [%s]", _path));
         }
      }

      //----------------------------------------------
      #pragma endregion

   };

   ///////////////////////////////////////////
   //
   // struct Program
   //
   ///////////////////////////////////////////

   Program::Program (Device const& DeviceArg)
   :  _pimpl (new Impl_t (DeviceArg))
   {
   }

   Program::Program (Device const& DeviceArg, CString const& SourceArg, bool IsFilePathArg /*= true*/)
   :  _pimpl (IsFilePathArg ? new Impl_t (DeviceArg, SourceArg, IsFilePathArg) : new Impl_t (DeviceArg, SourceArg))
   {
   }

   Program::~Program ()
   {
      delete _pimpl;
   }

   void Program::Init (Device const& DeviceArg)
   {
      if (!_pimpl)
      {
         _pimpl   = new Impl_t (DeviceArg);
      }
   }

   void  Program::LoadSourceFile (CString const& FilePathArg)
   {
      _pimpl->LoadSourceFile (FilePathArg);
   }

   void  Program::AddOption (CString const& OptionArg)
   {
      _pimpl->AddOption (OptionArg);
   }

   void  Program::AddHeader (CString const& HeaderArg)
   {
      _pimpl->AddHeader (HeaderArg);
   }

   void  Program::AddHeaderPath (CString const& PathArg)
   {
      _pimpl->AddHeaderPath (PathArg);
   }

   bool Program::Compile (LPCTSTR FileNameArg /*= nullptr*/)
   {
      return _pimpl->Compile (FileNameArg);
   }

   bool Program::Compile (CString const& SourceArg, LPCTSTR FileNameArg /*= nullptr*/)
   {
      return _pimpl->Compile (SourceArg, FileNameArg);
   }

   CString const& Program::GetLog () const noexcept
   {
      return _pimpl->_log;
   }

   const void* Program::GetImage () const noexcept
   {
      return _pimpl->_image.get ();
   }

   size_t   Program::GetImageSize () const noexcept
   {
      return _pimpl->_image_size;
   }

   ///////////////////////////////////////////
   //
   // struct Device
   //
   ///////////////////////////////////////////
   
   Device::Device (Impl_t* ImplPtrArg)
   :  _pimpl (ImplPtrArg)
   {
   }

   Device::Device (const int IdArg)
   :  _pimpl (Impl_t::GetDevice (IdArg))
   {
   }

   int  Device::GetId () const noexcept
   {
      return _pimpl ? _pimpl->_id : INVALID;
   }

   Device::Props_t const& Device::GetProperties () const noexcept
   {
      return _pimpl->_props;
   }

   CUdevice Device::GetDevice () const noexcept
   {
      return _pimpl->_device;
   }

   CUmodule Device::GetModule () const noexcept
   {
      return _pimpl->_module;
   }

   void  Device::Init ()
   {
      _pimpl->Init ();
   }

   void  Device::Init (int IdArg)
   {
      if (!_pimpl)
      {
         _pimpl   = Impl_t::GetDevice (IdArg);
         _pimpl->Init ();
      }
   }

   void  Device::Load (Program& ProgArg)
   {
      _pimpl->Load (ProgArg);
   }

   void  Device::Unload ()
   {
       _pimpl->Unload ();
   }

   void  Device::Reset ()
   {
      _pimpl->Reset ();
   }

   void  Device::Synchronize ()
   {
      __CDC(::cuCtxSynchronize());
   }

   int  Device::GetDeviceCount ()
   {
      return Impl_t::GetDeviceCount ();
   }

   Device::DevicesInfo_t const&  Device::GetDevicesInfo ()
   {
      return Impl_t::GetDevicesInfo ();
   }

   void  Device::AddLibrary (CString const& PathArg)
   {
      _pimpl->AddLibrary (PathArg);
   }

   void  Device::AddPtx (char* DataArg, size_t DataSizeArg, CString const& NameArg)
   {
      _pimpl->AddData (DataArg, DataSizeArg, NameArg, CU_JIT_INPUT_PTX);
   }

   void  Device::Add (Program& ProgArg, CString const& NameArg)
   {
      _pimpl->AddData (const_cast <void*>(ProgArg.GetImage ()), ProgArg.GetImageSize (), NameArg, CU_JIT_INPUT_CUBIN);
   }

   void  Device::LoadData (void*& CubinArg, size_t& CubinSizeArg)
   {
      _pimpl->LoadData (CubinArg, CubinSizeArg);
   }

   bool  Device::GetGlobal (
      CUdeviceptr&   PtrArg
   ,  size_t&        SizeArg
   ,  CString const& NameArg
   ) const
   {
      return _pimpl->GetGlobal (PtrArg, SizeArg, NameArg);
   }

   ///////////////////////////////////////////
   //
   // struct Kernel
   //
   ///////////////////////////////////////////

   Kernel::Kernel (Device const& DeviceArg, CString NameArg)
   :  _name (NameArg.Trim ())
   {
      __CDC (::cuModuleGetFunction(&_ptr, DeviceArg.GetModule (), _name));
   }

   Kernel::~Kernel ()
   {
   }

   void  Kernel::Execute (
      dim3 const& GridArg
   ,  dim3 const& BlockArg
   ,  void**      Args
   ,  CUstream    StreamArg /*= nullptr*/
   )
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

   ///////////////////////////////////////////
   //
   // class NVRTCH::Stream
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

}  // namespace NVRTCH
