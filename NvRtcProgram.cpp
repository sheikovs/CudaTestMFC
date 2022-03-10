#include "pch.h"
#include "NvRtcProgram.h"
#include "CDRHelpers.h"
#include <fstream>
#include <set>

namespace NVRTCH_1
{
   ///////////////////////////////////////////
   //
   // struct NVRTCH::KernelArgsImpl
   //
   ///////////////////////////////////////////

   //struct KernelArgsImpl
   //{
   //   #pragma region Types & Data
   //   //--------------------------------------------------------

   //   using uint_t   = unsigned int;
   //   using Arg_t    = void*;
   //   using Args_t   = std::vector <Arg_t>;

   //   CUstream Stream   = nullptr;
   //   dim3     Blocks;
   //   dim3     Threads;
   //   Args_t   Args;

   //   //--------------------------------------------------------
   //   #pragma endregion

   //   KernelArgsImpl () = default;
   //   KernelArgsImpl (
   //         dim3 const& BlocksArg
   //      ,  dim3 const& ThreadsArg
   //      ,  CUstream    StreamArg /*= nullptr*/
   //   )
   //   :  Stream  (StreamArg)
   //   ,  Blocks  (BlocksArg)
   //   ,  Threads (ThreadsArg)
   //   {
   //   }

   //   void  Add (Arg_t PtrArg)
   //   {
   //      Args.push_back (PtrArg);
   //   }

   //   size_t   size () const noexcept
   //   {
   //      return Args.size ();
   //   }

   //   Arg_t*   get () noexcept
   //   {
   //      return !Args.empty () ? Args.data () : nullptr;
   //   }
   //};

   ///////////////////////////////////////////
   //
   // class NVRTCH::KernelArgs
   //
   ///////////////////////////////////////////

   //KernelArgs::KernelArgs ()
   //:  ImplPx (std::make_shared <Impl_t>())
   //{
   //}

   //void  KernelArgs::__AddArg (void* PtrArg)
   //{
   //   ImplPx->Add (PtrArg);
   //}

   //size_t   KernelArgs::size () const noexcept
   //{
   //   return ImplPx->size ();
   //}

   //void**   KernelArgs::get  () noexcept
   //{
   //   return ImplPx->get ();
   //}

   ///////////////////////////////////////////
   //
   // struct NVRTCH::KernelImpl
   //
   ///////////////////////////////////////////

   struct KernelImpl
   {
      CUfunction  Impl  = nullptr;
      CString     Name;

      KernelImpl (CUmodule ModuleArg, CString const& NameArg)
      :  Name (NameArg)
      {
         __CDC (::cuModuleGetFunction(&Impl, ModuleArg, Name));
      }

      void  Execute (
            dim3 const& GridArg
         ,  dim3 const& BlockArg
         ,  void**      Args
         ,  CUstream    StreamArg /*= nullptr*/
      )
      {
         __CDC (::cuLaunchKernel(
               Impl
            ,  GridArg.x, GridArg.y, GridArg.z
            ,  BlockArg.x, BlockArg.y, BlockArg.z
            ,  0           // Shared Mem Size
            ,  StreamArg 
            ,  Args
            ,  nullptr     // extra
         ));
      }

      //void  Execute (KernelArgs& Args)
      //{
      //   auto& KArgs  = *Args.ImplPx;
      //   Execute (KArgs.Blocks, KArgs.Threads, KArgs.get (), KArgs.Stream);
      //}
   };

   ///////////////////////////////////////////
   //
   // class NVRTCH::Kernel
   //
   ///////////////////////////////////////////

   Kernel::Kernel (ImplPx_t&& ImplPxArg)
   :  ImplPx (std::move (ImplPxArg))
   {
   }

   Kernel::Kernel (CUmodule ModuleArg, CString const& NameArg)
   :  ImplPx (std::make_shared <Impl_t>(ModuleArg, NameArg))
   {
   }

   void  Kernel::Execute (
         dim3 const& GridArg
      ,  dim3 const& BlockArg
      ,  void**      Args
      ,  CUstream    StreamArg /*= nullptr*/
   )
   {
      ImplPx->Execute (GridArg, BlockArg, Args, StreamArg);
   }

   //void  Kernel::Execute (KernelArgs& Args)
   //{
   //   ImplPx->Execute (Args);
   //}

   ///////////////////////////////////////////
   //
   // struct NVRTCH::ProgramImpl
   //
   ///////////////////////////////////////////

   struct ProgramBinData
   {
      #pragma region Data
      //----------------------------------------------

      BytePx_t ImagePx;       // PTX or Binary data
      size_t   ImageSize {};  // Image size
      void*    Cubin = nullptr;
      size_t   CubinSize {};
      CString  Log;

      //----------------------------------------------
      #pragma endregion

      #pragma region Interface
      //----------------------------------------------

      void  Create (size_t SizeArg)
      {
         Reset ();
         if (ImageSize = SizeArg; ImageSize)
         {
            ImagePx.reset (new char [ImageSize]);
         }
      }

      void  Reset ()
      {
         ImagePx.reset ();
         ImageSize   = 0U;
         Cubin       = nullptr;
         CubinSize   = 0U;
         Log.Empty ();
      }

      //----------------------------------------------
      #pragma endregion

   };

   ///////////////////////////////////////////
   //
   // struct NVRTCH::ProgHeaders
   //
   ///////////////////////////////////////////

   struct ProgHeaders
   {
      #pragma region Types
      //----------------------------------------------

      using Set_t       = std::set <CString, CmpCStrings>;
      using ItemPtr_t   = LPCTSTR;
      using ItemsPtr_t  = const char * const *;
      using ItemsPx_t   = std::unique_ptr <ItemPtr_t []>;

      //----------------------------------------------
      #pragma endregion

      #pragma region Data
      //----------------------------------------------

      ItemsPx_t   HeadersPx;
      ItemsPx_t   PathsPx;
      Set_t       Headers;
      Set_t       Paths;

      //----------------------------------------------
      #pragma endregion

      #pragma region Interface
      //----------------------------------------------

      void  AddHeader (CString const& HeaderArg)
      {
         if (Headers.insert (HeaderArg).second)
         {
            HeadersPx.reset ();
         }
      }

      void  AddPath (CString const& PathArg)
      {
         if (Paths.insert (PathArg).second)
         {
            PathsPx.reset ();
         }
      }

      ItemsPtr_t  GetHeaders ()
      {
         return __Get (Headers, HeadersPx);
      }

      ItemsPtr_t  GetPaths ()
      {
         return __Get (Paths, PathsPx);
      }

      int   Count () const noexcept
      {
         return __STC(int, Headers.size ());
      }

      //----------------------------------------------
      #pragma endregion

   private:

      #pragma region Helpers
      //----------------------------------------------

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

      //----------------------------------------------
      #pragma endregion
   };

   ///////////////////////////////////////////
   //
   // struct NVRTCH::ProgOptions
   //
   ///////////////////////////////////////////

   struct ProgOptions
   {
      #pragma region Types
      //----------------------------------------------

      using Options_t      = std::vector <CString>;

      using OptionPtr_t    = LPCTSTR;
      using OptionsPtr_t   = const char * const *;
      using OptionsPx_t    = std::unique_ptr <OptionPtr_t []>;

      //----------------------------------------------
      #pragma endregion

      #pragma region Data
      //----------------------------------------------

      Options_t   Options;
      OptionsPx_t OptionsPx;

      //----------------------------------------------
      #pragma endregion

      #pragma region Ctor/Dtor
      //----------------------------------------------

      ProgOptions ()  = default;

      __NO_COPY(ProgOptions);

      ~ProgOptions ()
      {
      }

      //----------------------------------------------
      #pragma endregion

      #pragma region Interface
      //----------------------------------------------

      void  AddOption (CString OptionArg)
      {
         if (!OptionArg.Trim ().IsEmpty () && !Exists (OptionArg))
         {
            Options.push_back (OptionArg);
         }
      }

      bool  Exists (CString OptionArg) const noexcept
      {
         if (!OptionArg.Trim ().IsEmpty ())
         {
            for (auto const& Opt : Options)
            {
               if (Opt.CompareNoCase (OptionArg) == 0)
               {
                  return true;
               }
            }      
         }

         return false;
      }

      int   size () const noexcept
      {
         return static_cast <int>(Options.size ());
      }

      void  reset ()
      {
         if (OptionsPx)
         {
            OptionsPx.reset ();
         }
         Options.clear ();
      }

      OptionsPtr_t   get ()
      {
         if (!OptionsPx && !Options.empty ())
         {
            auto const  Size (Options.size ());
            OptionsPx.reset (new OptionPtr_t [Size]);

            for (size_t i = 0; i < Size; ++i)
            {
               OptionsPx [i] = (LPCTSTR)Options [i];
            }
         }

         return OptionsPx.get ();
      }

      ProgOptions& operator += (CString const& OptionArg)
      {
         AddOption (OptionArg);
         return *this;
      }

      //----------------------------------------------
      #pragma endregion
   };

   ///////////////////////////////////////////
   //
   // struct NVRTCH::ProgramImpl
   //
   ///////////////////////////////////////////

   struct ProgramImpl
   {
      #pragma region Data
      //----------------------------------------------

      using Module_t = CDRH::Module;

      //----------------------------------------------
      #pragma endregion

      #pragma region Data
      //----------------------------------------------

      nvrtcProgram   Impl     = nullptr;
      BytePx_t       SourcePx;
      ProgramBinData BinData;
      CString        Name;
      ProgHeaders    Headers;
      ProgOptions    Options;
      Module_t       Module;

      //----------------------------------------------
      #pragma endregion

      #pragma region Ctor/Dtor
      //----------------------------------------------

      ProgramImpl (CString const& NameArg)
      :  Name (NameArg)
      {
      }

      ProgramImpl (CString const& NameArg, CString const& SourceArg, bool IsFilePathArg)
      :  Name (NameArg)
      {
         IsFilePathArg ? __LoadSourceFile (SourceArg) : __CopySource (SourceArg);
      }

      ~ProgramImpl ()
      {
         Reset ();
      }

      //----------------------------------------------
      #pragma endregion

      #pragma region Interface
      //----------------------------------------------

      void  SetSource (CString const& SourceArg)
      {
         Reset ();
         __CopySource (SourceArg);
      }

      bool  Compile ()
      {
         bool  Result   = __Compile ();

         if (Result)
         {
            Module.CreateFromImage (GetImagePtr ());
            Module.AddCubin (GetImagePtr (), GetImageSize (), "MainEntry.Cubin");
         }

         return Result; 
      }

      void  Link ()
      {
         Module.LoadData (BinData.Cubin, BinData.CubinSize);
      }

      void  Unlink ()
      {
         Module.Reset ();
      }

      void  Reset ()
      {
         BinData.Reset ();

         if (Impl)
         {
            __CNC (::nvrtcDestroyProgram (&Impl));
            Impl  = nullptr;
         }
      }

      char* GetImagePtr () noexcept
      {
         return BinData.ImagePx.get ();
      }

      size_t GetImageSize () const noexcept
      {
         return BinData.ImagePx ? BinData.ImageSize : 0U;
      }

      //----------------------------------------------
      #pragma endregion

   private:

      #pragma region Helpers
      //----------------------------------------------

      void  __CopySource (CString const& SourceArg)
      {
         auto const  Size  = SourceArg.GetLength ();

         SourcePx.reset (new char [Size + 1]);
         ::memcpy (SourcePx.get (), (LPCTSTR)SourceArg, Size);
         SourcePx.get ()[Size] = '\x0';
      }

      void  __LoadSourceFile (CString const& PathArg)
      {
         std::ifstream  SrcFile ((LPCTSTR)PathArg, std::ios::in | std::ios::binary | std::ios::ate);

         if (SrcFile.is_open())
         {
            auto const     Pos   (SrcFile.tellg ());
            size_t const   Size  (static_cast<size_t>(Pos));

            SourcePx.reset (new char [Size + 1]);

            SrcFile.seekg (0, std::ios::beg);
            SrcFile.read  (SourcePx.get (), Size);

            SourcePx.get ()[Size] = '\x0';

            SrcFile.close ();
         }
         else
         {
            throw std::runtime_error (::_F("Failed to open file [%s]", PathArg));
         }
      }

      void  __Create ()
      {
         Reset ();

         __CNC(::nvrtcCreateProgram (
               &Impl
            ,  SourcePx.get ()
            ,  (LPCTSTR)Name
            ,  Headers.Count ()
            ,  Headers.GetPaths ()
            ,  Headers.GetHeaders ()
         ));

         BinData.Reset ();
      }

      #pragma region Compile/Link
      //.............................................

      bool  __Compile ()
      {
         bool  Result   = false;

         auto  LOnError = [this] (CString const& MsgArg)
         {
            BinData.Log.AppendFormat ("%sError compiling [%s]: %s"
               ,  BinData.Log.IsEmpty () ? "" : "\r\n"
               ,  Name
               ,  MsgArg
            );

            return false;
         };

         try
         {
            __Create ();

            auto const  Rc (::nvrtcCompileProgram (Impl, Options.size (), Options.get ()));

            __GetLog ();

            if (Result = (Rc == NVRTC_SUCCESS); Result)
            {
               size_t   Size {};

               __CNC (::nvrtcGetCUBINSize (Impl, &Size));

               BinData.Create (Size);

               __CNC (::nvrtcGetCUBIN (Impl, BinData.ImagePx.get ()));
            }
            else
            {
               __CNC (Rc);
            }
         }
         __CATCH_RET(LOnError);

         return Result; 
      }

      void  __GetLog ()
      {
         size_t   LogSize {};

         __CNC (::nvrtcGetProgramLogSize (Impl, &LogSize));

         BytePx_t Log (new char [LogSize + 1]);

         __CNC (::nvrtcGetProgramLog (Impl, Log.get ()));

         Log.get ()[LogSize]   = '\x0';

         BinData.Log = Log.get ();
      }

      //.............................................
      #pragma endregion

      //----------------------------------------------
      #pragma endregion
   };

   ///////////////////////////////////////////
   //
   // class NVRTCH::Program
   //
   ///////////////////////////////////////////

   Program::Program (CString const& NameArg)
   :  ImplPx (std::make_shared <Impl_t>(NameArg))
   {
   }

   Program::Program (CString const& NameArg, CString const& SourceArg, bool IsFilePathArg /*= true*/)
   :  ImplPx (std::make_shared <Impl_t>(NameArg, SourceArg, IsFilePathArg))
   {
   }

   Program::~Program ()
   {
   }

   void  Program::SetSource  (CString const& SourceArg)
   {
      ImplPx->SetSource (SourceArg);
   }

   #pragma region Interface
   //----------------------------------------------

   #pragma region Compile
   //...........................................

   void     Program::AddOption (CString const& OptionArg)
   {
      ImplPx->Options.AddOption (OptionArg);
   }

   void  Program::AddHeader (CString const& HeaderArg)
   {
      ImplPx->Headers.AddHeader (HeaderArg);
   }

   void  Program::AddHeaderPath (CString const& PathArg)
   {
      ImplPx->Headers.AddPath (PathArg);
   }

   bool  Program::Compile ()
   {
      return ImplPx->Compile ();
   }

   CString const& Program::GetLog () const noexcept
   {
      return ImplPx->BinData.Log;
   }

   //...........................................
   #pragma endregion

   #pragma region Link
   //...........................................

   void  Program::AddLibrary (CString const& PathArg)
   {
      ImplPx->Module.AddLibrary (PathArg);
   }

   void  Program::AddPtx (char* DataArg, size_t DataSizeArg, CString const& NameArg)
   {
      ImplPx->Module.AddPtx (DataArg, DataSizeArg, NameArg);
   }

   void  Program::AddCubin (char* DataArg, size_t DataSizeArg, CString const& NameArg)
   {
      ImplPx->Module.AddCubin (DataArg, DataSizeArg, NameArg);
   }

   void  Program::LoadData (void*& CubinArg, size_t& CubinSizeArg)
   {
      ImplPx->Module.LoadData (CubinArg, CubinSizeArg);
   }

   bool  Program::GetGlobal (CString const& NameArg, CUdeviceptr& PtrArg, size_t& SizeArg) const
   {
      return ImplPx->Module.GetGlobal (NameArg, PtrArg, SizeArg);
   }

   void  Program::Link ()
   {
      ImplPx->Link ();
   }

   void  Program::Unlink ()
   {
      ImplPx->Unlink ();
   }

   //...........................................
   #pragma endregion

   Kernel   Program::GetKernel (CString const& NameArg)
   {
      return Kernel (ImplPx->Module, NameArg);
   }

   //----------------------------------------------
   #pragma endregion

}  // namespace NVRTCH_1