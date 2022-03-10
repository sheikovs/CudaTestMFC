#ifndef __CDRHELPERS__
#define __CDRHELPERS__

#include "Common.h"
#include <vector>
#include <memory>

namespace CDRH
{
   using BytePx_t = std::unique_ptr <char []>;

   inline
   static constexpr int  INVALID   = -1;

   ///////////////////////////////////////////
   //
   // struct CDRH::DeviceInfo
   //
   ///////////////////////////////////////////

   struct DeviceInfo
   {
      using Props_t  = cudaDeviceProp;

      int      Id   = INVALID;
      Props_t  Properties {};
   };

   ///////////////////////////////////////////
   //
   // class CDRH::Device
   //
   ///////////////////////////////////////////

   struct DeviceImpl;

   class Device
   {
      #pragma region Types
      //----------------------------------------------

      using Impl_t   = DeviceImpl;
      using ImplPx_t = std::shared_ptr <Impl_t>;

   public:

      using Props_t        = DeviceInfo::Props_t;
      using DevicesInfo_t  = std::vector <DeviceInfo>;

      //----------------------------------------------
      #pragma endregion

   private:

      ImplPx_t ImplPx;

   public:

      Device () = default;
      Device (int IdArg);

      int            GetId     () const noexcept;
      CUdevice       GetDevice () const noexcept; 

      Props_t const& GetProperties () const noexcept;

      static size_t  GetCount  () noexcept;
      static bool    Exists    (int IdArg) noexcept;
      static Device  SetDevice (int IdArg = 0) noexcept;

      static Props_t const&   GetProperties (int IdArg);
      static void             Synchronize   ();
   };

   ///////////////////////////////////////////
   //
   // class CDRH::Module
   //
   ///////////////////////////////////////////

   struct ModuleImpl;

   class Module
   {
      using Impl_t   = ModuleImpl;
      using ImplPx_t = std::shared_ptr <Impl_t>;

      ImplPx_t ImplPx;

   public:

      Module ()   = default;
      Module (const void* ImageArg);

      ~Module ();

      void  CreateFromImage (const void* ImageArg);

      #pragma region Interface
      //----------------------------------------------

      CUmodule get () noexcept;

      operator CUmodule () noexcept
      {
         return get ();
      }

      void  AddLibrary (CString const& PathArg);
      void  AddPtx     (char* DataArg, size_t DataSizeArg, CString const& NameArg);
      void  AddCubin   (char* DataArg, size_t DataSizeArg, CString const& NameArg);
      void  LoadData   (void*& CubinArg, size_t& CubinSizeArg);

      bool  GetGlobal (CString const& NameArg, CUdeviceptr& PtrArg, size_t& SizeArg) const;

      void  Link  ();
      void  Reset ();

      //..........................................
      #pragma endregion

   };

   ///////////////////////////////////////////
   //
   // class CDRH::Stream
   //
   ///////////////////////////////////////////

   class Stream
   {
      CUstream _stream  = nullptr;

   public:

      __NO_COPY(Stream);

      Stream () = default;

      Stream (unsigned int FlagsArg)   //  = CU_STREAM_NON_BLOCKING
      {
         __Create (FlagsArg);
      }

      ~Stream ()
      {
         __Wait (true);
      }

      void  Create (unsigned int FlagsArg = CU_STREAM_NON_BLOCKING)
      {
         __Create (FlagsArg);
      }

      explicit operator bool () const noexcept
      {
         return _stream != nullptr;
      }

      operator CUstream () noexcept
      {
         return _stream;
      }

      void  Wait (bool DestroyArg = true)
      {
         __Wait (DestroyArg);
      }

   private:

      void  __Create (unsigned int FlagsArg);

      void  __Wait   (bool DestroyArg);
   };

}  // namespace CDRH

#endif // !__CDRHELPERS__

