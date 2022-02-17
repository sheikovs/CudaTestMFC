#ifndef __NVRTC_HELPERS_H__
#define __NVRTC_HELPERS_H__

#include "Common.h"
#include <vector>

namespace NVRTCH
{
   using BytePx_t = std::unique_ptr <char []>;

   inline
   static constexpr int  INVALID   = -1;

   ///////////////////////////////////////////
   //
   // struct ProgOptions
   //
   ///////////////////////////////////////////

   struct ProgOptions
   {
      using Options_t      = std::vector <CString>;

      using OptionPtr_t    = LPCTSTR;
      using OptionsPtr_t   = const char * const *;
      using OptionsPx      = std::unique_ptr <OptionPtr_t []>;

      Options_t   _options;
      OptionsPx   _px;

      ProgOptions ()  = default;
      __NO_COPY(ProgOptions);

      ~ProgOptions ();

      void           AddOption (CString OptionArg);
      bool           Exists    (CString OptionArg) const noexcept;

      int            size () const noexcept
      {
         return static_cast <int>(_options.size ());
      }

      void           reset ();

      OptionsPtr_t   get ();

      ProgOptions& operator += (CString const& OptionArg)
      {
         AddOption (OptionArg);
         return *this;
      }
   };

   ///////////////////////////////////////////
   //
   // class NVRTCH::Program
   //
   ///////////////////////////////////////////

   struct ProgramImpl;
   class  Device;

   class Program
   {
      using Impl_t = ProgramImpl;

      Impl_t*  _pimpl   = nullptr;

      __NO_COPY(Program);

   public:

      Program  () = default;
      Program  (Device const& DeviceArg);
      Program  (Device const& DeviceArg, CString const& SourceArg, bool IsFilePathArg = true);
      ~Program ();

      void  Init (Device const& DeviceArg);

      void  LoadSourceFile (CString const& FilePathArg);
      void  AddOption      (CString const& OptionArg);
      void  AddHeader      (CString const& HeaderArg);
      void  AddHeaderPath  (CString const& PathArg);

      bool  Compile        (LPCTSTR FileNameArg = nullptr);
      bool  Compile        (CString const& SourceArg, LPCTSTR FileNameArg = nullptr);

      CString const&   GetLog       () const noexcept;
      const void*      GetImage     () const noexcept;
      size_t           GetImageSize () const noexcept;

      explicit operator bool () const noexcept
      {
         return _pimpl != nullptr;
      }
   };

   ///////////////////////////////////////////
   //
   // struct DeviceInfo
   //
   ///////////////////////////////////////////

   struct DeviceInfo
   {
      using Props_t  = cudaDeviceProp;

      int      _id   = INVALID;
      Props_t  _props {};
   };

   ///////////////////////////////////////////
   //
   // class NVRTCH::Device
   //
   ///////////////////////////////////////////

   struct DeviceImpl;

   class Device
   {
      #pragma region Types
      //----------------------------------------------

      using Impl_t   = DeviceImpl;

   public:

      using Props_t        = DeviceInfo::Props_t;
      using DevicesInfo_t  = std::vector <DeviceInfo>;

      //----------------------------------------------
      #pragma endregion

      #pragma region Data
      //----------------------------------------------

   private:

      Impl_t*  _pimpl   = nullptr;

      //----------------------------------------------
      #pragma endregion

      #pragma region Construction/Destruction
      //----------------------------------------------

      Device (Impl_t* ImplPtrArg);

   public:

      Device ()   = default;

      Device (const int IdArg);

      //----------------------------------------------
      #pragma endregion

      #pragma region Interface
      //----------------------------------------------

      #pragma region Instance
      //..............................................


      int            GetId         () const noexcept;
      CUdevice       GetDevice     () const noexcept;
      CUmodule       GetModule     () const noexcept;
      Props_t const& GetProperties () const noexcept;

      void           Init   ();      
      void           Init   (int IdArg);
      void           Load   (Program& ProgArg);
      void           Unload ();
      void           Reset  ();

      void           AddLibrary (CString const& PathArg);
      void           AddPtx     (char* DataArg, size_t DataSizeArg, CString const& NameArg);
      void           Add        (Program& ProgArg, CString const& NameArg);
      void           LoadData   (void*& CubinArg, size_t& CubinSizeArg);

      bool           GetGlobal (CUdeviceptr& PtrArg, size_t& SizeArg, CString const& NameArg) const;

      explicit operator bool () const noexcept
      {
         return _pimpl != nullptr;
      }

      //..............................................
      #pragma endregion

      #pragma region Static
      //..............................................

      static void          Synchronize    ();
      static int           GetDeviceCount ();
      static 
      DevicesInfo_t const& GetDevicesInfo ();

      //..............................................
      #pragma endregion

      //----------------------------------------------
      #pragma endregion

   };

   void  ResetDeviceCache ();

   ///////////////////////////////////////////
   //
   // class NVRTCH::Kernel
   //
   ///////////////////////////////////////////

   class Kernel
   {
      CUfunction  _ptr   = nullptr;
      CString     _name;

      __NO_COPY(Kernel);

   public:

      Kernel (Device const& DeviceArg, CString NameArg);

      ~Kernel ();
 
      void  Execute (
               dim3 const& GridArg
            ,  dim3 const& BlockArg
            ,  void**      Args
            ,  CUstream    StreamArg = nullptr
            );
   };

   ///////////////////////////////////////////
   //
   // class NVRTCH::Stream
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

}  // namespace NVRTCH


#endif // !__NVRTC_HELPERS_H__

