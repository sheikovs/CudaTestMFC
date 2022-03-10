#ifndef __NVRTCPROGRAM__


#include "Common.h"
#include "KernelArg.h"
#include <memory>
#include <vector>

namespace NVRTCH_1
{
   using BytePx_t = std::unique_ptr <char []>;

   inline
   static constexpr int  INVALID   = -1;

   ///////////////////////////////////////////
   //
   // class NVRTCH::Program
   //
   ///////////////////////////////////////////

   struct ProgramImpl;
   class  Kernel;

   class Program
   {
   private:

      #pragma region Types & Data
      //----------------------------------------------

      using Impl_t   = ProgramImpl;
      using ImplPx_t = std::shared_ptr <Impl_t>;

   private:

      ImplPx_t ImplPx;

      //----------------------------------------------
      #pragma endregion

   public:

      #pragma region Ctor/Dtor
      //----------------------------------------------

      Program  (CString const& NameArg);
      Program  (CString const& NameArg, CString const& SourceArg, bool IsFilePathArg = true);
      ~Program ();

      void  SetSource (CString const& SourceArg);

      //----------------------------------------------
      #pragma endregion

      #pragma region Interface
      //----------------------------------------------

      #pragma region Compile
      //...........................................

      void  AddOption (CString const& OptionArg);

      template <typename ... TArgs>
      void  AddOptions (TArgs const& ... Args)
      {
         (AddOption (Args) && ... );
      }

      void  AddHeader     (CString const& HeaderArg);
      void  AddHeaderPath (CString const& PathArg);

      bool  Compile ();

      CString const& GetLog () const noexcept;

      //...........................................
      #pragma endregion

      #pragma region Link
      //...........................................

      void  AddLibrary (CString const& PathArg);
      void  AddPtx     (char* DataArg, size_t DataSizeArg, CString const& NameArg);
      void  AddCubin   (char* DataArg, size_t DataSizeArg, CString const& NameArg);
      void  LoadData   (void*& CubinArg, size_t& CubinSizeArg);
      bool  GetGlobal  (CString const& NameArg, CUdeviceptr& PtrArg, size_t& SizeArg) const;

      void  Link       ();
      void  Unlink     ();

      //...........................................
      #pragma endregion

      Kernel   GetKernel (CString const& NameArg);

      //----------------------------------------------
      #pragma endregion
   };

   ///////////////////////////////////////////
   //
   // class NVRTCH::KernelArgs
   //
   ///////////////////////////////////////////

   struct KernelImpl;
   struct KernelArgsImpl;

   //class KernelArgs
   //{
   //   using Impl_t   = KernelArgsImpl;
   //   using ImplPx_t = std::shared_ptr <Impl_t>;

   //   friend   KernelImpl;

   //   ImplPx_t ImplPx;

   //public:

   //   KernelArgs ();

   //   template <typename ... TArgs>
   //   void  Add (TArgs ... Args)
   //   {
   //      (__TAdd (Args) && ... );
   //   }

   //   size_t   size () const noexcept;
   //   void**   get  () noexcept;

   //private:

   //   template <typename T>
   //   void  __TAdd (T& Arg)
   //   {
   //      __AddArg (reinterpret_cast <void*>(&Arg));
   //   }

   //   void  __AddArg (void* PtrArg);
   //};

   ///////////////////////////////////////////
   //
   // class NVRTCH::Kernel
   //
   ///////////////////////////////////////////

   class Kernel
   {
   private:

      using Impl_t   = KernelImpl;
      using ImplPx_t = std::shared_ptr <Impl_t>;

      friend Program;

      ImplPx_t ImplPx;

   private:

      Kernel (ImplPx_t&& ImplPxArg);
      Kernel (CUmodule ModuleArg, CString const& NameArg);

   public:

      void  Execute (
            dim3 const& GridArg
         ,  dim3 const& BlockArg
         ,  void**      Args
         ,  CUstream    StreamArg = nullptr
      );

      //void  Execute (KernelArgs& Args);

   };

}  // namespace NVRTCH_1

#endif // !__NVRTCPROGRAM__

