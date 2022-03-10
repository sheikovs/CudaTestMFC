#ifndef __KERNELARG__
#define __KERNELARG__
#include "Common.h"
#include "CDRMemory.h"

namespace NVRTCH_1
{
   ///////////////////////////////////////////
   //
   // struct NVRTCH::_KArgBase
   //
   ///////////////////////////////////////////

   struct _KArgBase
   {
      #pragma region Types & Data
      //---------------------------------------

      using Byte_t = unsigned __int8;

      const Byte_t   SizeOf {};

      //---------------------------------------
      #pragma endregion

      #pragma region CTor/Dtor
      //---------------------------------------

      _KArgBase (size_t SizeOfArg)
      :  SizeOf (static_cast <Byte_t>(SizeOfArg))
      {
      }

      virtual ~_KArgBase ()
      {
      }

      //---------------------------------------
      #pragma endregion

      #pragma region Interface
      //---------------------------------------

      virtual size_t Size       () const noexcept = 0;
      virtual void*  Get        () noexcept       = 0;
      virtual void*  GetHostPtr (size_t& SizeArg) = 0;
      virtual bool   IsValue    () const noexcept = 0;

      //---------------------------------------
      #pragma endregion
   };

   ///////////////////////////////////////////
   //
   // struct NVRTCH::_TKValueArg
   //
   ///////////////////////////////////////////

   template <typename T>
   struct _TKValueArg : _KArgBase
   {
      #pragma region Types & Data
      //---------------------------------------

      T  Value {};

      //---------------------------------------
      #pragma endregion

      #pragma region CTors/Dtor
      //---------------------------------------

      _TKValueArg (const T ValArg)
      :  _KArgBase (sizeof (T))
      ,  Value     (ValArg)
      {
      }

      virtual ~_TKValueArg ()
      {
      }

      //---------------------------------------
      #pragma endregion

      #pragma region _KArgBase Implementation
      //---------------------------------------

      virtual size_t   Size () const noexcept override
      {
         return 1U;
      }

      virtual void*    Get  () noexcept override
      {
         return reinterpret_cast <void*>(&Value);
      }

      virtual void*  GetHostPtr (size_t& SizeArg) override
      {
         SizeArg  = this->SizeOf;
         return &Value;
      }

      virtual bool   IsValue  () const noexcept override
      {
         return true;
      }

      //---------------------------------------
      #pragma endregion
   };

   ///////////////////////////////////////////
   //
   // struct NVRTCH::_TKPtrArg
   //
   ///////////////////////////////////////////

   template <typename T>
   struct _TKPtrArg : _KArgBase
   {
      #pragma region Types & Data
      //---------------------------------------

      using DMem_t = CDRH::TCuDeviceMem <T>;

      DMem_t   DMem;
      void*    HPtr  = nullptr;

      //---------------------------------------
      #pragma endregion

      #pragma region CTors/Dtor
      //---------------------------------------

      _TKPtrArg (const T* PtrArg, size_t CountArg)
      :  _KArgBase (sizeof (T))
      ,  DMem      (PtrArg, CountArg)
      {
      }

      virtual ~_TKPtrArg ()
      {
         if (HPtr) ::free (HPtr);
      }

      //---------------------------------------
      #pragma endregion

      #pragma region _KArgBase Implementation
      //---------------------------------------

      virtual size_t   Size () const noexcept override
      {
         return DMem.size ();
      }

      virtual void*    Get  () noexcept override
      {
         return reinterpret_cast <void*>(&DMem._ptr);
      }

      virtual void*  GetHostPtr (size_t& SizeArg) override
      {
         SizeArg  = DMem.size_of ();

         if (!HPtr)
         {
            HPtr  = ::malloc (SizeArg);
            __CDC(cuMemcpyDtoH (HPtr, DMem._ptr, SizeArg));
         }

         return HPtr;
      }

      virtual bool   IsValue  () const noexcept override
      {
         return false;
      }

      //---------------------------------------
      #pragma endregion
   };

   ///////////////////////////////////////////
   //
   // struct NVRTCH::_TKFuncArg
   //
   ///////////////////////////////////////////

   template <typename T>
   struct _TKFuncArg : _KArgBase
   {
      #pragma region Types & Data
      //---------------------------------------

      //T  DFunc  = nullptr;
      CUdeviceptr DFunc {};

      //---------------------------------------
      #pragma endregion

      #pragma region CTors/Dtor
      //---------------------------------------

      _TKFuncArg (T FuncArg)
      :  _KArgBase (sizeof (T))
      {
         //DFunc = FuncArg;
         T  HFunc  = FuncArg;
         //DFunc = FuncArg;
         //__CC(::cudaMemcpyFromSymbol (&HFunc, FuncArg, sizeof (T)));
         //__CC(::cudaMemcpyToSymbol   (&DFunc, &HFunc,  sizeof (T)));
         //__CC(::cudaMemcpyToSymbol   (&DFunc, &FuncArg,  sizeof (T)));
         __CDC (cuMemAlloc   (&DFunc, sizeof (T)));
         __CDC (cuMemcpyHtoD (DFunc, &FuncArg, sizeof (T)));
      }

      virtual ~_TKFuncArg ()
      {
         __CDC (cuMemFree(DFunc));
      }

      //---------------------------------------
      #pragma endregion

      #pragma region _KArgBase Implementation
      //---------------------------------------

      virtual size_t   Size () const noexcept override
      {
         return 1U;
      }

      virtual void*    Get  () noexcept override
      {
         return reinterpret_cast <void*>(DFunc);
      }

      virtual void*  GetHostPtr (size_t& SizeArg) override
      {
         SizeArg  = this->SizeOf;
         return nullptr;
      }

      virtual bool   IsValue  () const noexcept override
      {
         return true;
      }

      //---------------------------------------
      #pragma endregion
   };

   ///////////////////////////////////////////
   //
   // class NVRTCH::KernelArg
   //
   ///////////////////////////////////////////

   class KernelArg
   {
      #pragma region Types & Data
      //----------------------------------------------

      using Impl_t = _KArgBase;

      Impl_t*  Ptr = nullptr;

      //----------------------------------------------
      #pragma endregion

   private:

      #pragma region Ctor/Dtor
      //----------------------------------------------

      KernelArg ()                              = default;
      KernelArg (KernelArg const&)              = delete;
      KernelArg&  operator = (KernelArg const&) = delete;

   public:

      KernelArg (Impl_t* PtrAtr)
      :  Ptr (PtrAtr)
      {
      }

      template <typename T>
      KernelArg (const T ValArg)
      :  Ptr (new _TKValueArg <T>(ValArg))
      {
      }

      template <typename T>
      KernelArg (const T* PtrArg, size_t SizeArg)
      :  Ptr (new _TKPtrArg <T>(PtrArg, SizeArg))
      {
      }

      KernelArg (KernelArg&& RhsArg) noexcept;
      KernelArg&  operator = (KernelArg&& RhsArg) noexcept;

      ~KernelArg ();

      //----------------------------------------------
      #pragma endregion

      #pragma region Interface
      //----------------------------------------------

      bool     IsValue () const noexcept;
      size_t   Size () const noexcept;
      void*    Get  () const noexcept;
      void     Swap (KernelArg& RhsArg) noexcept;

      void*    GetHostPtr (size_t& SizeArg);

      template <typename T>
      T*       TGetHostPtr (size_t& SizeArg)
      {
         T* HPtr  = reinterpret_cast <T*>(GetHostPtr (SizeArg));
         SizeArg /= sizeof (T);
         return HPtr;
      }

      //----------------------------------------------
      #pragma endregion
   };

   ///////////////////////////////////////////
   //
   // class NVRTCH::KernelArg
   //
   ///////////////////////////////////////////

   class KernelArgs
   {
      #pragma region Types & Data
      //---------------------------------------

      using Args_t = std::vector <KernelArg>;
      using Data_t = std::vector <void*>;

      Args_t   Args;
      Data_t   Data;

      //---------------------------------------
      #pragma endregion

   public:

      KernelArgs () = default;

      #pragma region Interface
      //---------------------------------------

      template <typename T>
      void  Add (const T ValArg)
      {
         Args.emplace_back (ValArg);
      }

      template <typename T>
      void  AddPtr (const T* PtrArg, size_t SizeArg)
      {
         Args.emplace_back (PtrArg, SizeArg);
      }

      template <typename T>
      void  AddPtr (const T* PtrArg)
      {
         Args.emplace_back (PtrArg, 1U);
      }

      template <typename T>
      void  AddFunction (T FuncArg)
      {
         Args.emplace_back (new _TKFuncArg <T>(FuncArg));
      }

      size_t            size () const noexcept;
      void**            get  () noexcept;
      KernelArg const&  at   (int IndexArg) const noexcept;
      KernelArg&  operator [](int IndexArg) noexcept;

      //---------------------------------------
      #pragma endregion
   };

}  // namespace NVRTCH_1

#endif // !__KERNELARG__

