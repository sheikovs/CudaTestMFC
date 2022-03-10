#ifndef __MEMHELPERS__
#define __MEMHELPERS__

#include "Common.h"
#include <thrust/device_ptr.h>

///////////////////////////////////////////
//
// struct _TMemBase
//
///////////////////////////////////////////

template <typename TValue, typename TPtr = TValue*>
struct _TMemBase
{
   #pragma region Types & Data
   //---------------------------------------

   using value_type  = TValue;
   using pointer     = TPtr;
   using Self_t      = _TMemBase <TValue, TPtr>;

   size_t   _count {};
   size_t   _bytes {};
   pointer  _ptr   {};

   //---------------------------------------
   #pragma endregion

   #pragma region Ctros/Dtor
   //---------------------------------------

   _TMemBase  () = default;
   ~_TMemBase () = default;

   _TMemBase (Self_t const&)           = delete;
   Self_t&  operator =(Self_t const&)  = delete;

   //---------------------------------------
   #pragma endregion

   #pragma region Interface
   //---------------------------------------

   size_t   size () const noexcept
   {
      return _ptr ? _count : 0U;
   }

   size_t   size_of () const noexcept
   {
      return _ptr ? _bytes : 0U;
   }

   pointer  get () noexcept
   {
      return _ptr;
   }

   operator pointer& () noexcept
   {
      return _ptr;
   }

   explicit operator bool () const noexcept
   {
      return _ptr != pointer ();
   }

   value_type const& operator [](size_t IndexArg) const noexcept
   {
      return _ptr [IndexArg];
   }

   value_type&       operator [](size_t IndexArg) noexcept
   {
      return _ptr [IndexArg];
   }

   //---------------------------------------
   #pragma endregion

protected:

   #pragma region Helpers
   //---------------------------------------

   void  __Reset () noexcept
   {
      _ptr     = pointer ();
      _count   = 0U;
      _bytes   = 0U;
   }

   //---------------------------------------
   #pragma endregion

};

///////////////////////////////////////////
//
// Forward declaration
//
///////////////////////////////////////////

template <typename TValue>
struct TCuDeviceMem;

///////////////////////////////////////////
//
// struct THostMem
//
///////////////////////////////////////////

template <typename TValue>
struct THostMem
:  public _TMemBase <TValue>
{
   #pragma region Types & Data
   //---------------------------------------

   using base_t      = _TMemBase <TValue>;
   using value_type  = typename base_t::value_type;
   using pointer     = typename base_t::pointer;
   using Self_t      = THostMem <value_type>;
   using DeviceMem_t = TCuDeviceMem <value_type>;

   //---------------------------------------
   #pragma endregion

   #pragma region Ctros/Dtor
   //---------------------------------------

   THostMem () = default;

   THostMem (size_t SizeArg)
   {
      reset (SizeArg);
   }

   THostMem (Self_t const& RhsArg)
   {
      __Copy (RhsArg);
   }

   THostMem (DeviceMem_t const& DMemArg)
   {
      __Copy (DMemArg);
   }

   Self_t&  operator =(Self_t const& RhsArg)
   {
      __Copy (RhsArg);
      return *this;
   }

   Self_t&  operator =(DeviceMem_t const& DMemArg)
   {
      __Copy (DMemArg);
      return *this;
   }

   ~THostMem ()
   {
      reset ();
   }

   //---------------------------------------
   #pragma endregion

   #pragma region Interface
   //---------------------------------------

   void  reset (size_t SizeArg)
   {
      reset ();
      auto const  Tmp (SizeArg * sizeof (value_type));
      if (base_t::_ptr = reinterpret_cast <pointer>(::malloc(Tmp)); base_t::_ptr)
      {
         base_t::_count   = SizeArg;
         base_t::_bytes   = Tmp;
      }
   }

   void  reset () noexcept
   {
      if (base_t::_ptr)
      {
         ::free(base_t::_ptr);
         base_t::__Reset ();
      }
   }

   pointer  begin () noexcept
   {
      return base_t::_ptr;
   }

   pointer  end () noexcept
   {
      return base_t::_ptr + base_t::_count;
   }

   //---------------------------------------
   #pragma endregion

protected:

   #pragma region Helpers
   //---------------------------------------

   void  __Copy (Self_t const& RhsArg)
   {
      if(RhsArg)
      {
         reset (RhsArg.size ());
         ::memcpy (base_t::_ptr, RhsArg._ptr, base_t::_bytes);
      }
   }

   void  __Copy (DeviceMem_t const& DMemArg)
   {
      if(DMemArg)
      {
         reset (DMemArg.size ());
         __CDC(cuMemcpyDtoH(base_t::_ptr, DMemArg._ptr, base_t::_bytes));
      }
   }

   //---------------------------------------
   #pragma endregion

};

///////////////////////////////////////////
//
// struct TCuDeviceMem
//
///////////////////////////////////////////

template <typename TValue>
struct TCuDeviceMem
:  public _TMemBase <TValue, CUdeviceptr>
{
   #pragma region Types & Data
   //---------------------------------------

   using base_t         = _TMemBase <TValue, CUdeviceptr>;
   using value_type     = typename base_t::value_type;
   using pointer        = typename base_t::pointer;
   using Self_t         = TCuDeviceMem <value_type>;
   using HostMem_t      = THostMem <value_type>;

   //---------------------------------------
   #pragma endregion

   #pragma region Ctros/Dtor
   //---------------------------------------

   TCuDeviceMem () = default;

   TCuDeviceMem (size_t SizeArg)
   {
      reset (SizeArg);
   }

   TCuDeviceMem (HostMem_t const& HMemArg)
   {
      __Copy (HMemArg);
   }

   TCuDeviceMem (const value_type* ValPtrArg, size_t SizeArg)
   {
      __Copy (ValArg);
   }

   Self_t&  operator =(HostMem_t const& HMemArg)
   {
      __Copy (HMemArg);
      return *this;
   }

   Self_t&  operator =(value_type const& ValArg)
   {
      __Copy (&ValArg, 1U);
      return *this;
   }

   ~TCuDeviceMem ()
   {
      reset ();
   }

   TCuDeviceMem (Self_t const&)        = delete;
   Self_t&  operator =(Self_t const&)  = delete;

   //---------------------------------------
   #pragma endregion

   #pragma region Interface
   //---------------------------------------

   pointer  get (size_t OffsetArg) noexcept
   {
      return base_t::_ptr + (OffsetArg * sizeof (value_type));
   }

   void  reset (size_t SizeArg)
   {
      reset ();
      auto const  Tmp (SizeArg * sizeof (value_type));
      __CDC(cuMemAlloc(&(base_t::_ptr), Tmp));
      base_t::_count = SizeArg;
      base_t::_bytes = Tmp;
   }

   void  reset ()
   {
      if (_ptr)
      {
         __CDC(cuMemFree(base_t::_ptr));
         base_t::__Reset ();
      }
   }

   //---------------------------------------
   #pragma endregion

protected:

   #pragma region Helpers
   //---------------------------------------

   void  __Copy (HostMem_t const& HMemArg)
   {
      if(HMemArg)
      {
         reset (HMemArg.size ());
         __CDC(cuMemcpyHtoD(base_t::_ptr, HMemArg._ptr, base_t::_bytes));
      }
   }

   void  __Copy (const value_type* ValPtrArg, size_t SizeArg)
   {
      reset (SizeArg);
      __CDC(cuMemcpyHtoD(base_t::_ptr, ValPtrArg, base_t::_bytes));
   }

   //---------------------------------------
   #pragma endregion

};

///////////////////////////////////////////
//
// struct TPinnedMem
//
///////////////////////////////////////////

template <typename TValue>
struct TPinnedMem
:  public _TMemBase <TValue>
{
   #pragma region Types & Data
   //---------------------------------------

   using base_t      = _TMemBase <TValue>;
   using value_type  = typename base_t::value_type;
   using pointer     = typename base_t::pointer;
   using Self_t      = TPinnedMem <value_type>;
   using HostMem_t   = THostMem <value_type>;

   pointer  _d_ptr   = nullptr;

   //---------------------------------------
   #pragma endregion

   #pragma region Ctros/Dtor
   //---------------------------------------

   TPinnedMem () = default;

   TPinnedMem (size_t SizeArg)
   {
      reset (SizeArg);
   }

   TPinnedMem (HostMem_t const& HMemArg)
   {
      __Copy (HMemArg);
   }

   Self_t&  operator =(HostMem_t const& HMemArg)
   {
      __Copy (DMemArg);
      return *this;
   }

   ~TPinnedMem ()
   {
      reset (false);
   }

   TPinnedMem (Self_t const&)          = delete;
   Self_t&  operator =(Self_t const&)  = delete;

   //---------------------------------------
   #pragma endregion

   #pragma region Interface
   //---------------------------------------

   pointer  d_get () noexcept
   {
      return _d_ptr;
   }

   void  reset (size_t SizeArg)
   {
      reset ();

      if (SizeArg)
      {
         auto const  Tmp (SizeArg * sizeof (value_type));
         __CC(::cudaHostAlloc((void**)&(this->_ptr), Tmp, cudaHostAllocMapped));
         __CC(::cudaHostGetDevicePointer (&_d_ptr, this->_ptr, 0));
         base_t::_count = SizeArg;
         base_t::_bytes = Tmp;
      }
   }

   void  reset (bool ThrowArg = true) noexcept
   {
      if (base_t::_ptr)
      {
         auto const Rc  = ::cudaFreeHost (base_t::_ptr);
         ThrowArg ? __CC (Rc) : __CCE (Rc);
         _d_ptr   = nullptr;
         base_t::__Reset ();
      }
   }

   pointer  begin () noexcept
   {
      return base_t::_ptr;
   }

   pointer  end () noexcept
   {
      return base_t::_ptr + base_t::_count;
   }

   pointer  d_begin () noexcept
   {
      return base_t::_d_ptr;
   }

   pointer  d_end () noexcept
   {
      return base_t::_d_ptr + base_t::_count;
   }

   //---------------------------------------
#pragma endregion

protected:

   #pragma region Helpers
   //---------------------------------------

   void  __Copy (HostMem_t const& HMemArg)
   {
      if(HMemArg.size ())
      {
         reset (HMemArg.size ());
         ::memcpy (base_t::_ptr, HMemArg._ptr, base_t::size_of ());
      }
   }

   //---------------------------------------
   #pragma endregion

};
#endif // !__MEMHELPERS__
