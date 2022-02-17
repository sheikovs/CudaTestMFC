#ifndef __CUDA_HELPERS__
#define __CUDA_HELPERS__

#include "Common.h"
#include <thrust/device_vector.h>

//////////////////////////////////////////
//
// class CudaStream
//
///////////////////////////////////////////

class CudaStream
{
   cudaStream_t _stream    = nullptr;
   bool         _dtor_wait = true;

public:

   CudaStream ()
   {
      __CC (::cudaStreamCreate(&_stream));
   }

   ~CudaStream ()
   {
      __Reset (false);
   }

   CudaStream (CudaStream const& )              = delete;
   CudaStream& operator = (CudaStream const& )  = delete;

   cudaStream_t   get () noexcept
   {
      return _stream;
   }

   void  reset ()
   {
      __Reset (true);
   }

   void  wait ()
   {
      __Wait (true);
   }

   explicit operator cudaStream_t const& () const noexcept
   {
      return _stream;
   }

   explicit operator cudaStream_t() noexcept
   {
      return _stream;
   }

   explicit operator bool () const noexcept
   {
      return _stream != nullptr;
   }

private:

   void  __Wait (bool ThrowArg)
   {
      if (_stream && _dtor_wait)
      {
         _dtor_wait  = false;
         auto const Rc  = ::cudaStreamSynchronize (_stream);
         ThrowArg ? __CC (Rc) : __CCE (Rc);
      }
   }

   void  __Reset (bool ThrowArg)
   {
      if (_stream)
      {
         __Wait  (ThrowArg);
         auto const Rc  = ::cudaStreamDestroy (_stream);
         ThrowArg ? __CC (Rc) : __CCE (Rc);
         _stream  = nullptr;
      }
   }
};

//////////////////////////////////////////
//
// class _TAllocator
//
///////////////////////////////////////////

template <typename T>
class _TAllocator
{
public:

   #pragma region Types
   //-------------------------------------------------

   using value_type  = T;
   using pointer     = T*;

protected:

   using Self_t      = _TAllocator <T>;

   //-------------------------------------------------
   #pragma endregion

   #pragma region Data
   //-------------------------------------------------

   size_t   _count  {};
   size_t   _sizeof {};
   pointer  _ptr  = nullptr;

   //-------------------------------------------------
   #pragma endregion

public:

   #pragma region Ctors/Dtor/Assignments
   //-------------------------------------------------

   _TAllocator  () = default;
   ~_TAllocator () = default;

   _TAllocator (Self_t const&)         = delete;
   Self_t&  operator =(Self_t const&)  = delete;

   //-------------------------------------------------
   #pragma endregion

   #pragma region Interface
   //-------------------------------------------------

   size_t   size () const noexcept
   {
      return _count;
   }

   size_t   size_of () const noexcept
   {
      return _sizeof;
   }

   pointer        get () noexcept
   {
      return _ptr;
   }

   const pointer  get () const noexcept
   {
      return _ptr;
   }

   explicit operator bool () const noexcept
   {
      return _ptr != nullptr;
   }

   //-------------------------------------------------
   #pragma endregion

protected:

   void  __Reset () noexcept
   {
      _ptr     = nullptr;
      _count   = 0U;
      _sizeof  = 0U;
   }
};

///////////////////////////////////////////
//
// class THAAllocator (Host Aligned Allocator)
//
///////////////////////////////////////////

template <typename T>
class THAAllocator
:  public _TAllocator <T>
{
   #pragma region Types & Data
   //-------------------------------------------------

   using base_t         = _TAllocator <T>;
   using value_type     = typename base_t::value_type;
   using pointer        = typename base_t::pointer;
   using iterator       = thrust::device_ptr<value_type>;

   pointer  _d_ptr = nullptr;

public:

   static constexpr size_t A_BLOCK  = 4096;

   //-------------------------------------------------
   #pragma endregion

   #pragma region Ctors/Dtor
   //-------------------------------------------------

   THAAllocator () = default;

   THAAllocator (size_t CountArg)
   :  base_t ()
   {
      reset (CountArg);
   }

   ~THAAllocator ()
   {
      reset ();
   }

   //-------------------------------------------------
   #pragma endregion

   #pragma region Interface
   //-------------------------------------------------

   iterator begin () const
   {
      return iterator (_d_ptr);
   }

   iterator end () const
   {
      return iterator (_d_ptr + _count);
   }

   pointer  get_dptr () noexcept
   {
      return _d_ptr;
   }

   const pointer  get_dptr () const noexcept
   {
      return _d_ptr;
   }

   void  reset (size_t CountArg)
   {
      reset ();
      if (CountArg)
      {
         _sizeof        = (((CountArg * sizeof(value_type)) + A_BLOCK - 1) / A_BLOCK) * A_BLOCK;
         base_t::_ptr   = static_cast<value_type*>(::_aligned_malloc (_sizeof, A_BLOCK));
         base_t::_count = CountArg;
         __Register ();
      }
   }

   void  reset ()
   {
      if (base_t::_ptr)
      {
         __UnRegister ();
         ::_aligned_free (base_t::_ptr);
         base_t::__Reset ();
      }
   }

   //-------------------------------------------------
   #pragma endregion

private:

   void  __Register ()
   {
      __CC (::cudaHostRegister(base_t::_ptr, _sizeof, cudaHostRegisterMapped));
      __CC (::cudaHostGetDevicePointer (&_d_ptr, base_t::_ptr, 0));
   }

   void  __UnRegister ()
   {
      __CC(::cudaHostUnregister (base_t::_ptr));
      _d_ptr   = nullptr;
   }
};


#endif // !__CUDA_HELPERS__

