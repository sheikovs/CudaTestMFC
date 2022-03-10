#ifndef __CDRMEMORY__
#define __CDRMEMORY__

#include "MemHelper.cuh"

namespace CDRH
{
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
      using HostMem_t      = THostMem   <value_type>;
      using TPinnedMem_t   = TPinnedMem <value_type>;

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
         __TCopy (HMemArg);
      }

      TCuDeviceMem (TPinnedMem_t const& HMemArg)
      {
         __TCopy (HMemArg);
      }

      TCuDeviceMem (const value_type* ValPtrArg, size_t SizeArg)
      {
         __Copy (ValPtrArg, SizeArg);
      }

      Self_t&  operator =(HostMem_t const& HMemArg)
      {
         __TCopy (HMemArg);
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
         if (base_t::_ptr)
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

      template <typename TMem>
      void  __TCopy (TMem const& MemArg)
      {
         if(MemArg)
         {
            reset (MemArg.size ());
            __CDC(cuMemcpyHtoD(base_t::_ptr, MemArg._ptr, base_t::_bytes));
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


}  // namespace CDRH

#endif // !__CDRMEMORY__
