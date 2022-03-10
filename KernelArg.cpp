#include "pch.h"
#include "CommonDefs.h"
#include "KernelArg.h"

namespace NVRTCH_1
{
   ///////////////////////////////////////////
   //
   // class NVRTCH::KernelArg
   //
   ///////////////////////////////////////////

   KernelArg::KernelArg (KernelArg&& RhsArg) noexcept
   {
      Swap (RhsArg);
   }

   KernelArg&  KernelArg::operator = (KernelArg&& RhsArg) noexcept
   {
      if (this != &RhsArg)
      {
         Swap (RhsArg);
      }
      return *this;
   }

   KernelArg::~KernelArg ()
   {
      delete Ptr;
   }

   bool     KernelArg::IsValue  () const noexcept
   {
      return Ptr->IsValue ();
   }

   size_t   KernelArg::Size () const noexcept
   {
      return Ptr ? Ptr->Size () : 0U;
   }

   void*    KernelArg::Get () const noexcept
   {
      return Ptr ? Ptr->Get () : nullptr;
   }

   void*    KernelArg::GetHostPtr (size_t& SizeArg)
   {
      return Ptr ? Ptr->GetHostPtr (SizeArg) : nullptr;
   }

   void     KernelArg::Swap (KernelArg& RhsArg) noexcept
   {
      TSWAP (Ptr, RhsArg.Ptr);
   }

   ///////////////////////////////////////////
   //
   // class NVRTCH::KernelArg
   //
   ///////////////////////////////////////////

   size_t   KernelArgs::size () const noexcept
   {
      return Data.size ();
   }

   void**   KernelArgs::get  () noexcept
   {
      if (Data.empty () && !Args.empty ())
      {
         for (auto const& Arg : Args)
         {
            Data.push_back (Arg.Get ());
         }
      }

      return !Data.empty () ? Data.data () : nullptr;
   }

   KernelArg const&  KernelArgs::at (int IndexArg) const noexcept
   {
      return Args.at (IndexArg);
   }

   KernelArg&  KernelArgs::operator [](int IndexArg) noexcept
   {
      return Args [IndexArg];
   }

}  // namespace NVRTCH_1