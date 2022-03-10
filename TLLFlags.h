#ifndef __TLLFLAGS__
#define __TLLFLAGS__

namespace TLL
{

   /*//////////////////////////////////////////////////////////////////////
   //
   // Helpers
   //
   //////////////////////////////////////////////////////////////////////*/

   #pragma region Helpers:  
   //-------------------------------------------------------

   template <typename T>
   constexpr   T  TFlagVal (const T ValArg)
   {
      return ValArg > 2 ? T(T(2) << (ValArg - 2)) : ValArg;
   }

   template <typename T>
   constexpr   T  TFlagNext (const T ValArg)
   {
      return ValArg << 1;
   }

   template <typename T>
   constexpr   T  TFlagNext (const T ValArg, const unsigned IndexArg)
   {
      return IndexArg >= 0 ? ValArg << IndexArg : ValArg;
   }

   template <typename T>
   constexpr   T  TGetBitAt (const unsigned IndexArg)
   {
      return IndexArg > 0 ? T(T(2) << (IndexArg - 1)) : T(1);
   }

   //-------------------------------------------------------
   #pragma endregion

   /*//////////////////////////////////////////////////////////////////////
   //
   // struct TLL::TFlags
   //
   //////////////////////////////////////////////////////////////////////*/

   template <typename T>
   struct   TFlags
   {
      using value_t  = T;
   
      static const   T  FL_EMPTY = T{};

   protected:

      T  FlagsInternal {};

   public:

      #pragma region CTORs:  
      //.......................................................

      constexpr TFlags () noexcept
      {
      }

      TFlags (T FlagsArg) noexcept
      :  FlagsInternal (FlagsArg)
      {
      }

      //.......................................................
      #pragma endregion

      #pragma region Members:  
      //.......................................................

      bool  IsEmpty () const noexcept
      {
         return FlagsInternal == FL_EMPTY;
      }

      T  Get () const noexcept
      {
         return FlagsInternal;
      }

      T  Set (T FlagsArg) noexcept
      {
         FlagsInternal |= FlagsArg;

         return FlagsInternal;
      }

      T SetAt (const unsigned IndexArg)
      {
         return SetAt (FlagsInternal, IndexArg);
      }

      T Extract (T MaskArg) const noexcept
      {
         return (FlagsInternal & MaskArg);
      }

      bool Compare (T RhsArg, T MaskArg) const noexcept
      {
         return Compare (FlagsInternal, RhsArg, MaskArg);
      }

      T  Reset (T FlagsArg) noexcept
      {
         auto  Tmp      = FlagsInternal;

         FlagsInternal  = FlagsArg;

         return Tmp;
      }

      void Clear () noexcept
      {
         __ClearAll ();
      }

      T Clear (T FlagsArg) noexcept
      {
         FlagsInternal &= ~ FlagsArg;

         return FlagsInternal;
      }

      T ClearAt (const unsigned IndexArg)
      {
         return ClearAt (FlagsInternal, IndexArg);
      }

      bool  IsSet (T MaskArg) const noexcept
      {
         return IsSet(FlagsInternal, MaskArg);
      }

      bool IsAny (T MaskArg) const noexcept
      {
         return  IsAny (FlagsInternal, MaskArg);
      }

      bool At (unsigned IdxArg) const noexcept
      {
         return  IsSet (FlagsInternal, TGetBitAt <T>(IdxArg));
      }

      bool operator [] (unsigned IdxArg) const noexcept
      {
         return  At (IdxArg);
      }

      //.......................................................
      #pragma endregion

      #pragma region Static:  
      //.......................................................

      static   T  Set (T& FlagsArg, T MaskArg) noexcept
      {
         FlagsArg |= MaskArg;

         return FlagsArg;
      }

      static   T  Clear (T& FlagsArg, T MaskArg) noexcept
      {
         FlagsArg &= ~ MaskArg;

         return FlagsArg;
      }

      static bool IsSet (T FlagsArg, T MaskArg) noexcept
      {
         return (FlagsArg & MaskArg) == MaskArg;
      }

      static bool IsAny (T FlagsArg, T MaskArg) noexcept
      {
         return (FlagsArg & MaskArg) != 0;
      }

      static T Extract (T FlagsArg, T MaskArg) noexcept
      {
         return (FlagsArg & MaskArg);
      }

      static bool Compare (T LhsArg, T RhsArg, T MaskArg) noexcept
      {
         return Extract (LhsArg, MaskArg) == Extract (RhsArg, MaskArg);
      }

      static T SetAt (T& FlagsArg, const unsigned IndexArg)
      {
         return Set (FlagsArg, TGetBitAt <T> (IndexArg));
      }

      static T ClearAt (T& FlagsArg, const unsigned IndexArg)
      {
         return Clear (FlagsArg, TGetBitAt <T> (IndexArg));
      }

   protected:

      void  __ClearAll () noexcept
      {
         FlagsInternal  = FL_EMPTY;
      }

      //.......................................................
      #pragma endregion
   };
}


#endif   // __TLLFLAGS__
