#ifndef __PARSER_H__
#define __PARSER_H__

#include "TLLFlags.h"
#include <vector>
#include <tuple>
#include <map>
#include <set>

///////////////////////////////////////////
//
// struct KeyWord
//
///////////////////////////////////////////

struct KeyWord
{
   #pragma region Types
   //---------------------------------------

   using FlagValue_t = unsigned __int32;
   using Flags_t     = TLL::TFlags <FlagValue_t>;

   //---------------------------------------
   #pragma endregion

   #pragma region Data
   //---------------------------------------

   Flags_t  _Flags;
   int      _IntVal   {};
   float    _FloatVal {};
   CString  _Name;
   CString  _VarName;

   //---------------------------------------
   #pragma endregion

   #pragma region CTors
   //---------------------------------------

   KeyWord ()  = default;

   KeyWord (CString const& NameArg, CString const& VarNameArg, FlagValue_t FlagsArg);

   //---------------------------------------
   #pragma endregion

   #pragma region Interface
   //---------------------------------------

   CString const& GetName () const noexcept
   {
      return _Name;
   }

   bool  IsKeyWord (LPCTSTR KeyWordArg)
   {
      return _VarName.CompareNoCase (KeyWordArg) == 0;
   }

   bool  IsInt      () const noexcept;
   bool  IsFloat    () const noexcept;
   bool  IsFunction () const noexcept;
   bool  IsReadOnly () const noexcept;
   bool  IsRW       () const noexcept;


   int   GetIntValue () const noexcept
   {
      return _IntVal;
   }

   float GetFloatValue () const noexcept
   {
      return _FloatVal;
   }

   void  SetValue (int ValArg) noexcept
   {
      _IntVal  = ValArg;
   }

   void  SetValue (float ValArg) noexcept
   {
      _FloatVal   = ValArg;
   }

   //---------------------------------------
   #pragma endregion

};

///////////////////////////////////////////
//
// struct _CmpStrings
//
///////////////////////////////////////////

struct _CmpStrings
{
   bool operator ()(CString const LhsArg, CString const RhsArg) const noexcept;
};

struct Dictionary
{
   using Map_t = std::map <CString, KeyWord, _CmpStrings>;

   Map_t Map;

   Dictionary ();

   KeyWord* Find (CString NameArg) const;

private:

   void  __Init ();
};

///////////////////////////////////////////
//
// struct _Parser
//
///////////////////////////////////////////

struct _Parser
{
   #pragma region Types
   //---------------------------------------

   using KetWordPtr_t   = KeyWord*;
   using KeyWordsSet_t  = std::set <KetWordPtr_t>;
   using Args_t         = std::vector <KetWordPtr_t>;

   struct _FData
   {
      int      EndPos {};
      CString  Token;
   };

   //---------------------------------------
   #pragma endregion

   #pragma region Data
   //---------------------------------------

   inline
   static LPCTSTR FUNC_NAME   = "__cuda_Entry";

   CString        _Input;
   CString        _Output;
   Dictionary     _KWDict;
   KeyWordsSet_t  _KWSet;
   mutable Args_t _Args;

   //---------------------------------------
   #pragma endregion

   #pragma region Interface
   //---------------------------------------

   _Parser () = default;

   bool Parse (CString const& InputArg, CString& ResultArg);

   Args_t&  GetArguments () const noexcept
   {
      return _Args;
   }

   //---------------------------------------
   #pragma endregion

private:

   #pragma region Helpers
   //---------------------------------------

   void     __Clear ();

   void     __Resolve ();
   bool     __GetKeyWord (int SPosArg, _FData& DataArg);
   void     __AddKeyWord (_FData const& DataArg);
   void     __Replace ();
   void     __Finalize ();
   CString  __GetKW (KeyWord const& KwArg) const;
   CString  __GetArguments () const;
   CString  __GetArgument (KeyWord const& KwArg) const;

   [[noreturn]] 
   static void __Throw (CString const& MsgArg);

   //---------------------------------------
   #pragma endregion
};

#endif // !__PARSER_H__

