#ifndef __PARSER_H__
#define __PARSER_H__

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
   enum
   {
         T_UNDEF
      ,  T_FLOAT
      ,  T_INT
      ,  T_FUNCTION
   };

   enum
   {
         A_UNDEF
      ,  A_CONST
      ,  A_IN_ONLY
      ,  A_RW
   };

   int      _type    = T_UNDEF;
   int      _access  = A_UNDEF;
   int      _i_val {};
   float    _f_val {};
   CString  _name;
   CString  _var_name;

   KeyWord ()  = default;

   KeyWord (CString const& NameArg, CString const& VarNameArg, int TypeArg, int AccessArg)
   :  _type     (TypeArg)
   ,  _access   (AccessArg)
   ,  _name     (NameArg)
   ,  _var_name (VarNameArg)
   {
   }

   bool  IsInt () const noexcept
   {
      return _type == T_INT;
   }

   bool  IsFloat () const noexcept
   {
      return _type == T_FLOAT;
   }

   bool  IsReadOnly () const noexcept
   {
      return _access == A_IN_ONLY;
   }

   bool  IsRW () const noexcept
   {
      return _access == A_RW;
   }

   void  SetValue (int ValArg) noexcept
   {
      _i_val   = ValArg;
   }

   void  SetValue (float ValArg) noexcept
   {
      _f_val   = ValArg;
   }
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
   using KetWordPtr_t   = KeyWord*;
   using KeyWordsSet_t  = std::set <KetWordPtr_t>;
   using Args_t         = std::vector <KetWordPtr_t>;

   struct _FData
   {
      int      EndPos {};
      CString  Token;
   };

   inline
   static LPCTSTR FUNC_NAME   = "__cuda_Entry";

   CString        _input;
   CString        _output;
   Dictionary     _dic;
   KeyWordsSet_t  _kw_set;
   mutable Args_t _args;

   _Parser () = default;

   bool Parse (CString const& InputArg, CString& ResultArg);

   Args_t&  GetArguments () const noexcept
   {
      return _args;
   }

private:

   void  __Clear ();

   void  __Resolve ();
   bool  __GetKeyWord (int SPosArg, _FData& DataArg);
   void  __AddKeyWord (_FData const& DataArg);
   void  __Replace ();
   void  __Finalize ();
   CString  __GetKW (KeyWord const& KwArg) const;
   CString  __GetArguments () const;
   CString  __GetArgument (KeyWord const& KwArg) const;

   [[noreturn]] static void __Throw (CString const& MsgArg);
};

///////////////////////////////////////////
//
// class FuncParser
//
///////////////////////////////////////////

struct FuncParser
{
   static bool  Parse (CString const& InArg, CString& OutArg);
};

///////////////////////////////////////////
//
// class ScriptParser
//
///////////////////////////////////////////

struct ScriptParserImpl;

class ScriptParser
{
public:

   using ArgItem_t   = std::tuple <CString, float, bool>;
   using Args_t      = std::vector <ArgItem_t>;

private:

   using Impl_t      = ScriptParserImpl;

   Impl_t*  _pimpl   = nullptr;

public:

   ScriptParser  () = default;
   ~ScriptParser ();

   bool           Parse (CString const& SrcArg);
   CString const& GetFunctionName () const noexcept; 
   Args_t&        GetArguments    () const noexcept;
};

#endif // !__PARSER_H__

