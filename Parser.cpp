#include "pch.h"
#include "Parser.h"
#include "Common.h"
#include <vector>
#include <stdexcept>
#include <ctype.h>
#include <string>

///////////////////////////////////////////
//
// struct KeyWord
//
///////////////////////////////////////////

static constexpr KeyWord::FlagValue_t   FL_UNDEF   = 0;
static constexpr KeyWord::FlagValue_t   FL_INT     = 1;
static constexpr KeyWord::FlagValue_t   FL_FLOAT   = 2;
static constexpr KeyWord::FlagValue_t   FL_FUNC    = 4;

static constexpr KeyWord::FlagValue_t   FL_RO      = 16; // Read only
static constexpr KeyWord::FlagValue_t   FL_RW      = 32; // Read/Write


KeyWord::KeyWord (CString const& NameArg, CString const& VarNameArg, FlagValue_t FlagsArg)
:  _Flags    (FlagsArg)
,  _Name     (NameArg)
,  _VarName  (VarNameArg)
{
}

bool  KeyWord::IsInt () const noexcept
{
   return _Flags.IsSet (FL_INT);
}

bool  KeyWord::IsFloat () const noexcept
{
   return _Flags.IsSet (FL_FLOAT);
}

bool  KeyWord::IsFunction () const noexcept
{
   return _Flags.IsSet (FL_FUNC);
}

bool  KeyWord::IsReadOnly () const noexcept
{
   return !_Flags.IsSet (FL_RW);
}

bool  KeyWord::IsRW () const noexcept
{
   return _Flags.IsSet (FL_RW);
}

///////////////////////////////////////////
//
// struct Dictionary
//
///////////////////////////////////////////

bool _CmpStrings::operator ()(CString const LhsArg, CString const RhsArg) const noexcept
{
   return LhsArg.CompareNoCase (RhsArg) < 0;
}

Dictionary::Dictionary ()
{
   __Init ();
}

KeyWord* Dictionary::Find (CString NameArg) const
{
   KeyWord* Ptr   = nullptr;

   if (auto Itr = Map.find (NameArg.Trim ());Itr != Map.end ())
   {
      Ptr   = const_cast <KeyWord*>(&(Itr->second));
   }

   return Ptr;
}

void  Dictionary::__Init ()
{
   Map.emplace ("@Size",     KeyWord ("@Size",     "kw_Size",     FL_INT));
   Map.emplace ("@Result",   KeyWord ("@Result",   "kw_Result",   FL_FLOAT | FL_RW));
   Map.emplace ("@ValInt",   KeyWord ("@ValInt",   "kw_ValInt",   FL_INT));
   Map.emplace ("@ValFloat", KeyWord ("@ValFloat", "kw_ValFloat", FL_FLOAT));
   Map.emplace ("@BinFunc",  KeyWord ("@BinFunc",  "kw_BinFunc",  FL_FUNC));
}

///////////////////////////////////////////
//
// struct _BLData
//
///////////////////////////////////////////

struct _BLData
{
   CString const& InLine;
   CString        BodyLine;
   int            EndPos {};
   CString        Token;

   _BLData (CString const& LineArg)
   :  InLine   (LineArg)
   {
   }
};

///////////////////////////////////////////
//
// struct _Parser
//
///////////////////////////////////////////

bool _Parser::Parse (CString const& InputArg, CString& ResultArg)
{
   __Clear ();

   _Input   = InputArg;

   auto LOnError = [] (CString const& MsgArg)
   {
      ::__OnError (MsgArg);
      return false;
   };

   try
   {
      __Resolve ();
      __Replace ();
      __Finalize ();
   }
   __CATCH_RET(LOnError); 

   ResultArg   = _Output;

   return true;
}

void  _Parser::__Clear ()
{
   _Input .Empty  ();
   _Output.Empty ();
   _KWSet .clear ();
   _Args  .clear ();
}

void  _Parser::__Resolve ()
{
   int      CPos  = 0;
   int      FPos  = 0;
   _FData   KWData;

   while (FPos >= CPos)
   {
      if (FPos  = _Input.Find ('@', CPos); FPos >= CPos)
      {
         if (__GetKeyWord (FPos, KWData))
         {
            __AddKeyWord (KWData);
            CPos = FPos  = KWData.EndPos;
         }
         else
         {
            __Throw ("Invalid syntax");
         }
      }
   }
}

bool _Parser::__GetKeyWord (int SPosArg, _FData& DataArg)
{
   int         Count {};
   bool        Done  {};

   for (int i = SPosArg; !Done && i < _Input.GetLength (); ++i)
   {
      auto const Char (_Input [i]);
      if (::isalpha (Char) || Char == '@' || Char == '_')
      {
         ++Count;
      }
      else
      {
         Done  = true;
      }
   }

   if (Count)
   {
      DataArg.EndPos  = SPosArg + Count;
      DataArg.Token   = _Input.Mid (SPosArg, Count);
   }

   return Count > 0;
}

void  _Parser::__AddKeyWord (_FData const& DataArg)
{
   auto KWPtr = _KWDict.Find (DataArg.Token);

   if (KWPtr)
   {
      _KWSet.insert (KWPtr);
   }
   else
   {
      __Throw (::_F("Invalid Key Word [%s]", DataArg.Token));
   }
}

void  _Parser::__Replace ()
{
   _Output  = _Input;

   for (auto const KwPtr : _KWSet)
   {
      auto const Kw    (__GetKW (*KwPtr));
      auto const Count (_Output.Replace (KwPtr->_Name, Kw));
   }
}

void  _Parser::__Finalize ()
{
   auto const Args (__GetArguments ());

   CString  Tmp = ::_F(
         "%sextern \"C\" __global__ void %s (%s)\r\n{\r\n%s\r\n}\r\n"
      ,  "#include <CommonFunc.h>\r\n\r\n"
      ,  FUNC_NAME
      ,  Args
      ,  _Output
      );

   _Output  = Tmp;
}

CString  _Parser::__GetKW (KeyWord const& KwArg) const
{
   CString  Rc;

   if (KwArg.IsFunction ())
   {
      Rc = KwArg._VarName; // Rc.Format ("(*%s)", KwArg._VarName);
   }
   else if (KwArg.IsRW ())
   {
      Rc.Format ("*(%s)", KwArg._VarName);
   }
   else
   {
      Rc = KwArg._VarName;
   }

   return Rc;
}

CString  _Parser::__GetArguments () const
{
   CString  Args;

   _Args.reserve (_KWSet.size ());

   for (auto KwPtr : _KWSet)
   {
      _Args.push_back (KwPtr);
   }

   for (auto KwPtr : _Args)
   {
      auto const Arg (__GetArgument (*KwPtr));
      
      if (Args.IsEmpty ())
      {
         Args  = Arg;
      }
      else
      {
         Args.AppendFormat (", %s", Arg);
      }
   }

   return Args;
}

CString  _Parser::__GetArgument (KeyWord const& KwArg) const
{
   CString  Arg;

   if      (KwArg.IsInt ())      Arg = "int"; 
   else if (KwArg.IsFloat ())    Arg = "float";
   else if (KwArg.IsFunction ()) Arg = "BinFunc_t";
   else
   {
      __Throw (::_F("Invalid variable type [%s] in [%s]", KwArg._Name, __FUNCTION__));
   }

   KwArg.IsRW () 
   ?  Arg.AppendFormat ("* %s", KwArg._VarName) 
   :  Arg.AppendFormat (" %s",  KwArg._VarName);

   return Arg;
}

void _Parser::__Throw (CString const& MsgArg)
{
   throw std::runtime_error ((LPCTSTR)MsgArg);
}
