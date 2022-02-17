#include "pch.h"
#include "Parser.h"
#include "Common.h"
#include <vector>
#include <stdexcept>
#include <ctype.h>
#include <string>

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
   Map.emplace ("@Size",     KeyWord ("@Size",     "kw_Size",     KeyWord::T_INT,   KeyWord::A_IN_ONLY));
   Map.emplace ("@Result",   KeyWord ("@Result",   "kw_Result",   KeyWord::T_FLOAT, KeyWord::A_RW));
   Map.emplace ("@ValInt",   KeyWord ("@ValInt",   "kw_ValInt",   KeyWord::T_INT,   KeyWord::A_IN_ONLY));
   Map.emplace ("@ValFloat", KeyWord ("@ValFloat", "kw_ValFloat", KeyWord::T_FLOAT, KeyWord::A_IN_ONLY));
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
// struct ParserBase
//
///////////////////////////////////////////

struct ParserBase
{
   using StrList_t   = std::vector <CString>;

   inline static CString   OPERATORS = "+=-*/";

   CString     Source;
   CString     FName;
   StrList_t   LinesList;
   StrList_t   ArgsList;

   ParserBase ()  = default;
   ParserBase (CString const& SourceArg)
   :  Source (SourceArg)
   {
   }

   [[noreturn]] static void __Throw (CString const& MsgArg)
   {
      throw std::runtime_error ((LPCTSTR)MsgArg);
   }

protected:

   #pragma region Helpers
   //------------------------------------------------

   void  __GetArguments (CString const& InArg, int SPosArg)
   {
      int      EPos  = InArg.Find (')');

      auto LOnMissArgs = [] ()
      {
         ParserBase::__Throw ("Missing Function Arguments");
      };

      if (EPos > SPosArg)
      {
         CString  Args  = __GetSubString (InArg, SPosArg, EPos).Trim ();

         if (Args.IsEmpty ())
         {
            LOnMissArgs ();
         }

         if (ArgsList = __Tokenize (Args, "(,"); ArgsList.empty ())
         {
            LOnMissArgs ();
         }
      }
      else
      {
         LOnMissArgs ();
      }
   }

   #pragma region Static Helpers
   //...........................................

   StrList_t  __Tokenize (CString const& StrArg, LPCTSTR DelimsArg) const
   {
      int         Pos {};
      CString     Line (StrArg.Tokenize (DelimsArg, Pos));
      StrList_t   List;

      while (Line != "")
      {
         if (!Line.Trim ().IsEmpty ())
         {
            List.push_back (Line);
         }

         Line	= StrArg.Tokenize (DelimsArg, Pos);
      }

      return List;
   }

   CString  __GetSubString (CString const& InArg, int SPosArg, int EPosSrg) const
   {
      return InArg.Mid (SPosArg, EPosSrg - SPosArg).Trim ();
   }

   static bool __IsOperator (char CharArg)
   {
      return OPERATORS.Find (CharArg) >= 0;
   }

   static bool __GetNumber (int SPosArg, _BLData& DataArg)
   {
      int         Count {};
      bool        Done  {};
      auto const& InLine (DataArg.InLine);

      for (int i = SPosArg; !Done && i < InLine.GetLength (); ++i)
      {
         auto const Char (InLine [i]);
         if ((Char >= '0' && Char <= '9') || Char == '.')
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
         DataArg.Token   = InLine.Mid (SPosArg, Count);
      }

      return Count > 0;
   }

   static bool __GetWord (int SPosArg, _BLData& DataArg)
   {
      int         Count {};
      bool        Done  {};
      auto const& InLine (DataArg.InLine);

      for (int i = SPosArg; !Done && i < InLine.GetLength (); ++i)
      {
         auto const Char (InLine [i]);
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
         DataArg.Token   = InLine.Mid (SPosArg, Count);
      }

      return Count > 0;
   }

   //...........................................
   #pragma endregion

   //------------------------------------------------
   #pragma endregion
};

///////////////////////////////////////////
//
// struct _Parser
//
///////////////////////////////////////////

bool _Parser::Parse (CString const& InputArg, CString& ResultArg)
{
   __Clear ();

   _input   = InputArg;

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

   ResultArg   = _output;

   return true;
}

void  _Parser::__Clear ()
{
   _input .Empty  ();
   _output.Empty ();
   _kw_set.clear ();
   _args  .clear ();
}

void  _Parser::__Resolve ()
{
   int      CPos  = 0;
   int      FPos  = 0;
   _FData   KWData;

   while (FPos >= CPos)
   {
      if (FPos  = _input.Find ('@', CPos); FPos >= CPos)
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

   for (int i = SPosArg; !Done && i < _input.GetLength (); ++i)
   {
      auto const Char (_input [i]);
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
      DataArg.Token   = _input.Mid (SPosArg, Count);
   }

   return Count > 0;
}

void  _Parser::__AddKeyWord (_FData const& DataArg)
{
   auto KWPtr = _dic.Find (DataArg.Token);

   if (KWPtr)
   {
      _kw_set.insert (KWPtr);
   }
   else
   {
      __Throw (::_F("Invalid Key Word [%s]", DataArg.Token));
   }
}

void  _Parser::__Replace ()
{
   _output  = _input;

   for (auto const KwPtr : _kw_set)
   {
      auto const Kw (__GetKW (*KwPtr));
      auto const Count (_output.Replace (KwPtr->_name, Kw));
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
      , _output
      );

   _output  = Tmp;
}

CString  _Parser::__GetKW (KeyWord const& KwArg) const
{
   CString  Kw;

   switch (KwArg._access)
   {
   case KeyWord::A_IN_ONLY:   Kw = KwArg._var_name; break;
   case KeyWord::A_RW:        Kw.Format ("*(%s)", KwArg._var_name); break;
   default: __Throw (::_F("Invalid access type [%s-%i] in [%s]", KwArg._name, KwArg._access, __FUNCTION__));
   }

   return Kw;
}

CString  _Parser::__GetArguments () const
{
   CString  Args;

   _args.reserve (_kw_set.size ());

   for (auto KwPtr : _kw_set)
   {
      _args.push_back (KwPtr);
   }

   for (auto KwPtr : _args)
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

   switch (KwArg._type)
   {
   case KeyWord::T_INT:    Arg = "int";   break;
   case KeyWord::T_FLOAT:  Arg = "float"; break;
   default: __Throw (::_F("Invalid variable type [%s-%i] in [%s]", KwArg._name, KwArg._type, __FUNCTION__));
   }

   switch (KwArg._access)
   {
   case KeyWord::A_IN_ONLY:   Arg.AppendFormat (" %s",  KwArg._var_name); break;
   case KeyWord::A_RW:        Arg.AppendFormat ("* %s", KwArg._var_name); break;
   default: __Throw (::_F("Invalid access type [%s-%i] in [%s]", KwArg._name, KwArg._access, __FUNCTION__));
   }

   return Arg;
}

void _Parser::__Throw (CString const& MsgArg)
{
   throw std::runtime_error ((LPCTSTR)MsgArg);
}

///////////////////////////////////////////
//
// struct FuncParserImpl
//
///////////////////////////////////////////

struct FuncParserImpl
:  public ParserBase
{
   using Base_t      = ParserBase;
   using StrList_t   = Base_t::StrList_t;

   CString     Out;
   CString     Body;

   FuncParserImpl (CString const& SourceArg)
   :  Base_t (SourceArg)
   {
   }

   bool  Parse ()
   {
      auto LOnError = [] (CString const& MsgArg)
      {
         ::__OnError (MsgArg);
         return false;
      };

      try
      {
         __GetLines ();

         __ParseFistLine ();

         __ParseBody ();

         __Finalize ();
      }
      __CATCH_RET(LOnError);

      return true;
   }

   #pragma region Parse Input
   //..............................................

   void  __GetLines ()
   {
      LPCTSTR  DELIMS = "\r\n;";

      LinesList   = __Tokenize (Base_t::Source, DELIMS);

      if (LinesList.size () >= 4)
      {
         CString  Tmp;

         auto LCheckBody = [&Tmp] (LPCTSTR StrArg)
         {
            if (Tmp != StrArg) FuncParserImpl::__Throw ("Missing function body");
         };

         Tmp   = LinesList [1];
         LCheckBody ("{");

         Tmp   = LinesList.back ();
         LCheckBody ("}");
      }
      else
      {
         __Throw ("Invalid input");
      }
   }

   void  __Finalize ()
   {
      Out.AppendFormat ("{\r\n%s}", Body);
   }

   //..............................................
   #pragma endregion

   #pragma region Parse First Line
   //..............................................

   void  __ParseFistLine ()
   {
      CString  First (LinesList.front ());

      int      Pos  = First.Find ('(');

      if (Pos < 1)
      {
         __Throw ("Missing Function name");
      }

      FName = __GetSubString (First, 0, Pos);
      Base_t::__GetArguments (First, Pos + 1);
 
      __AddFirstLine ();

   }

   void  __AddFirstLine ()
   {
      LPCTSTR  TYPE  = "float*";
      CString  Args;

      for (auto const& Arg : ArgsList)
      {
         if (Args.IsEmpty ())
         {
            Args.Format ("%s %s", TYPE, Arg);
         }
         else
         {
            Args.AppendFormat (",%s %s", TYPE, Arg);
         }
      }

      Out.Format ("extern \"C\" __global__ void %s (%s)\r\n", FName, Args);
   }

   //..............................................
   #pragma endregion

   #pragma region Parse Body
   //..............................................

   void  __ParseBody ()
   {
      for (size_t i = 2; i < LinesList.size () - 1; ++i)
      {
         __ParseBodyLine (LinesList.at (i));
      }
   }

   void  __ParseBodyLine (CString const& LineArg)
   {
      auto LOnError  = [&LineArg] ()
      {
         FuncParserImpl::__Throw (::_F("Syntax error in [%s]", LineArg));
      };

      _BLData  Data (LineArg);

      auto LOnVar = [this, &Data, &LOnError] (int PosArg)
      {
         if (!__GetWord (PosArg, Data)) LOnError ();
         __CheckVar (Data.Token);
         Data.BodyLine.AppendFormat ("*%s", Data.Token);
      };

      for (int i = 0; i < LineArg.GetLength (); ++i)
      {
         auto const  Char (LineArg [i]);

         if (i == 0)
         {
            LOnVar (i);
            i  = Data.EndPos;
         }
         else
         {
            if (::isblank (Char) || __IsOperator (Char))
            {
               Data.BodyLine.AppendFormat ("%c", Char);
            }
            else if (::isalpha (Char))
            {
               LOnVar (i);
               i  = Data.EndPos;
            }
            else if (::isdigit (Char))
            {
               if (!__GetNumber (i, Data)) LOnError ();

               Data.BodyLine.AppendFormat ("%s", Data.Token);

               i  = Data.EndPos;
            }
            else
            {
               __Throw (::_F("Syntax error: illegal character [%c] in  [%s]", Char, LineArg));
            }
         }
      }

      __AddBodyLine (Data.BodyLine);
   }

   void  __AddBodyLine (CString const& LineArg)
   {
      Body.AppendFormat ("   %s;\r\n", LineArg);
   }

   bool  __CheckVar (CString& VarArg)
   {
      for (auto const& Arg : ArgsList)
      {
         if (Arg.CompareNoCase (VarArg) == 0)
         {
            VarArg   = Arg;
            return true;
         }
      }

      __Throw (::_F("Unkown variable [%s]", VarArg));

      return false;
   }

   //..............................................
   #pragma endregion

};

///////////////////////////////////////////
//
// struct FuncParser
//
///////////////////////////////////////////

bool  FuncParser::Parse (CString const& InArg, CString& OutArg)
{
   FuncParserImpl Impl (InArg);

   auto const     Rc   (Impl.Parse ());

   if (Rc)
   {
      OutArg   = Impl.Out;
   }

   return Rc;
}

///////////////////////////////////////////
//
// struct ScriptParserImpl
//
///////////////////////////////////////////

struct ScriptParserImpl
:  public ParserBase
{
   #pragma region Types & Data
   //------------------------------------------------

   #pragma region Types
   //..............................................

   using Base_t      = ParserBase;
   using StrList_t   = Base_t::StrList_t;
   using VarItem_t   = std::pair <float, bool>;
   using VarsMap_t   = std::map  <CString, VarItem_t>;
   using DefMap_t    = std::map  <CString, float>;

   using ArgItem_t   = ScriptParser::ArgItem_t;
   using Args_t      = ScriptParser::Args_t;

   //..............................................
   #pragma endregion

   VarsMap_t   VarsMap;
   DefMap_t    DefValMap;
   Args_t      Args;

   //------------------------------------------------
   #pragma endregion

   ScriptParserImpl (CString const& SrcArg)
   :  Base_t (SrcArg)
   {
      DefValMap.emplace ("@DEF_X", 2.225f);
      DefValMap.emplace ("@DEF_Y", 100.0f);
   }

   bool  Parse ()
   {
      auto LOnError = [] (CString const& MsgArg)
      {
         ::__OnError (MsgArg);
         return false;
      };

      try
      {
         __GetLines   ();
         __ParseLines ();

         __Validate   ();

         __Finalize   ();
       }
      __CATCH_RET(LOnError);

      return true;
   }

private:

   #pragma region Helpers
   //------------------------------------------------

   #pragma region Parse Source
   //..............................................

   void  __GetLines ()
   {
      LPCTSTR  DELIMS = "\r\n;";

      LinesList   = __Tokenize (Base_t::Source, DELIMS);

      if (LinesList.empty ())
      {
         __Throw ("Invalid input");
      }
   }

   void  __Validate () const
   {

      if (Base_t::FName.IsEmpty ())
      {
         __Throw ("Missing Function Name");
      }

      if (Base_t::ArgsList.empty ())
      {
         __Throw ("Missing Function Arguments");
      }
   }

   void  __Finalize ()
   {
      Args.reserve (Base_t::ArgsList.size ());

      for (auto const& Arg : Base_t::ArgsList)
      {
         CString     Tmp (Arg);
         auto const  Itr (VarsMap.find (Tmp.MakeUpper ()));

         if (Itr == VarsMap.cend ()) __Throw (::_F("Undefined Argument [%s]", Tmp));

         auto const& [Value, IsOut] = Itr->second;

         Args.emplace_back (Arg, Value, IsOut);
      }
   }

   void  __ParseLines ()
   {
      for (auto const& Line : LinesList)
      {
         __ParseLine (Line);
      }
   }

   void  __ParseLine (CString const& LineArg)
   {
      auto LOnError  = [&LineArg] (CString const& MsgArg = CString ())
      {
         ParserBase::__Throw (::_F("Syntax error in [%s] %s", LineArg, MsgArg));
      };

      try
      {
         _BLData  Data (LineArg);

         if (!__GetWord (0, Data)) LOnError ();

         if (Data.Token.CompareNoCase ("IN") == 0)
         {
            __ParseVariable (Data.EndPos + 1, Data);
         }
         else if (Data.Token.CompareNoCase ("OUT") == 0)
         {
            __ParseVariable (Data.EndPos + 1, Data, true);
         }
         else
         {
            __ParseFunction (Data);
         }
      }
      __CATCH (LOnError)

   }

   //..............................................
   #pragma endregion

   #pragma region Parse Variable
   //..............................................

   void  __ParseVariable (int PosArg, _BLData& DataArg, bool IsOutArg = false)
   {
      bool  VarFound {};

      auto const& Line (DataArg.InLine);

      CString     VarName;
      bool        AssFound {};
      bool        ValFound {};

      VarItem_t   Item (0.0f, IsOutArg);

      for (int i = PosArg; !ValFound && i < Line.GetLength (); ++i)
      {
         auto const  Char (Line [i]);

         if (::isalpha (Char))
         {
            if (!VarName.IsEmpty () || !__GetWord (i, DataArg)) __Throw (": illegal token");
            VarName  = DataArg.Token;
            i        = DataArg.EndPos;
         }
         else if (Char == '=')
         {
            if (!VarName.IsEmpty ()) AssFound = true;
            else __Throw (": illegal assignment");
         }
         else if (::isdigit (Char) || Char == '@')
         {
            if (!VarName.IsEmpty () && AssFound)
            {
               ValFound = __ParseDigit (i, DataArg, Item.first);
               i        = DataArg.EndPos;
            }
         }
         else if (!::isblank (Char))
         {
            __Throw (::_F(": illegal character [%c]", Char));
         }
      }

      VarsMap.emplace (VarName.MakeUpper (), Item);

      //if (!VarsMap.emplace (VarName.MakeUpper (), Item).second)
      //{
      //   __Throw (::_F(": Variable [%s] already declared", VarName));
      //}
   }

   bool  __ParseDigit (int PosArg, _BLData& DataArg, float& ValueArg)
   {
      auto const  Char (DataArg.InLine [PosArg]);

      bool        Rc (::isdigit (Char));

      if (Rc)
      {
         if (Rc = __GetNumber (PosArg, DataArg); Rc)
         {
            ValueArg = static_cast <float>(::atof ((LPCTSTR)DataArg.Token));
         }
      }
      else // Predefined Var
      {
         if (Rc = __GetWord (PosArg, DataArg); Rc)
         {
            if (!__GetDefVal (DataArg.Token, ValueArg))
            {
               __Throw (::_F(": Invalid Value [%s]", DataArg.Token));
            }
         }      
      }

      if (!Rc) __Throw (::_F(": Invalid Value Character [%c]", Char));

      return true;
   }

   bool  __GetDefVal (CString NameArg, float& ValArg)
   {
      auto const Itr = DefValMap.find (NameArg.Trim ().MakeUpper ());

      if (Itr != DefValMap.cend ())
      {
         ValArg   = Itr->second;
         return true;
      }

      return false;
   }

   //..............................................
   #pragma endregion

   #pragma region Parse Variable
   //..............................................

   void  __ParseFunction (_BLData& DataArg)
   {
      if (!FName.IsEmpty ()) __Throw (_F("Function already defined as [%s]", FName));

      FName = DataArg.Token;

      int      Pos  = DataArg.InLine.Find ('(', DataArg.EndPos);

      if (Pos < 1)
      {
         __Throw ("Missing Function arguments");
      }

      Base_t::__GetArguments (DataArg.InLine, Pos + 1);

      for (auto const& Arg : Base_t::ArgsList)
      {
         CString  Tmp (Arg);
         
         if (VarsMap.find (Tmp.MakeUpper ()) == VarsMap.cend ())
         {
            __Throw (::_F("Undefined Function Parameter [%s]", Arg));
         }
      }
   }

   //..............................................
   #pragma endregion

   //------------------------------------------------
   #pragma endregion

};


///////////////////////////////////////////
//
// class ScriptParser
//
///////////////////////////////////////////

ScriptParser::~ScriptParser ()
{
   delete _pimpl;
}

bool  ScriptParser::Parse (CString const& SrcArg)
{
   if (!_pimpl) _pimpl = new Impl_t (SrcArg);

   return _pimpl->Parse ();
}

CString const& ScriptParser::GetFunctionName () const noexcept
{
   return _pimpl->FName;
}

ScriptParser::Args_t&  ScriptParser::GetArguments () const noexcept
{
   return _pimpl->Args;
}
