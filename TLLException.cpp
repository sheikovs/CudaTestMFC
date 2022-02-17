//#include "stdafx.h"
#include "pch.h"
#include "TLLException.h"

#pragma warning( disable : 4793)

TLLException::TLLException (CString const& MsgArg)
:  Msg (MsgArg)
{

}

TLLException::TLLException (LPCTSTR FormatArg, ...)
{
   va_list  Args;
   va_start (Args, FormatArg);

   Msg.FormatV (FormatArg, Args);
   
   va_end (Args);
}

BOOL TLLException::GetErrorMessage (LPTSTR ErrorArg, UINT MaxErrorArg, PUINT HelpContextArg /* = nullptr*/)
{
   BOOL const Result = ErrorArg != nullptr && MaxErrorArg > 0 ? TRUE : FALSE;

   if (Result)
   {
      ::_tcsncpy_s (ErrorArg, MaxErrorArg, (LPCTSTR)Msg, _TRUNCATE);
   }
   
   return Result;
}
