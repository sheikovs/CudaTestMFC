#ifndef __CPPB_COMMON__
#define __CPPB_COMMON__

#include "cuda_runtime.h"
#include <cuda.h>
#include <nvrtc.h>
#include  <stdexcept>
#include "CommonDefs.h"
#include <string>
#include <sstream>

extern   void           HandleError (cudaError_t ErrArg, const char* FileArg, int LineArg, bool ThrowArg);
extern   void           CudaError   (CUresult    ErrArg, const char* FileArg, int LineArg, bool ThrowArg);

extern   void           __OnError  (const char* MsgArg);
extern   std::locale&   GetLocale ();

extern   void           __CheckDriverCall(CUresult RcArg, LPCTSTR FileArg, const int LineArg);
extern   void           __CheckNVTC (nvrtcResult RcArg);

extern   void				__AddLog (CString const& MsgArg);
extern   void				__AddLog (LPCTSTR FormatArg, ...);
extern   CString        _F (LPCTSTR FormatArg, ...);

template <typename T>
std::string TGetString (T ValArg)
{
   std::stringstream Ss;
   Ss.imbue(GetLocale ());
   Ss << ValArg;
   return Ss.str ();
}

template <typename T>
CString TGetCString (T ValArg)
{
   std::stringstream Ss;
   Ss.imbue(GetLocale ());
   Ss << ValArg;
   return CString (Ss.str ().c_str ());
}

///////////////////////////////////////////
//
// struct _CmpStrings
//
///////////////////////////////////////////

struct CmpCStrings
{
   bool operator ()(CString const LhsArg, CString const RhsArg) const noexcept
   {
      return LhsArg.CompareNoCase (RhsArg) < 0;
   }
};


#define __TO_CSTR(X)  TGetString (X).c_str ()

#pragma region Macros
//------------------------------------------------------------------

#define __CC(ERR)    (::HandleError (ERR, __FILE__, __LINE__, true ))
#define __CCE(ERR)   (::HandleError (ERR, __FILE__, __LINE__, false ))

#define __CD(ERR)    (::CudaError ( ERR, __FILE__, __LINE__, true ))
#define __CDE(ERR)   (::CudaError ( ERR, __FILE__, __LINE__, false ))


#define __CDC(RC)    (::__CheckDriverCall ((RC), __FILE__, __LINE__))
#define __CNC(RC)    (::__CheckNVTC ((RC)))

#ifndef __CATCH

#define __CATCH(FN) \
catch (std::exception const& Ex)    { FN(Ex.what ()); } \
catch (...)                         { FN("Unkown error"); }

#endif // !__CATCH

#ifndef __CATCH_RET

#define __CATCH_RET(FN) \
catch (std::exception const& Ex)    { return FN(Ex.what ()); } \
catch (...)                         { return FN("Unkown error"); }

#endif // !__CATCH_RET

#ifndef __NO_COPY
#define __NO_COPY(CNAME)                        \
   CNAME(CNAME const&)              = delete;   \
   CNAME operator = (CNAME const&)  = delete;

#endif // !__NO_COPY


#ifndef __MAKE_STATIC
#define __MAKE_STATIC(CNAME)                    \
   CNAME()              = delete;               \
   CNAME(CNAME const&)              = delete;   \
   CNAME operator = (CNAME const&)  = delete;

#endif // !__STATIC

#ifndef __STC
#define __STC(TYPE,VAL) static_cast<TYPE>(VAL)
#endif // !__STC

//------------------------------------------------------------------
#pragma endregion

#endif // !__CPPB_COMMON__
