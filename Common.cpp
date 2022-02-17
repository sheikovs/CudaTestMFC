#include "pch.h"
#include "Common.h"
#include <locale>
#include <clocale>

static CString  __CreateErrMsg (
   cudaError_t    ErrArg
,  const char*    FileArg
,  int            LineArg
)
{
   return ::_F ("[%s]] in %s  at line %i"
            ,  ::cudaGetErrorString(ErrArg)
            ,  FileArg
            ,  LineArg
            );
}

void HandleError (cudaError_t ErrArg, const char* FileArg, int LineArg, bool ThrowArg) 
{
   if (ErrArg != cudaSuccess) 
   {
      auto const  Msg = ::__CreateErrMsg (ErrArg, FileArg, LineArg);

      if (ThrowArg)
      {
         throw std::runtime_error ((LPCTSTR)Msg);
      }
      else
      {
         ::AfxMessageBox (Msg, MB_OK | MB_ICONERROR);
      }
   }
}

void  CudaError (CUresult ErrArg, const char* FileArg, int LineArg, bool ThrowArg)
{
   if (ErrArg != CUDA_SUCCESS) 
   {
      const char* ErrStr  = nullptr;

      ::cuGetErrorString (ErrArg, &ErrStr);

      auto const  Msg = ::_F (
         "Driver API error = %04d [%s] in %s  at line %i"
         ,  ErrArg
         ,  ErrStr
         ,  FileArg
         ,  LineArg
         );

      if (ThrowArg)
      {
         throw std::runtime_error ((LPCTSTR)Msg);
      }
      else
      {
         ::AfxMessageBox (Msg, MB_OK | MB_ICONERROR);
      }
   }
}

void  __CheckDriverCall (CUresult RcArg, LPCTSTR FileArg, const int LineArg) 
{
   if (CUDA_SUCCESS != RcArg) 
   {
      LPCTSTR     ErrStr (nullptr);

      ::cuGetErrorString(RcArg, &ErrStr);

      auto const  Msg (
         ::_F("Driver API error = %04d [%s] from file <%s>, line %i"
            ,  RcArg
            ,  ErrStr
            ,  FileArg
            ,  LineArg
         ));

      throw std::runtime_error ((LPCTSTR)Msg);
   }
}

void  __CheckNVTC (nvrtcResult RcArg)
{
   if (RcArg != NVRTC_SUCCESS)
   {
      auto const  Msg (::_F("Error: %s", ::nvrtcGetErrorString(RcArg)));
      throw std::runtime_error ((LPCTSTR)Msg);
   }
}

void  __OnError (const char* MsgArg)
{
   ::AfxMessageBox (MsgArg, MB_OK | MB_ICONERROR);
}


std::locale&   GetLocale ()
{
   static std::locale   Locale (std::setlocale(LC_ALL, "en_US.UTF-8"));

   return Locale;
}
