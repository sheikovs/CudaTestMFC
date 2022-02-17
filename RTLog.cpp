#include <QWidget>
#include <QString>
#include "RTLog.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

struct RtLogImpl
{
   inline static QWidget* _ptr = nullptr;

   static void  Log (QString const& MsgArg)
   {
      if (_ptr)
      {
         QMetaObject::invokeMethod (_ptr, "OnLog", Q_ARG(QString, MsgArg));
      }
   }
};

void  RtLog::SetLogWidget (QWidget* PtrArg)
{
   RtLogImpl::_ptr = PtrArg;
}

void  RtLog::Log (QString const& MsgArg)
{
   RtLogImpl::Log (MsgArg);
}

void  RtLog  (const char* FormatArg, ... )
{
   const size_t   B_SIZE  = 1024;
   char           Buffer [B_SIZE];
   va_list        Args;

   va_start (Args, FormatArg);

   ::vsprintf (Buffer, FormatArg, Args);

   va_end (Args);

   RtLogImpl::Log (QString (Buffer));
}

void  RtLog  (QString const& MsgArg)
{
   RtLogImpl::Log (MsgArg);
}