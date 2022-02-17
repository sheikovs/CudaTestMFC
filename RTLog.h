#ifndef __RT_LOG_H__
#define __RT_LOG_H__
#include "CommonDefs.h"

class RtLog
{
   MAKE_STATIC (RtLog)

public:

   static void SetLogWidget (QWidget* PtrArg);

   static void Log (QString const& MsgArg);
};


extern void  RtLog  (const char* FormatArg, ... );
extern void  RtLog  (QString const& MsgArg);

#endif // !__RT_LOG_H__

