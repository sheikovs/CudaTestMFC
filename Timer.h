#ifndef __TIMER__
#define __TIMER__

#include <iostream>
#include <iomanip>
#include <chrono>

///////////////////////////////////////////////////
//
// class Timer
//
///////////////////////////////////////////////////

class Timer
{
   using Clock_t  = std::chrono::steady_clock;
   using Tp_t     = std::chrono::steady_clock::time_point;

   int64_t  _dur {};
   Tp_t     _start;

public:

   Timer ()
   : _start (Clock_t::now ())
   {
   }

   int64_t  get ()
   {
      if (!_dur)
      {
         auto const e   = Clock_t::now ();
         _dur  = std::chrono::duration_cast<std::chrono::microseconds>(e - _start).count();;
      }

      return _dur;
   }
};

#endif // !__TIMER__

