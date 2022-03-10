#ifndef __COMMON_DEFS_H__
#define __COMMON_DEFS_H__

#ifndef MAKE_STATIC
#define MAKE_STATIC(CN)                      \
   private:                                  \
      CN ()                      = delete;   \
      CN (CN const&)             = delete;   \
      CN operator= (CN const&)   = delete;

#endif // !MAKE_STATIC

#ifndef TSWAP
#define  TSWAP   std::swap
#endif // !TSWAP

#ifndef TMOVE
#define  TMOVE   std::move
#endif // !TMOVE

#ifndef TFORWARD
#define TFORWARD  std::forward
#endif // !TFORWARD

#ifndef __STC
#define __STC(TYPE,VAL) static_cast<TYPE>(VAL)
#endif // !__STC

#ifndef AFX_EXT_CLASS
#define  AFX_EXT_CLASS
#endif // !AFX_EXT_CLASS

#endif // !__COMMON_DEFS_H__

