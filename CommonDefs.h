#ifndef __COMMON_DEFS_H__
#define __COMMON_DEFS_H__

#ifndef MAKE_STATIC
#define MAKE_STATIC(CN)                      \
   private:                                  \
      CN ()                      = delete;   \
      CN (CN const&)             = delete;   \
      CN operator= (CN const&)   = delete;

#endif // !MAKE_STATIC

#endif // !__COMMON_DEFS_H__

