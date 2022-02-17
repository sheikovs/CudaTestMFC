#ifndef __GPU_H__
#define __GPU_H__

struct cudaDeviceProp;

class Gpu
{
   inline static int _deviceId   = -1;

public:

   Gpu  (bool InitArg);
   ~Gpu ();

   static int   GetId () noexcept
   {
      return _deviceId;
   }

   static cudaDeviceProp const&   GetProperties () noexcept;

   static bool Exists () noexcept
   {
      return _deviceId >= 0;
   }

   static void  Init ();
};

#endif // !__GPU_H__

