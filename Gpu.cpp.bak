#include "pch.h"
#include "Common.h"
#include "Gpu.h"

static cudaDeviceProp   _DeviceProp;

static bool GetDeviceInfo (cudaDeviceProp& PropArg, int& DeviceIdArg)
{
   int   Count {};

   __CC (::cudaGetDeviceCount (&Count));

   if (Count > 0)
   {
      DeviceIdArg = 0;

      __CC (::cudaGetDeviceProperties (&PropArg, DeviceIdArg));

   }

   return Count > 0;
}

Gpu::Gpu  (bool InitArg)
{
   if (InitArg)
   {
      Init ();
   }
}

Gpu::~Gpu  ()
{
   if (_deviceId >= 0)
   {
      __CC (::cudaDeviceReset ());
   }
}

void  Gpu::Init ()
{
   if (_deviceId < 0)
   {
      if (::GetDeviceInfo (_DeviceProp, _deviceId))
      {
         __CC(::cudaSetDevice (_deviceId));

         if (_DeviceProp.canMapHostMemory)
         {
            __CC(::cudaSetDeviceFlags (cudaDeviceMapHost));
         }
      }
   }
}

cudaDeviceProp const&   Gpu::GetProperties () noexcept
{
   return _DeviceProp;
}
