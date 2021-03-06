#ifndef __TESTS_H__
#define __TESTS_H__

#include "CTestDialog.h"
#include "CDRHelpers.h"

class SAXPYTest
:  public _ITest
{
public:

   using Device_t = CDRH::Device;

   Device_t _device;

public:

   SAXPYTest ();

   virtual  void  OnInit () override;
   virtual  void  OnRun  () override;

   virtual  cudaDeviceProp GetCudaProperties () override;
};

class BasicsTest
:  public _ITest
{
public:

public:

   BasicsTest ();

   virtual ~BasicsTest ()
   {
   }

   virtual  void  OnInit () override;
   virtual  void  OnRun  () override;

   virtual  cudaDeviceProp GetCudaProperties () override;

};

#endif // !__TESTS_H__

