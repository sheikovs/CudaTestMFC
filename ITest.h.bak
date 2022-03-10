#ifndef __ITEST_H__
#define __ITEST_H__
#include "cuda_runtime.h"
#include <vector>

///////////////////////////////////////////
//
// struct _ITestData
//
///////////////////////////////////////////

struct _ITestData
{
   virtual  ~_ITestData ()
   {
   }

   virtual  size_t			GetCalcSize  () const noexcept   = 0;
   virtual  unsigned int	GetGridSize  () const noexcept   = 0;
   virtual  unsigned int	GetBlockSize () const noexcept   = 0;
};

///////////////////////////////////////////
//
// struct _ITestReport
//
///////////////////////////////////////////

struct _ITestReport
{
   virtual  ~_ITestReport ()
   {
   }

   using Column_t		= std::tuple <CString, int, int>;
   using ColumnsVc_t = std::vector <Column_t>;
   using ItemsVc_t	= std::vector <CString>;

   virtual  void	MakeHeader (ColumnsVc_t const& ColumnsArg)   = 0;
   virtual  void	AddRow     (ItemsVc_t   const& ItemsArg)     = 0;

   virtual  void  SetLog    (CString const& MsgArg)                           = 0;
   virtual  void  AddLog    (CString const& MsgArg, bool NewLineArg = true)   = 0;
   virtual  void  AppendLog (LPCTSTR FormatArg, ...)                          = 0;
   virtual  void  ClearLog  ()                                                = 0;
};

///////////////////////////////////////////
//
// struct _ITestImpl
//
///////////////////////////////////////////

struct _ITestImpl
:  public _ITestData
,  public _ITestReport
{
   virtual ~_ITestImpl ()
   {
   }
};

///////////////////////////////////////////
//
// struct _IGpu
//
///////////////////////////////////////////

struct _IGpu
{
   virtual ~_IGpu ()
   {
   }

   virtual  cudaDeviceProp GetCudaProperties () = 0;

};

///////////////////////////////////////////
//
// struct _ITest
//
///////////////////////////////////////////

class _ITest
:  public _ITestData
,  public _ITestReport
,  public _IGpu
{
public:

   using Column_t		= _ITestReport::Column_t;
   using ColumnsVc_t = _ITestReport::ColumnsVc_t;
   using ItemsVc_t	= _ITestReport::ItemsVc_t;

protected:

   _ITestImpl* _Pimpl = nullptr;

public:

   _ITest ()   = default;

   _ITest (_ITestImpl* ImplPtrArg)
   :  _Pimpl (ImplPtrArg)
   {
   }

   virtual ~_ITest ()
   {
   }

   void  SetImpl (_ITestImpl* ImplPtrArg)
   {
      _Pimpl = ImplPtrArg;
   }

   virtual  void           OnInit () = 0;
   virtual  void           OnRun  () = 0;

   virtual  size_t			GetCalcSize  () const noexcept override;
   virtual  unsigned int	GetGridSize  () const noexcept override;
   virtual  unsigned int	GetBlockSize () const noexcept override;

   virtual  void	MakeHeader (ColumnsVc_t const& ColumnsArg) override;
   virtual  void	AddRow     (ItemsVc_t   const& ItemsArg)   override;

   virtual  void  SetLog    (CString const& MsgArg) override;
   virtual  void  AddLog    (CString const& MsgArg, bool NewLineArg = true) override;
   virtual  void  AppendLog (LPCTSTR FormatArg, ...) override;
   virtual  void  ClearLog  () override;

   int   GetMaxBlocks (int SizeArg, int BlockSizeArg) const;

};

#endif // !__ITEST_H__

