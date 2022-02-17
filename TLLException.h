#ifndef __TLLEXCEPTION__
#define __TLLEXCEPTION__

class /* AFX_EXT_CLASS*/ TLLException
:  public CException
{
private:

   CString  Msg;

protected:

   TLLException ()   = default;

public:

   TLLException (CString const& MsgArg);
   TLLException (LPCTSTR FormatArg, ... );
   TLLException (TLLException const& ) = default;

   virtual ~TLLException ()
   {
   }

   virtual BOOL GetErrorMessage (LPTSTR ErrorArg, UINT MaxErrorArg, PUINT HelpContextArg = nullptr) override;

   CString const& GetErrorMessage () const noexcept
   {
      return Msg;
   }
};

#endif
