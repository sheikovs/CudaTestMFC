#pragma once
#include "afxdialogex.h"
#include "NvRtcProgram.h"
#include "CDRHelpers.h"

struct _Parser;

// CFuncTestDlg dialog

class CFuncTestDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CFuncTestDlg)

	using Device_t		= CDRH::Device;
	using Program_t   = NVRTCH_1::Program;
	using Kernel_t    = NVRTCH_1::Kernel;

private:

	Device_t		_Device;
	Program_t	_Program;
	_Parser*		_ParserPtr {};

	int	_SizeVar		= 1;
	int	_IntVar		= 1;
	float _FloatVar	= 1.0f;
   float _GV_1       = 12.25;
   float _GV_2       = 111.75;
	float _ResultVar	= 0.0f;
	bool	_IsLinked   = false;

public:
	CFuncTestDlg(CWnd* pParent = nullptr);   // standard constructor
	virtual ~CFuncTestDlg();

private:

	void	__AddLog (CString const& MsgArg, bool NextLineArg = true);
	void	__SetLog (CString const& MsgArg);

	void	__Run    ();
	void	__Link   ();

	void	__UpdateSizeVar   (bool GetArg);
	void	__UpdateIntVar    (bool GetArg);
	void	__UpdateFloatVar  (bool GetArg);
	void	__UpdateGV_1      (bool GetArg);
	void	__UpdateGV_2      (bool GetArg);
	void	__UpdateResultVar (bool GetArg);

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_PARSER };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()
public:
	CEdit _FuncEdit;
	CButton _BtnCompile;
	virtual BOOL OnInitDialog();
	afx_msg void OnEnChangeFuncEdit();
	afx_msg void OnBnClickedBtnCompile();
	CEdit _ParserLog;
	afx_msg void OnBnClickedClearParserLog();
	CButton _BtnRunScript;
	afx_msg void OnBnClickedBtnScriptRun();
	CEdit _SizeEdit;
	CEdit _EditIntVal;
	CEdit _EditFloatVal;
	CEdit _EditResult;
	CButton _BtnUpdateVars;
	afx_msg void OnBnClickedBtnUpdateVars();
   CEdit _EditGV_1;
   CEdit _EditGV_2;
};
