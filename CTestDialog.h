#pragma once
#include "afxdialogex.h"
#include "ITest.h"

// CTestDialog dialog

class CTestDialog : public CDialogEx, public _ITestImpl
{
	DECLARE_DYNAMIC(CTestDialog)

public:

	using Column_t		= _ITestReport::Column_t;
	using ColumnsVc_t = _ITestReport::ColumnsVc_t;
	using ItemsVc_t	= _ITestReport::ItemsVc_t;

private:

	_ITest*	_test_ptr	= nullptr;
	size_t	_columns {};
	bool		_intialized {};

public:

	size_t	_Mult	= 1;
	int		_MpCount {};		// (MP) Multiprocessors deviceProp.multiProcessorCount
	int		_Cores {};			// total cores
	int		_MpMaxThreads;		// deviceProp.maxThreadsPerMultiProcessor
	int		_BlockMaxThreads;	// deviceProp.maxThreadsPerBlock
	int		_WarpSize;			// deviceProp.warpSize
	int		_MaxThreads;		// deviceProp.warpSize

public:
	CTestDialog(_ITest* TestPtrArg, CWnd* pParent = nullptr);   // standard constructor
	virtual ~CTestDialog();

	virtual size_t			GetCalcSize  () const noexcept override;
	virtual unsigned int	GetGridSize  () const noexcept override;
	virtual unsigned int	GetBlockSize () const noexcept override;

	virtual void	MakeHeader (ColumnsVc_t const& ColumnsArg) override;
	virtual void	AddRow     (ItemsVc_t   const& ItemsArg) override;


	virtual void SetLog    (CString const& MsgArg) override;
	virtual void AddLog    (CString const& MsgArg, bool NewLineArg = true) override;
	virtual void AppendLog (LPCTSTR FormatArg, ...) override;
	virtual void ClearLog () override;

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_TEST_DLG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	void	__InitControls ();
	void	__OnUpdate (bool CalcGridArg);

	DECLARE_MESSAGE_MAP()

public:

	CListCtrl	_ListCtrl;
	CEdit			_LogCtrl;
	CEdit			_LoopsEdit;
	afx_msg void OnBnClickedRunBtn();
	afx_msg void OnEnChangeLoopsEdit();
	virtual BOOL OnInitDialog();
	CSpinButtonCtrl _SizeSpin;
	CEdit _GridSizeEdit;
	CSpinButtonCtrl _GridSizeSpin;
	CEdit _BlockSizeEdit;
	CSpinButtonCtrl _BlockSizeSpin;
	CStatic _ThreadsCount;
	afx_msg void OnEnChangeGridSizeEdit();
	afx_msg void OnEnChangeBlockSizeEdit();
	CStatic _WarpSizeLabel;
	CStatic _BlockThreadsLabel;
	afx_msg void OnEnChangeSizeEdit();
	CComboBox _SizeMultCB;
	afx_msg void OnCbnSelchangeSizeMultCombo();
	CStatic _MaxBlocks;
	CStatic _SizeGroup;
	afx_msg void OnBnClickedClearLog();
};
