
// MainDlg.h : header file
//

#pragma once

// CMainDlg dialog
class CMainDlg : public CDialogEx
{
// Construction
public:

	CMainDlg (CWnd* pParent = nullptr);	// standard constructor

	virtual	~CMainDlg ();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_CUDATESTMFC_DIALOG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support

public:

	void SetLog   (CString const& MsgArg);
	void AddLog   (CString const& MsgArg);
	void ClearLog ();


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedSaxpyRun();
	afx_msg void OnBnClickedBasicTests();
	afx_msg void OnBnClickedBtnFuncTest();
};
