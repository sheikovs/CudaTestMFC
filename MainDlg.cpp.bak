
// MainDlg.cpp : implementation file
//

#include "pch.h"
#include "framework.h"
#include "CudaTestMFC.h"
#include "MainDlg.h"
#include "afxdialogex.h"
#include "CFuncTestDlg.h"
#include <locale.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

namespace NVRTCH
{
	void  ResetDeviceCache ();
}
extern void  SaxpyRun_entry   ();
extern void  BasicsTest_entry ();

// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMainDlg dialog
static CMainDlg*	_ThisDlgPtr = nullptr;


CMainDlg::CMainDlg(CWnd* pParent /*=nullptr*/)
:	CDialogEx(IDD_CUDATESTMFC_DIALOG, pParent)
{
	m_hIcon		= AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	_ThisDlgPtr	= this;
}

CMainDlg::~CMainDlg ()
{
	NVRTCH::ResetDeviceCache ();
}

void CMainDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CMainDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_SAXPY_RUN, &CMainDlg::OnBnClickedSaxpyRun)
	ON_BN_CLICKED(IDC_BASIC_TESTS, &CMainDlg::OnBnClickedBasicTests)
	ON_BN_CLICKED(IDC_BTN_FUNC_TEST, &CMainDlg::OnBnClickedBtnFuncTest)
END_MESSAGE_MAP()


// CMainDlg message handlers

BOOL CMainDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here

	::setlocale(LC_ALL,     "en_US.UTF-8");
	::setlocale(LC_NUMERIC, "en_US.UTF-8");

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CMainDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CMainDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CMainDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void CMainDlg::SetLog (CString const& MsgArg)
{
	GetDlgItem (IDC_LOG)->SetWindowText (MsgArg);
}

void CMainDlg::AddLog (CString const& MsgArg)
{
	CString  t;

	GetDlgItem (IDC_LOG)->GetWindowText (t);

	t.Append (MsgArg);

	SetLog (t);
}

void CMainDlg::ClearLog ()
{
	GetDlgItem (IDC_LOG)->SetWindowText ("");
}


void  __AddLog (CString const& MsgArg)
{
	_ThisDlgPtr->AddLog (MsgArg);
}

void  __AddLog (LPCTSTR FormatArg, ...)
{
	va_list  Args;
	va_start (Args, FormatArg);

	CString  Str;

	Str.FormatV (FormatArg, Args);

	va_end (Args);

	_ThisDlgPtr->AddLog (Str);
}

CString _F (LPCTSTR FormatArg, ...)
{
	va_list  Args;
	va_start (Args, FormatArg);

	CString  Str;

	Str.FormatV (FormatArg, Args);

	va_end (Args);

	return Str;
}


void CMainDlg::OnBnClickedSaxpyRun()
{
	::SaxpyRun_entry ();
}


void CMainDlg::OnBnClickedBasicTests()
{
	::BasicsTest_entry ();
}


void CMainDlg::OnBnClickedBtnFuncTest()
{
	CFuncTestDlg	Dlg;

	Dlg.DoModal ();
}
