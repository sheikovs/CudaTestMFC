// CTestDialog.cpp : implementation file
//

#include "pch.h"
#include "CudaTestMFC.h"
#include "afxdialogex.h"
#include "CTestDialog.h"
#include "ITest.h"
#include "Common.h"
#include <algorithm>

static constexpr int MAX_BLOCKS	= 65536;

// CTestDialog dialog

IMPLEMENT_DYNAMIC(CTestDialog, CDialogEx)

CTestDialog::CTestDialog(_ITest* TestPtrArg, CWnd* pParent /*=nullptr*/)
:	CDialogEx(IDD_TEST_DLG, pParent)
,	_test_ptr (TestPtrArg)
{
	if (_test_ptr) _test_ptr->SetImpl (this);
}

CTestDialog::~CTestDialog()
{
}

size_t	CTestDialog::GetCalcSize () const noexcept
{
	return _SizeSpin.GetPos () * _Mult;
}

unsigned int	CTestDialog::GetGridSize  () const noexcept
{
	return static_cast <unsigned int>(_GridSizeSpin.GetPos32 ());
}

unsigned int	CTestDialog::GetBlockSize () const noexcept
{
	return static_cast <unsigned int>(_BlockSizeSpin.GetPos () * _WarpSize);
}

void	CTestDialog::MakeHeader (ColumnsVc_t const& ColumnsArg)
{
	if (!_columns)
	{
		for (int i = 0; i < ColumnsArg.size (); ++i)
		{
			auto const&	[Name, Align, Width]	= ColumnsArg.at (i);
			_ListCtrl.InsertColumn (i, Name, Align, Width);
		}
		_columns	= ColumnsArg.size ();
	}
}

void	CTestDialog::AddRow (ItemsVc_t const& ItemsArg)
{
	if (_columns && !ItemsArg.empty ())
	{
		auto const	Columns (std::min <size_t> (_columns, ItemsArg.size ()));
		int			Item;

		for (int i = 0; i < Columns; ++i)
		{
			if (i == 0)
			{
				Item	= _ListCtrl.InsertItem (i, ItemsArg.at (i));
			}
			else
			{
				_ListCtrl.SetItemText (Item, i, ItemsArg.at (i));
			}
		}
	}
}

void CTestDialog::SetLog (CString const& MsgArg)
{
	_LogCtrl.SetWindowText (MsgArg);
}

void CTestDialog::AddLog (CString const& MsgArg, bool NewLineArg /*= true*/)
{
	CString  Tmp;

	_LogCtrl.GetWindowText (Tmp);

	if (NewLineArg && !Tmp.IsEmpty ())
	{
		Tmp.AppendFormat ("\r\n%s", MsgArg);
	}
	else
	{
		Tmp.Append (MsgArg);
	}

	SetLog (Tmp);
}

void  CTestDialog::AppendLog (LPCTSTR FormatArg, ...)
{
	va_list  Args;
	va_start (Args, FormatArg);

	CString  Str;

	Str.FormatV (FormatArg, Args);

	va_end (Args);

	AddLog (Str);
}

void CTestDialog::ClearLog ()
{
	_LogCtrl.SetWindowText (CString ());
}

void CTestDialog::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_RC_LIST, _ListCtrl);
	DDX_Control(pDX, IDC_RC_LOG, _LogCtrl);
	DDX_Control(pDX, IDC_LOOPS_EDIT, _LoopsEdit);
	DDX_Control(pDX, IDC_SIZE_SPIN, _SizeSpin);
	DDX_Control(pDX, IDC_GRID_SIZE_EDIT, _GridSizeEdit);
	DDX_Control(pDX, IDC_GRID_SIZE_SPIN, _GridSizeSpin);
	DDX_Control(pDX, IDC_BLOCK_SIZE_EDIT, _BlockSizeEdit);
	DDX_Control(pDX, IDC_BLOCK_SIZE_SPIN, _BlockSizeSpin);
	DDX_Control(pDX, IDC_THREADS, _ThreadsCount);
	DDX_Control(pDX, IDC_WARP_SIZE, _WarpSizeLabel);
	DDX_Control(pDX, IDC_BLOCK_THREADS, _BlockThreadsLabel);
	DDX_Control(pDX, IDC_SIZE_MULT_COMBO, _SizeMultCB);
	DDX_Control(pDX, IDC_MAX_BLOCKS, _MaxBlocks);
	DDX_Control(pDX, IDC_SIZE_GROUP, _SizeGroup);
}


BEGIN_MESSAGE_MAP(CTestDialog, CDialogEx)
	ON_BN_CLICKED(IDC_RUN_BTN, &CTestDialog::OnBnClickedRunBtn)
	ON_EN_CHANGE(IDC_LOOPS_EDIT, &CTestDialog::OnEnChangeLoopsEdit)
	ON_EN_CHANGE(IDC_GRID_SIZE_EDIT, &CTestDialog::OnEnChangeGridSizeEdit)
	ON_EN_CHANGE(IDC_BLOCK_SIZE_EDIT, &CTestDialog::OnEnChangeBlockSizeEdit)
	ON_EN_CHANGE(IDC_SIZE_EDIT, &CTestDialog::OnEnChangeSizeEdit)
	ON_CBN_SELCHANGE(IDC_SIZE_MULT_COMBO, &CTestDialog::OnCbnSelchangeSizeMultCombo)
	ON_BN_CLICKED(IDC_CLEAR_LOG, &CTestDialog::OnBnClickedClearLog)
END_MESSAGE_MAP()


// CTestDialog message handlers


void CTestDialog::OnBnClickedRunBtn()
{
	if (_test_ptr) _test_ptr->OnRun ();
}


void CTestDialog::OnEnChangeLoopsEdit()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialogEx::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}


BOOL CTestDialog::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	::setlocale(LC_ALL,     "en_US.UTF-8");
	::setlocale(LC_NUMERIC, "en_US.UTF-8");

	_ListCtrl.SetExtendedStyle(LVS_EX_FULLROWSELECT );

	if (_test_ptr)
	{
		_test_ptr->OnInit ();
		__InitControls    ();
	}

	_intialized	= true;
	__OnUpdate (true);

	return TRUE;  // return TRUE unless you set the focus to a control
					  // EXCEPTION: OCX Property Pages should return FALSE
}

void	CTestDialog::__InitControls ()
{
	constexpr int INIT_THREADS	 = 256;

	auto Props	= _test_ptr->GetCudaProperties ();

	_MpCount         = Props.multiProcessorCount;
	_MpMaxThreads    = Props.maxThreadsPerMultiProcessor;
	_BlockMaxThreads = Props.maxThreadsPerBlock;
	_WarpSize        = Props.warpSize;

	// Size
#ifdef DEBUG
	const	int	Size	= 33;
#else
	const	int	Size	= 1024;
#endif // DEBUG

	{
		_SizeSpin.SetRange (1, 10000);
		_SizeSpin.SetPos   (Size);

		for (int	Mult = 1; Mult	<= 1000000; Mult *= 10)
		{
			int	Idx	= _SizeMultCB.AddString (::TGetCString (Mult));
			_SizeMultCB.SetItemData (Idx, Mult);
		}

		_Mult	= 1;

		_SizeMultCB.SetCurSel(0);
	}

	_WarpSizeLabel.SetWindowText (::_F("x %i", _WarpSize));

	_MaxThreads	= _MpMaxThreads * _MpCount;

	// Block
	int	InitThreads (((Size + _WarpSize - 1) / _WarpSize) * _WarpSize);
	int	MaxRange	= _BlockMaxThreads / _WarpSize;
	int	BlockPos = InitThreads / _WarpSize;
	_BlockSizeSpin.SetRange (1, MaxRange);
	_BlockSizeSpin.SetPos   (BlockPos);
	int	BlockSize	= BlockPos * _WarpSize;
	_BlockThreadsLabel.SetWindowText (::_F("%i", BlockSize));

	// Grid Size: MAX_BLOCKS
	MaxRange	= _test_ptr->GetMaxBlocks (Size, BlockSize);
	_GridSizeSpin.SetRange32 (1, MaxRange);
	_GridSizeSpin.SetPos32   (MaxRange);

	_ThreadsCount.SetWindowText (::_F("%s", ::TGetCString (MaxRange * BlockSize)));
	_SizeGroup   .SetWindowText (::_F("Size [%s]", ::TGetCString (Size)));
}


void CTestDialog::OnEnChangeGridSizeEdit()
{
	__OnUpdate (false);
}

void CTestDialog::OnEnChangeBlockSizeEdit()
{
	__OnUpdate (true);
}

void	CTestDialog::__OnUpdate (bool CalcGridArg)
{
	if (_intialized)
	{
		int	Size			= static_cast <int>(GetCalcSize ());
		int	BlockPos		= _BlockSizeSpin.GetPos ();
		int	GridPos		= _GridSizeSpin. GetPos32 ();
		int	BlockSize	= BlockPos * _WarpSize;
		int	Threads		= BlockSize * GridPos;

		if (CalcGridArg)
		{
			int	MaxBlocks	= _test_ptr->GetMaxBlocks (Size, BlockSize);

			if (MaxBlocks < GridPos)
			{
				_GridSizeSpin. SetPos32 (MaxBlocks);
				Threads	= BlockSize * MaxBlocks;
			}

			_GridSizeSpin.SetRange32 (1, MaxBlocks);
			_MaxBlocks.SetWindowText (::_F("%s", ::TGetCString (MaxBlocks)));
		}

		_BlockThreadsLabel.SetWindowText (::_F("%s", ::TGetCString  (BlockSize)));
		_ThreadsCount.     SetWindowText (::_F("%s", ::TGetCString (Threads)));
		_SizeGroup        .SetWindowText (::_F("Size [%s]", ::TGetCString (Size)));
	}
}

void CTestDialog::OnEnChangeSizeEdit()
{
	__OnUpdate (true);
}


void CTestDialog::OnCbnSelchangeSizeMultCombo()
{
	if (_intialized)
	{
		int Idx	= _SizeMultCB. GetCurSel();
		_Mult		= _SizeMultCB.GetItemData (Idx);
		__OnUpdate (true);
	}
}


void CTestDialog::OnBnClickedClearLog()
{
	ClearLog ();
	_ListCtrl.DeleteAllItems ();
}
