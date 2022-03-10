// CFuncTestDlg.cpp : implementation file
//

#include "pch.h"
#include "CudaTestMFC.h"
#include "afxdialogex.h"
#include "CFuncTestDlg.h"
#include "Parser.h"
#include "Common.h"
#include "Timer.h"
#include "MemHelper.cuh"
#include "FuncTest.cuh"

// CFuncTestDlg dialog

IMPLEMENT_DYNAMIC(CFuncTestDlg, CDialogEx)

CFuncTestDlg::CFuncTestDlg(CWnd* pParent /*=nullptr*/)
:	CDialogEx(IDD_PARSER, pParent)
,	_Program ("main_prog")
{
}

CFuncTestDlg::~CFuncTestDlg()
{
	delete _ParserPtr;
}

void CFuncTestDlg::DoDataExchange(CDataExchange* pDX)
{
   CDialogEx::DoDataExchange(pDX);
   DDX_Control(pDX, IDC_FUNC_EDIT, _FuncEdit);
   DDX_Control(pDX, IDC_BTN_COMPILE, _BtnCompile);
   DDX_Control(pDX, IDC_PARSER_LOG, _ParserLog);
   DDX_Control(pDX, IDC_BTN_SCRIPT_RUN, _BtnRunScript);
   DDX_Control(pDX, IDC_EDIT_SIZE, _SizeEdit);
   DDX_Control(pDX, IDC_EDIT_INTVAL, _EditIntVal);
   DDX_Control(pDX, IDC_EDIT_FLOATVAL, _EditFloatVal);
   DDX_Control(pDX, IDC_EDIT_RESULT, _EditResult);
   DDX_Control(pDX, IDC_BTN_UPDATE_VARS, _BtnUpdateVars);
   DDX_Control(pDX, IDC_EDIT_GB_1, _EditGV_1);
   DDX_Control(pDX, IDC_EDIT_GV_2, _EditGV_2);
}


BEGIN_MESSAGE_MAP(CFuncTestDlg, CDialogEx)
	ON_EN_CHANGE(IDC_FUNC_EDIT, &CFuncTestDlg::OnEnChangeFuncEdit)
	ON_BN_CLICKED(IDC_BTN_COMPILE, &CFuncTestDlg::OnBnClickedBtnCompile)
	ON_BN_CLICKED(IDC_CLEAR_PARSER_LOG, &CFuncTestDlg::OnBnClickedClearParserLog)
	ON_BN_CLICKED(IDC_BTN_SCRIPT_RUN, &CFuncTestDlg::OnBnClickedBtnScriptRun)
	ON_BN_CLICKED(IDC_BTN_UPDATE_VARS, &CFuncTestDlg::OnBnClickedBtnUpdateVars)
END_MESSAGE_MAP()


// CFuncTestDlg message handlers


BOOL CFuncTestDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// TODO:  Add extra initialization here

	_BtnCompile  .EnableWindow (FALSE);
	_BtnRunScript.EnableWindow (FALSE);

	try
	{
		_Device	= Device_t::SetDevice ();

		auto const&	DProp (_Device.GetProperties ());

		_Program.AddOption (::_F("--gpu-architecture=sm_%d%d", DProp.major, DProp.minor));
		_Program.AddOption ("--relocatable-device-code=true");
		_Program.AddOption ("--extra-device-vectorization");
		_Program.AddOption ("--std=c++17");
		_Program.AddOption ("--include-path=C:/Temp/cudafiles/");

		//_Program.AddHeader     ("CommonFunc.cu");
		//_Program.AddHeaderPath ("C:/tmp/cudafiles/");

		__UpdateSizeVar   (false);
		__UpdateIntVar    (false);
		__UpdateFloatVar  (false);
      __UpdateGV_1      (false);
      __UpdateGV_2      (false);
		__UpdateResultVar (false);

	}
	__CATCH(__OnError);

	return TRUE;  // return TRUE unless you set the focus to a control
					  // EXCEPTION: OCX Property Pages should return FALSE
}


void CFuncTestDlg::OnEnChangeFuncEdit()
{
	CString	Input;

	_FuncEdit.GetWindowText  (Input);
	_BtnCompile.EnableWindow (Input.Trim ().GetLength () >= 4);
}


void CFuncTestDlg::OnBnClickedBtnCompile()
{
	CString	SourceData, ParsedData;
	Timer    Tm;

	_FuncEdit.GetWindowText (SourceData);

	_BtnRunScript.EnableWindow (FALSE);

	if (!_ParserPtr) _ParserPtr = new _Parser;

	if (_ParserPtr->Parse (SourceData, ParsedData))
	{
		__AddLog (::_F("Parsing Succeeded: \r\n%s", ParsedData));

		auto LOnError = [this] (CString const& MsgArg)
		{
			__AddLog (::_F("Compilation Failed: [%s]", MsgArg));

			if (auto Log = _Program.GetLog (); !Log.Trim ().IsEmpty ())
			{
				__AddLog (Log);
			}
		};

		try
		{
			_Program.SetSource (ParsedData);
			_IsLinked	= false;

			if (_Program.Compile ())
			{
				auto const	Et  (Tm.get ());
				float const Etf (Et / 1000000.0f);

				__AddLog (::_F("Compilation Succeeded (%.4f sec.)\r\n", Etf));

				_BtnRunScript.EnableWindow (TRUE);
			}
			else
			{
				LOnError ("Unknow error");
			}
		}
		__CATCH (LOnError);
	}

}

void	CFuncTestDlg::__AddLog (CString const& MsgArg, bool NextLineArg /*= true*/)
{
	CString Tmp;
	_ParserLog.GetWindowText (Tmp);
	if (Tmp.IsEmpty ())
	{
		Tmp	= MsgArg;
	}
	else
	{
		Tmp.AppendFormat ("%s%s", NextLineArg ? "\r\n" : "", MsgArg);
	}
	__SetLog (Tmp);
}

void	CFuncTestDlg::__SetLog (CString const& MsgArg)
{
	_ParserLog.SetWindowText (MsgArg);
}



void CFuncTestDlg::OnBnClickedClearParserLog()
{
	__SetLog ("");
}

void CFuncTestDlg::OnBnClickedBtnScriptRun ()
{
	Timer	Tm;

	__Run ();

	auto const Ms	= Tm.get ();

	__AddLog (::_F("Run time: (%.4f) sec", static_cast <float>(Ms / 1000000.0f)));
}

extern __device__ DFunc_t  __MultFunc;

void	CFuncTestDlg::__Run ()
{
	using InArgs_t    = NVRTCH_1::KernelArgs;
	using OutArgs_t	= std::vector <size_t>;

	auto LOnError	= [this] (CString const& MsgArg)
	{
		__AddLog (::_F("Error [%s]", MsgArg));
	};

	try
	{
		__Link ();

		auto const&	FName (_Parser::FUNC_NAME);
		Kernel_t    KTmp  (_Program.GetKernel (FName));

		auto&			Args  (_ParserPtr->GetArguments ());
		auto const  Count (Args.size ());

		InArgs_t		InArgs;
		OutArgs_t	OutArgs;

		for (size_t i = 0; i < Count; ++i)
		{
			auto& Arg	= *Args [i];
			float	Val	{};

			if (Arg.IsKeyWord ("kw_Size"))
			{
				Arg.SetValue (_SizeVar);

				__AddLog (::_F("\t%s = %i", Arg.GetName (), Arg.GetIntValue ()));
				InArgs.Add (Arg.GetIntValue ());
			}
			else if (Arg.IsKeyWord ("kw_ValInt"))
			{
				Arg.SetValue (_IntVar);
				__AddLog (::_F("\t%s = %i", Arg.GetName (), Arg.GetIntValue ()));
				InArgs.Add (Arg.GetIntValue ());
			}
			else if (Arg.IsKeyWord ("kw_ValFloat"))
			{
				Arg.SetValue (_FloatVar);
				__AddLog (::_F("\t%s = %f", Arg.GetName (), Arg.GetFloatValue ()));
				InArgs.Add (Arg.GetFloatValue ());
			}
			else if (Arg.IsKeyWord ("kw_Result"))
			{
				Arg.SetValue (0.0f);
				__AddLog (::_F("\t%s = %f", Arg.GetName (), Arg.GetFloatValue ()));
				InArgs.AddPtr (&Arg._FloatVal);
			}
			else if (Arg.IsKeyWord ("kw_BinFunc"))
			{
				DFunc_t  HOp	= nullptr;
				::GetBinFunc (HOp);
				InArgs.Add (HOp);
			}
			else
			{
				throw std::runtime_error ((LPCTSTR)::_F("Invalid argument [%s]", Arg.GetName ()));
			}

			if (Arg.IsRW ()) OutArgs.push_back (i);
		}

		dim3     Grid  (1);
		dim3     Block (1);

		KTmp.Execute (Grid, Block, InArgs.get ());

		Device_t::Synchronize ();

		if (!OutArgs.empty ())
		{
			CString	Report ("\r\nOutput:\r\n");

			for (size_t i = 0; i < OutArgs.size (); ++i)
			{
				auto const	Idx    (OutArgs.at (i));

				auto&			Arg     (*Args [Idx]); 

				if (Arg.IsKeyWord ("kw_Result"))
				{
					auto&		OutArg (InArgs [Idx]);
					size_t	ArgSize {};

					if (float* ValPtr = OutArg.TGetHostPtr <float> (ArgSize); ValPtr)
					{
						_ResultVar	= *ValPtr;
						Report.AppendFormat ("\t%s = %s\r\n", Arg.GetName (), ::TGetCString (_ResultVar));
						__UpdateResultVar (false);
					}
				}
			}
			__AddLog (Report);
		}
	}
	__CATCH (LOnError);
}

void	CFuncTestDlg::__Link ()
{
#ifdef DEBUG
	CString const	LibPath ("C:\\Temp\\cudafiles\\Debug\\CommonFunc.fatbin");
#else	
	CString const	LibPath ("C:\\Temp\\cudafiles\\Release\\CommonFunc.fatbin");
#endif // DEBUG

	if (_IsLinked)
	{
		_Program.AddLibrary (LibPath);

		_Program.Link ();

		{
			CUdeviceptr	Ptr  {};
			size_t      Size {};

			if (_Program.GetGlobal ("GlobalVal_1", Ptr, Size))
			{
			}
		}

		_IsLinked	= true;
	}
}

void CFuncTestDlg::OnBnClickedBtnUpdateVars ()
{
	__UpdateSizeVar   (true);
	__UpdateIntVar    (true);
	__UpdateFloatVar  (true);
   __UpdateGV_1      (true);
   __UpdateGV_2      (true);
	__UpdateResultVar (true);
}

void	CFuncTestDlg::__UpdateSizeVar (bool GetArg)
{
	CString	Tmp;

	if (GetArg)
	{
		_SizeEdit.GetWindowText (Tmp);
		_SizeVar	= ::atoi (Tmp);
	}

	Tmp.Format ("%i", _SizeVar);

	_SizeEdit.SetWindowText (Tmp);
}

void	CFuncTestDlg::__UpdateIntVar (bool GetArg)
{
	CString	Tmp;

	if (GetArg)
	{
		_EditIntVal.GetWindowText (Tmp);
		_IntVar	= ::atoi (Tmp);
	}

	Tmp.Format ("%i", _IntVar);

	_EditIntVal.SetWindowText (Tmp);
}

void	CFuncTestDlg::__UpdateFloatVar (bool GetArg)
{
	CString	Tmp;

	if (GetArg)
	{
		_EditFloatVal.GetWindowText (Tmp);
		_FloatVar	= ::atof (Tmp);
	}

	Tmp.Format ("%.4f", _FloatVar);

	_EditFloatVal.SetWindowText (Tmp);
}

void	CFuncTestDlg::__UpdateGV_1 (bool GetArg)
{
	CString	Tmp;

	if (GetArg)
	{
		_EditGV_1.GetWindowText (Tmp);
		_GV_1	= ::atof (Tmp);
	}

	Tmp.Format ("%.4f", _GV_1);

	_EditGV_1.SetWindowText (Tmp);
}

void	CFuncTestDlg::__UpdateGV_2 (bool GetArg)
{
	CString	Tmp;

	if (GetArg)
	{
		_EditGV_2.GetWindowText (Tmp);
		_GV_2	= ::atof (Tmp);
	}

	Tmp.Format ("%.4f", _GV_2);

	_EditGV_2.SetWindowText (Tmp);
}

void	CFuncTestDlg::__UpdateResultVar (bool GetArg)
{
	CString	Tmp;

	if (GetArg)
	{
		_EditResult.GetWindowText (Tmp);
		_ResultVar	= ::atof (Tmp);
	}

	Tmp.Format ("%.4f", _ResultVar);

	_EditResult.SetWindowText (Tmp);
}