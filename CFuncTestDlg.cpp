// CFuncTestDlg.cpp : implementation file
//

#include "pch.h"
#include "CudaTestMFC.h"
#include "afxdialogex.h"
#include "CFuncTestDlg.h"
#include "Parser.h"
#include "Common.h"
#include "NvRtcHelpers.h"
#include "Timer.h"
#include "MemHelper.cuh"


// CFuncTestDlg dialog

IMPLEMENT_DYNAMIC(CFuncTestDlg, CDialogEx)

CFuncTestDlg::CFuncTestDlg(CWnd* pParent /*=nullptr*/)
: CDialogEx(IDD_PARSER, pParent)
{

}

CFuncTestDlg::~CFuncTestDlg()
{
	delete _parser_ptr;
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
		_device. Init (0);
		_program.Init (_device);

		_program.AddOption ("--relocatable-device-code=true");
		_program.AddOption ("--extra-device-vectorization");
		_program.AddOption ("--std=c++17");
		_program.AddOption ("--include-path=C:/Temp/cudafiles/");

		//_program.AddHeader     ("CommonFunc.cu");
		//_program.AddHeaderPath ("C:/tmp/cudafiles/");

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

	_FuncEdit.GetWindowText (Input);
	_BtnCompile.EnableWindow (Input.Trim ().GetLength () >= 4);
}


void CFuncTestDlg::OnBnClickedBtnCompile()
{
	CString	Input, Out;

	Timer    Tm;

	_FuncEdit.GetWindowText (Input);

	_BtnRunScript.EnableWindow (FALSE);

	if (!_parser_ptr) _parser_ptr = new _Parser;

	if (_parser_ptr->Parse (Input, Out))
	{
		__AddLog (::_F("Parsing Succeeded: \r\n%s", Out));

		auto LOnError = [this] (CString const& MsgArg)
		{
			__AddLog (::_F("Compilation Failed: [%s]", MsgArg));

			if (auto Log = _program.GetLog (); !Log.Trim ().IsEmpty ())
			{
				__AddLog (Log);
			}
		};

		try
		{
			if (_program.Compile (Out, "Test_Func"))
			{
				auto const Et = Tm.get ();

				float const Etf	= Et / 1000000.0f;

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

void	CFuncTestDlg::__Run ()
{
	using Ptr_t			= void*;
	using ArgPx_t		= std::unique_ptr <Ptr_t []>;
	using DMem_t		= TCuDeviceMem    <float>;
	using DMemPx_t		= std::unique_ptr <DMem_t []>;
	using HMem_t		= THostMem <float>;
	using HMemPx_t		= std::unique_ptr <HMem_t []>;
	using OutArgs_t	= std::vector <size_t>;

	auto LOnError	= [this] (CString const& MsgArg)
	{
		__AddLog (::_F("Error [%s]", MsgArg));
	};

	try
	{
#ifdef DEBUG
		CString const	LibPath ("C:\\Temp\\cudafiles\\Debug\\CommonFunc.fatbin");
#else	
		CString const	LibPath ("C:\\Temp\\cudafiles\\Release\\CommonFunc.fatbin");
#endif // DEBUG
      //_device.Reset ();

		_device.AddLibrary (LibPath);
		_device.Add (_program, "CudaEntry.cubin");

		void*		CubinPtr = nullptr;
		size_t	CubinSize {};

		_device.LoadData (CubinPtr, CubinSize);

		{
			CUdeviceptr	Ptr  {};
			size_t      Size {};

			if (_device.GetGlobal (Ptr, Size, "GlobalVal_1"))
			{
            __CDC (cuMemcpyHtoD (Ptr, &_GV_1, sizeof (_GV_1)));
			}

			if (_device.GetGlobal (Ptr, Size, "GlobalVal_2"))
			{
            __CDC (cuMemcpyHtoD (Ptr, &_GV_2, sizeof (_GV_2)));
			}
		}

		auto const&	FName (_Parser::FUNC_NAME);
		Kernel_t    KTmp (_device, FName);

		auto&			Args  (_parser_ptr->GetArguments ());
		auto const  Count (Args.size ());

		DMemPx_t		DMemPx (new DMem_t [Count]);
		ArgPx_t		ArgPx  (new Ptr_t  [Count]);
		OutArgs_t	OutArgs;

		for (size_t i = 0; i < Count; ++i)
		{
			auto& Arg	= *Args [i];
			float	Val	{};

			if (Arg._var_name.CompareNoCase ("kw_Size") == 0)
			{
				Arg.SetValue (_SizeVar);

				__AddLog (::_F("\t%s = %i", Arg._name, Arg._i_val));
				ArgPx  [i]	= &(Arg._i_val);
			}
			else if (Arg._var_name.CompareNoCase ("kw_ValInt") == 0)
			{
				Arg.SetValue (_IntVar);
				__AddLog (::_F("\t%s = %i", Arg._name, Arg._i_val));
				ArgPx  [i]	= &(Arg._i_val);
			}
			else if (Arg._var_name.CompareNoCase ("kw_ValFloat") == 0)
			{
				Arg.SetValue (_FloatVar);
				__AddLog (::_F("\t%s = %f", Arg._name, Arg._f_val));
				ArgPx  [i]	= &(Arg._f_val);
			}
			else if (Arg._var_name.CompareNoCase ("kw_Result") == 0)
			{
				Arg.SetValue (0.0f);
				__AddLog (::_F("\t%s = %f", Arg._name, Arg._f_val));
				DMemPx [i]	= Arg._f_val;
				ArgPx  [i]	= &DMemPx [i]._ptr;
			}
			else
			{
				throw std::runtime_error ((LPCTSTR)::_F("Invalid argument [%s]", Arg._name));
			}

			if (Arg.IsRW ()) OutArgs.push_back (i);
		}

		dim3     Grid  (1);
		dim3     Block (1);

		KTmp.Execute (Grid, Block, ArgPx.get ());

		_device.Synchronize ();

		if (!OutArgs.empty ())
		{
			CString	Report ("\r\nOutput:\r\n");
			HMemPx_t	HMemPx (new HMem_t [OutArgs.size ()]);

			for (size_t i = 0; i < OutArgs.size (); ++i)
			{
				auto const	Idx (OutArgs.at (i));
				HMemPx [i]	= DMemPx [OutArgs.at (i)];

				auto& Arg	= *Args [Idx];

				if (Arg._var_name.CompareNoCase ("kw_Result") == 0)
				{
					_ResultVar	= static_cast <float>(*HMemPx [i]._ptr);
					Report.AppendFormat ("\t%s = %s\r\n", Arg._name, ::TGetCString (_ResultVar));
					__UpdateResultVar (false);
				}
			}		
			__AddLog (Report);
		}
	}
	__CATCH (LOnError);

   _device.Unload ();
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