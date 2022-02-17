#include "pch.h"
#include "Common.h"
#include "ITest.h"
#include <algorithm>

size_t			_ITest::GetCalcSize  () const noexcept
{
	return _Pimpl->GetCalcSize ();
}

unsigned int	_ITest::GetGridSize  () const noexcept
{
	return _Pimpl->GetGridSize ();
}

unsigned int	_ITest::GetBlockSize () const noexcept
{
	return _Pimpl->GetBlockSize ();
}

void	_ITest::MakeHeader (ColumnsVc_t const& ColumnsArg)
{
	_Pimpl->MakeHeader (ColumnsArg);
}


void	_ITest::AddRow (ItemsVc_t const& ItemsArg)
{
	_Pimpl->AddRow (ItemsArg);
}

void _ITest::SetLog (CString const& MsgArg)
{
	_Pimpl->SetLog (MsgArg);
}

void _ITest::AddLog    (CString const& MsgArg, bool NewLineArg /*= true*/)
{
	_Pimpl->AddLog (MsgArg, NewLineArg);
}

void _ITest::ClearLog  ()
{
	_Pimpl->ClearLog  ();
}

void  _ITest::AppendLog (LPCTSTR FormatArg, ...)
{
	va_list  Args;
	va_start (Args, FormatArg);

	CString  Str;

	Str.FormatV (FormatArg, Args);

	va_end (Args);

	AddLog (Str);
}

int   _ITest::GetMaxBlocks (int SizeArg, int BlockSizeArg) const
{
	constexpr int MAX_BLOCKS	= 65536;
	auto const	MaxBlocks	= (SizeArg + BlockSizeArg - 1) / BlockSizeArg;
	return std::min <int>(MAX_BLOCKS, MaxBlocks);
}
