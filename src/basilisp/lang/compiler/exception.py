import ast
import os
from enum import Enum
from types import TracebackType
from typing import Any, Optional, Union

import attr

from basilisp.lang import keyword as kw
from basilisp.lang import map as lmap
from basilisp.lang.compiler.nodes import Node
from basilisp.lang.exception import format_exception
from basilisp.lang.interfaces import IExceptionInfo, IMeta, IPersistentMap, ISeq
from basilisp.lang.obj import lrepr
from basilisp.lang.reader import (
    READER_COL_KW,
    READER_END_COL_KW,
    READER_END_LINE_KW,
    READER_LINE_KW,
)
from basilisp.lang.source import format_source_context
from basilisp.lang.typing import LispForm

_FILE = kw.keyword("file")
_PHASE = kw.keyword("phase")
_FORM = kw.keyword("form")
_LISP_AST = kw.keyword("lisp_ast")
_PY_AST = kw.keyword("py_ast")
_LINE = kw.keyword("line")
_COL = kw.keyword("col")
_END_LINE = kw.keyword("end-line")
_END_COL = kw.keyword("end-col")


class CompilerPhase(Enum):
    ANALYZING = kw.keyword("analyzing")
    CODE_GENERATION = kw.keyword("code-generation")
    INLINING = kw.keyword("inlining")
    MACROEXPANSION = kw.keyword("macroexpansion")
    COMPILING_PYTHON = kw.keyword("compiling-python")


@attr.frozen
class _loc:
    line: Optional[int] = None
    col: Optional[int] = None
    end_line: Optional[int] = None
    end_col: Optional[int] = None

    def __bool__(self):
        return (
            self.line is not None
            or self.col is not None
            or self.end_line is not None
            or self.end_col is not None
        )


@attr.define(str=False)
class CompilerException(IExceptionInfo):
    msg: str
    phase: CompilerPhase
    filename: str
    form: Union[LispForm, None, ISeq] = None
    lisp_ast: Optional[Node] = None
    py_ast: Optional[Union[ast.expr, ast.stmt]] = None

    @property
    def data(self) -> IPersistentMap:
        d: dict[kw.Keyword, Any] = {_PHASE: self.phase.value}
        d[_FILE] = self.filename
        loc = None
        if self.form is not None:
            d[_FORM] = self.form
            loc = (
                _loc(
                    self.form.meta.val_at(READER_LINE_KW),
                    self.form.meta.val_at(READER_COL_KW),
                    self.form.meta.val_at(READER_END_LINE_KW),
                    self.form.meta.val_at(READER_END_COL_KW),
                )
                if isinstance(self.form, IMeta) and self.form.meta
                else None
            )
        if self.lisp_ast is not None:  # pragma: no cover
            d[_LISP_AST] = self.lisp_ast
            loc = loc or _loc(
                self.lisp_ast.env.line,
                self.lisp_ast.env.col,
                self.lisp_ast.env.end_line,
                self.lisp_ast.env.end_col,
            )
        if self.py_ast is not None:  # pragma: no cover
            d[_PY_AST] = self.py_ast
            loc = loc or _loc(
                self.py_ast.lineno,
                self.py_ast.col_offset,
                self.py_ast.end_lineno,
                self.py_ast.end_col_offset,
            )
        if loc:  # pragma: no cover
            d[_LINE] = loc.line
            d[_COL] = loc.col
            d[_END_LINE] = loc.end_line
            d[_END_COL] = loc.end_col
        return lmap.map(d)

    def __str__(self):
        return f"{self.msg} {lrepr(self.data)}"


@format_exception.register(CompilerException)
def format_compiler_exception(  # pylint: disable=too-many-branches,unused-argument
    e: CompilerException,
    tp: Optional[type[Exception]] = None,
    tb: Optional[TracebackType] = None,
    disable_color: Optional[bool] = None,
) -> list[str]:
    """Format a compiler exception as a list of newline-terminated strings.

    If `disable_color` is True, no color formatting will be applied to the source
    code."""
    context_exc: Optional[BaseException] = e.__cause__

    lines = [os.linesep]
    if context_exc is not None:
        lines.append(f"  exception: {type(context_exc)} from {type(e)}{os.linesep}")
    else:
        lines.append(f"  exception: {type(e)}{os.linesep}")
    lines.append(f"      phase: {e.phase.value}{os.linesep}")
    if context_exc is None:
        lines.append(f"    message: {e.msg}{os.linesep}")
    elif e.phase in {CompilerPhase.MACROEXPANSION, CompilerPhase.INLINING}:
        if isinstance(context_exc, CompilerException):
            lines.append(f"    message: {e.msg}: {context_exc.msg}{os.linesep}")
        else:
            lines.append(f"    message: {e.msg}: {context_exc}{os.linesep}")
    else:
        lines.append(f"    message: {e.msg}: {context_exc}{os.linesep}")
    if e.form is not None:
        lines.append(f"       form: {e.form!r}{os.linesep}")

    d = e.data
    line = d.val_at(_LINE)
    end_line = d.val_at(_END_LINE)
    if line is not None and end_line is not None and line != end_line:
        line_nums = f"{line}-{end_line}"
    elif line is not None:
        line_nums = str(line)
    else:
        line_nums = ""

    lines.append(
        f"   location: {e.filename}:{line_nums or 'NO_SOURCE_LINE'}{os.linesep}"
    )

    # Print context source lines around the error. Use the current exception to
    # derive source lines, but use the inner cause exception to place a marker
    # around the error.
    if line is not None and (
        context_lines := format_source_context(
            e.filename, line, end_line=end_line, disable_color=disable_color
        )
    ):
        lines.append(f"    context:{os.linesep}")
        lines.append(os.linesep)
        lines.extend(context_lines)

    return lines
