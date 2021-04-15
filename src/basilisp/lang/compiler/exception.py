from enum import Enum
from typing import Any, Dict, Optional, Union

import attr

from basilisp import _pyast as ast
from basilisp.lang import keyword as kw
from basilisp.lang import map as lmap
from basilisp.lang.compiler.nodes import Node
from basilisp.lang.interfaces import IExceptionInfo, IMeta, IPersistentMap, ISeq
from basilisp.lang.obj import lrepr
from basilisp.lang.reader import READER_COL_KW, READER_LINE_KW
from basilisp.lang.typing import LispForm

_PHASE = kw.keyword("phase")
_FORM = kw.keyword("form")
_LISP_AST = kw.keyword("lisp_ast")
_PY_AST = kw.keyword("py_ast")
_LINE = kw.keyword("line")
_COL = kw.keyword("col")


class CompilerPhase(Enum):
    ANALYZING = kw.keyword("analyzing")
    CODE_GENERATION = kw.keyword("code-generation")
    MACROEXPANSION = kw.keyword("macroexpansion")
    COMPILING_PYTHON = kw.keyword("compiling-python")


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _loc:
    line: Optional[int] = None
    col: Optional[int] = None

    def __bool__(self):
        return self.line is not None and self.col is not None


@attr.s(auto_attribs=True, slots=True, str=False)
class CompilerException(IExceptionInfo):
    msg: str
    phase: CompilerPhase
    form: Union[LispForm, None, ISeq] = None
    lisp_ast: Optional[Node] = None
    py_ast: Optional[ast.AST] = None

    @property
    def data(self) -> IPersistentMap:
        d: Dict[kw.Keyword, Any] = {_PHASE: self.phase.value}
        loc = None
        if self.form is not None:
            d[_FORM] = self.form
            loc = (
                _loc(
                    self.form.meta.val_at(READER_LINE_KW),
                    self.form.meta.val_at(READER_COL_KW),
                )
                if isinstance(self.form, IMeta) and self.form.meta
                else None
            )
        if self.lisp_ast is not None:  # pragma: no cover
            d[_LISP_AST] = self.lisp_ast
            loc = loc or _loc(self.lisp_ast.env.line, self.lisp_ast.env.col)
        if self.py_ast is not None:  # pragma: no cover
            d[_PY_AST] = self.py_ast
            loc = loc or _loc(self.py_ast.lineno, self.py_ast.col_offset)
        if loc:  # pragma: no cover
            d[_LINE] = loc.line
            d[_COL] = loc.col
        return lmap.map(d)

    def __str__(self):
        return f"{self.msg} {lrepr(self.data)}"
