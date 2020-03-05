from enum import Enum
from typing import Optional, Union

import attr

import basilisp._pyast as ast
import basilisp.lang.keyword as kw
import basilisp.lang.map as lmap
from basilisp.lang.compiler.nodes import Node
from basilisp.lang.interfaces import ISeq
from basilisp.lang.typing import LispForm


class CompilerPhase(Enum):
    ANALYZING = kw.keyword("analyzing")
    CODE_GENERATION = kw.keyword("code-generation")
    MACROEXPANSION = kw.keyword("macroexpansion")
    COMPILING_PYTHON = kw.keyword("compiling-python")


@attr.s(auto_attribs=True, repr=False, slots=True, str=False)
class CompilerException(Exception):
    msg: str
    phase: CompilerPhase
    form: Union[LispForm, None, ISeq] = None
    lisp_ast: Optional[Node] = None
    py_ast: Optional[ast.AST] = None

    def __repr__(self):
        return (
            f"basilisp.lang.compiler.exception.CompilerException({self.msg}, "
            f"{self.phase}, {self.form}, {self.lisp_ast}, {self.py_ast})"
        )

    def __str__(self):
        m = lmap.map(
            {
                kw.keyword("phase"): self.phase.value,
                **({kw.keyword("form"): self.form} if self.form is not None else {}),
                **(
                    {kw.keyword("lisp_ast"): self.lisp_ast}
                    if self.lisp_ast is not None
                    else {}
                ),
                **(
                    {kw.keyword("py_ast"): self.py_ast}
                    if self.py_ast is not None
                    else {}
                ),
            }
        )
        return f"{self.msg} {m}"
