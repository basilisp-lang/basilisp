import ast
from enum import Enum
from typing import Optional, Union

import attr

import basilisp.lang.keyword as kw
from basilisp.lang.compyler.nodes import Node
from basilisp.lang.seq import Seq
from basilisp.lang.typing import LispForm


class CompilerPhase(Enum):
    PARSING = kw.keyword("parsing")
    CODE_GENERATION = kw.keyword("code-generation")
    MACROEXPANSION = kw.keyword("macroexpansion")


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CompilerException(Exception):
    msg: str
    phase: CompilerPhase
    form: Union[LispForm, None, Seq] = None
    lisp_ast: Optional[Node] = None
    py_ast: Optional[ast.AST] = None
