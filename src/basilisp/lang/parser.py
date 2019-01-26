import uuid
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from typing import Pattern

import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.seq as lseq
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
from basilisp.lang.typing import LispForm

# Node descriptors
OP = kw.keyword("op")
FORM = kw.keyword("form")
ENV = kw.keyword("env")
CHILDREN = kw.keyword("children")
RAW_FORMS = kw.keyword("raw-forms")
TOP_LEVEL = kw.keyword("top-level")
LITERAL_Q = kw.keyword("literal?")
TYPE = kw.keyword("type")
VAL = kw.keyword("val")
META = kw.keyword("meta")

# Node types
BINDING = kw.keyword("binding")
CATCH = kw.keyword("catch")
CONST = kw.keyword("const")
DEF = kw.keyword("def")
DO = kw.keyword("do")
FN = kw.keyword("fn")
FN_METHOD = kw.keyword("fn-method")
HOST_CALL = kw.keyword("host-call")
HOST_FIELD = kw.keyword("host-field")
HOST_INTEROP = kw.keyword("host-interop")
IF = kw.keyword("if")
INVOKE = kw.keyword("invoke")
LET = kw.keyword("let")
LETFN = kw.keyword("letfn")
LOCAL = kw.keyword("local")
LOOP = kw.keyword("loop")
MAP = kw.keyword("map")
MAYBE_CLASS = kw.keyword("maybe-class")
MAYBE_HOST_FORM = kw.keyword("maybe-host-form")
NEW = kw.keyword("new")
QUOTE = kw.keyword("quote")
RECUR = kw.keyword("recur")
SET = kw.keyword("set")
SET_BANG = kw.keyword("set!")
THROW = kw.keyword("throw")
TRY = kw.keyword("try")
VAR = kw.keyword("var")
VECTOR = kw.keyword("vector")
WITH_META = kw.keyword("with-meta")


def _const_node(form: LispForm) -> lmap.Map:
    descriptor = lmap.map({
        OP: CONST,
        FORM: form,
        LITERAL_Q: True,
        TYPE: "",
        VAL: form,
        CHILDREN: vec.Vector.empty()
    })

    if form.meta is not None:
        meta_ast = parse_ast(form.meta)
        return descriptor.assoc(META, meta_ast, CHILDREN, vec.v(meta_ast))

    return descriptor


def parse_ast(form: LispForm) -> lmap.Map:  # pylint: disable=too-many-branches
    """Take a Lisp form as an argument and produce zero or more Python
    AST nodes.

    This is the primary entrypoint for generating AST nodes from Lisp
    syntax. It may be called recursively to compile child forms.

    All of the various types of elements which are compiled by this
    function delegate to external functions, which are functions of
    two arguments (a `CompilerContext` object, and the form to compile).
    Each function returns a generator, which may contain 0 or more AST
    nodes."""
    if isinstance(form, llist.List):
        yield from _list_ast(ctx, form)
        return
    elif isinstance(form, lseq.Seq):
        yield from _list_ast(ctx, llist.list(form))
        return
    elif isinstance(form, vec.Vector):
        yield from _vec_ast(ctx, form)
        return
    elif isinstance(form, lmap.Map):
        yield from _map_ast(ctx, form)
        return
    elif isinstance(form, lset.Set):
        yield from _set_ast(ctx, form)
        return
    elif isinstance(
        form,
        (
            bool,
            complex,
            datetime,
            Decimal,
            float,
            Fraction,
            int,
            kw.Keyword,
            Pattern,
            sym.Symbol,
            str,
            type(None),
            uuid.UUID,
        ),
    ):
        yield from _const_node(form)
        return
    else:
        raise TypeError(f"Unexpected form type {type(form)}: {form}")
