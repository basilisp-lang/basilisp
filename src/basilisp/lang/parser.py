import uuid
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from typing import Pattern

import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.runtime as runtime
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
ITEMS = kw.keyword("items")
KEYS = kw.keyword("keys")
VALS = kw.keyword("vals")
NAME = kw.keyword("name")
INIT = kw.keyword("init")
DOC = kw.keyword("doc")

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

# Constant node types (not already covered by the above)
NIL = kw.keyword("nil")
BOOL = kw.keyword("bool")
KEYWORD = kw.keyword("keyword")
SYMBOL = kw.keyword("symbol")
STRING = kw.keyword("string")
NUMBER = kw.keyword("number")
RECORD = kw.keyword("record")
SEQ = kw.keyword("seq")
CHAR = kw.keyword("char")
REGEX = kw.keyword("regex")
CLASS = kw.keyword("class")
INST = kw.keyword("inst")
UUID = kw.keyword("uuid")
UNKNOWN = kw.keyword("unknown")

# Special form symbols
_AMPERSAND = sym.symbol("&")
_CATCH = sym.symbol("catch")
_DEF = sym.symbol("def")
_DO = sym.symbol("do")
_FINALLY = sym.symbol("finally")
_FN = sym.symbol("fn*")
_IF = sym.symbol("if")
_IMPORT = sym.symbol("import*")
_INTEROP_CALL = sym.symbol(".")
_INTEROP_PROP = sym.symbol(".-")
_LET = sym.symbol("let*")
_LOOP = sym.symbol("loop*")
_QUOTE = sym.symbol("quote")
_RECUR = sym.symbol("recur")
_THROW = sym.symbol("throw")
_TRY = sym.symbol("try")
_VAR = sym.symbol("var")


class ParserException(Exception):
    def __init__(self, msg):
        self.msg = msg


def _def_node(form: lseq.Seq) -> lmap.Map:
    assert form.first == _DEF

    nelems = sum([1 for _ in form])
    if nelems not in (2, 3, 4):
        raise ParserException(
            f"def forms must have between 2 and 4 elements, as in: (def name docstring? init?)"
        )

    name = runtime.nth(form, 1)
    if not isinstance(name, sym.Symbol):
        raise ParserException(f"def names must be symbols, not {type(name)}")

    if nelems == 2:
        init = None
        doc = None
        children = vec.Vector.empty()
    elif nelems == 3:
        init = runtime.nth(form, 2)
        doc = None
        children = vec.v(INIT)
    else:
        init = runtime.nth(form, 3)
        doc = runtime.nth(form, 2)
        children = vec.v(INIT)

    descriptor = lmap.map(
        {
            OP: DEF,
            FORM: form,
            NAME: name,
            VAR: None,  # TODO: identify the var
            INIT: init,
            DOC: doc,
            CHILDREN: children,
        }
    )

    if name.meta is not None:
        meta_ast = parse_ast(name.meta)

        meta_op = meta_ast.entry(OP)
        if meta_op == MAP or (meta_op == CONST and meta_ast.entry(TYPE) == MAP):
            existing_children: vec.Vector = descriptor.entry(CHILDREN)
            return descriptor.assoc(
                META,
                meta_ast,
                CHILDREN,
                vec.vector(runtime.cons(META, existing_children)),
            )

        raise ParserException(f"Meta applied to constant must be a map")

    return descriptor


_SPECIAL_FORM_HANDLERS = lmap.map({_DEF: _def_node})


def _list_node(form: lseq.Seq) -> lmap.Map:
    handle_special_form = _SPECIAL_FORM_HANDLERS.entry(form.first)
    if handle_special_form is not None:
        return handle_special_form(form)

    pass


def _map_node(form: lmap.Map) -> lmap.Map:
    keys, vals = [], []
    for k, v in form.items():
        keys.append(parse_ast(k))
        vals.append(parse_ast(v))

    return lmap.map(
        {
            OP: MAP,
            FORM: form,
            KEYS: vec.vector(keys),
            VALS: vec.vector(vals),
            CHILDREN: vec.v(KEYS, VALS),
        }
    )


def _set_node(form: lset.Set) -> lmap.Map:
    return lmap.map(
        {
            OP: SET,
            FORM: form,
            ITEMS: vec.vector(map(parse_ast, form)),
            CHILDREN: vec.v(ITEMS),
        }
    )


def _vector_node(form: vec.Vector) -> lmap.Map:
    return lmap.map(
        {
            OP: VECTOR,
            FORM: form,
            ITEMS: vec.vector(map(parse_ast, form)),
            CHILDREN: vec.v(ITEMS),
        }
    )


_CONST_NODE_TYPES = lmap.map(
    {
        bool: BOOL,
        complex: NUMBER,
        datetime: INST,
        Decimal: NUMBER,
        float: NUMBER,
        Fraction: NUMBER,
        int: NUMBER,
        kw.Keyword: KEYWORD,
        Pattern: REGEX,
        sym.Symbol: SYMBOL,
        str: STRING,
        type(None): NIL,
        uuid.UUID: UUID,
    }
)


def _const_node(form: LispForm) -> lmap.Map:
    descriptor = lmap.map(
        {
            OP: CONST,
            FORM: form,
            LITERAL_Q: True,
            TYPE: _CONST_NODE_TYPES.entry(type(form), UNKNOWN),
            VAL: form,
            CHILDREN: vec.Vector.empty(),
        }
    )

    if hasattr(form, "meta") and form.meta is not None:
        meta_ast = parse_ast(form.meta)

        meta_op = meta_ast.entry(OP)
        if meta_op == MAP or (meta_op == CONST and meta_ast.entry(TYPE) == MAP):
            return descriptor.assoc(META, meta_ast, CHILDREN, vec.v(META))

        raise ParserException(f"Meta applied to constant must be a map")

    return descriptor


def parse_ast(form: LispForm) -> lmap.Map:  # pylint: disable=too-many-branches
    """Take a Lisp form as an argument and produce a Basilisp syntax
    tree matching the clojure.tools.analyzer AST spec."""
    if isinstance(form, (llist.List, lseq.Seq)):
        return _list_node(form)
    elif isinstance(form, vec.Vector):
        return _vector_node(form)
    elif isinstance(form, lmap.Map):
        return _map_node(form)
    elif isinstance(form, lset.Set):
        return _set_node(form)
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
        return _const_node(form)
    else:
        raise TypeError(f"Unexpected form type {type(form)}: {form}")
