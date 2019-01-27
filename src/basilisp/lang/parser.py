import uuid
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from functools import partial
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
BODY_Q = kw.keyword("body?")
STATEMENTS = kw.keyword("statements")
RET = kw.keyword("ret")
TEST = kw.keyword("test")
THEN = kw.keyword("then")
ELSE = kw.keyword("else")
ARGS = kw.keyword("args")
EXPR = kw.keyword("expr")
EXPRS = kw.keyword("exprs")
EXCEPTION = kw.keyword("exception")
BODY = kw.keyword("body")
CATCHES = kw.keyword("catches")
FINALLY = kw.keyword("finally")
FIELD = kw.keyword("field")
TARGET = kw.keyword("target")
M_OR_F = kw.keyword("m-or-f")
ASSIGNABLE_Q = kw.keyword("assignable?")
METHOD = kw.keyword("method")

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


class ParserContext:
    __slots__ = ()

    def __init__(self) -> None:
        pass

    @property
    def current_ns(self) -> runtime.Namespace:
        return runtime.get_current_ns()


def _def_node(ctx: ParserContext, form: lseq.Seq) -> lmap.Map:
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
        meta_ast = parse_ast(ctx, name.meta)

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


def _do_ast(ctx: ParserContext, form: lseq.Seq) -> lmap.Map:
    assert form.first == _DO
    *statements, ret = map(partial(parse_ast, ctx), form.rest)
    return lmap.map(
        {
            OP: DO,
            FORM: form,
            STATEMENTS: vec.vector(statements),
            RET: ret,
            BODY_Q: False,
            CHILDREN: vec.v(STATEMENTS, RET),
        }
    )


def _host_call_ast(ctx: ParserContext, form: lseq.Seq) -> lmap.Map:
    assert isinstance(form.first, sym.Symbol)
    assert form.first.name.startswith(".")

    if not sum([1 for _ in form]) >= 2:
        raise ParserException("host interop calls must be 2 or more elements long")

    return lmap.map(
        {
            OP: HOST_CALL,
            FORM: form,
            METHOD: parse_ast(ctx, form.first[1:]),
            TARGET: parse_ast(ctx, runtime.nth(form, 1)),
            ARGS: vec.vector(map(partial(parse_ast, ctx), runtime.nthrest(form, 2))),
            CHILDREN: vec.v(TARGET, ARGS),
        }
    )


def _host_prop_ast(ctx: ParserContext, form: lseq.Seq) -> lmap.Map:
    assert isinstance(form.first, sym.Symbol)
    assert form.first.name.startswith(".-")

    if not sum([1 for _ in form]) == 2:
        raise ParserException("host interop prop must be exactly 2 elements long")

    return lmap.map(
        {
            OP: HOST_FIELD,
            FORM: form,
            FIELD: parse_ast(ctx, sym.symbol(form.first.name[2:])),
            TARGET: parse_ast(ctx, runtime.nth(form, 1)),
            ASSIGNABLE_Q: True,
            CHILDREN: vec.v(TARGET),
        }
    )


def _host_interop_ast(ctx: ParserContext, form: lseq.Seq) -> lmap.Map:
    assert form.first == _INTEROP_CALL
    nelems = sum([1 for _ in form])
    assert nelems >= 3

    maybe_m_or_f = runtime.nth(form, 2)
    if isinstance(maybe_m_or_f, sym.Symbol):
        # The clojure.tools.analyzer spec is unclear about whether or not a form
        # like (. target -field) should be emitted as a :host-field or as a
        # :host-interop node. I have elected to emit :host-field, since it is
        # more specific.
        if maybe_m_or_f.name.startswith("-"):
            if nelems != 3:
                raise ParserException(
                    "host field accesses must be exactly 3 elements long"
                )

            return lmap.map(
                {
                    OP: HOST_FIELD,
                    FORM: form,
                    FIELD: parse_ast(ctx, sym.symbol(maybe_m_or_f.name[1:])),
                    TARGET: parse_ast(ctx, runtime.nth(form, 1)),
                    ASSIGNABLE_Q: True,
                    CHILDREN: vec.v(TARGET),
                }
            )

        # Other symbolic members or fields can call through and will be handled
        # below
    elif isinstance(maybe_m_or_f, (llist.List, lseq.Seq)):
        # Likewise, I emit :host-call for forms like (. target (method arg1 ...)).
        method = maybe_m_or_f.first
        if not isinstance(method, sym.Symbol):
            raise ParserException("host call method must be a symbol")

        return lmap.map(
            {
                OP: HOST_CALL,
                FORM: form,
                METHOD: parse_ast(
                    ctx,
                    sym.symbol(method.name[1:])
                    if method.name.startswith("-")
                    else method,
                ),
                TARGET: parse_ast(ctx, runtime.nth(form, 1)),
                ARGS: vec.vector(map(partial(parse_ast, ctx), maybe_m_or_f.rest)),
                CHILDREN: vec.v(TARGET, ARGS),
            }
        )

    if nelems != 3:
        raise ParserException("host interop forms must be 3 or more elements long")

    m_or_f = _const_node(ctx, runtime.nth(form, 2))
    if m_or_f.entry(TYPE) != SYMBOL:
        raise ParserException("host interop member or field must be a symbol")

    return lmap.map(
        {
            OP: HOST_INTEROP,
            FORM: form,
            TARGET: parse_ast(ctx, runtime.nth(form, 1)),
            M_OR_F: m_or_f,
            ASSIGNABLE_Q: True,
            CHILDREN: vec.v(TARGET),
        }
    )


def _if_ast(ctx: ParserContext, form: lseq.Seq) -> lmap.Map:
    assert form.first == _IF

    nelems = sum([1 for _ in form])
    if nelems not in (3, 4):
        raise ParserException(
            "if forms must have either 3 or 4 elements, as in: (if test then else?)"
        )

    if nelems == 3:
        else_node = parse_ast(ctx, runtime.nth(form, 3))
    else:
        else_node = _const_node(ctx, None)

    return lmap.map(
        {
            OP: IF,
            FORM: form,
            TEST: parse_ast(ctx, runtime.nth(form, 1)),
            THEN: parse_ast(ctx, runtime.nth(form, 2)),
            ELSE: else_node,
            CHILDREN: vec.v(TEST, THEN, ELSE),
        }
    )


def _invoke_ast(ctx: ParserContext, form: lseq.Seq) -> lmap.Map:
    descriptor = lmap.map(
        {
            OP: INVOKE,
            FORM: form,
            FN: None,  # TODO: get the actual function reference
            ARGS: vec.vector(map(partial(parse_ast, ctx), form.rest)),
            META: [],
            CHILDREN: vec.v(FN, ARGS),
        }
    )

    if hasattr(form, "meta") and form.meta is not None:
        meta_ast = parse_ast(ctx, form.meta)

        meta_op = meta_ast.entry(OP)
        if meta_op == MAP or (meta_op == CONST and meta_ast.entry(TYPE) == MAP):
            return descriptor.assoc(META, meta_ast)

        raise ParserException(f"Meta  must be a map")

    return descriptor


def _quote_ast(ctx: ParserContext, form: lseq.Seq) -> lmap.Map:
    assert form.first == _QUOTE

    return lmap.map(
        {
            OP: QUOTE,
            FORM: form,
            EXPR: parse_ast(ctx, runtime.nth(form, 1)),
            LITERAL_Q: True,
            CHILDREN: vec.v(EXPR),
        }
    )


def _throw_ast(ctx: ParserContext, form: lseq.Seq) -> lmap.Map:
    assert form.first == _THROW

    return lmap.map(
        {
            OP: THROW,
            FORM: form,
            EXCEPTION: parse_ast(ctx, runtime.nth(form, 1)),
            CHILDREN: vec.v(EXCEPTION),
        }
    )


def _catch_ast(ctx: ParserContext, form: lseq.Seq) -> lmap.Map:
    assert form.first == _CATCH

    return lmap.map(
        {
            OP: THROW,
            FORM: form,
            EXCEPTION: parse_ast(ctx, runtime.nth(form, 1)),
            CHILDREN: vec.v(EXCEPTION),
        }
    )


def _try_ast(ctx: ParserContext, form: lseq.Seq) -> lmap.Map:
    assert form.first == _TRY

    try_exprs = []
    catches = []
    finallys = []
    for expr in form.rest:
        if isinstance(expr, (llist.List, lseq.Seq)):
            if expr.first == _CATCH:
                if finallys:
                    raise ParserException(
                        "catch forms may not appear after finally forms in a try"
                    )
                catches.append(_catch_ast(ctx, expr))
                continue
            elif expr.first == _FINALLY:
                if finallys:
                    raise ParserException(
                        "try forms may not contain multiple finally forms"
                    )
                finallys.extend(map(parse_ast, expr.rest))
                continue

        parsed = parse_ast(ctx, expr)

        if catches:
            raise ParserException(
                "try body expressions may not appear after catch forms"
            )
        if finallys:
            raise ParserException(
                "try body expressions may not appear after finally forms"
            )

        try_exprs.append(parsed)

    assert all(
        [node.entry(OP) == CATCH for node in catches]
    ), "All catch statements must be catch ops"

    if len(finallys) > 1:
        raise ParserException("try forms may have only 0 or 1 finally forms")

    *try_statements, try_ret = try_exprs
    return lmap.map(
        {
            OP: TRY,
            FORM: form,
            BODY: lmap.map(
                {
                    OP: DO,
                    FORM: form,
                    STATEMENTS: vec.vector(try_statements),
                    RET: try_ret,
                    BODY_Q: False,
                    CHILDREN: vec.v(STATEMENTS, RET),
                }
            ),
            CATCHES: vec.vector(catches),
            FINALLY: finallys,
            CHILDREN: vec.v(EXCEPTION),
        }
    )


_SPECIAL_FORM_HANDLERS = lmap.map(
    {
        _DEF: _def_node,
        _DO: _do_ast,
        _IF: _if_ast,
        _INTEROP_CALL: _host_interop_ast,
        _QUOTE: _quote_ast,
        _THROW: _throw_ast,
        _TRY: _try_ast,
    }
)


def _list_node(ctx: ParserContext, form: lseq.Seq) -> lmap.Map:
    handle_special_form = _SPECIAL_FORM_HANDLERS.entry(form.first)
    if handle_special_form is not None:
        return handle_special_form(ctx, form)

    s = form.first
    if isinstance(s, sym.Symbol):
        if s.name.startswith(".-"):
            return _host_prop_ast(ctx, form)
        elif s.name.startswith("."):
            return _host_call_ast(ctx, form)

    return _invoke_ast(ctx, form)


def _map_node(ctx: ParserContext, form: lmap.Map) -> lmap.Map:
    keys, vals = [], []
    for k, v in form.items():
        keys.append(parse_ast(ctx, k))
        vals.append(parse_ast(ctx, v))

    return lmap.map(
        {
            OP: MAP,
            FORM: form,
            KEYS: vec.vector(keys),
            VALS: vec.vector(vals),
            CHILDREN: vec.v(KEYS, VALS),
        }
    )


def _set_node(ctx: ParserContext, form: lset.Set) -> lmap.Map:
    return lmap.map(
        {
            OP: SET,
            FORM: form,
            ITEMS: vec.vector(map(partial(parse_ast, ctx), form)),
            CHILDREN: vec.v(ITEMS),
        }
    )


def _vector_node(ctx: ParserContext, form: vec.Vector) -> lmap.Map:
    return lmap.map(
        {
            OP: VECTOR,
            FORM: form,
            ITEMS: vec.vector(map(partial(parse_ast, ctx), form)),
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


def _const_node(ctx: ParserContext, form: LispForm) -> lmap.Map:
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
        meta_ast = parse_ast(ctx, form.meta)

        meta_op = meta_ast.entry(OP)
        if meta_op == MAP or (meta_op == CONST and meta_ast.entry(TYPE) == MAP):
            return descriptor.assoc(META, meta_ast, CHILDREN, vec.v(META))

        raise ParserException(f"Meta applied to constant must be a map")

    return descriptor


def parse_ast(ctx: ParserContext, form: LispForm) -> lmap.Map:
    """Take a Lisp form as an argument and produce a Basilisp syntax
    tree matching the clojure.tools.analyzer AST spec."""
    if isinstance(form, (llist.List, lseq.Seq)):
        return _list_node(ctx, form)
    elif isinstance(form, vec.Vector):
        return _vector_node(ctx, form)
    elif isinstance(form, lmap.Map):
        return _map_node(ctx, form)
    elif isinstance(form, lset.Set):
        return _set_node(ctx, form)
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
        return _const_node(ctx, form)
    else:
        raise TypeError(f"Unexpected form type {type(form)}: {form}")
