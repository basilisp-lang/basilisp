import collections
import contextlib
import logging
import uuid
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from functools import partial, wraps
from typing import Pattern, Union, Deque, Optional, NamedTuple, Dict, Callable, cast

import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.reader as reader
import basilisp.lang.runtime as runtime
import basilisp.lang.seq as lseq
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
from basilisp.lang.compyler.constants import *
from basilisp.lang.compyler.nodes import (
    Const,
    Node,
    ConstType,
    Map as MapNode,
    Set as SetNode,
    Vector as VectorNode,
    Local,
    Throw,
    Quote,
    Invoke,
    If,
    HostCall,
    HostField,
    Do,
    HostInterop,
    Catch,
    Binding,
    LocalType,
    MaybeClass,
    Def,
    WithMeta,
    FnMethod,
    Fn,
    Let,
    Try,
    MaybeHostForm,
    VarRef,
)
from basilisp.lang.runtime import Var
from basilisp.lang.typing import LispForm
from basilisp.lang.util import genname
from basilisp.util import Maybe, partition

# Parser logging
logger = logging.getLogger(__name__)

DEFAULT_COMPILER_FILE_PATH = "NO_SOURCE_PATH"

# Parser options
WARN_ON_UNUSED_NAMES = "warn_on_unused_names"

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

_BUILTINS_NS = "builtins"


def count(seq: lseq.Seq) -> int:
    return sum([1 for _ in seq])


class ParserException(Exception):
    def __init__(self, msg):
        self.msg = msg


# Symbols to be ignored for unused symbol warnings
_IGNORED_SYM = sym.symbol("_")
_MACRO_ENV_SYM = sym.symbol("&env")
_MACRO_FORM_SYM = sym.symbol("&form")
_NO_WARN_UNUSED_SYMS = lset.s(_IGNORED_SYM, _MACRO_ENV_SYM, _MACRO_FORM_SYM)


class SymbolTableEntry(NamedTuple):
    context: LocalType
    symbol: sym.Symbol
    used: bool = False
    warn_if_unused: bool = True


class SymbolTable:
    LOCAL_CONTEXTS = lset.set(
        [
            SYM_CTX_LOCAL_ARG,
            SYM_CTX_LOCAL_CATCH,
            SYM_CTX_LOCAL_FN,
            SYM_CTX_LOCAL_LET,
            SYM_CTX_LOCAL_LETFN,
            SYM_CTX_LOCAL_LOOP,
        ]
    )

    __slots__ = ("_name", "_parent", "_table", "_children")

    def __init__(
        self,
        name: str,
        parent: "SymbolTable" = None,
        table: Dict[sym.Symbol, SymbolTableEntry] = None,
        children: Dict[str, "SymbolTable"] = None,
    ) -> None:
        self._name = name
        self._parent = parent
        self._table = {} if table is None else table
        self._children = {} if children is None else children

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self):
        return (
            f"SymbolTable({self._name}, parent={repr(self._parent.name)}, "
            f"table={repr(self._table)}, children={len(self._children)})"
        )

    def new_symbol(
        self, s: sym.Symbol, ctx: LocalType, warn_if_unused: bool = True
    ) -> "SymbolTable":
        assert ctx in SymbolTable.LOCAL_CONTEXTS
        if s in self._table:
            self._table[s] = self._table[s]._replace(
                context=ctx, symbol=s, warn_if_unused=warn_if_unused
            )
        else:
            self._table[s] = SymbolTableEntry(ctx, s, warn_if_unused=warn_if_unused)
        return self

    def find_symbol(self, s: sym.Symbol) -> Optional[SymbolTableEntry]:
        if s in self._table:
            return self._table[s]
        if self._parent is None:
            return None
        return self._parent.find_symbol(s)

    def mark_used(self, s: sym.Symbol) -> None:
        """Mark the symbol s used in the current table or the first ancestor table
        which contains the symbol."""
        if s in self._table:
            old: SymbolTableEntry = self._table[s]
            if old.used:
                return
            self._table[s] = old._replace(used=True)
        elif self._parent is not None:
            self._parent.mark_used(s)
        else:
            assert False, f"Symbol {s} not defined in any symbol table"

    def _warn_unused_names(self):
        """Log a warning message for locally bound names whose values are not used
        by the time the symbol table frame is being popped off the stack.

        The symbol table contains locally-bound symbols, recur point symbols, and
        symbols bound to var-args in generated Python functions. Only the locally-
        bound symbols are eligible for an unused warning, since it is not common
        that recur points will be used and user code is not permitted to directly
        access the var-args symbol (the compiler inserts an intermediate symbol
        which user code uses).

        Warnings will not be issued for symbols named '_', '&form', and '&env'. The
        latter symbols appear in macros and a great many macros will never use them."""
        assert logger.isEnabledFor(
            logging.WARNING
        ), "Only warn when logger is configured for WARNING level"
        ns = runtime.get_current_ns()
        for _, entry in self._table.items():
            if entry.context not in SymbolTable.LOCAL_CONTEXTS:
                continue
            if entry.symbol in _NO_WARN_UNUSED_SYMS:
                continue
            if entry.warn_if_unused and not entry.used:
                code_loc = (
                    Maybe(entry.symbol.meta)
                    .map(lambda m: f": {m.entry(reader.READER_LINE_KW)}")
                    .or_else_get("")
                )
                logger.warning(
                    f"symbol '{entry.symbol}' defined but not used ({ns}{code_loc})"
                )

    def append_frame(self, name: str, parent: "SymbolTable" = None) -> "SymbolTable":
        new_frame = SymbolTable(name, parent=parent)
        self._children[name] = new_frame
        return new_frame

    def pop_frame(self, name: str) -> None:
        del self._children[name]

    @contextlib.contextmanager
    def new_frame(self, name, warn_on_unused_names):
        """Context manager for creating a new stack frame. If warn_on_unused_names is
        True and the logger is enabled for WARNING, call _warn_unused_names() on the
        child SymbolTable before it is popped."""
        new_frame = self.append_frame(name, parent=self)
        yield new_frame
        if warn_on_unused_names and logger.isEnabledFor(logging.WARNING):
            new_frame._warn_unused_names()
        self.pop_frame(name)


class ParserContext:
    __slots__ = ("_filename", "_is_quoted", "_opts", "_st")

    def __init__(
        self, filename: Optional[str] = None, opts: Optional[Dict[str, bool]] = None
    ) -> None:
        self._filename = Maybe(filename).or_else_get(DEFAULT_COMPILER_FILE_PATH)
        self._is_quoted: Deque[bool] = collections.deque([])
        self._opts = Maybe(opts).map(lmap.map).or_else_get(lmap.Map.empty())
        self._st = collections.deque([SymbolTable("<Top>")])

    @property
    def current_ns(self) -> runtime.Namespace:
        return runtime.get_current_ns()

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def warn_on_unused_names(self) -> bool:
        """If True, warn when local names are unused."""
        return self._opts.entry(WARN_ON_UNUSED_NAMES, True)

    @property
    def is_quoted(self) -> bool:
        try:
            return self._is_quoted[-1] is True
        except IndexError:
            return False

    @contextlib.contextmanager
    def quoted(self):
        self._is_quoted.append(True)
        yield
        self._is_quoted.pop()

    @property
    def symbol_table(self) -> SymbolTable:
        return self._st[-1]

    @contextlib.contextmanager
    def new_symbol_table(self, name):
        old_st = self.symbol_table
        with old_st.new_frame(name, self.warn_on_unused_names) as st:
            self._st.append(st)
            yield st
            self._st.pop()


ParseFunction = Callable[[ParserContext, LispForm], lmap.Map]


def _with_meta(gen_node):
    """Wraps the node generated by gen_node in a :with-meta AST node if the
    original form has .

    :with-meta AST nodes are used for non-quoted collection literals and for
    function expressions."""

    @wraps(gen_node)
    def with_meta(ctx: ParserContext, form: lmap.Map) -> Node:
        assert not ctx.is_quoted, "with-meta nodes are not used in quoted expressions"

        descriptor = gen_node(ctx, form)

        if hasattr(form, "meta") and form.meta is not None:  # type: ignore
            meta_ast = parse_ast(ctx, form.meta)  # type: ignore

            if isinstance(meta_ast, MapNode) or (
                isinstance(meta_ast, Const) and meta_ast.type == ConstType.MAP
            ):
                return WithMeta(form=form, meta=meta_ast, expr=descriptor)

            raise ParserException(f"meta must be a map")

        return descriptor

    return with_meta


def _def_node(ctx: ParserContext, form: lseq.Seq) -> Def:
    assert form.first == _DEF

    nelems = count(form)
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
        init = parse_ast(ctx, runtime.nth(form, 2))
        doc = None
        children = vec.v(INIT)
    else:
        init = parse_ast(ctx, runtime.nth(form, 3))
        doc = runtime.nth(form, 2)
        children = vec.v(INIT)

    descriptor = Def(form=form, name=name, init=init, doc=doc, children=children)

    if name.meta is not None:
        meta_ast = parse_ast(ctx, name.meta)

        if not isinstance(meta_ast, Const) or meta_ast.type != ConstType.MAP:
            existing_children = cast(vec.Vector, descriptor.children)
            return descriptor.assoc(
                meta=meta_ast,
                children=vec.vector(runtime.cons(META, existing_children)),
            )

        raise ParserException(f"Meta applied to constant must be a map")

    return descriptor


def _do_ast(ctx: ParserContext, form: lseq.Seq) -> Do:
    assert form.first == _DO
    *statements, ret = map(partial(parse_ast, ctx), form.rest)
    return Do(form=form, statements=vec.vector(statements), ret=ret)


def _fn_method_ast(
    ctx: ParserContext, form: lseq.Seq, fnname: Optional[sym.Symbol] = None
) -> FnMethod:
    with ctx.new_symbol_table("fn-method"):
        params = form.first
        if not isinstance(params, vec.Vector):
            raise ParserException("function arity arguments must be a vector")

        vargs, has_vargs, vargs_idx = None, False, 0
        param_nodes = []
        for i, s in enumerate(params):
            if not isinstance(s, sym.Symbol):
                raise ParserException("function arity parameter name must be a symbol")

            if s == _AMPERSAND:
                has_vargs = True
                vargs_idx = i
                break

            param_nodes.append(
                Local(
                    form=s,
                    name=s,
                    local=LocalType.ARG,
                    arg_id=i,
                    is_assignable=False,
                    is_variadic=False,
                )
            )

            ctx.symbol_table.new_symbol(s, LocalType.ARG)

        if has_vargs:
            try:
                vargs_sym = params[vargs_idx + 1]

                param_nodes.append(
                    Local(
                        form=vargs_sym,
                        name=vargs_sym,
                        local=LocalType.ARG,
                        arg_id=i,
                        is_assignable=False,
                        is_variadic=True,
                    )
                )

                ctx.symbol_table.new_symbol(vargs_sym, LocalType.ARG)
            except IndexError:
                raise ParserException(
                    "Expected variadic argument name after '&'"
                ) from None

        *stmts, ret = map(partial(parse_ast, ctx), form.rest)
        return FnMethod(
            form=form,
            loop_id=sym.symbol(genname("fn_arity" if fnname is None else fnname.name)),
            params=vec.vector(param_nodes),
            is_variadic=has_vargs,
            fixed_arity=len(param_nodes) - int(has_vargs),
            body=Do(
                form=form.rest, statements=vec.vector(stmts), ret=ret, is_body=True
            ),
        )


def _fn_ast(ctx: ParserContext, form: lseq.Seq) -> Fn:
    assert form.first == _FN

    idx = 1

    with ctx.new_symbol_table("fn"):
        name = runtime.nth(form, idx)
        if isinstance(name, sym.Symbol):
            ctx.symbol_table.new_symbol(name, LocalType.FN, warn_if_unused=False)
            name_node: Optional[Binding] = Binding(
                form=name, name=name, local=LocalType.FN
            )
        elif isinstance(name, (llist.List, vec.Vector)):
            name = None
            name_node = None
            idx += 1
        else:
            raise ParserException(
                "fn form must match: (fn* name? [arg*] body*) or (fn* name? method*)"
            )

        arity_or_args = runtime.nth(form, idx)
        if isinstance(arity_or_args, llist.List):
            methods = vec.vector(
                map(
                    partial(_fn_method_ast, ctx, fnname=name),
                    runtime.nthrest(form, idx),
                )
            )
        elif isinstance(arity_or_args, vec.Vector):
            methods = vec.v(
                _fn_method_ast(ctx, runtime.nthrest(form, idx), fnname=name)
            )
        else:
            raise ParserException(
                "fn form expects either multiple arities or a vector of arguments"
            )

        return Fn(
            form=form,
            is_variadic=any([method.is_variadic for method in methods]),
            max_fixed_arity=max([node.entry(FIXED_ARITY) for node in methods]),
            methods=methods,
            local=name_node,
        )


def _host_call_ast(ctx: ParserContext, form: lseq.Seq) -> HostCall:
    assert isinstance(form.first, sym.Symbol)

    method = form.first
    assert method.name.startswith(".")

    if not count(form) >= 2:
        raise ParserException("host interop calls must be 2 or more elements long")

    return HostCall(
        form=form,
        method=sym.symbol(method.name[1:]),
        target=parse_ast(ctx, runtime.nth(form, 1)),
        args=vec.vector(map(partial(parse_ast, ctx), runtime.nthrest(form, 2))),
    )


def _host_prop_ast(ctx: ParserContext, form: lseq.Seq) -> HostField:
    assert isinstance(form.first, sym.Symbol)

    field = form.first
    assert field.name.startswith(".-")

    if not count(form) == 2:
        raise ParserException("host interop prop must be exactly 2 elements long")

    return HostField(
        form=form,
        field=sym.symbol(field.name[2:]),
        target=parse_ast(ctx, runtime.nth(form, 1)),
        is_assignable=True,
    )


def _host_interop_ast(
    ctx: ParserContext, form: lseq.Seq
) -> Union[HostCall, HostField, HostInterop]:
    assert form.first == _INTEROP_CALL
    nelems = count(form)
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

            return HostField(
                form=form,
                field=sym.symbol(maybe_m_or_f.name[1:]),
                target=parse_ast(ctx, runtime.nth(form, 1)),
                is_assignable=True,
            )

        # Other symbolic members or fields can call through and will be handled
        # below
    elif isinstance(maybe_m_or_f, (llist.List, lseq.Seq)):
        # Likewise, I emit :host-call for forms like (. target (method arg1 ...)).
        method = maybe_m_or_f.first
        if not isinstance(method, sym.Symbol):
            raise ParserException("host call method must be a symbol")

        return HostCall(
            form=form,
            method=sym.symbol(method.name[1:])
            if method.name.startswith("-")
            else method,
            target=parse_ast(ctx, runtime.nth(form, 1)),
            args=vec.vector(map(partial(parse_ast, ctx), maybe_m_or_f.rest)),
        )

    if nelems != 3:
        raise ParserException("host interop forms must be 3 or more elements long")

    m_or_f = runtime.nth(form, 2)
    if not isinstance(m_or_f, sym.Symbol):
        raise ParserException("host interop member or field must be a symbol")

    return HostInterop(
        form=form,
        target=parse_ast(ctx, runtime.nth(form, 1)),
        m_or_f=m_or_f,
        is_assignable=True,
    )


def _if_ast(ctx: ParserContext, form: lseq.Seq) -> If:
    assert form.first == _IF

    nelems = count(form)
    if nelems not in (3, 4):
        raise ParserException(
            "if forms must have either 3 or 4 elements, as in: (if test then else?)"
        )

    if nelems == 4:
        else_node = parse_ast(ctx, runtime.nth(form, 3))
    else:
        else_node = _const_node(ctx, None)

    return If(
        form=form,
        test=parse_ast(ctx, runtime.nth(form, 1)),
        then=parse_ast(ctx, runtime.nth(form, 2)),
        else_=else_node,
    )


def _invoke_ast(ctx: ParserContext, form: Union[llist.List, lseq.Seq]) -> Invoke:
    descriptor = Invoke(
        form=form,
        fn=parse_ast(ctx, form.first),
        args=vec.vector(map(partial(parse_ast, ctx), form.rest)),
    )

    if hasattr(form, "meta") and form.meta is not None:  # type: ignore
        meta_ast = parse_ast(ctx, form.meta)  # type: ignore

        if not isinstance(meta_ast, Const) or meta_ast.type != ConstType.MAP:
            return descriptor.assoc(meta=meta_ast)

        raise ParserException(f"Meta  must be a map")

    return descriptor


def _let_ast(ctx: ParserContext, form: lseq.Seq) -> Let:
    assert form.first == _LET
    nelems = count(form)

    if nelems < 3:
        raise ParserException("let forms must have bindings and at least one body form")

    bindings = runtime.nth(form, 1)
    if not isinstance(bindings, vec.Vector):
        raise ParserException("let bindings must be a vector")
    elif len(bindings) == 0:
        raise ParserException("let form must have at least one pair of bindings")
    elif len(bindings) % 2 != 0:
        raise ParserException("let bindings must appear in name-value pairs")

    with ctx.new_symbol_table("let"):
        binding_nodes = []
        for name, value in partition(bindings, 2):
            if not isinstance(name, sym.Symbol):
                raise ParserException("let binding name must be a symbol")

            binding_nodes.append(
                Binding(
                    form=name,
                    name=name,
                    local=LocalType.LET,
                    init=parse_ast(ctx, value),
                )
            )

            ctx.symbol_table.new_symbol(name, LocalType.LET)

        *statements, ret = map(partial(parse_ast, ctx), runtime.nthrest(form, 2))
        return Let(
            form=form,
            bindings=vec.vector(binding_nodes),
            body=Do(
                form=runtime.nthrest(form, 2),
                statements=vec.vector(statements),
                ret=ret,
                is_body=True,
            ),
        )


def _quote_ast(ctx: ParserContext, form: lseq.Seq) -> Quote:
    assert form.first == _QUOTE

    with ctx.quoted():
        expr = parse_ast(ctx, runtime.nth(form, 1))
        assert isinstance(expr, Const), "Quoted expressions must yield :const nodes"
        return Quote(form=form, expr=expr, is_literal=True)


def _throw_ast(ctx: ParserContext, form: lseq.Seq) -> Throw:
    assert form.first == _THROW
    return Throw(form=form, exception=parse_ast(ctx, runtime.nth(form, 1)))


def _catch_ast(ctx: ParserContext, form: lseq.Seq) -> Catch:
    assert form.first == _CATCH
    nelems = count(form)

    if nelems < 4:
        raise ParserException(
            "catch forms must contain at least 4 elements: (catch class local body*)"
        )

    catch_cls = parse_ast(ctx, runtime.nth(form, 1))
    if not isinstance(catch_cls, MaybeClass):
        raise ParserException("catch forms must name a class type to catch")

    local_name = runtime.nth(form, 2)
    if not isinstance(local_name, sym.Symbol):
        raise ParserException("catch local must be a symbol")

    with ctx.new_symbol_table("catch"):
        ctx.symbol_table.new_symbol(local_name, LocalType.CATCH)

        *catch_statements, catch_ret = map(
            partial(parse_ast, ctx), runtime.nthrest(form, 3)
        )
        return Catch(
            form=form,
            class_=catch_cls,
            local=Binding(form=local_name, name=local_name, local=LocalType.CATCH),
            body=Do(
                form=runtime.nthrest(form, 3),
                statements=vec.vector(catch_statements),
                ret=catch_ret,
                is_body=True,
            ),
        )


def _try_ast(ctx: ParserContext, form: lseq.Seq) -> Try:
    assert form.first == _TRY

    try_exprs = []
    catches = []
    finally_: Optional[Do] = None
    for expr in form.rest:
        if isinstance(expr, (llist.List, lseq.Seq)):
            if expr.first == _CATCH:
                if finally_:
                    raise ParserException(
                        "catch forms may not appear after finally forms in a try"
                    )
                catches.append(_catch_ast(ctx, expr))
                continue
            elif expr.first == _FINALLY:
                if finally_ is not None:
                    raise ParserException(
                        "try forms may not contain multiple finally forms"
                    )
                *finally_stmts, finally_ret = map(partial(parse_ast, ctx), expr.rest)
                finally_ = Do(
                    form=expr.rest,
                    statements=vec.vector(finally_stmts),
                    ret=finally_ret,
                    is_body=True,
                )
                continue

        parsed = parse_ast(ctx, expr)

        if catches:
            raise ParserException(
                "try body expressions may not appear after catch forms"
            )
        if finally_:
            raise ParserException(
                "try body expressions may not appear after finally forms"
            )

        try_exprs.append(parsed)

    assert all(
        [isinstance(node, Catch) for node in catches]
    ), "All catch statements must be catch ops"

    *try_statements, try_ret = try_exprs
    return Try(
        form=form,
        body=Do(
            form=form, statements=vec.vector(try_statements), ret=try_ret, is_body=True
        ),
        catches=vec.vector(catches),
        finally_=finally_,
        children=vec.v(BODY, CATCHES, FINALLY)
        if finally_ is not None
        else vec.v(BODY, CATCHES),
    )


SpecialFormNode = Union[
    Def, Do, Fn, If, HostCall, HostField, HostInterop, Invoke, Let, Quote, Throw, Try
]
SpecialFormHandler = Callable[[ParserContext, lseq.Seq], SpecialFormNode]
_SPECIAL_FORM_HANDLERS: Dict[sym.Symbol, SpecialFormHandler] = {
    _DEF: _def_node,
    _DO: _do_ast,
    _FN: _fn_ast,
    _IF: _if_ast,
    _INTEROP_CALL: _host_interop_ast,
    _LET: _let_ast,
    _QUOTE: _quote_ast,
    _THROW: _throw_ast,
    _TRY: _try_ast,
}


def _list_node(
    ctx: ParserContext, form: lseq.Seq
) -> Union[
    Def, Do, Fn, If, HostCall, HostField, HostInterop, Invoke, Let, Quote, Throw, Try
]:
    s = form.first
    if isinstance(s, sym.Symbol):
        handle_special_form = _SPECIAL_FORM_HANDLERS.get(s)
        if handle_special_form is not None:
            return handle_special_form(ctx, form)
        elif s.name.startswith(".-"):
            return _host_prop_ast(ctx, form)
        elif s.name.startswith("."):
            return _host_call_ast(ctx, form)

    return _invoke_ast(ctx, form)


def _resolve_sym(
    ctx: ParserContext, form: sym.Symbol
) -> Union[MaybeClass, MaybeHostForm, VarRef]:
    """Resolve a Basilisp symbol as a Var or Python name."""
    # Support special class-name syntax to instantiate new classes
    #   (Classname. *args)
    #   (aliased.Classname. *args)
    #   (fully.qualified.Classname. *args)
    if form.ns is None and form.name.endswith("."):
        try:
            ns, name = form.name[:-1].rsplit(".", maxsplit=1)
            form = sym.symbol(name, ns=ns)
        except ValueError:
            form = sym.symbol(form.name[:-1])

    if form.ns is not None:
        if form.ns == ctx.current_ns.name:
            v = ctx.current_ns.find(sym.symbol(form.name))
            if v is not None:
                return VarRef(form=form, var=v, is_assignable=v.dynamic)

        ns_sym = sym.symbol(form.ns)
        if ns_sym in ctx.current_ns.imports or ns_sym in ctx.current_ns.import_aliases:
            v = Var.find(form)
            if v is not None:
                return VarRef(form=form, var=v, is_assignable=v.dynamic)
            if ns_sym in ctx.current_ns.import_aliases:
                ns_sym: sym.Symbol = ctx.current_ns.import_aliases[  # type: ignore
                    ns_sym
                ]
        elif ns_sym in ctx.current_ns.aliases:
            aliased_ns: runtime.Namespace = ctx.current_ns.aliases[ns_sym]
            v = Var.find(sym.symbol(form.name, ns=aliased_ns.name))
            if v is not None:
                return VarRef(form=form, var=v, is_assignable=v.dynamic)

        return MaybeHostForm(form=form, class_=ns_sym, field=sym.symbol(form.name))
    else:
        # Look up the symbol in the namespace mapping of the current namespace.
        v = ctx.current_ns.find(form)
        if v is not None:
            return VarRef(form=form, var=v, is_assignable=v.dynamic)

        return MaybeClass(form=form, class_=form)


def _symbol_node(
    ctx: ParserContext, form: sym.Symbol
) -> Union[Const, Local, MaybeClass, MaybeHostForm, VarRef]:
    if ctx.is_quoted:
        return _const_node(ctx, form)

    sym_entry = ctx.symbol_table.find_symbol(form)
    if sym_entry is not None:
        return Local(form=form, name=form, local=sym_entry.context, is_assignable=False)

    return _resolve_sym(ctx, form)


@_with_meta
def _map_node(ctx: ParserContext, form: lmap.Map) -> MapNode:
    keys, vals = [], []
    for k, v in form.items():
        keys.append(parse_ast(ctx, k))
        vals.append(parse_ast(ctx, v))

    return MapNode(form=form, keys=vec.vector(keys), vals=vec.vector(vals))


@_with_meta
def _set_node(ctx: ParserContext, form: lset.Set) -> SetNode:
    return SetNode(form=form, items=vec.vector(map(partial(parse_ast, ctx), form)))


@_with_meta
def _vector_node(ctx: ParserContext, form: vec.Vector) -> VectorNode:
    return VectorNode(form=form, items=vec.vector(map(partial(parse_ast, ctx), form)))


_CONST_NODE_TYPES = lmap.map(
    {
        bool: BOOL,
        complex: NUMBER,
        datetime: INST,
        Decimal: DECIMAL,
        float: NUMBER,
        Fraction: FRACTION,
        int: NUMBER,
        kw.Keyword: KEYWORD,
        lmap.Map: MAP,
        lset.Set: SET,
        Pattern: REGEX,
        sym.Symbol: SYMBOL,
        str: STRING,
        type(None): NIL,
        uuid.UUID: UUID,
        vec.Vector: VECTOR,
    }
)


def _const_node(ctx: ParserContext, form: LispForm) -> Const:
    assert (
        ctx.is_quoted and isinstance(form, (sym.Symbol, vec.Vector, lmap.Map, lset.Set))
    ) or isinstance(
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
            str,
            type(None),
            uuid.UUID,
        ),
    )

    descriptor = Const(
        form=form,
        is_literal=True,
        type=_CONST_NODE_TYPES.entry(type(form), UNKNOWN),
        val=form,
    )

    if hasattr(form, "meta") and form.meta is not None:  # type: ignore
        meta_ast = _const_node(ctx, form.meta)  # type: ignore

        if not isinstance(meta_ast, Const) or meta_ast.type != ConstType.MAP:
            raise ParserException(f"Meta applied to constant must be a map")

        return descriptor.assoc(meta=meta_ast, children=vec.v(META))

    return descriptor


def parse_ast(ctx: ParserContext, form: LispForm) -> Node:
    """Take a Lisp form as an argument and produce a Basilisp syntax
    tree matching the clojure.tools.analyzer AST spec."""
    if isinstance(form, (llist.List, lseq.Seq)):
        return _list_node(ctx, form)
    elif isinstance(form, vec.Vector):
        if ctx.is_quoted:
            return _const_node(ctx, form)
        return _vector_node(ctx, form)
    elif isinstance(form, lmap.Map):
        if ctx.is_quoted:
            return _const_node(ctx, form)
        return _map_node(ctx, form)
    elif isinstance(form, lset.Set):
        if ctx.is_quoted:
            return _const_node(ctx, form)
        return _set_node(ctx, form)
    elif isinstance(form, sym.Symbol):
        return _symbol_node(ctx, form)
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
