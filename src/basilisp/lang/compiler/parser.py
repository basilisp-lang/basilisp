import builtins
import collections
import contextlib
import logging
import re
import sys
import uuid
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from functools import partial, wraps
from typing import (
    Pattern,
    Union,
    Deque,
    Optional,
    Dict,
    Callable,
    cast,
    Any,
    Collection,
    Set,
    Tuple,
)

import attr

import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.reader as reader
import basilisp.lang.runtime as runtime
import basilisp.lang.seq as lseq
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
from basilisp.lang.compiler.constants import (
    SpecialForm,
    AMPERSAND,
    SYM_MACRO_META_KEY,
    DEFAULT_COMPILER_FILE_PATH,
    SYM_DYNAMIC_META_KEY,
    COL_KW,
    FILE_KW,
    LINE_KW,
    NAME_KW,
    NS_KW,
    SYM_NO_WARN_WHEN_UNUSED_META_KEY,
    DOC_KW,
)
from basilisp.lang.compiler.exception import CompilerException, CompilerPhase
from basilisp.lang.compiler.nodes import (
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
    ReaderLispForm,
    SetBang,
    Assignable,
    Loop,
    NodeOp,
    Recur,
    SpecialFormNode,
    Import,
    ImportAlias,
    LetFn,
    NodeEnv,
)
from basilisp.lang.runtime import Var
from basilisp.lang.typing import LispForm
from basilisp.lang.util import count, genname, munge
from basilisp.util import Maybe, partition

# Parser logging
logger = logging.getLogger(__name__)

# Parser options
WARN_ON_SHADOWED_NAME = "warn_on_shadowed_name"
WARN_ON_SHADOWED_VAR = "warn_on_shadowed_var"
WARN_ON_UNUSED_NAMES = "warn_on_unused_names"

# Lisp AST node keywords
INIT = kw.keyword("init")
META = kw.keyword("meta")
FIXED_ARITY = kw.keyword("fixed-arity")
BODY = kw.keyword("body")
CATCHES = kw.keyword("catches")
FINALLY = kw.keyword("finally")

# Constants used in parsing
AS = kw.keyword("as")
_BUILTINS_NS = "builtins"

# Symbols to be ignored for unused symbol warnings
_IGNORED_SYM = sym.symbol("_")
_MACRO_ENV_SYM = sym.symbol("&env")
_MACRO_FORM_SYM = sym.symbol("&form")
_NO_WARN_UNUSED_SYMS = lset.s(_IGNORED_SYM, _MACRO_ENV_SYM, _MACRO_FORM_SYM)


ParserException = partial(CompilerException, phase=CompilerPhase.PARSING)


@attr.s(auto_attribs=True, slots=True)
class RecurPoint:
    loop_id: str
    args: Collection[Binding] = ()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SymbolTableEntry:
    context: LocalType
    symbol: sym.Symbol
    used: bool = False
    warn_if_unused: bool = True


# pylint: disable=unsupported-membership-test,unsupported-delete-operation,unsupported-assignment-operation
@attr.s(auto_attribs=True, slots=True)
class SymbolTable:
    name: str
    _parent: Optional["SymbolTable"] = None
    _table: Dict[sym.Symbol, SymbolTableEntry] = attr.ib(factory=dict)
    _children: Dict[str, "SymbolTable"] = attr.ib(factory=dict)

    def new_symbol(
        self, s: sym.Symbol, ctx: LocalType, warn_if_unused: bool = True
    ) -> "SymbolTable":
        if s in self._table:
            self._table[s] = attr.evolve(
                self._table[s], context=ctx, symbol=s, warn_if_unused=warn_if_unused
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
            self._table[s] = attr.evolve(old, used=True)
        elif self._parent is not None:
            self._parent.mark_used(s)
        else:  # pragma: no cover
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
    __slots__ = ("_filename", "_is_quoted", "_opts", "_recur_points", "_st")

    def __init__(
        self, filename: Optional[str] = None, opts: Optional[Dict[str, bool]] = None
    ) -> None:
        self._filename = Maybe(filename).or_else_get(DEFAULT_COMPILER_FILE_PATH)
        self._is_quoted: Deque[bool] = collections.deque([])
        self._opts = Maybe(opts).map(lmap.map).or_else_get(lmap.Map.empty())
        self._recur_points: Deque[RecurPoint] = collections.deque([])
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
    def warn_on_shadowed_name(self) -> bool:
        """If True, warn when a name is shadowed in an inner scope.

        Implies warn_on_shadowed_var."""
        return self._opts.entry(WARN_ON_SHADOWED_NAME, False)

    @property
    def warn_on_shadowed_var(self) -> bool:
        """If True, warn when a def'ed Var name is shadowed in an inner scope.

        Implied by warn_on_shadowed_name. The value of warn_on_shadowed_name
        supersedes the value of this flag."""
        return self.warn_on_shadowed_name or self._opts.entry(
            WARN_ON_SHADOWED_VAR, False
        )

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
    def recur_point(self) -> Optional[RecurPoint]:
        try:
            return self._recur_points[-1]
        except IndexError:
            return None

    @contextlib.contextmanager
    def new_recur_point(self, loop_id: str, args: Collection[Any] = ()):
        self._recur_points.append(RecurPoint(loop_id, args=args))
        yield
        self._recur_points.pop()

    @property
    def symbol_table(self) -> SymbolTable:
        return self._st[-1]

    def put_new_symbol(  # pylint: disable=too-many-arguments
        self,
        s: sym.Symbol,
        sym_ctx: LocalType,
        warn_on_shadowed_name: bool = True,
        warn_on_shadowed_var: bool = True,
        warn_if_unused: bool = True,
    ):
        """Add a new symbol to the symbol table.

        This function allows individual warnings to be disabled for one run
        by supplying keyword arguments temporarily disabling those warnings.
        In certain cases, we do not want to issue warnings again for a
        previously checked case, so this is a simple way of disabling these
        warnings for those cases.

        If WARN_ON_SHADOWED_NAME compiler option is active and the
        warn_on_shadowed_name keyword argument is True, then a warning will be
        emitted if a local name is shadowed by another local name. Note that
        WARN_ON_SHADOWED_NAME implies WARN_ON_SHADOWED_VAR.

        If WARN_ON_SHADOWED_VAR compiler option is active and the
        warn_on_shadowed_var keyword argument is True, then a warning will be
        emitted if a named var is shadowed by a local name."""
        st = self.symbol_table
        if warn_on_shadowed_name and self.warn_on_shadowed_name:
            if st.find_symbol(s) is not None:
                logger.warning(f"name '{s}' shadows name from outer scope")
        if (
            warn_on_shadowed_name or warn_on_shadowed_var
        ) and self.warn_on_shadowed_var:
            if self.current_ns.find(s) is not None:
                logger.warning(f"name '{s}' shadows def'ed Var from outer scope")
        if s.meta is not None and s.meta.entry(SYM_NO_WARN_WHEN_UNUSED_META_KEY, None):
            warn_if_unused = False
        st.new_symbol(s, sym_ctx, warn_if_unused=warn_if_unused)

    @contextlib.contextmanager
    def new_symbol_table(self, name):
        old_st = self.symbol_table
        with old_st.new_frame(name, self.warn_on_unused_names) as st:
            self._st.append(st)
            yield st
            self._st.pop()

    def get_node_env(self):
        return NodeEnv(ns=self.current_ns, file=self.filename)


def _is_macro(v: Var) -> bool:
    """Return True if the Var holds a macro function."""
    return (
        Maybe(v.meta)
        .map(lambda m: m.get(SYM_MACRO_META_KEY, None))  # type: ignore
        .or_else_get(False)
    )


ParseFunction = Callable[[ParserContext, Union[LispForm, lseq.Seq]], Node]


def _loc(form: Union[LispForm, lseq.Seq]) -> Optional[Tuple[int, int]]:
    """Fetch the location of the form in the original filename from the
    input form, if it has metadata."""
    try:
        meta = form.meta  # type: ignore
        line = meta.get(reader.READER_LINE_KW)  # type: ignore
        col = meta.get(reader.READER_COL_KW)  # type: ignore
    except AttributeError:
        return None
    else:
        assert isinstance(line, int) and isinstance(col, int)
        return line, col


def _with_loc(f: ParseFunction):
    """Attach any available location information from the input form to
    the node environment returned from the parsing function."""

    @wraps(f)
    def _parse_form(ctx: ParserContext, form: Union[LispForm, lseq.Seq]) -> Node:
        form_loc = _loc(form)
        if form_loc is None:
            return f(ctx, form)
        else:
            return f(ctx, form).fix_missing_locations(form_loc)

    return _parse_form


def _clean_meta(meta: Optional[lmap.Map]) -> Optional[lmap.Map]:
    """Remove reader metadata from the form's meta map."""
    if meta is None:
        return None
    else:
        new_meta = meta.discard(reader.READER_LINE_KW, reader.READER_COL_KW)
        return None if len(new_meta) == 0 else new_meta


def _with_meta(gen_node):
    """Wraps the node generated by gen_node in a :with-meta AST node if the
    original form has .

    :with-meta AST nodes are used for non-quoted collection literals and for
    function expressions."""

    @wraps(gen_node)
    def with_meta(ctx: ParserContext, form: lmap.Map) -> Node:
        assert not ctx.is_quoted, "with-meta nodes are not used in quoted expressions"

        descriptor = gen_node(ctx, form)

        if hasattr(form, "meta"):  # type: ignore
            form_meta = _clean_meta(form.meta)
            if form_meta is not None:
                meta_ast = _parse_ast(ctx, form_meta)  # type: ignore
                assert isinstance(meta_ast, MapNode) or (
                    isinstance(meta_ast, Const) and meta_ast.type == ConstType.MAP
                )
                return WithMeta(
                    form=form, meta=meta_ast, expr=descriptor, env=ctx.get_node_env()
                )

        return descriptor

    return with_meta


def _def_ast(  # pylint: disable=too-many-locals
    ctx: ParserContext, form: lseq.Seq
) -> Def:
    assert form.first == SpecialForm.DEF

    nelems = count(form)
    if nelems not in (2, 3, 4):
        raise ParserException(
            f"def forms must have between 2 and 4 elements, as in: (def name docstring? init?)",
            form=form,
        )

    name = runtime.nth(form, 1)
    if not isinstance(name, sym.Symbol):
        raise ParserException(f"def names must be symbols, not {type(name)}", form=name)

    if nelems == 2:
        init = None
        doc = None
        children = vec.Vector.empty()
    elif nelems == 3:
        init = _parse_ast(ctx, runtime.nth(form, 2))
        doc = None
        children = vec.v(INIT)
    else:
        init = _parse_ast(ctx, runtime.nth(form, 3))
        doc = runtime.nth(form, 2)
        if not isinstance(doc, str):
            raise ParserException("def docstring must be a string", form=doc)
        children = vec.v(INIT)

    # Attach metadata relevant for the current process below.
    def_loc = _loc(form)
    def_node_env = ctx.get_node_env()
    def_meta = _clean_meta(
        name.meta.update(
            lmap.map(
                {
                    COL_KW: def_loc[1] if def_loc is not None else None,
                    FILE_KW: def_node_env.file,
                    LINE_KW: def_loc[0] if def_loc is not None else None,
                    NAME_KW: name,
                    NS_KW: ctx.current_ns,
                }
            )
        )
    )
    assert def_meta is not None, "def metadata must be defined at this point"
    if doc is not None:
        def_meta = def_meta.assoc(DOC_KW, doc)

    # Generation fails later if we use the same symbol we received, since
    # its meta may contain values which fail to compile.
    bare_name = sym.symbol(name.name)

    ns_sym = sym.symbol(ctx.current_ns.name)
    var = Var.intern_unbound(
        ns_sym,
        bare_name,
        dynamic=def_meta.entry(SYM_DYNAMIC_META_KEY, False),
        meta=def_meta,
    )
    descriptor = Def(
        form=form,
        name=bare_name,
        var=var,
        init=init,
        doc=doc,
        children=children,
        env=def_node_env,
    )

    # We still have to compile the meta here down to Python source code, so
    # anything which is not constant below needs to be valid Basilisp code
    # at the site it is called.
    #
    # We are roughly generating code like this:
    #
    # (def ^{:col  1
    #        :file "<REPL Input>"
    #        :line 1
    #        :name 'some-name
    #        :ns   ((.- basilisp.lang.runtime/Namespace get) 'user)}
    #       some-name
    #       "some value")
    meta_ast = _parse_ast(
        ctx,
        def_meta.update(
            {
                NAME_KW: llist.l(SpecialForm.QUOTE, bare_name),
                NS_KW: llist.l(
                    llist.l(
                        SpecialForm.INTEROP_PROP,
                        sym.symbol("Namespace", "basilisp.lang.runtime"),
                        sym.symbol("get"),
                    ),
                    llist.l(SpecialForm.QUOTE, sym.symbol(ctx.current_ns.name)),
                ),
            }
        ),
    )

    assert (isinstance(meta_ast, Const) and meta_ast.type == ConstType.MAP) or (
        isinstance(meta_ast, MapNode)
    )
    existing_children = cast(vec.Vector, descriptor.children)
    return descriptor.assoc(
        meta=meta_ast, children=vec.vector(runtime.cons(META, existing_children))
    )


def _do_ast(ctx: ParserContext, form: lseq.Seq) -> Do:
    assert form.first == SpecialForm.DO
    *statements, ret = map(partial(_parse_ast, ctx), form.rest)
    return Do(
        form=form, statements=vec.vector(statements), ret=ret, env=ctx.get_node_env()
    )


def __fn_method_ast(  # pylint: disable=too-many-branches
    ctx: ParserContext, form: lseq.Seq, fnname: Optional[sym.Symbol] = None
) -> FnMethod:
    with ctx.new_symbol_table("fn-method"):
        params = form.first
        if not isinstance(params, vec.Vector):
            raise ParserException(
                "function arity arguments must be a vector", form=params
            )

        has_vargs, vargs_idx = False, 0
        param_nodes = []
        for i, s in enumerate(params):
            if not isinstance(s, sym.Symbol):
                raise ParserException(
                    "function arity parameter name must be a symbol", form=s
                )

            if s == AMPERSAND:
                has_vargs = True
                vargs_idx = i
                break

            param_nodes.append(
                Binding(
                    form=s,
                    name=s.name,
                    local=LocalType.ARG,
                    arg_id=i,
                    is_variadic=False,
                    env=ctx.get_node_env(),
                )
            )

            ctx.put_new_symbol(s, LocalType.ARG)

        if has_vargs:
            try:
                vargs_sym = params[vargs_idx + 1]

                if not isinstance(vargs_sym, sym.Symbol):
                    raise ParserException(
                        "function rest parameter name must be a symbol", form=vargs_sym
                    )

                param_nodes.append(
                    Binding(
                        form=vargs_sym,
                        name=vargs_sym.name,
                        local=LocalType.ARG,
                        arg_id=vargs_idx + 1,
                        is_variadic=True,
                        env=ctx.get_node_env(),
                    )
                )

                ctx.put_new_symbol(vargs_sym, LocalType.ARG)
            except IndexError:
                raise ParserException(
                    "Expected variadic argument name after '&'", form=params
                ) from None

        fn_loop_id = genname("fn_arity" if fnname is None else fnname.name)
        with ctx.new_recur_point(fn_loop_id, param_nodes):
            body = list(map(partial(_parse_ast, ctx), form.rest))
            if body:
                *stmts, ret = body
            else:
                stmts, ret = [], _const_node(ctx, None)

            method = FnMethod(
                form=form,
                loop_id=fn_loop_id,
                params=vec.vector(param_nodes),
                is_variadic=has_vargs,
                fixed_arity=len(param_nodes) - int(has_vargs),
                body=Do(
                    form=form.rest,
                    statements=vec.vector(stmts),
                    ret=ret,
                    is_body=True,
                    # Use the argument vector or first body statement, whichever
                    # exists, for metadata.
                    env=ctx.get_node_env(),
                ),
                # Use the argument vector for fetching line/col since the
                # form itself is a sequence with no meaningful metadata.
                env=ctx.get_node_env(),
            )
            method.visit(_assert_recur_is_tail)
            return method


def _fn_ast(  # pylint: disable=too-many-branches  # noqa: MC0001
    ctx: ParserContext, form: lseq.Seq
) -> Fn:
    assert form.first == SpecialForm.FN

    idx = 1

    with ctx.new_symbol_table("fn"):
        try:
            name = runtime.nth(form, idx)
        except IndexError:
            raise ParserException(
                "fn form must match: (fn* name? [arg*] body*) or (fn* name? method*)",
                form=form,
            )

        if isinstance(name, sym.Symbol):
            ctx.put_new_symbol(name, LocalType.FN, warn_if_unused=False)
            name_node: Optional[Binding] = Binding(
                form=name, name=name.name, local=LocalType.FN, env=ctx.get_node_env()
            )
            idx += 1
        elif isinstance(name, (llist.List, vec.Vector)):
            name = None
            name_node = None
        else:
            raise ParserException(
                "fn form must match: (fn* name? [arg*] body*) or (fn* name? method*)",
                form=form,
            )

        try:
            arity_or_args = runtime.nth(form, idx)
        except IndexError:
            raise ParserException(
                "fn form expects either multiple arities or a vector of arguments",
                form=form,
            )

        if isinstance(arity_or_args, llist.List):
            methods = vec.vector(
                map(
                    partial(__fn_method_ast, ctx, fnname=name),
                    runtime.nthrest(form, idx),
                )
            )
        elif isinstance(arity_or_args, vec.Vector):
            methods = vec.v(
                __fn_method_ast(ctx, runtime.nthrest(form, idx), fnname=name)
            )
        else:
            raise ParserException(
                "fn form must match: (fn* name? [arg*] body*) or (fn* name? method*)",
                form=form,
            )

        assert count(methods) > 0, "fn must have at least one arity"

        fixed_arities: Set[int] = set()
        fixed_arity_for_variadic: Optional[int] = None
        num_variadic = 0
        for method in methods:
            if fixed_arity_for_variadic is not None:
                if method.fixed_arity >= fixed_arity_for_variadic:
                    raise ParserException(
                        "fn may not have a method with fixed arity greater than "
                        "fixed arity of variadic function",
                        form=method.form,
                    )
            if method.is_variadic:
                if num_variadic > 0:
                    raise ParserException(
                        "fn may have at most 1 variadic arity", form=method.form
                    )
                fixed_arity_for_variadic = method.fixed_arity
                num_variadic += 1
            else:
                if method.fixed_arity in fixed_arities:
                    raise ParserException(
                        "fn may not have multiple methods with the same fixed arity",
                        form=method.form,
                    )
                fixed_arities.add(method.fixed_arity)

        if fixed_arity_for_variadic is not None and any(
            [fixed_arity_for_variadic < arity for arity in fixed_arities]
        ):
            raise ParserException(
                "variadic arity may not have fewer fixed arity arguments than any other arities",
                form=form,
            )

        return Fn(
            form=form,
            is_variadic=num_variadic == 1,
            max_fixed_arity=max([node.fixed_arity for node in methods]),
            methods=methods,
            local=name_node,
            # Use the function name for metadata if it exists or otherwise
            # try the `fn*` symbol.
            env=ctx.get_node_env(),
        )


def _host_call_ast(ctx: ParserContext, form: lseq.Seq) -> HostCall:
    assert isinstance(form.first, sym.Symbol)

    method = form.first
    assert isinstance(method, sym.Symbol), "host interop field must be a symbol"
    assert method.name.startswith(".")

    if not count(form) >= 2:
        raise ParserException(
            "host interop calls must be 2 or more elements long", form=form
        )

    return HostCall(
        form=form,
        method=method.name[1:],
        target=_parse_ast(ctx, runtime.nth(form, 1)),
        args=vec.vector(map(partial(_parse_ast, ctx), runtime.nthrest(form, 2))),
        env=ctx.get_node_env(),
    )


def _host_prop_ast(ctx: ParserContext, form: lseq.Seq) -> HostField:
    assert isinstance(form.first, sym.Symbol)

    field = form.first
    assert isinstance(field, sym.Symbol), "host interop field must be a symbol"

    nelems = count(form)
    assert field.name.startswith(".-")

    if field.name == ".-":
        try:
            field = runtime.nth(form, 2)
        except IndexError:
            raise ParserException(
                "host interop prop must be exactly 3 elems long: (.- target field)",
                form=form,
            )
        else:
            if not isinstance(field, sym.Symbol):
                raise ParserException("host interop field must be a symbol", form=form)

        if not nelems == 3:
            raise ParserException(
                "host interop prop must be exactly 3 elems long: (.- target field)",
                form=form,
            )

        return HostField(
            form=form,
            field=field.name,
            target=_parse_ast(ctx, runtime.nth(form, 1)),
            is_assignable=True,
            env=ctx.get_node_env(),
        )
    else:
        if not nelems == 2:
            raise ParserException(
                "host interop prop must be exactly 2 elements long: (.-field target)",
                form=form,
            )

        return HostField(
            form=form,
            field=field.name[2:],
            target=_parse_ast(ctx, runtime.nth(form, 1)),
            is_assignable=True,
            env=ctx.get_node_env(),
        )


def _host_interop_ast(  # pylint: disable=too-many-branches
    ctx: ParserContext, form: lseq.Seq
) -> Union[HostCall, HostField]:
    assert form.first == SpecialForm.INTEROP_CALL
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
                    "host field accesses must be exactly 3 elements long", form=form
                )

            return HostField(
                form=form,
                field=maybe_m_or_f.name[1:],
                target=_parse_ast(ctx, runtime.nth(form, 1)),
                is_assignable=True,
                env=ctx.get_node_env(),
            )
        else:
            return HostCall(
                form=form,
                method=maybe_m_or_f.name,
                target=_parse_ast(ctx, runtime.nth(form, 1)),
                args=vec.vector(
                    map(partial(_parse_ast, ctx), runtime.nthrest(form, 3))
                ),
                env=ctx.get_node_env(),
            )
    elif isinstance(maybe_m_or_f, (llist.List, lseq.Seq)):
        # Likewise, I emit :host-call for forms like (. target (method arg1 ...)).
        method = maybe_m_or_f.first
        if not isinstance(method, sym.Symbol):
            raise ParserException("host call method must be a symbol", form=method)

        return HostCall(
            form=form,
            method=method.name[1:] if method.name.startswith("-") else method.name,
            target=_parse_ast(ctx, runtime.nth(form, 1)),
            args=vec.vector(map(partial(_parse_ast, ctx), maybe_m_or_f.rest)),
            env=ctx.get_node_env(),
        )
    else:
        raise ParserException(
            "host interop forms must take the form: "
            "(. instance (method args*)), "
            "(. instance method args*), "
            "(. instance -field), ",
            form=form,
        )


def _if_ast(ctx: ParserContext, form: lseq.Seq) -> If:
    assert form.first == SpecialForm.IF

    nelems = count(form)
    if nelems not in (3, 4):
        raise ParserException(
            "if forms must have either 3 or 4 elements, as in: (if test then else?)",
            form=form,
        )

    if nelems == 4:
        else_node = _parse_ast(ctx, runtime.nth(form, 3))
    else:
        else_node = _const_node(ctx, None)

    return If(
        form=form,
        test=_parse_ast(ctx, runtime.nth(form, 1)),
        then=_parse_ast(ctx, runtime.nth(form, 2)),
        else_=else_node,
        env=ctx.get_node_env(),
    )


def _import_ast(  # pylint: disable=too-many-branches
    ctx: ParserContext, form: lseq.Seq
) -> Import:
    assert form.first == SpecialForm.IMPORT

    aliases = []
    for f in form.rest:
        if isinstance(f, sym.Symbol):
            module_name = f
            module_alias = module_name.name.split(".", maxsplit=1)[0]
        elif isinstance(f, vec.Vector):
            if len(f) != 3:
                raise ParserException(
                    "import alias must take the form: [module :as alias]", form=f
                )
            module_name = f.entry(0)
            if not isinstance(module_name, sym.Symbol):
                raise ParserException("Python module name must be a symbol", form=f)
            if not AS == f.entry(1):
                raise ParserException("expected :as alias for Python import", form=f)
            module_alias_sym = f.entry(2)
            if not isinstance(module_alias_sym, sym.Symbol):
                raise ParserException("Python module alias must be a symbol", form=f)
            module_alias = module_alias_sym.name
        else:
            raise ParserException("symbol or vector expected for import*", form=f)

        aliases.append(
            ImportAlias(
                form=f,
                name=module_name.name,
                alias=module_alias,
                env=ctx.get_node_env(),
            )
        )

    return Import(form=form, aliases=aliases, env=ctx.get_node_env())


def _invoke_ast(ctx: ParserContext, form: Union[llist.List, lseq.Seq]) -> Node:
    fn = _parse_ast(ctx, form.first)

    if fn.op == NodeOp.VAR and isinstance(fn, VarRef):
        if _is_macro(fn.var):
            try:
                expanded = fn.var.value(form, *form.rest)
                expanded_ast = _parse_ast(ctx, expanded)

                # Verify that macroexpanded code also does not have any
                # non-tail recur forms
                if ctx.recur_point is not None:
                    _assert_recur_is_tail(expanded_ast)

                return expanded_ast.assoc(
                    raw_forms=cast(vec.Vector, expanded_ast.raw_forms).cons(form)
                )
            except Exception as e:
                raise CompilerException(
                    "error occurred during macroexpansion",
                    form=form,
                    phase=CompilerPhase.MACROEXPANSION,
                ) from e

    return Invoke(
        form=form,
        fn=fn,
        args=vec.vector(map(partial(_parse_ast, ctx), form.rest)),
        env=ctx.get_node_env(),
    )


def _let_ast(ctx: ParserContext, form: lseq.Seq) -> Let:
    assert form.first == SpecialForm.LET
    nelems = count(form)

    if nelems < 3:
        raise ParserException(
            "let forms must have bindings and at least one body form", form=form
        )

    bindings = runtime.nth(form, 1)
    if not isinstance(bindings, vec.Vector):
        raise ParserException("let bindings must be a vector", form=bindings)
    elif len(bindings) == 0:
        raise ParserException(
            "let form must have at least one pair of bindings", form=bindings
        )
    elif len(bindings) % 2 != 0:
        raise ParserException(
            "let bindings must appear in name-value pairs", form=bindings
        )

    with ctx.new_symbol_table("let"):
        binding_nodes = []
        for name, value in partition(bindings, 2):
            if not isinstance(name, sym.Symbol):
                raise ParserException("let binding name must be a symbol", form=name)

            binding_nodes.append(
                Binding(
                    form=name,
                    name=name.name,
                    local=LocalType.LET,
                    init=_parse_ast(ctx, value),
                    children=vec.v(INIT),
                    env=ctx.get_node_env(),
                )
            )

            ctx.put_new_symbol(name, LocalType.LET)

        let_body = runtime.nthrest(form, 2)
        *statements, ret = map(partial(_parse_ast, ctx), let_body)
        return Let(
            form=form,
            bindings=vec.vector(binding_nodes),
            body=Do(
                form=let_body,
                statements=vec.vector(statements),
                ret=ret,
                is_body=True,
                env=ctx.get_node_env(),
            ),
            env=ctx.get_node_env(),
        )


def _loop_ast(ctx: ParserContext, form: lseq.Seq) -> Loop:
    assert form.first == SpecialForm.LOOP
    nelems = count(form)

    if nelems < 3:
        raise ParserException(
            "loop forms must have bindings and at least one body form", form=form
        )

    bindings = runtime.nth(form, 1)
    if not isinstance(bindings, vec.Vector):
        raise ParserException("loop bindings must be a vector", form=bindings)
    elif len(bindings) % 2 != 0:
        raise ParserException(
            "loop bindings must appear in name-value pairs", form=bindings
        )

    loop_id = genname("loop")
    with ctx.new_symbol_table(loop_id):
        binding_nodes = []
        for name, value in partition(bindings, 2):
            if not isinstance(name, sym.Symbol):
                raise ParserException("loop binding name must be a symbol", form=name)

            binding_nodes.append(
                Binding(
                    form=name,
                    name=name.name,
                    local=LocalType.LOOP,
                    init=_parse_ast(ctx, value),
                    env=ctx.get_node_env(),
                )
            )

            ctx.put_new_symbol(name, LocalType.LOOP)

        with ctx.new_recur_point(loop_id, binding_nodes):
            loop_body = runtime.nthrest(form, 2)
            *statements, ret = map(partial(_parse_ast, ctx), loop_body)
            loop_node = Loop(
                form=form,
                bindings=vec.vector(binding_nodes),
                body=Do(
                    form=loop_body,
                    statements=vec.vector(statements),
                    ret=ret,
                    is_body=True,
                    env=ctx.get_node_env(),
                ),
                loop_id=loop_id,
                env=ctx.get_node_env(),
            )
            loop_node.visit(_assert_recur_is_tail)
            return loop_node


def _quote_ast(ctx: ParserContext, form: lseq.Seq) -> Quote:
    assert form.first == SpecialForm.QUOTE

    with ctx.quoted():
        expr = _parse_ast(ctx, runtime.nth(form, 1))
        assert isinstance(expr, Const), "Quoted expressions must yield :const nodes"
        return Quote(form=form, expr=expr, is_literal=True, env=ctx.get_node_env())


def _assert_no_recur(node: Node) -> None:
    """Assert that `recur` forms do not appear in any position of this or
    child AST nodes."""
    if node.op == NodeOp.RECUR:
        raise ParserException(
            "recur must appear in tail position", form=node.form, lisp_ast=node
        )
    elif node.op in {NodeOp.FN, NodeOp.LOOP}:
        pass
    else:
        node.visit(_assert_no_recur)


def _assert_recur_is_tail(node: Node) -> None:  # pylint: disable=too-many-branches
    """Assert that `recur` forms only appear in the tail position of this
    or child AST nodes.

    `recur` forms may only appear in `do` nodes (both literal and synthetic
    `do` nodes) and in either the :then or :else expression of an `if` node."""
    if node.op == NodeOp.DO:
        assert isinstance(node, Do)
        for child in node.statements:
            _assert_no_recur(child)
        _assert_recur_is_tail(node.ret)
    elif node.op in {NodeOp.FN, NodeOp.FN_METHOD}:
        assert isinstance(node, (Fn, FnMethod))
        node.visit(_assert_recur_is_tail)
    elif node.op == NodeOp.IF:
        assert isinstance(node, If)
        _assert_no_recur(node.test)
        _assert_recur_is_tail(node.then)
        _assert_recur_is_tail(node.else_)
    elif node.op in {NodeOp.LET, NodeOp.LETFN}:
        assert isinstance(node, (Let, LetFn))
        for binding in node.bindings:
            assert binding.init is not None
            _assert_no_recur(binding.init)
        _assert_recur_is_tail(node.body)
    elif node.op == NodeOp.LOOP:
        assert isinstance(node, Loop)
        for binding in node.bindings:
            assert binding.init is not None
            _assert_no_recur(binding.init)
    elif node.op == NodeOp.RECUR:
        pass
    elif node.op == NodeOp.TRY:
        assert isinstance(node, Try)
        _assert_recur_is_tail(node.body)
        for catch in node.catches:
            _assert_recur_is_tail(catch)
        if node.finally_:
            _assert_no_recur(node.finally_)
    else:
        node.visit(_assert_no_recur)


def _recur_ast(ctx: ParserContext, form: lseq.Seq) -> Recur:
    assert form.first == SpecialForm.RECUR

    if ctx.recur_point is None:
        raise ParserException("no recur point defined for recur", form=form)

    if len(ctx.recur_point.args) != count(form.rest):
        raise ParserException(
            "recur arity does not match last recur point arity", form=form
        )

    exprs = vec.vector(map(partial(_parse_ast, ctx), form.rest))
    return Recur(
        form=form, exprs=exprs, loop_id=ctx.recur_point.loop_id, env=ctx.get_node_env()
    )


def _set_bang_ast(ctx: ParserContext, form: lseq.Seq) -> SetBang:
    assert form.first == SpecialForm.SET_BANG
    nelems = count(form)

    if nelems != 3:
        raise ParserException(
            "set! forms must contain exactly 3 elements: (set! target value)", form=form
        )

    target = _parse_ast(ctx, runtime.nth(form, 1))
    if not isinstance(target, Assignable):
        raise ParserException(
            f"cannot set! targets of type {type(target)}", form=target
        )

    if not target.is_assignable:
        raise ParserException(
            f"cannot set! target which is not assignable", form=target
        )

    return SetBang(
        form=form,
        target=target,
        val=_parse_ast(ctx, runtime.nth(form, 2)),
        env=ctx.get_node_env(),
    )


def _throw_ast(ctx: ParserContext, form: lseq.Seq) -> Throw:
    assert form.first == SpecialForm.THROW
    return Throw(
        form=form,
        exception=_parse_ast(ctx, runtime.nth(form, 1)),
        env=ctx.get_node_env(),
    )


def _catch_ast(ctx: ParserContext, form: lseq.Seq) -> Catch:
    assert form.first == SpecialForm.CATCH
    nelems = count(form)

    if nelems < 4:
        raise ParserException(
            "catch forms must contain at least 4 elements: (catch class local body*)",
            form=form,
        )

    catch_cls = _parse_ast(ctx, runtime.nth(form, 1))
    if not isinstance(catch_cls, (MaybeClass, MaybeHostForm)):
        raise ParserException(
            "catch forms must name a class type to catch", form=catch_cls
        )

    local_name = runtime.nth(form, 2)
    if not isinstance(local_name, sym.Symbol):
        raise ParserException("catch local must be a symbol", form=local_name)

    with ctx.new_symbol_table("catch"):
        ctx.put_new_symbol(local_name, LocalType.CATCH)

        catch_body = runtime.nthrest(form, 3)
        *catch_statements, catch_ret = map(partial(_parse_ast, ctx), catch_body)
        return Catch(
            form=form,
            class_=catch_cls,
            local=Binding(
                form=local_name,
                name=local_name.name,
                local=LocalType.CATCH,
                env=ctx.get_node_env(),
            ),
            body=Do(
                form=catch_body,
                statements=vec.vector(catch_statements),
                ret=catch_ret,
                is_body=True,
                env=ctx.get_node_env(),
            ),
            env=ctx.get_node_env(),
        )


def _try_ast(  # pylint: disable=too-many-branches
    ctx: ParserContext, form: lseq.Seq
) -> Try:
    assert form.first == SpecialForm.TRY

    try_exprs = []
    catches = []
    finally_: Optional[Do] = None
    for expr in form.rest:
        if isinstance(expr, (llist.List, lseq.Seq)):
            if expr.first == SpecialForm.CATCH:
                if finally_:
                    raise ParserException(
                        "catch forms may not appear after finally forms in a try",
                        form=expr,
                    )
                catches.append(_catch_ast(ctx, expr))
                continue
            elif expr.first == SpecialForm.FINALLY:
                if finally_ is not None:
                    raise ParserException(
                        "try forms may not contain multiple finally forms", form=expr
                    )
                *finally_stmts, finally_ret = map(partial(_parse_ast, ctx), expr.rest)
                finally_ = Do(
                    form=expr.rest,
                    statements=vec.vector(finally_stmts),
                    ret=finally_ret,
                    is_body=True,
                    env=ctx.get_node_env(),
                )
                continue

        parsed = _parse_ast(ctx, expr)

        if catches:
            raise ParserException(
                "try body expressions may not appear after catch forms", form=expr
            )
        if finally_:
            raise ParserException(
                "try body expressions may not appear after finally forms", form=expr
            )

        try_exprs.append(parsed)

    assert all(
        [isinstance(node, Catch) for node in catches]
    ), "All catch statements must be catch ops"

    *try_statements, try_ret = try_exprs
    return Try(
        form=form,
        body=Do(
            form=form,
            statements=vec.vector(try_statements),
            ret=try_ret,
            is_body=True,
            env=ctx.get_node_env(),
        ),
        catches=vec.vector(catches),
        finally_=finally_,
        children=vec.v(BODY, CATCHES, FINALLY)
        if finally_ is not None
        else vec.v(BODY, CATCHES),
        env=ctx.get_node_env(),
    )


def _var_ast(ctx: ParserContext, form: lseq.Seq) -> VarRef:
    assert form.first == SpecialForm.VAR

    nelems = count(form)
    if nelems != 2:
        raise ParserException(
            "var special forms must contain 2 elements: (var sym)", form=form
        )

    var_sym = runtime.nth(form, 1)
    if not isinstance(var_sym, sym.Symbol):
        raise ParserException("vars may only be resolved for symbols", form=form)

    if var_sym.ns is None:
        var = runtime.resolve_var(sym.symbol(var_sym.name, ctx.current_ns.name))
    else:
        var = runtime.resolve_var(var_sym)

    if var is None:
        raise ParserException(f"cannot resolve var {var_sym}", form=form)

    return VarRef(form=var_sym, var=var, return_var=True, env=ctx.get_node_env())


SpecialFormHandler = Callable[[ParserContext, lseq.Seq], SpecialFormNode]
_SPECIAL_FORM_HANDLERS: Dict[sym.Symbol, SpecialFormHandler] = {
    SpecialForm.DEF: _def_ast,
    SpecialForm.DO: _do_ast,
    SpecialForm.FN: _fn_ast,
    SpecialForm.IF: _if_ast,
    SpecialForm.IMPORT: _import_ast,
    SpecialForm.INTEROP_CALL: _host_interop_ast,
    SpecialForm.LET: _let_ast,
    SpecialForm.LOOP: _loop_ast,
    SpecialForm.QUOTE: _quote_ast,
    SpecialForm.RECUR: _recur_ast,
    SpecialForm.SET_BANG: _set_bang_ast,
    SpecialForm.THROW: _throw_ast,
    SpecialForm.TRY: _try_ast,
    SpecialForm.VAR: _var_ast,
}


def _list_node(ctx: ParserContext, form: lseq.Seq) -> Node:
    if ctx.is_quoted:
        return _const_node(ctx, form)

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


def __resolve_namespaced_symbol(  # pylint: disable=too-many-branches
    ctx: ParserContext, form: sym.Symbol
) -> Union[MaybeClass, MaybeHostForm, VarRef]:
    """Resolve a namespaced symbol into a Python name or Basilisp Var."""
    assert form.ns is not None

    if form.ns == ctx.current_ns.name:
        v = ctx.current_ns.find(sym.symbol(form.name))
        if v is not None:
            return VarRef(form=form, var=v, env=ctx.get_node_env())
    elif form.ns == _BUILTINS_NS:
        return MaybeClass(
            form=form,
            class_=munge(form.name, allow_builtins=True),
            env=ctx.get_node_env(),
        )

    if "." in form.name:
        raise ParserException(
            "symbol names may not contain the '.' operator", form=form
        )

    ns_sym = sym.symbol(form.ns)
    if ns_sym in ctx.current_ns.imports or ns_sym in ctx.current_ns.import_aliases:
        # We still import Basilisp code, so we'll want to make sure
        # that the symbol isn't referring to a Basilisp Var first
        v = Var.find(form)
        if v is not None:
            return VarRef(form=form, var=v, env=ctx.get_node_env())

        # Fetch the full namespace name for the aliased namespace/module.
        # We don't need this for actually generating the link later, but
        # we _do_ need it for fetching a reference to the module to check
        # for membership.
        if ns_sym in ctx.current_ns.import_aliases:
            ns = ctx.current_ns.import_aliases[ns_sym]
            assert ns is not None
            ns_name = ns.name
        else:
            ns_name = ns_sym.name

        safe_module_name = munge(ns_name)
        assert (
            safe_module_name in sys.modules
        ), f"Module '{safe_module_name}' is not imported"
        ns_module = sys.modules[safe_module_name]
        safe_name = munge(form.name)

        # Try without allowing builtins first
        if safe_name in ns_module.__dict__:
            return MaybeHostForm(
                form=form,
                class_=munge(ns_sym.name),
                field=safe_name,
                env=ctx.get_node_env(),
            )

        # Then allow builtins
        safe_name = munge(form.name, allow_builtins=True)
        if safe_name not in ns_module.__dict__:
            raise ParserException("can't identify aliased form", form=form)

        # Aliased imports generate code which uses the import alias, so we
        # don't need to care if this is an import or an alias.
        return MaybeHostForm(
            form=form,
            class_=munge(ns_sym.name),
            field=safe_name,
            env=ctx.get_node_env(),
        )
    elif ns_sym in ctx.current_ns.aliases:
        aliased_ns: runtime.Namespace = ctx.current_ns.aliases[ns_sym]
        v = Var.find(sym.symbol(form.name, ns=aliased_ns.name))
        if v is None:
            raise ParserException(
                f"unable to resolve symbol '{sym.symbol(form.name, ns_sym.name)}' in this context",
                form=form,
            )
        return VarRef(form=form, var=v, env=ctx.get_node_env())
    else:
        raise ParserException(
            f"unable to resolve symbol '{form}' in this context", form=form
        )


def __resolve_bare_symbol(
    ctx: ParserContext, form: sym.Symbol
) -> Union[MaybeClass, VarRef]:
    """Resolve a non-namespaced symbol into a Python name or a local
    Basilisp Var."""
    assert form.ns is None

    # Look up the symbol in the namespace mapping of the current namespace.
    v = ctx.current_ns.find(form)
    if v is not None:
        return VarRef(form=form, var=v, env=ctx.get_node_env())

    if "." in form.name:
        raise ParserException(
            "symbol names may not contain the '.' operator", form=form
        )

    if form.name in builtins.__dict__:
        return MaybeClass(
            form=form,
            class_=munge(form.name, allow_builtins=True),
            env=ctx.get_node_env(),
        )

    assert form.name not in ctx.current_ns.module.__dict__
    raise ParserException(
        f"unable to resolve symbol '{form}' in this context", form=form
    )


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
        return __resolve_namespaced_symbol(ctx, form)
    else:
        return __resolve_bare_symbol(ctx, form)


def _symbol_node(
    ctx: ParserContext, form: sym.Symbol
) -> Union[Const, Local, MaybeClass, MaybeHostForm, VarRef]:
    if ctx.is_quoted:
        return _const_node(ctx, form)

    sym_entry = ctx.symbol_table.find_symbol(form)
    if sym_entry is not None:
        ctx.symbol_table.mark_used(form)
        return Local(
            form=form,
            name=form.name,
            local=sym_entry.context,
            is_assignable=False,
            env=ctx.get_node_env(),
        )

    return _resolve_sym(ctx, form)


@_with_meta
def _map_node(ctx: ParserContext, form: lmap.Map) -> MapNode:
    keys, vals = [], []
    for k, v in form.items():
        keys.append(_parse_ast(ctx, k))
        vals.append(_parse_ast(ctx, v))

    return MapNode(
        form=form, keys=vec.vector(keys), vals=vec.vector(vals), env=ctx.get_node_env()
    )


@_with_meta
def _set_node(ctx: ParserContext, form: lset.Set) -> SetNode:
    return SetNode(
        form=form,
        items=vec.vector(map(partial(_parse_ast, ctx), form)),
        env=ctx.get_node_env(),
    )


@_with_meta
def _vector_node(ctx: ParserContext, form: vec.Vector) -> VectorNode:
    return VectorNode(
        form=form,
        items=vec.vector(map(partial(_parse_ast, ctx), form)),
        env=ctx.get_node_env(),
    )


_CONST_NODE_TYPES = {  # type: ignore
    bool: ConstType.BOOL,
    complex: ConstType.NUMBER,
    datetime: ConstType.INST,
    Decimal: ConstType.DECIMAL,
    float: ConstType.NUMBER,
    Fraction: ConstType.FRACTION,
    int: ConstType.NUMBER,
    kw.Keyword: ConstType.KEYWORD,
    llist.List: ConstType.SEQ,
    lmap.Map: ConstType.MAP,
    lset.Set: ConstType.SET,
    lseq.Seq: ConstType.SEQ,
    type(re.compile("")): ConstType.REGEX,
    sym.Symbol: ConstType.SYMBOL,
    str: ConstType.STRING,
    type(None): ConstType.NIL,
    uuid.UUID: ConstType.UUID,
    vec.Vector: ConstType.VECTOR,
}


def _const_node(ctx: ParserContext, form: ReaderLispForm) -> Const:
    assert (
        (
            ctx.is_quoted
            and isinstance(
                form, (sym.Symbol, vec.Vector, llist.List, lmap.Map, lset.Set, lseq.Seq)
            )
        )
        or (isinstance(form, (llist.List, lseq.Seq)) and form.is_empty)
        or isinstance(
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
    )

    node_type = _CONST_NODE_TYPES.get(type(form), ConstType.UNKNOWN)
    assert node_type != ConstType.UNKNOWN, "Only allow known constant types"
    descriptor = Const(
        form=form,
        is_literal=True,
        type=cast(ConstType, node_type),
        val=form,
        env=ctx.get_node_env(),
    )

    if hasattr(form, "meta"):  # type: ignore
        form_meta = _clean_meta(form.meta)  # type: ignore
        if form_meta is not None:
            meta_ast = _const_node(ctx, form_meta)
            assert isinstance(meta_ast, MapNode) or (
                isinstance(meta_ast, Const) and meta_ast.type == ConstType.MAP
            )
            return descriptor.assoc(meta=meta_ast, children=vec.v(META))

    return descriptor


@_with_loc
def _parse_ast(  # pylint: disable=too-many-branches
    ctx: ParserContext, form: Union[LispForm, lseq.Seq]
) -> Node:
    if isinstance(form, (llist.List, lseq.Seq)):
        # Special case for unquoted empty list
        if form == llist.List.empty():
            with ctx.quoted():
                return _const_node(ctx, form)
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
    else:  # pragma: no cover
        raise ParserException(f"Unexpected form type {type(form)}", form=form)


def parse_ast(ctx: ParserContext, form: LispForm) -> Node:
    """Take a Lisp form as an argument and produce a Basilisp syntax
    tree matching the clojure.tools.analyzer AST spec."""
    return _parse_ast(ctx, form).assoc(top_level=True)
