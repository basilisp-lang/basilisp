import builtins
import collections
import contextlib
import inspect
import logging
import re
import sys
import uuid
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Collection,
    Deque,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    MutableSet,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

import attr

import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.reader as reader
import basilisp.lang.runtime as runtime
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
from basilisp.lang.compiler.constants import (
    AMPERSAND,
    ARGLISTS_KW,
    COL_KW,
    DEFAULT_COMPILER_FILE_PATH,
    DOC_KW,
    FILE_KW,
    LINE_KW,
    NAME_KW,
    NS_KW,
    SYM_ABSTRACT_META_KEY,
    SYM_ASYNC_META_KEY,
    SYM_CLASSMETHOD_META_KEY,
    SYM_DEFAULT_META_KEY,
    SYM_DYNAMIC_META_KEY,
    SYM_KWARGS_META_KEY,
    SYM_MACRO_META_KEY,
    SYM_MUTABLE_META_KEY,
    SYM_NO_WARN_ON_SHADOW_META_KEY,
    SYM_NO_WARN_WHEN_UNUSED_META_KEY,
    SYM_PRIVATE_META_KEY,
    SYM_PROPERTY_META_KEY,
    SYM_STATICMETHOD_META_KEY,
    VAR_IS_PROTOCOL_META_KEY,
    SpecialForm,
)
from basilisp.lang.compiler.exception import CompilerException, CompilerPhase
from basilisp.lang.compiler.nodes import (
    Assignable,
    Await,
    Binding,
    Catch,
    Const,
    ConstType,
    Def,
    DefType,
    DefTypeBase,
    DefTypeClassMethod,
    DefTypeMember,
    DefTypeMethod,
    DefTypeMethodArity,
    DefTypeProperty,
    DefTypePythonMember,
    DefTypeStaticMethod,
    Do,
    Fn,
    FnArity,
    FunctionContext,
    HostCall,
    HostField,
    If,
    Import,
    ImportAlias,
    Invoke,
    KeywordArgs,
    KeywordArgSupport,
    Let,
    LetFn,
    Local,
    LocalType,
    Loop,
    Map as MapNode,
    MaybeClass,
    MaybeHostForm,
    Node,
    NodeEnv,
    NodeOp,
    NodeSyntacticPosition,
    PyDict,
    PyList,
    PySet,
    PyTuple,
    Quote,
    Recur,
    Reify,
    Require,
    RequireAlias,
    Set as SetNode,
    SetBang,
    SpecialFormNode,
    Throw,
    Try,
    VarRef,
    Vector as VectorNode,
    WithMeta,
    deftype_or_reify_python_member_names,
)
from basilisp.lang.interfaces import IMeta, IRecord, ISeq, IType, IWithMeta
from basilisp.lang.runtime import Var
from basilisp.lang.typing import CompilerOpts, LispForm, ReaderForm
from basilisp.lang.util import OBJECT_DUNDER_METHODS, count, genname, is_abstract, munge
from basilisp.logconfig import TRACE
from basilisp.util import Maybe, partition

# Analyzer logging
logger = logging.getLogger(__name__)

# Analyzer options
WARN_ON_SHADOWED_NAME = kw.keyword("warn-on-shadowed-name")
WARN_ON_SHADOWED_VAR = kw.keyword("warn-on-shadowed-var")
WARN_ON_UNUSED_NAMES = kw.keyword("warn-on-unused-names")

# Lisp AST node keywords
INIT = kw.keyword("init")
META = kw.keyword("meta")
FIXED_ARITY = kw.keyword("fixed-arity")
BODY = kw.keyword("body")
CATCHES = kw.keyword("catches")
FINALLY = kw.keyword("finally")

# Constants used in analyzing
AS = kw.keyword("as")
IMPLEMENTS = kw.keyword("implements")
INTERFACE = kw.keyword("interface")
STAR_STAR = sym.symbol("**")
_DOUBLE_DOT_MACRO_NAME = ".."
_BUILTINS_NS = "python"

# Symbols to be ignored for unused symbol warnings
_IGNORED_SYM = sym.symbol("_")
_MACRO_ENV_SYM = sym.symbol("&env")
_MACRO_FORM_SYM = sym.symbol("&form")
_NO_WARN_UNUSED_SYMS = lset.s(_IGNORED_SYM, _MACRO_ENV_SYM, _MACRO_FORM_SYM)


AnalyzerException = partial(CompilerException, phase=CompilerPhase.ANALYZING)


@attr.s(auto_attribs=True, slots=True)
class RecurPoint:
    loop_id: str
    args: Collection[Binding] = ()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SymbolTableEntry:
    binding: Binding
    used: bool = False
    warn_if_unused: bool = True

    @property
    def symbol(self) -> sym.Symbol:
        return self.binding.form

    @property
    def context(self) -> LocalType:
        return self.binding.local


# pylint: disable=unsupported-membership-test,unsupported-delete-operation,unsupported-assignment-operation
@attr.s(auto_attribs=True, slots=True)
class SymbolTable:
    name: str
    _is_context_boundary: bool = False
    _parent: Optional["SymbolTable"] = None
    _table: MutableMapping[sym.Symbol, SymbolTableEntry] = attr.ib(factory=dict)

    def new_symbol(
        self, s: sym.Symbol, binding: Binding, warn_if_unused: bool = True
    ) -> "SymbolTable":
        assert s == binding.form, "Binding symbol must match passed symbol"

        if s in self._table:
            self._table[s] = attr.evolve(
                self._table[s], binding=binding, warn_if_unused=warn_if_unused
            )
        else:
            self._table[s] = SymbolTableEntry(binding, warn_if_unused=warn_if_unused)
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
                    .map(lambda m: f": {m.val_at(reader.READER_LINE_KW)}")
                    .or_else_get("")
                )
                logger.warning(
                    f"symbol '{entry.symbol}' defined but not used ({ns}{code_loc})"
                )

    @contextlib.contextmanager
    def new_frame(
        self, name: str, is_context_boundary: bool, warn_on_unused_names: bool
    ):
        """Context manager for creating a new stack frame. If warn_on_unused_names is
        True and the logger is enabled for WARNING, call _warn_unused_names() on the
        child SymbolTable before it is popped."""
        new_frame = SymbolTable(
            name, is_context_boundary=is_context_boundary, parent=self
        )
        yield new_frame
        if warn_on_unused_names and logger.isEnabledFor(logging.WARNING):
            new_frame._warn_unused_names()

    def _as_env_map(self) -> MutableMapping[sym.Symbol, lmap.Map]:
        locals_ = {} if self._parent is None else self._parent._as_env_map()
        locals_.update({k: v.binding.to_map() for k, v in self._table.items()})
        return locals_

    def as_env_map(self) -> lmap.Map:
        """Return a map of symbols to the local binding objects in the
        local symbol table as of this call."""
        return lmap.map(self._as_env_map())

    @property
    def context_boundary(self) -> "SymbolTable":
        """Return the nearest context boundary parent symbol table to this one. If the
        current table is a context boundary, it will be returned directly.

        Context boundary symbol tables are symbol tables defined at the top level for
        major Python execution boundaries, such as modules (namespaces), functions
        (sync and async), and methods.

        Certain symbols (such as imports) are globally available in the execution
        context they are defined in once they have been created, context boundary
        symbol tables serve as the anchor points where we hoist these global symbols
        so they do not go out of scope when the local table frame is popped."""
        if self._is_context_boundary:
            return self
        assert (
            self._parent is not None
        ), "Top symbol table must always be a context boundary"
        return self._parent.context_boundary


class AnalyzerContext:
    __slots__ = (
        "_allow_unresolved_symbols",
        "_filename",
        "_func_ctx",
        "_is_quoted",
        "_macro_ns",
        "_opts",
        "_recur_points",
        "_should_macroexpand",
        "_st",
        "_syntax_pos",
    )

    def __init__(
        self,
        filename: Optional[str] = None,
        opts: Optional[CompilerOpts] = None,
        should_macroexpand: bool = True,
        allow_unresolved_symbols: bool = False,
    ) -> None:
        self._allow_unresolved_symbols = allow_unresolved_symbols
        self._filename = Maybe(filename).or_else_get(DEFAULT_COMPILER_FILE_PATH)
        self._func_ctx: Deque[FunctionContext] = collections.deque([])
        self._is_quoted: Deque[bool] = collections.deque([])
        self._macro_ns: Deque[Optional[runtime.Namespace]] = collections.deque([])
        self._opts = (
            Maybe(opts).map(lmap.map).or_else_get(lmap.Map.empty())  # type: ignore
        )
        self._recur_points: Deque[RecurPoint] = collections.deque([])
        self._should_macroexpand = should_macroexpand
        self._st = collections.deque([SymbolTable("<Top>", is_context_boundary=True)])
        self._syntax_pos = collections.deque([NodeSyntacticPosition.EXPR])

    @property
    def current_ns(self) -> runtime.Namespace:
        return runtime.get_current_ns()

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def warn_on_unused_names(self) -> bool:
        """If True, warn when local names are unused."""
        return self._opts.val_at(WARN_ON_UNUSED_NAMES, True)

    @property
    def warn_on_shadowed_name(self) -> bool:
        """If True, warn when a name is shadowed in an inner scope.

        Implies warn_on_shadowed_var."""
        return self._opts.val_at(WARN_ON_SHADOWED_NAME, False)

    @property
    def warn_on_shadowed_var(self) -> bool:
        """If True, warn when a def'ed Var name is shadowed in an inner scope.

        Implied by warn_on_shadowed_name. The value of warn_on_shadowed_name
        supersedes the value of this flag."""
        return self.warn_on_shadowed_name or self._opts.val_at(
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
    def current_macro_ns(self) -> Optional[runtime.Namespace]:
        """Return the current transient namespace available during macroexpansion.

        If None, the analyzer should only use the current namespace for symbol
        resolution."""
        try:
            return self._macro_ns[-1]
        except IndexError:
            return None

    @contextlib.contextmanager
    def macro_ns(self, ns: Optional[runtime.Namespace]):
        """Set the transient namespace which is available to the analyzer during a
        macroexpansion phase.

        If set to None, prohibit the analyzer from using another namespace for symbol
        resolution.

        During macroexpansion, new forms referenced from the macro namespace would
        be unavailable to the namespace containing the original macro invocation.
        The macro namespace is a temporary override pointing to the namespace of the
        macro definition which can be used to resolve these transient references."""
        self._macro_ns.append(ns)
        try:
            yield
        finally:
            self._macro_ns.pop()

    @property
    def should_allow_unresolved_symbols(self) -> bool:
        """If True, the analyzer will allow unresolved symbols. This is primarily
        useful for contexts by the `macroexpand` and `macroexpand-1` core
        functions. This would not be a good setting to use for normal compiler
        scenarios."""
        return self._allow_unresolved_symbols

    @property
    def should_macroexpand(self) -> bool:
        """Return True if macros should be expanded."""
        return self._should_macroexpand

    @property
    def func_ctx(self) -> Optional[FunctionContext]:
        """Return the current function or method context of the current node, if one.
        Return None otherwise.

        It is possible that the current function is defined inside other functions,
        so this does not imply anything about the nesting level of the current node."""
        try:
            return self._func_ctx[-1]
        except IndexError:
            return None

    @property
    def is_async_ctx(self) -> bool:
        """Return True if the current node appears inside of an async function
        definition. Return False otherwise.

        It is possible that the current function is defined inside other functions,
        so this does not imply anything about the nesting level of the current node."""
        return self.func_ctx == FunctionContext.ASYNC_FUNCTION

    @contextlib.contextmanager
    def new_func_ctx(self, context_type: FunctionContext):
        """Context manager which can be used to set a function or method context for
        child nodes to examine. A new function context is pushed onto the stack each
        time the Analyzer finds a new function or method definition, so there may be
        many nested function contexts."""
        self._func_ctx.append(context_type)
        yield
        self._func_ctx.pop()

    @property
    def recur_point(self) -> Optional[RecurPoint]:
        """Return the current recur point which applies to the current node, if there
        is one."""
        try:
            return self._recur_points[-1]
        except IndexError:
            return None

    @contextlib.contextmanager
    def new_recur_point(self, loop_id: str, args: Collection[Any] = ()):
        """Context manager which can be used to set a recur point for child nodes.
        A new recur point is pushed onto the stack each time the Analyzer finds a
        form which supports recursion (such as `fn*` or `loop*`), so there may be
        many recur points, though only one may be active at any given time for a
        node."""
        self._recur_points.append(RecurPoint(loop_id, args=args))
        yield
        self._recur_points.pop()

    @property
    def symbol_table(self) -> SymbolTable:
        return self._st[-1]

    def put_new_symbol(  # pylint: disable=too-many-arguments
        self,
        s: sym.Symbol,
        binding: Binding,
        warn_on_shadowed_name: bool = True,
        warn_on_shadowed_var: bool = True,
        warn_if_unused: bool = True,
        symbol_table: Optional[SymbolTable] = None,
    ):
        """Add a new symbol to the symbol table.

        This function allows individual warnings to be disabled for one run
        by supplying keyword arguments temporarily disabling those warnings.
        In certain cases, we do not want to issue warnings again for a
        previously checked case, so this is a simple way of disabling these
        warnings for those cases.

        There are cases where undesired warnings may be triggered non-locally,
        so the Python keyword arguments cannot be used to suppress unwanted
        warnings. For these cases, symbols may include the `:no-warn-on-shadow`
        metadata key to indicate that warnings for shadowing names from outer
        scopes should be suppressed. It is not currently possible to suppress
        Var shadowing warnings at the symbol level.

        If WARN_ON_SHADOWED_NAME compiler option is active and the
        warn_on_shadowed_name keyword argument is True, then a warning will be
        emitted if a local name is shadowed by another local name. Note that
        WARN_ON_SHADOWED_NAME implies WARN_ON_SHADOWED_VAR.

        If WARN_ON_SHADOWED_VAR compiler option is active and the
        warn_on_shadowed_var keyword argument is True, then a warning will be
        emitted if a named var is shadowed by a local name."""
        st = symbol_table or self.symbol_table
        no_warn_on_shadow = (
            Maybe(s.meta)
            .map(lambda m: m.val_at(SYM_NO_WARN_ON_SHADOW_META_KEY, False))
            .or_else_get(False)
        )
        if (
            not no_warn_on_shadow
            and warn_on_shadowed_name
            and self.warn_on_shadowed_name
        ):
            if st.find_symbol(s) is not None:
                logger.warning(f"name '{s}' shadows name from outer scope")
        if (
            warn_on_shadowed_name or warn_on_shadowed_var
        ) and self.warn_on_shadowed_var:
            if self.current_ns.find(s) is not None:
                logger.warning(f"name '{s}' shadows def'ed Var from outer scope")
        if s.meta is not None and s.meta.val_at(SYM_NO_WARN_WHEN_UNUSED_META_KEY, None):
            warn_if_unused = False
        st.new_symbol(s, binding, warn_if_unused=warn_if_unused)

    @contextlib.contextmanager
    def new_symbol_table(self, name: str, is_context_boundary: bool = False):
        old_st = self.symbol_table
        with old_st.new_frame(
            name, is_context_boundary, self.warn_on_unused_names,
        ) as st:
            self._st.append(st)
            yield st
            self._st.pop()

    @contextlib.contextmanager
    def hide_parent_symbol_table(self):
        """Hide the immediate parent symbol table by temporarily popping
        it off the stack.

        Obviously doing this could have serious adverse consequences if another
        new symbol table is added to the stack during this operation, so it
        should essentially NEVER be used.

        Right now, it is being used to hide fields and `this` symbol from
        static and class methods."""
        old_st = self._st.pop()
        try:
            yield self.symbol_table
        finally:
            self._st.append(old_st)

    @contextlib.contextmanager
    def expr_pos(self):
        """Context manager which indicates to immediate child nodes that they
        are in an expression syntactic position."""
        self._syntax_pos.append(NodeSyntacticPosition.EXPR)
        try:
            yield
        finally:
            self._syntax_pos.pop()

    @contextlib.contextmanager
    def stmt_pos(self):
        """Context manager which indicates to immediate child nodes that they
        are in a statement syntactic position."""
        self._syntax_pos.append(NodeSyntacticPosition.STMT)
        try:
            yield
        finally:
            self._syntax_pos.pop()

    @contextlib.contextmanager
    def parent_pos(self):
        """Context manager which indicates to immediate child nodes that they
        are in an equivalent syntactic position as their parent node.

        This context manager copies the top position and pushes a new value onto
        the stack, so the parent node's syntax position will not be lost."""
        self._syntax_pos.append(self.syntax_position)
        try:
            yield
        finally:
            self._syntax_pos.pop()

    @property
    def syntax_position(self) -> NodeSyntacticPosition:
        """Return the syntax position of the current node as indicated by its
        parent node."""
        return self._syntax_pos[-1]

    def get_node_env(self, pos: Optional[NodeSyntacticPosition] = None) -> NodeEnv:
        """Return the current Node environment.

        If a synax position is given, it will be included in the environment.
        Otherwise, the position will be set to None."""
        return NodeEnv(
            ns=self.current_ns, file=self.filename, pos=pos, func_ctx=self.func_ctx
        )


MetaGetter = Callable[[Union[IMeta, Var]], bool]
AnalyzeFunction = Callable[[AnalyzerContext, Union[LispForm, ISeq]], Node]


def _meta_getter(meta_kw: kw.Keyword) -> MetaGetter:
    """Return a function which checks an object with metadata for a boolean
    value by meta_kw."""

    def has_meta_prop(o: Union[IMeta, Var]) -> bool:
        return bool(
            Maybe(o.meta).map(lambda m: m.val_at(meta_kw, None)).or_else_get(False)
        )

    return has_meta_prop


_is_artificially_abstract = _meta_getter(SYM_ABSTRACT_META_KEY)
_is_async = _meta_getter(SYM_ASYNC_META_KEY)
_is_mutable = _meta_getter(SYM_MUTABLE_META_KEY)
_is_py_classmethod = _meta_getter(SYM_CLASSMETHOD_META_KEY)
_is_py_property = _meta_getter(SYM_PROPERTY_META_KEY)
_is_py_staticmethod = _meta_getter(SYM_STATICMETHOD_META_KEY)
_is_macro = _meta_getter(SYM_MACRO_META_KEY)


def _loc(form: Union[LispForm, ISeq]) -> Optional[Tuple[int, int]]:
    """Fetch the location of the form in the original filename from the
    input form, if it has metadata."""
    # Technically, IMeta is sufficient for fetching `form.meta` but the
    # reader only applies line and column metadata to IWithMeta instances
    if isinstance(form, IWithMeta):
        meta = form.meta
        if meta is not None:
            line = meta.get(reader.READER_LINE_KW)
            col = meta.get(reader.READER_COL_KW)
            if isinstance(line, int) and isinstance(col, int):
                return line, col
    return None


def _with_loc(f: AnalyzeFunction):
    """Attach any available location information from the input form to
    the node environment returned from the parsing function."""

    @wraps(f)
    def _analyze_form(ctx: AnalyzerContext, form: Union[LispForm, ISeq]) -> Node:
        form_loc = _loc(form)
        if form_loc is None:
            return f(ctx, form)
        else:
            return f(ctx, form).fix_missing_locations(form_loc)

    return _analyze_form


def _clean_meta(meta: Optional[lmap.Map]) -> Optional[lmap.Map]:
    """Remove reader metadata from the form's meta map."""
    if meta is None:
        return None
    else:
        new_meta = meta.dissoc(reader.READER_LINE_KW, reader.READER_COL_KW)
        return None if len(new_meta) == 0 else new_meta


def _body_ast(
    ctx: AnalyzerContext, form: Union[llist.List, ISeq]
) -> Tuple[Iterable[Node], Node]:
    """Analyze the form and produce a body of statement nodes and a single
    return expression node.

    If the body is empty, return a constant node containing nil.

    If the parent indicates that it is in a statement syntactic position
    (and thus that it cannot return a value), the final node will be marked
    as a statement (rather than an expression) as well."""
    body_list = list(form)
    if body_list:
        *stmt_forms, ret_form = body_list

        with ctx.stmt_pos():
            body_stmts = list(map(partial(_analyze_form, ctx), stmt_forms))

        with ctx.parent_pos():
            body_expr = _analyze_form(ctx, ret_form)

        body = body_stmts + [body_expr]
    else:
        body = []

    if body:
        *stmts, ret = body
    else:
        stmts, ret = [], _const_node(ctx, None)
    return stmts, ret


def _call_args_ast(  # pylint: disable=too-many-branches
    ctx: AnalyzerContext, form: ISeq
) -> Tuple[Iterable[Node], KeywordArgs]:
    """Return a tuple of positional arguments and keyword arguments, splitting at the
    keyword argument marker symbol '**'."""
    with ctx.expr_pos():
        nmarkers = sum(int(e == STAR_STAR) for e in form)
        if nmarkers > 1:
            raise AnalyzerException(
                "function and method invocations may have at most 1 keyword argument marker '**'",
                form=form,
            )
        elif nmarkers == 1:
            kwarg_marker = False
            pos, kws = [], []
            for arg in form:
                if arg == STAR_STAR:
                    kwarg_marker = True
                    continue
                if kwarg_marker:
                    kws.append(arg)
                else:
                    pos.append(arg)

            args = vec.vector(map(partial(_analyze_form, ctx), pos))
            kw_map = {}
            try:
                for k, v in partition(kws, 2):
                    if isinstance(k, kw.Keyword):
                        munged_k = munge(k.name, allow_builtins=True)
                    elif isinstance(k, str):
                        munged_k = munge(k, allow_builtins=True)
                    else:
                        raise AnalyzerException(
                            f"keys for keyword arguments must be keywords or strings, not '{type(k)}'",
                            form=k,
                        )

                    if munged_k in kw_map:
                        raise AnalyzerException(
                            f"duplicate keyword argument key in function or method invocation",
                            form=k,
                        )

                    kw_map[munged_k] = _analyze_form(ctx, v)

            except ValueError:
                raise AnalyzerException(
                    "keyword arguments must appear in key/value pairs", form=form
                ) from ValueError
            else:
                kwargs = lmap.map(kw_map)
        else:
            args = vec.vector(map(partial(_analyze_form, ctx), form))
            kwargs = lmap.Map.empty()

        return args, kwargs


def _with_meta(gen_node):
    """Wraps the node generated by gen_node in a :with-meta AST node if the
    original form has meta.

    :with-meta AST nodes are used for non-quoted collection literals and for
    function expressions."""

    @wraps(gen_node)
    def with_meta(
        ctx: AnalyzerContext,
        form: Union[llist.List, lmap.Map, ISeq, lset.Set, vec.Vector],
    ) -> Node:
        assert not ctx.is_quoted, "with-meta nodes are not used in quoted expressions"

        descriptor = gen_node(ctx, form)

        if isinstance(form, IMeta):
            assert isinstance(form.meta, (lmap.Map, type(None)))
            form_meta = _clean_meta(form.meta)
            if form_meta is not None:
                meta_ast = _analyze_form(ctx, form_meta)
                assert isinstance(meta_ast, MapNode) or (
                    isinstance(meta_ast, Const) and meta_ast.type == ConstType.MAP
                )
                return WithMeta(
                    form=form,
                    meta=meta_ast,
                    expr=descriptor,
                    env=ctx.get_node_env(pos=ctx.syntax_position),
                )

        return descriptor

    return with_meta


def _await_ast(ctx: AnalyzerContext, form: ISeq) -> Await:
    assert form.first == SpecialForm.AWAIT

    if not ctx.is_async_ctx:
        raise AnalyzerException(
            f"await forms may not appear in non-async context", form=form
        )

    nelems = count(form)
    if nelems != 2:
        raise AnalyzerException(
            f"await forms must contain 2 elements, as in: (await expr)", form=form
        )

    with ctx.expr_pos():
        expr = _analyze_form(ctx, runtime.nth(form, 1))

    return Await(form=form, expr=expr, env=ctx.get_node_env(pos=ctx.syntax_position),)


def _def_ast(  # pylint: disable=too-many-branches,too-many-locals
    ctx: AnalyzerContext, form: ISeq
) -> Def:
    assert form.first == SpecialForm.DEF

    nelems = count(form)
    if nelems not in (2, 3, 4):
        raise AnalyzerException(
            f"def forms must have between 2 and 4 elements, as in: (def name docstring? init?)",
            form=form,
        )

    name = runtime.nth(form, 1)
    if not isinstance(name, sym.Symbol):
        raise AnalyzerException(
            f"def names must be symbols, not {type(name)}", form=name
        )

    children: vec.Vector[kw.Keyword]
    if nelems == 2:
        init = None
        doc = None
        children = vec.Vector.empty()
    elif nelems == 3:
        with ctx.expr_pos():
            init = _analyze_form(ctx, runtime.nth(form, 2))
        doc = None
        children = vec.v(INIT)
    else:
        with ctx.expr_pos():
            init = _analyze_form(ctx, runtime.nth(form, 3))
        doc = runtime.nth(form, 2)
        if not isinstance(doc, str):
            raise AnalyzerException("def docstring must be a string", form=doc)
        children = vec.v(INIT)

    # Cache the current namespace
    current_ns = ctx.current_ns

    # Attach metadata relevant for the current process below.
    #
    # The reader line/col metadata will be attached to the form itself in the
    # happy path case (top-level bare def or top-level def returned from a
    # macro). Less commonly, we may have to rely on metadata attached to the
    # def symbol if, for example, the def form is returned by a macro wrapped
    # in a do or let form. In that case, the macroexpansion process won't have
    # any metadata to pass along to the expanded form, but the symbol itself
    # is likely to have metadata. In rare cases, we may not be able to get
    # any metadata. This may happen if the form and name were both generated
    # programmatically.
    def_loc = _loc(form) or _loc(name) or (None, None)
    if def_loc == (None, None):
        logger.warning(f"def line and column metadata not provided for Var {name}")
    def_node_env = ctx.get_node_env(pos=ctx.syntax_position)
    def_meta = _clean_meta(
        name.meta.update(  # type: ignore [union-attr]
            lmap.map(
                {
                    COL_KW: def_loc[1],
                    FILE_KW: def_node_env.file,
                    LINE_KW: def_loc[0],
                    NAME_KW: name,
                    NS_KW: current_ns,
                }
            )
        )
    )
    assert def_meta is not None, "def metadata must be defined at this point"
    if doc is not None:
        def_meta = def_meta.assoc(DOC_KW, doc)

    # Var metadata is set both for the running Basilisp instance
    # and cached as Python bytecode to be reread again. Argument lists
    # are quoted so as not to resolve argument symbols. For the compiled
    # and cached bytecode, this causes no trouble. However, for the case
    # where we directly set the Var meta for the running Basilisp instance
    # this causes problems since we'll end up getting something like
    # `(quote ([] [v]))` rather than simply `([] [v])`.
    arglists_meta = def_meta.val_at(ARGLISTS_KW)  # type: ignore
    if isinstance(arglists_meta, llist.List):
        assert arglists_meta.first == SpecialForm.QUOTE
        var_meta = def_meta.update(  # type: ignore
            {ARGLISTS_KW: runtime.nth(arglists_meta, 1)}
        )
    else:
        var_meta = def_meta

    # Generation fails later if we use the same symbol we received, since
    # its meta may contain values which fail to compile.
    bare_name = sym.symbol(name.name)

    ns_sym = sym.symbol(current_ns.name)
    var = Var.intern_unbound(
        ns_sym,
        bare_name,
        dynamic=def_meta.val_at(SYM_DYNAMIC_META_KEY, False),  # type: ignore
        meta=var_meta,
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
    #        :ns   ((.- basilisp.lang.runtime/Namespace get) 'basilisp.user)}
    #       some-name
    #       "some value")
    meta_ast = _analyze_form(
        ctx,
        def_meta.update(  # type: ignore
            {
                NAME_KW: llist.l(SpecialForm.QUOTE, bare_name),
                NS_KW: llist.l(
                    llist.l(
                        SpecialForm.INTEROP_PROP,
                        sym.symbol("Namespace", "basilisp.lang.runtime"),
                        sym.symbol("get"),
                    ),
                    llist.l(SpecialForm.QUOTE, sym.symbol(current_ns.name)),
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


def __deftype_method_param_bindings(
    ctx: AnalyzerContext, params: vec.Vector, special_form: sym.Symbol
) -> Tuple[bool, int, List[Binding]]:
    """Generate parameter bindings for `deftype*` or `reify*` methods.

    Return a tuple containing a boolean, indicating if the parameter bindings
    contain a variadic binding, an integer indicating the fixed arity of the
    parameter bindings, and the list of parameter bindings.

    Special cases for individual method types must be handled by their
    respective handlers. This method will only produce vanilla ARG type
    bindings."""
    assert special_form in {SpecialForm.DEFTYPE, SpecialForm.REIFY}

    has_vargs, vargs_idx = False, 0
    param_nodes = []
    for i, s in enumerate(params):
        if not isinstance(s, sym.Symbol):
            raise AnalyzerException(
                f"{special_form} method parameter name must be a symbol", form=s
            )

        if s == AMPERSAND:
            has_vargs = True
            vargs_idx = i
            break

        binding = Binding(
            form=s,
            name=s.name,
            local=LocalType.ARG,
            arg_id=i,
            is_variadic=False,
            env=ctx.get_node_env(),
        )
        param_nodes.append(binding)
        ctx.put_new_symbol(s, binding)

    fixed_arity = len(param_nodes)

    if has_vargs:
        try:
            vargs_sym = params[vargs_idx + 1]

            if not isinstance(vargs_sym, sym.Symbol):
                raise AnalyzerException(
                    f"{special_form} method rest parameter name must be a symbol",
                    form=vargs_sym,
                )

            binding = Binding(
                form=vargs_sym,
                name=vargs_sym.name,
                local=LocalType.ARG,
                arg_id=vargs_idx + 1,
                is_variadic=True,
                env=ctx.get_node_env(),
            )
            param_nodes.append(binding)
            ctx.put_new_symbol(vargs_sym, binding)
        except IndexError:
            raise AnalyzerException(
                "Expected variadic argument name after '&'", form=params
            ) from None

    return has_vargs, fixed_arity, param_nodes


def __deftype_classmethod(
    ctx: AnalyzerContext,
    form: Union[llist.List, ISeq],
    method_name: str,
    args: vec.Vector,
    kwarg_support: Optional[KeywordArgSupport] = None,
) -> DefTypeClassMethod:
    """Emit a node for a :classmethod member of a `deftype*` form."""
    with ctx.hide_parent_symbol_table(), ctx.new_symbol_table(
        method_name, is_context_boundary=True
    ):
        try:
            cls_arg = args[0]
        except IndexError:
            raise AnalyzerException(
                "deftype* class method must include 'cls' argument", form=args
            )
        else:
            if not isinstance(cls_arg, sym.Symbol):
                raise AnalyzerException(
                    "deftype* class method 'cls' argument must be a symbol", form=args,
                )
            cls_binding = Binding(
                form=cls_arg,
                name=cls_arg.name,
                local=LocalType.ARG,
                env=ctx.get_node_env(),
            )
            ctx.put_new_symbol(cls_arg, cls_binding)

        params = args[1:]
        has_vargs, fixed_arity, param_nodes = __deftype_method_param_bindings(
            ctx, params, SpecialForm.DEFTYPE
        )
        with ctx.new_func_ctx(FunctionContext.CLASSMETHOD), ctx.expr_pos():
            stmts, ret = _body_ast(ctx, runtime.nthrest(form, 2))
        method = DefTypeClassMethod(
            form=form,
            name=method_name,
            params=vec.vector(param_nodes),
            fixed_arity=fixed_arity,
            is_variadic=has_vargs,
            kwarg_support=kwarg_support,
            body=Do(
                form=form.rest,
                statements=vec.vector(stmts),
                ret=ret,
                is_body=True,
                # Use the argument vector or first body statement, whichever
                # exists, for metadata.
                env=ctx.get_node_env(),
            ),
            class_local=cls_binding,
            env=ctx.get_node_env(),
        )
        method.visit(_assert_no_recur)
        return method


def __deftype_or_reify_method(  # pylint: disable=too-many-arguments,too-many-locals
    ctx: AnalyzerContext,
    form: Union[llist.List, ISeq],
    method_name: str,
    args: vec.Vector,
    special_form: sym.Symbol,
    kwarg_support: Optional[KeywordArgSupport] = None,
) -> DefTypeMethodArity:
    """Emit a node for a method member of a `deftype*` or `reify*` form."""
    assert special_form in {SpecialForm.DEFTYPE, SpecialForm.REIFY}

    with ctx.new_symbol_table(method_name, is_context_boundary=True):
        try:
            this_arg = args[0]
        except IndexError:
            raise AnalyzerException(
                f"{special_form} method must include 'this' or 'self' argument",
                form=args,
            )
        else:
            if not isinstance(this_arg, sym.Symbol):
                raise AnalyzerException(
                    f"{special_form} method 'this' argument must be a symbol", form=args
                )
            this_binding = Binding(
                form=this_arg,
                name=this_arg.name,
                local=LocalType.THIS,
                env=ctx.get_node_env(),
            )
            ctx.put_new_symbol(this_arg, this_binding, warn_if_unused=False)

        params = args[1:]
        has_vargs, fixed_arity, param_nodes = __deftype_method_param_bindings(
            ctx, params, special_form
        )

        loop_id = genname(method_name)
        with ctx.new_recur_point(loop_id, param_nodes):
            with ctx.new_func_ctx(FunctionContext.METHOD), ctx.expr_pos():
                stmts, ret = _body_ast(ctx, runtime.nthrest(form, 2))
            method = DefTypeMethodArity(
                form=form,
                name=method_name,
                this_local=this_binding,
                params=vec.vector(param_nodes),
                fixed_arity=fixed_arity,
                is_variadic=has_vargs,
                kwarg_support=kwarg_support,
                body=Do(
                    form=form.rest,
                    statements=vec.vector(stmts),
                    ret=ret,
                    is_body=True,
                    # Use the argument vector or first body statement, whichever
                    # exists, for metadata.
                    env=ctx.get_node_env(),
                ),
                loop_id=loop_id,
                env=ctx.get_node_env(),
            )
            method.visit(_assert_recur_is_tail)
            return method


def __deftype_or_reify_property(
    ctx: AnalyzerContext,
    form: Union[llist.List, ISeq],
    method_name: str,
    args: vec.Vector,
    special_form: sym.Symbol,
) -> DefTypeProperty:
    """Emit a node for a :property member of a `deftype*` or `reify*` form."""
    assert special_form in {SpecialForm.DEFTYPE, SpecialForm.REIFY}

    with ctx.new_symbol_table(method_name, is_context_boundary=True):
        try:
            this_arg = args[0]
        except IndexError:
            raise AnalyzerException(
                f"{special_form} property must include 'this' or 'self' argument",
                form=args,
            )
        else:
            if not isinstance(this_arg, sym.Symbol):
                raise AnalyzerException(
                    f"{special_form} property 'this' argument must be a symbol",
                    form=args,
                )
            this_binding = Binding(
                form=this_arg,
                name=this_arg.name,
                local=LocalType.THIS,
                env=ctx.get_node_env(),
            )
            ctx.put_new_symbol(this_arg, this_binding, warn_if_unused=False)

        params = args[1:]
        has_vargs, _, param_nodes = __deftype_method_param_bindings(
            ctx, params, special_form
        )

        if len(param_nodes) > 0:
            raise AnalyzerException(
                f"{special_form} properties may not specify arguments", form=form
            )

        assert not has_vargs, f"{special_form} properties may not have arguments"

        with ctx.new_func_ctx(FunctionContext.PROPERTY), ctx.expr_pos():
            stmts, ret = _body_ast(ctx, runtime.nthrest(form, 2))
        prop = DefTypeProperty(
            form=form,
            name=method_name,
            this_local=this_binding,
            params=vec.vector(param_nodes),
            body=Do(
                form=form.rest,
                statements=vec.vector(stmts),
                ret=ret,
                is_body=True,
                # Use the argument vector or first body statement, whichever
                # exists, for metadata.
                env=ctx.get_node_env(),
            ),
            env=ctx.get_node_env(),
        )
        prop.visit(_assert_no_recur)
        return prop


def __deftype_staticmethod(
    ctx: AnalyzerContext,
    form: Union[llist.List, ISeq],
    method_name: str,
    args: vec.Vector,
    kwarg_support: Optional[KeywordArgSupport] = None,
) -> DefTypeStaticMethod:
    """Emit a node for a :staticmethod member of a `deftype*` form."""
    with ctx.hide_parent_symbol_table(), ctx.new_symbol_table(
        method_name, is_context_boundary=True
    ):
        has_vargs, fixed_arity, param_nodes = __deftype_method_param_bindings(
            ctx, args, SpecialForm.DEFTYPE
        )
        with ctx.new_func_ctx(FunctionContext.STATICMETHOD), ctx.expr_pos():
            stmts, ret = _body_ast(ctx, runtime.nthrest(form, 2))
        method = DefTypeStaticMethod(
            form=form,
            name=method_name,
            params=vec.vector(param_nodes),
            fixed_arity=fixed_arity,
            is_variadic=has_vargs,
            kwarg_support=kwarg_support,
            body=Do(
                form=form.rest,
                statements=vec.vector(stmts),
                ret=ret,
                is_body=True,
                # Use the argument vector or first body statement, whichever
                # exists, for metadata.
                env=ctx.get_node_env(),
            ),
            env=ctx.get_node_env(),
        )
        method.visit(_assert_no_recur)
        return method


def __deftype_or_reify_prop_or_method_arity(  # pylint: disable=too-many-branches
    ctx: AnalyzerContext, form: Union[llist.List, ISeq], special_form: sym.Symbol
) -> Union[DefTypeMethodArity, DefTypePythonMember]:
    """Emit either a `deftype*` or `reify*` property node or an arity of a `deftype*`
    or `reify*` method.

    Unlike standard `fn*` definitions, multiple arities for a single method are
    not defined within some containing node. As such, we can only emit either a
    full property node (since properties may not be multi-arity) or the single
    arity of a method, classmethod, or staticmethod.

    The type of the member node is determined by the presence or absence of certain
    metadata elements on the input form (or the form's first member, typically a
    symbol naming that member)."""
    assert special_form in {SpecialForm.DEFTYPE, SpecialForm.REIFY}

    if not isinstance(form.first, sym.Symbol):
        raise AnalyzerException(
            f"{special_form} method must be named by symbol: (name [& args] & body)",
            form=form.first,
        )
    method_name = form.first.name

    is_classmethod = _is_py_classmethod(form.first) or (
        isinstance(form, IMeta) and _is_py_classmethod(form)
    )
    is_property = _is_py_property(form.first) or (
        isinstance(form, IMeta) and _is_py_property(form)
    )
    is_staticmethod = _is_py_staticmethod(form.first) or (
        isinstance(form, IMeta) and _is_py_staticmethod(form)
    )

    if special_form == SpecialForm.REIFY and (is_classmethod or is_staticmethod):
        raise AnalyzerException(
            f"{special_form} does not support classmethod or staticmethod members",
            form=form,
        )

    if not sum([is_classmethod, is_property, is_staticmethod]) in {0, 1}:
        raise AnalyzerException(
            f"{special_form} member may be only one of: :classmethod, :property, "
            "or :staticmethod",
            form=form,
        )

    args = runtime.nth(form, 1)
    if not isinstance(args, vec.Vector):
        raise AnalyzerException(
            f"{special_form} member arguments must be vector, not {type(args)}",
            form=args,
        )

    kwarg_meta = __fn_kwargs_support(form.first) or (
        isinstance(form, IMeta) and __fn_kwargs_support(form)
    )
    kwarg_support = None if isinstance(kwarg_meta, bool) else kwarg_meta

    if is_classmethod:
        return __deftype_classmethod(
            ctx, form, method_name, args, kwarg_support=kwarg_support
        )
    elif is_property:
        if kwarg_support is not None:
            raise AnalyzerException(
                f"{special_form} properties may not declare keyword argument support",
                form=form,
            )

        return __deftype_or_reify_property(ctx, form, method_name, args, special_form)
    elif is_staticmethod:
        return __deftype_staticmethod(
            ctx, form, method_name, args, kwarg_support=kwarg_support
        )
    else:
        return __deftype_or_reify_method(
            ctx, form, method_name, args, special_form, kwarg_support=kwarg_support
        )


def __deftype_or_reify_method_node_from_arities(  # pylint: disable=too-many-branches
    ctx: AnalyzerContext,
    form: Union[llist.List, ISeq],
    arities: List[DefTypeMethodArity],
    special_form: sym.Symbol,
) -> DefTypeMember:
    """Roll all of the collected `deftype*` or `reify*` arities up into a single
    method node."""
    assert special_form in {SpecialForm.DEFTYPE, SpecialForm.REIFY}

    fixed_arities: MutableSet[int] = set()
    fixed_arity_for_variadic: Optional[int] = None
    num_variadic = 0
    for arity in arities:
        if arity.is_variadic:
            if num_variadic > 0:
                raise AnalyzerException(
                    f"{special_form} method may have at most 1 variadic arity",
                    form=arity.form,
                )
            fixed_arity_for_variadic = arity.fixed_arity
            num_variadic += 1
        else:
            if arity.fixed_arity in fixed_arities:
                raise AnalyzerException(
                    f"{special_form} may not have multiple methods with the same "
                    "fixed arity",
                    form=arity.form,
                )
            fixed_arities.add(arity.fixed_arity)

    if fixed_arity_for_variadic is not None and any(
        fixed_arity_for_variadic < arity for arity in fixed_arities
    ):
        raise AnalyzerException(
            "variadic arity may not have fewer fixed arity arguments than any other arities",
            form=form,
        )

    assert (
        len(set(arity.name for arity in arities)) <= 1
    ), "arities must have the same name defined"

    if len(arities) > 1 and any(arity.kwarg_support is not None for arity in arities):
        raise AnalyzerException(
            f"multi-arity {special_form} methods may not declare support for "
            "keyword arguments",
            form=form,
        )

    return DefTypeMethod(
        form=form,
        name=arities[0].name,
        max_fixed_arity=max(arity.fixed_arity for arity in arities),
        arities=vec.vector(arities),
        is_variadic=num_variadic == 1,
        env=ctx.get_node_env(),
    )


def __deftype_or_reify_impls(  # pylint: disable=too-many-branches,too-many-locals  # noqa: MC0001
    ctx: AnalyzerContext, form: ISeq, special_form: sym.Symbol,
) -> Tuple[List[DefTypeBase], List[DefTypeMember]]:
    """Roll up `deftype*` and `reify*` declared bases and method implementations."""
    assert special_form in {SpecialForm.DEFTYPE, SpecialForm.REIFY}

    if runtime.to_seq(form) is None:
        return [], []

    if not isinstance(form.first, kw.Keyword) or form.first != IMPLEMENTS:
        raise AnalyzerException(
            f"{special_form} forms must declare which interfaces they implement",
            form=form,
        )

    implements = runtime.nth(form, 1)
    if not isinstance(implements, vec.Vector):
        raise AnalyzerException(
            f"{special_form} interfaces must be declared as "
            ":implements [Interface1 Interface2 ...]",
            form=implements,
        )

    interface_names: MutableSet[sym.Symbol] = set()
    interfaces = []
    for iface in implements:
        if not isinstance(iface, sym.Symbol):
            raise AnalyzerException(
                f"{special_form} interfaces must be symbols", form=iface
            )

        if iface in interface_names:
            raise AnalyzerException(
                f"{special_form} interfaces may only appear once in :implements vector",
                form=iface,
            )
        interface_names.add(iface)

        current_interface = _analyze_form(ctx, iface)
        if not isinstance(current_interface, (MaybeClass, MaybeHostForm, VarRef)):
            raise AnalyzerException(
                f"{special_form} interface implementation must be an existing interface",
                form=iface,
            )
        interfaces.append(current_interface)

    # Use the insertion-order preserving capabilities of a dictionary with 'True'
    # keys to act as an ordered set of members we've seen. We don't want to register
    # duplicates.
    member_order = {}
    methods: MutableMapping[str, List[DefTypeMethodArity]] = collections.defaultdict(
        list
    )
    py_members: MutableMapping[str, DefTypePythonMember] = {}
    for elem in runtime.nthrest(form, 2):
        if not isinstance(elem, ISeq):
            raise AnalyzerException(
                f"{special_form} must consist of interface or protocol names and methods",
                form=elem,
            )

        member = __deftype_or_reify_prop_or_method_arity(ctx, elem, special_form)
        member_order[member.name] = True
        if isinstance(
            member, (DefTypeClassMethod, DefTypeProperty, DefTypeStaticMethod)
        ):
            if member.name in py_members:
                raise AnalyzerException(
                    f"{special_form} class methods, properties, and static methods "
                    "may only have one arity defined",
                    form=elem,
                    lisp_ast=member,
                )
            elif member.name in methods:
                raise AnalyzerException(
                    f"{special_form} class method, property, or static method name "
                    "already defined as a method",
                    form=elem,
                    lisp_ast=member,
                )
            py_members[member.name] = member
        else:
            if member.name in py_members:
                raise AnalyzerException(
                    f"{special_form} method name already defined as a class method, "
                    "property, or static method",
                    form=elem,
                    lisp_ast=member,
                )
            methods[member.name].append(member)

    members: List[DefTypeMember] = []
    for member_name in member_order:
        arities = methods.get(member_name)
        if arities is not None:
            members.append(
                __deftype_or_reify_method_node_from_arities(
                    ctx, form, arities, special_form
                )
            )
            continue

        py_member = py_members.get(member_name)
        assert py_member is not None, "Member must be a method or property"
        members.append(py_member)

    return interfaces, members


_var_is_protocol = _meta_getter(VAR_IS_PROTOCOL_META_KEY)


def __is_deftype_member(mem) -> bool:
    """Return True if `mem` names a valid `deftype*` member."""
    return (
        inspect.isfunction(mem)
        or isinstance(mem, (property, staticmethod))
        or inspect.ismethod(mem)
    )


def __is_reify_member(mem) -> bool:
    """Return True if `mem` names a valid `reify*` member."""
    return inspect.isfunction(mem) or isinstance(mem, property)


def __deftype_and_reify_impls_are_all_abstract(  # pylint: disable=too-many-branches,too-many-locals
    special_form: sym.Symbol,
    fields: Iterable[str],
    interfaces: Iterable[DefTypeBase],
    members: Iterable[DefTypeMember],
) -> Tuple[bool, lset.Set[DefTypeBase]]:
    """Return a tuple of two items indicating the abstractness of the `deftype*` or
    `reify*` super-types. The first element is a boolean value which, if True,
    indicates that all bases have been statically verified abstract. If False, that
    value indicates at least one base could not be statically verified. The second
    element is the set of all super-types which have been marked as artificially
    abstract.

    In certain cases, such as in macro definitions and potentially inside of
    functions, the compiler will be unable to resolve the named super-type as an
    object during compilation and these checks will need to be deferred to runtime.
    In these cases, the compiler will wrap the emitted class in a decorator that
    performs the checks when the class is compiled by the Python compiler.

    The Python ecosystem is much less strict with its use of `abc.ABC` to define
    interfaces than Java (which has a native `interface` construct), so even in cases
    where a type may be _in practice_ an interface or ABC, the compiler would not
    permit you to declare such types as supertypes because they do not themselves
    inherit from `abc.ABC`. In these cases, users can mark the type as artificially
    abstract with the `:abstract` metadata key.

    For normal compile-time errors, an `AnalyzerException` will be raised."""
    assert special_form in {SpecialForm.DEFTYPE, SpecialForm.REIFY}

    unverifiably_abstract = set()
    artificially_abstract: Set[DefTypeBase] = set()
    artificially_abstract_base_members: Set[str] = set()
    is_member = {
        SpecialForm.DEFTYPE: __is_deftype_member,
        SpecialForm.REIFY: __is_reify_member,
    }[special_form]

    field_names = frozenset(fields)
    member_names = frozenset(deftype_or_reify_python_member_names(members))
    all_member_names = field_names.union(member_names)
    all_interface_methods: Set[str] = set()
    for interface in interfaces:
        if isinstance(interface, (MaybeClass, MaybeHostForm)):
            interface_type = interface.target
        else:
            assert isinstance(
                interface, VarRef
            ), "Interface must be MaybeClass, MaybeHostForm, or VarRef"
            if not interface.var.is_bound:
                logger.log(
                    TRACE,
                    f"{special_form} interface Var '{interface.form}' is not bound "
                    "and cannot be checked for abstractness; deferring to runtime",
                )
                unverifiably_abstract.add(interface)
                if _is_artificially_abstract(interface.form):
                    artificially_abstract.add(interface)
                continue

            # Protocols are defined as maps, with the interface being simply a member
            # of the map, denoted by the keyword `:interface`.
            if _var_is_protocol(interface.var):
                proto_map = interface.var.value
                assert isinstance(proto_map, lmap.Map)
                interface_type = proto_map.val_at(INTERFACE)
            else:
                interface_type = interface.var.value

        if interface_type is object:
            continue

        if is_abstract(interface_type):
            interface_names: FrozenSet[str] = interface_type.__abstractmethods__
            interface_property_names: FrozenSet[str] = frozenset(
                method
                for method in interface_names
                if isinstance(getattr(interface_type, method), property)
            )
            interface_method_names = interface_names - interface_property_names
            if not interface_method_names.issubset(member_names):
                missing_methods = ", ".join(interface_method_names - member_names)
                raise AnalyzerException(
                    f"{special_form} definition missing interface members for "
                    f"interface {interface.form}: {missing_methods}",
                    form=interface.form,
                    lisp_ast=interface,
                )
            elif not interface_property_names.issubset(all_member_names):
                missing_fields = ", ".join(interface_property_names - field_names)
                raise AnalyzerException(
                    f"{special_form} definition missing interface properties for "
                    f"interface {interface.form}: {missing_fields}",
                    form=interface.form,
                    lisp_ast=interface,
                )

            all_interface_methods.update(interface_names)
        elif _is_artificially_abstract(interface.form):
            # Given that artificially abstract bases aren't real `abc.ABC`s and do
            # not annotate their `abstractmethod`s, we can't assert right now that
            # any the type will satisfy the artificially abstract base. However,
            # we can collect any defined methods into a set for artificial bases
            # and assert that any extra methods are included in that set below.
            artificially_abstract.add(interface)
            artificially_abstract_base_members.update(
                map(
                    lambda v: v[0],
                    inspect.getmembers(interface_type, predicate=is_member),
                )
            )
        else:
            raise AnalyzerException(
                f"{special_form} interface must be Python abstract class or object",
                form=interface.form,
                lisp_ast=interface,
            )

    # We cannot compute if there are extra methods defined if there are any
    # unverifiably abstract bases, so we just skip this check.
    if not unverifiably_abstract:
        extra_methods = member_names - all_interface_methods - OBJECT_DUNDER_METHODS
        if extra_methods and not extra_methods.issubset(
            artificially_abstract_base_members
        ):
            extra_method_str = ", ".join(extra_methods)
            raise AnalyzerException(
                f"{special_form} definition for interface includes members not "
                f"part of defined interfaces: {extra_method_str}"
            )

    return not unverifiably_abstract, lset.set(artificially_abstract)


__DEFTYPE_DEFAULT_SENTINEL = object()


def _deftype_ast(  # pylint: disable=too-many-branches,too-many-locals
    ctx: AnalyzerContext, form: ISeq
) -> DefType:
    assert form.first == SpecialForm.DEFTYPE

    nelems = count(form)
    if nelems < 3:
        raise AnalyzerException(
            "deftype forms must have 3 or more elements, as in: "
            "(deftype* name fields :implements [bases+impls])",
            form=form,
        )

    name = runtime.nth(form, 1)
    if not isinstance(name, sym.Symbol):
        raise AnalyzerException(
            f"deftype* names must be symbols, not {type(name)}", form=name
        )
    ctx.put_new_symbol(
        name,
        Binding(
            form=name, name=name.name, local=LocalType.DEFTYPE, env=ctx.get_node_env()
        ),
        warn_if_unused=False,
    )

    fields = runtime.nth(form, 2)
    if not isinstance(fields, vec.Vector):
        raise AnalyzerException(
            f"deftype* fields must be vector, not {type(fields)}", form=fields
        )

    has_defaults = False
    with ctx.new_symbol_table(name.name):
        is_frozen = True
        param_nodes = []
        for field in fields:
            if not isinstance(field, sym.Symbol):
                raise AnalyzerException("deftype* fields must be symbols", form=field)

            field_default = (
                Maybe(field.meta)
                .map(
                    lambda m: m.val_at(SYM_DEFAULT_META_KEY, __DEFTYPE_DEFAULT_SENTINEL)
                )
                .value
            )
            if not has_defaults and field_default is not __DEFTYPE_DEFAULT_SENTINEL:
                has_defaults = True
            elif has_defaults and field_default is __DEFTYPE_DEFAULT_SENTINEL:
                raise AnalyzerException(
                    "deftype* fields without defaults may not appear after fields "
                    "without defaults",
                    form=field,
                )

            is_mutable = _is_mutable(field)
            if is_mutable:
                is_frozen = False

            binding = Binding(
                form=field,
                name=field.name,
                local=LocalType.FIELD,
                is_assignable=is_mutable,
                env=ctx.get_node_env(),
                init=analyze_form(ctx, field_default)
                if field_default is not __DEFTYPE_DEFAULT_SENTINEL
                else None,
            )
            param_nodes.append(binding)
            ctx.put_new_symbol(field, binding, warn_if_unused=False)

        interfaces, members = __deftype_or_reify_impls(
            ctx, runtime.nthrest(form, 3), SpecialForm.DEFTYPE
        )
        (
            verified_abstract,
            artificially_abstract,
        ) = __deftype_and_reify_impls_are_all_abstract(
            SpecialForm.DEFTYPE, map(lambda f: f.name, fields), interfaces, members
        )
        return DefType(
            form=form,
            name=name.name,
            interfaces=vec.vector(interfaces),
            fields=vec.vector(param_nodes),
            members=vec.vector(members),
            verified_abstract=verified_abstract,
            artificially_abstract=artificially_abstract,
            is_frozen=is_frozen,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )


def _do_ast(ctx: AnalyzerContext, form: ISeq) -> Do:
    assert form.first == SpecialForm.DO
    statements, ret = _body_ast(ctx, form.rest)
    return Do(
        form=form,
        statements=vec.vector(statements),
        ret=ret,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def __fn_method_ast(  # pylint: disable=too-many-branches,too-many-locals
    ctx: AnalyzerContext,
    form: ISeq,
    fnname: Optional[sym.Symbol] = None,
    is_async: bool = False,
) -> FnArity:
    with ctx.new_symbol_table("fn-method", is_context_boundary=True):
        params = form.first
        if not isinstance(params, vec.Vector):
            raise AnalyzerException(
                "function arity arguments must be a vector", form=params
            )

        has_vargs, vargs_idx = False, 0
        param_nodes = []
        for i, s in enumerate(params):
            if not isinstance(s, sym.Symbol):
                raise AnalyzerException(
                    "function arity parameter name must be a symbol", form=s
                )

            if s == AMPERSAND:
                has_vargs = True
                vargs_idx = i
                break

            binding = Binding(
                form=s,
                name=s.name,
                local=LocalType.ARG,
                arg_id=i,
                is_variadic=False,
                env=ctx.get_node_env(),
            )
            param_nodes.append(binding)
            ctx.put_new_symbol(s, binding)

        if has_vargs:
            try:
                vargs_sym = params[vargs_idx + 1]

                if not isinstance(vargs_sym, sym.Symbol):
                    raise AnalyzerException(
                        "function rest parameter name must be a symbol", form=vargs_sym
                    )

                binding = Binding(
                    form=vargs_sym,
                    name=vargs_sym.name,
                    local=LocalType.ARG,
                    arg_id=vargs_idx + 1,
                    is_variadic=True,
                    env=ctx.get_node_env(),
                )
                param_nodes.append(binding)
                ctx.put_new_symbol(vargs_sym, binding)
            except IndexError:
                raise AnalyzerException(
                    "Expected variadic argument name after '&'", form=params
                ) from None

        fn_loop_id = genname("fn_arity" if fnname is None else fnname.name)
        with ctx.new_recur_point(fn_loop_id, param_nodes):
            with ctx.new_func_ctx(
                FunctionContext.ASYNC_FUNCTION if is_async else FunctionContext.FUNCTION
            ), ctx.expr_pos():
                stmts, ret = _body_ast(ctx, form.rest)
            method = FnArity(
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


def __fn_kwargs_support(o: IMeta) -> Optional[KeywordArgSupport]:
    if o.meta is None:
        return None

    kwarg_support = o.meta.val_at(SYM_KWARGS_META_KEY)
    if kwarg_support is None:
        return None

    try:
        return KeywordArgSupport(kwarg_support)
    except ValueError:
        raise AnalyzerException(
            "fn keyword argument support metadata :kwarg must be one of: #{:apply :collect}",
            form=kwarg_support,
        )


@_with_meta  # noqa: MC0001
def _fn_ast(  # pylint: disable=too-many-branches
    ctx: AnalyzerContext, form: Union[llist.List, ISeq]
) -> Fn:
    assert form.first == SpecialForm.FN

    idx = 1

    with ctx.new_symbol_table("fn", is_context_boundary=True):
        try:
            name = runtime.nth(form, idx)
        except IndexError:
            raise AnalyzerException(
                "fn form must match: (fn* name? [arg*] body*) or (fn* name? method*)",
                form=form,
            )

        name_node: Optional[Binding]
        if isinstance(name, sym.Symbol):
            name_node = Binding(
                form=name, name=name.name, local=LocalType.FN, env=ctx.get_node_env()
            )
            is_async = _is_async(name) or isinstance(form, IMeta) and _is_async(form)
            kwarg_support = (
                __fn_kwargs_support(name)
                or isinstance(form, IMeta)
                and __fn_kwargs_support(form)
            )
            ctx.put_new_symbol(name, name_node, warn_if_unused=False)
            idx += 1
        elif isinstance(name, (llist.List, vec.Vector)):
            name = None
            name_node = None
            is_async = isinstance(form, IMeta) and _is_async(form)
            kwarg_support = isinstance(form, IMeta) and __fn_kwargs_support(form)
        else:
            raise AnalyzerException(
                "fn form must match: (fn* name? [arg*] body*) or (fn* name? method*)",
                form=form,
            )

        try:
            arity_or_args = runtime.nth(form, idx)
        except IndexError:
            raise AnalyzerException(
                "fn form expects either multiple arities or a vector of arguments",
                form=form,
            )

        if isinstance(arity_or_args, llist.List):
            arities = vec.vector(
                map(
                    partial(__fn_method_ast, ctx, fnname=name, is_async=is_async),
                    runtime.nthrest(form, idx),
                )
            )
        elif isinstance(arity_or_args, vec.Vector):
            arities = vec.v(
                __fn_method_ast(
                    ctx, runtime.nthrest(form, idx), fnname=name, is_async=is_async
                )
            )
        else:
            raise AnalyzerException(
                "fn form must match: (fn* name? [arg*] body*) or (fn* name? method*)",
                form=form,
            )

        nmethods = count(arities)
        assert nmethods > 0, "fn must have at least one arity"

        if kwarg_support is not None and nmethods > 1:
            raise AnalyzerException(
                "multi-arity functions may not declare support for keyword arguments",
                form=form,
            )

        fixed_arities: MutableSet[int] = set()
        fixed_arity_for_variadic: Optional[int] = None
        num_variadic = 0
        for arity in arities:
            if arity.is_variadic:
                if num_variadic > 0:
                    raise AnalyzerException(
                        "fn may have at most 1 variadic arity", form=arity.form
                    )
                fixed_arity_for_variadic = arity.fixed_arity
                num_variadic += 1
            else:
                if arity.fixed_arity in fixed_arities:
                    raise AnalyzerException(
                        "fn may not have multiple methods with the same fixed arity",
                        form=arity.form,
                    )
                fixed_arities.add(arity.fixed_arity)

        if fixed_arity_for_variadic is not None and any(
            fixed_arity_for_variadic < arity for arity in fixed_arities
        ):
            raise AnalyzerException(
                "variadic arity may not have fewer fixed arity arguments than any other arities",
                form=form,
            )

        return Fn(
            form=form,
            is_variadic=num_variadic == 1,
            max_fixed_arity=max(node.fixed_arity for node in arities),
            arities=arities,
            local=name_node,
            env=ctx.get_node_env(pos=ctx.syntax_position),
            is_async=is_async,
            kwarg_support=None if isinstance(kwarg_support, bool) else kwarg_support,
        )


def _host_call_ast(ctx: AnalyzerContext, form: ISeq) -> HostCall:
    assert isinstance(form.first, sym.Symbol)

    method = form.first
    assert isinstance(method, sym.Symbol), "host interop field must be a symbol"
    assert method.name.startswith(".")

    if not count(form) >= 2:
        raise AnalyzerException(
            "host interop calls must be 2 or more elements long", form=form
        )

    args, kwargs = _call_args_ast(ctx, runtime.nthrest(form, 2))
    return HostCall(
        form=form,
        method=method.name[1:],
        target=_analyze_form(ctx, runtime.nth(form, 1)),
        args=args,
        kwargs=kwargs,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _host_prop_ast(ctx: AnalyzerContext, form: ISeq) -> HostField:
    assert isinstance(form.first, sym.Symbol)

    field = form.first
    assert isinstance(field, sym.Symbol), "host interop field must be a symbol"

    nelems = count(form)
    assert field.name.startswith(".-")

    if field.name == ".-":
        try:
            field = runtime.nth(form, 2)
        except IndexError:
            raise AnalyzerException(
                "host interop prop must be exactly 3 elems long: (.- target field)",
                form=form,
            )
        else:
            if not isinstance(field, sym.Symbol):
                raise AnalyzerException(
                    "host interop field must be a symbol", form=form
                )

        if not nelems == 3:
            raise AnalyzerException(
                "host interop prop must be exactly 3 elems long: (.- target field)",
                form=form,
            )

        return HostField(
            form=form,
            field=field.name,
            target=_analyze_form(ctx, runtime.nth(form, 1)),
            is_assignable=True,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )
    else:
        if not nelems == 2:
            raise AnalyzerException(
                "host interop prop must be exactly 2 elements long: (.-field target)",
                form=form,
            )

        return HostField(
            form=form,
            field=field.name[2:],
            target=_analyze_form(ctx, runtime.nth(form, 1)),
            is_assignable=True,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )


def _host_interop_ast(  # pylint: disable=too-many-branches
    ctx: AnalyzerContext, form: ISeq
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
                raise AnalyzerException(
                    "host field accesses must be exactly 3 elements long", form=form
                )

            return HostField(
                form=form,
                field=maybe_m_or_f.name[1:],
                target=_analyze_form(ctx, runtime.nth(form, 1)),
                is_assignable=True,
                env=ctx.get_node_env(pos=ctx.syntax_position),
            )
        else:
            args, kwargs = _call_args_ast(ctx, runtime.nthrest(form, 3))
            return HostCall(
                form=form,
                method=maybe_m_or_f.name,
                target=_analyze_form(ctx, runtime.nth(form, 1)),
                args=args,
                kwargs=kwargs,
                env=ctx.get_node_env(pos=ctx.syntax_position),
            )
    elif isinstance(maybe_m_or_f, (llist.List, ISeq)):
        # Likewise, I emit :host-call for forms like (. target (method arg1 ...)).
        method = maybe_m_or_f.first
        if not isinstance(method, sym.Symbol):
            raise AnalyzerException("host call method must be a symbol", form=method)

        args, kwargs = _call_args_ast(ctx, maybe_m_or_f.rest)
        return HostCall(
            form=form,
            method=method.name[1:] if method.name.startswith("-") else method.name,
            target=_analyze_form(ctx, runtime.nth(form, 1)),
            args=args,
            kwargs=kwargs,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )
    else:
        raise AnalyzerException(
            "host interop forms must take the form: "
            "(. instance (method args*)), "
            "(. instance method args*), "
            "(. instance -field), ",
            form=form,
        )


def _if_ast(ctx: AnalyzerContext, form: ISeq) -> If:
    assert form.first == SpecialForm.IF

    nelems = count(form)
    if nelems not in (3, 4):
        raise AnalyzerException(
            "if forms must have either 3 or 4 elements, as in: (if test then else?)",
            form=form,
        )

    with ctx.expr_pos():
        test_node = _analyze_form(ctx, runtime.nth(form, 1))

    with ctx.parent_pos():
        then_node = _analyze_form(ctx, runtime.nth(form, 2))

        if nelems == 4:
            else_node = _analyze_form(ctx, runtime.nth(form, 3))
        else:
            else_node = _const_node(ctx, None)

    return If(
        form=form,
        test=test_node,
        then=then_node,
        else_=else_node,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _import_ast(  # pylint: disable=too-many-branches
    ctx: AnalyzerContext, form: ISeq
) -> Import:
    assert form.first == SpecialForm.IMPORT

    aliases = []
    for f in form.rest:
        if isinstance(f, sym.Symbol):
            module_name = f
            module_alias = None

            ctx.put_new_symbol(
                module_name,
                Binding(
                    form=module_name,
                    name=module_name.name,
                    local=LocalType.IMPORT,
                    env=ctx.get_node_env(),
                ),
                symbol_table=ctx.symbol_table.context_boundary,
            )
        elif isinstance(f, vec.Vector):
            if len(f) != 3:
                raise AnalyzerException(
                    "import alias must take the form: [module :as alias]", form=f
                )
            module_name = f.val_at(0)
            if not isinstance(module_name, sym.Symbol):
                raise AnalyzerException("Python module name must be a symbol", form=f)
            if not AS == f.val_at(1):
                raise AnalyzerException("expected :as alias for Python import", form=f)
            module_alias_sym = f.val_at(2)
            if not isinstance(module_alias_sym, sym.Symbol):
                raise AnalyzerException("Python module alias must be a symbol", form=f)
            module_alias = module_alias_sym.name

            ctx.put_new_symbol(
                module_alias_sym,
                Binding(
                    form=module_alias_sym,
                    name=module_alias,
                    local=LocalType.IMPORT,
                    env=ctx.get_node_env(),
                ),
                symbol_table=ctx.symbol_table.context_boundary,
            )
        else:
            raise AnalyzerException("symbol or vector expected for import*", form=f)

        aliases.append(
            ImportAlias(
                form=f,
                name=module_name.name,
                alias=module_alias,
                env=ctx.get_node_env(),
            )
        )

    return Import(
        form=form, aliases=aliases, env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _invoke_ast(ctx: AnalyzerContext, form: Union[llist.List, ISeq]) -> Node:
    with ctx.expr_pos():
        fn = _analyze_form(ctx, form.first)

    if fn.op == NodeOp.VAR and isinstance(fn, VarRef):
        if _is_macro(fn.var):
            if ctx.should_macroexpand:
                try:
                    macro_env = ctx.symbol_table.as_env_map()
                    expanded = fn.var.value(macro_env, form, *form.rest)
                    if isinstance(expanded, IWithMeta) and isinstance(form, IMeta):
                        old_meta = expanded.meta
                        expanded = expanded.with_meta(
                            old_meta.cons(form.meta) if old_meta else form.meta
                        )
                    with ctx.macro_ns(
                        fn.var.ns if fn.var.ns is not ctx.current_ns else None
                    ), ctx.expr_pos():
                        expanded_ast = _analyze_form(ctx, expanded)

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

    args, kwargs = _call_args_ast(ctx, form.rest)
    return Invoke(
        form=form,
        fn=fn,
        args=args,
        kwargs=kwargs,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _let_ast(ctx: AnalyzerContext, form: ISeq) -> Let:
    assert form.first == SpecialForm.LET
    nelems = count(form)

    if nelems < 2:
        raise AnalyzerException(
            "let forms must have bindings vector and 0 or more body forms", form=form
        )

    bindings = runtime.nth(form, 1)
    if not isinstance(bindings, vec.Vector):
        raise AnalyzerException("let bindings must be a vector", form=bindings)
    elif len(bindings) % 2 != 0:
        raise AnalyzerException(
            "let bindings must appear in name-value pairs", form=bindings
        )

    with ctx.new_symbol_table("let"):
        binding_nodes = []
        for name, value in partition(bindings, 2):
            if not isinstance(name, sym.Symbol):
                raise AnalyzerException("let binding name must be a symbol", form=name)

            binding = Binding(
                form=name,
                name=name.name,
                local=LocalType.LET,
                init=_analyze_form(ctx, value),
                children=vec.v(INIT),
                env=ctx.get_node_env(),
            )
            binding_nodes.append(binding)
            ctx.put_new_symbol(name, binding)

        let_body = runtime.nthrest(form, 2)
        stmts, ret = _body_ast(ctx, let_body)
        return Let(
            form=form,
            bindings=vec.vector(binding_nodes),
            body=Do(
                form=let_body,
                statements=vec.vector(stmts),
                ret=ret,
                is_body=True,
                env=ctx.get_node_env(),
            ),
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )


def __letfn_fn_body(ctx: AnalyzerContext, form: ISeq) -> Fn:
    """Produce an `Fn` node for a `letfn*` special form.

    `letfn*` forms use `let*`-like bindings. Each function binding name is
    added to the symbol table as a forward declaration before analyzing the
    function body. The function bodies are defined as

        (fn* name
          [...]
          ...)

    When the `name` is added to the symbol table for the function, a warning
    will be produced because it will previously have been defined in the
    `letfn*` binding scope. This function adds `:no-warn-on-shadow` metadata to
    the function name symbol to disable the compiler warning."""
    fn_sym = form.first

    fn_name = runtime.nth(form, 1)
    if not isinstance(fn_name, sym.Symbol):
        raise AnalyzerException("letfn function name must be a symbol", form=fn_name)

    fn_rest = runtime.nthrest(form, 2)

    fn_body = _analyze_form(
        ctx,
        fn_rest.cons(
            fn_name.with_meta(
                (fn_name.meta or lmap.Map.empty()).assoc(
                    SYM_NO_WARN_ON_SHADOW_META_KEY, True
                )
            )
        ).cons(fn_sym),
    )

    if not isinstance(fn_body, Fn):
        raise AnalyzerException(
            "letfn bindings must be functions", form=form, lisp_ast=fn_body
        )

    return fn_body


def _letfn_ast(  # pylint: disable=too-many-locals
    ctx: AnalyzerContext, form: ISeq
) -> LetFn:
    assert form.first == SpecialForm.LETFN
    nelems = count(form)

    if nelems < 2:
        raise AnalyzerException(
            "letfn forms must have bindings vector and 0 or more body forms", form=form
        )

    bindings = runtime.nth(form, 1)
    if not isinstance(bindings, vec.Vector):
        raise AnalyzerException("letfn bindings must be a vector", form=bindings)
    elif len(bindings) % 2 != 0:
        raise AnalyzerException(
            "letfn bindings must appear in name-value pairs", form=bindings
        )

    with ctx.new_symbol_table("letfn"):
        # Generate empty Binding nodes to put into the symbol table
        # as forward declarations. All functions in letfn* forms may
        # refer to all other functions regardless of order of definition.
        empty_binding_nodes = []
        for name, value in partition(bindings, 2):
            if not isinstance(name, sym.Symbol):
                raise AnalyzerException(
                    "letfn binding name must be a symbol", form=name
                )

            if not isinstance(value, llist.List):
                raise AnalyzerException(
                    "letfn binding value must be a list", form=value
                )

            binding = Binding(
                form=name,
                name=name.name,
                local=LocalType.LETFN,
                init=_const_node(ctx, None),
                children=vec.v(INIT),
                env=ctx.get_node_env(),
            )
            empty_binding_nodes.append((name, value, binding))
            ctx.put_new_symbol(
                name, binding,
            )

        # Once we've generated all of the filler Binding nodes, analyze the
        # function bodies and replace the Binding nodes with full nodes.
        binding_nodes = []
        for fn_name, fn_def, binding in empty_binding_nodes:
            fn_body = __letfn_fn_body(ctx, fn_def)
            new_binding = binding.assoc(init=fn_body)
            binding_nodes.append(new_binding)
            ctx.put_new_symbol(
                fn_name,
                new_binding,
                warn_on_shadowed_name=False,
                warn_on_shadowed_var=False,
            )

        letfn_body = runtime.nthrest(form, 2)
        stmts, ret = _body_ast(ctx, letfn_body)
        return LetFn(
            form=form,
            bindings=vec.vector(binding_nodes),
            body=Do(
                form=letfn_body,
                statements=vec.vector(stmts),
                ret=ret,
                is_body=True,
                env=ctx.get_node_env(),
            ),
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )


def _loop_ast(ctx: AnalyzerContext, form: ISeq) -> Loop:
    assert form.first == SpecialForm.LOOP
    nelems = count(form)

    if nelems < 2:
        raise AnalyzerException(
            "loop forms must have bindings vector and 0 or more body forms", form=form
        )

    bindings = runtime.nth(form, 1)
    if not isinstance(bindings, vec.Vector):
        raise AnalyzerException("loop bindings must be a vector", form=bindings)
    elif len(bindings) % 2 != 0:
        raise AnalyzerException(
            "loop bindings must appear in name-value pairs", form=bindings
        )

    loop_id = genname("loop")
    with ctx.new_symbol_table(loop_id):
        binding_nodes = []
        for name, value in partition(bindings, 2):
            if not isinstance(name, sym.Symbol):
                raise AnalyzerException("loop binding name must be a symbol", form=name)

            binding = Binding(
                form=name,
                name=name.name,
                local=LocalType.LOOP,
                init=_analyze_form(ctx, value),
                env=ctx.get_node_env(),
            )
            binding_nodes.append(binding)
            ctx.put_new_symbol(name, binding)

        with ctx.new_recur_point(loop_id, binding_nodes):
            loop_body = runtime.nthrest(form, 2)
            stmts, ret = _body_ast(ctx, loop_body)
            loop_node = Loop(
                form=form,
                bindings=vec.vector(binding_nodes),
                body=Do(
                    form=loop_body,
                    statements=vec.vector(stmts),
                    ret=ret,
                    is_body=True,
                    env=ctx.get_node_env(),
                ),
                loop_id=loop_id,
                env=ctx.get_node_env(pos=ctx.syntax_position),
            )
            loop_node.visit(_assert_recur_is_tail)
            return loop_node


def _quote_ast(ctx: AnalyzerContext, form: ISeq) -> Quote:
    assert form.first == SpecialForm.QUOTE
    nelems = count(form)

    if nelems != 2:
        raise AnalyzerException(
            "quote forms must have exactly two elements: (quote form)", form=form
        )

    with ctx.quoted():
        with ctx.expr_pos():
            expr = _analyze_form(ctx, runtime.nth(form, 1))
        assert isinstance(expr, Const), "Quoted expressions must yield :const nodes"
        return Quote(
            form=form,
            expr=expr,
            is_literal=True,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )


def _assert_no_recur(node: Node) -> None:
    """Assert that `recur` forms do not appear in any position of this or
    child AST nodes."""
    if node.op == NodeOp.RECUR:
        raise AnalyzerException(
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
    elif node.op in {
        NodeOp.FN,
        NodeOp.FN_ARITY,
        NodeOp.DEFTYPE_METHOD,
        NodeOp.DEFTYPE_METHOD_ARITY,
    }:
        assert isinstance(node, (Fn, FnArity, DefTypeMethod, DefTypeMethodArity))
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
    elif node.op == NodeOp.REIFY:
        assert isinstance(node, Reify)
        for child in node.members:
            _assert_recur_is_tail(child)
    elif node.op == NodeOp.TRY:
        assert isinstance(node, Try)
        _assert_recur_is_tail(node.body)
        for catch in node.catches:
            _assert_recur_is_tail(catch)
        if node.finally_:
            _assert_no_recur(node.finally_)
    else:
        node.visit(_assert_no_recur)


def _recur_ast(ctx: AnalyzerContext, form: ISeq) -> Recur:
    assert form.first == SpecialForm.RECUR

    if ctx.recur_point is None:
        raise AnalyzerException("no recur point defined for recur", form=form)

    if len(ctx.recur_point.args) != count(form.rest):
        raise AnalyzerException(
            "recur arity does not match last recur point arity", form=form
        )

    with ctx.expr_pos():
        exprs = vec.vector(map(partial(_analyze_form, ctx), form.rest))

    return Recur(
        form=form, exprs=exprs, loop_id=ctx.recur_point.loop_id, env=ctx.get_node_env()
    )


@_with_meta
def _reify_ast(ctx: AnalyzerContext, form: ISeq) -> Reify:
    assert form.first == SpecialForm.REIFY

    nelems = count(form)
    if nelems < 3:
        raise AnalyzerException(
            "reify forms must have 3 or more elements, as in: "
            "(reify* :implements [bases+impls])",
            form=form,
        )

    with ctx.new_symbol_table("reify"):
        interfaces, members = __deftype_or_reify_impls(
            ctx, runtime.nthrest(form, 1), SpecialForm.REIFY
        )
        (
            verified_abstract,
            artificially_abstract,
        ) = __deftype_and_reify_impls_are_all_abstract(
            SpecialForm.REIFY, (), interfaces, members
        )
        return Reify(
            form=form,
            interfaces=vec.vector(interfaces),
            members=vec.vector(members),
            verified_abstract=verified_abstract,
            artificially_abstract=artificially_abstract,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )


def _require_ast(  # pylint: disable=too-many-branches
    ctx: AnalyzerContext, form: ISeq
) -> Require:
    assert form.first == SpecialForm.REQUIRE

    aliases = []
    for f in form.rest:
        if isinstance(f, sym.Symbol):
            module_name = f
            module_alias = None
        elif isinstance(f, vec.Vector):
            if len(f) != 3:
                raise AnalyzerException(
                    "require alias must take the form: [namespace :as alias]", form=f
                )
            module_name = f.val_at(0)
            if not isinstance(module_name, sym.Symbol):
                raise AnalyzerException(
                    "Basilisp namespace name must be a symbol", form=f
                )
            if not AS == f.val_at(1):
                raise AnalyzerException("expected :as alias for Basilisp alias", form=f)
            module_alias_sym = f.val_at(2)
            if not isinstance(module_alias_sym, sym.Symbol):
                raise AnalyzerException(
                    "Basilisp namespace alias must be a symbol", form=f
                )
            module_alias = module_alias_sym.name
        else:
            raise AnalyzerException("symbol or vector expected for require*", form=f)

        aliases.append(
            RequireAlias(
                form=f,
                name=module_name.name,
                alias=module_alias,
                env=ctx.get_node_env(),
            )
        )

    return Require(
        form=form, aliases=aliases, env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _set_bang_ast(ctx: AnalyzerContext, form: ISeq) -> SetBang:
    assert form.first == SpecialForm.SET_BANG
    nelems = count(form)

    if nelems != 3:
        raise AnalyzerException(
            "set! forms must contain exactly 3 elements: (set! target value)", form=form
        )

    with ctx.expr_pos():
        target = _analyze_form(ctx, runtime.nth(form, 1))

    if not isinstance(target, Assignable):
        raise AnalyzerException(
            f"cannot set! targets of type {type(target)}", form=target
        )

    if not target.is_assignable:
        raise AnalyzerException(
            f"cannot set! target which is not assignable", form=target
        )

    with ctx.expr_pos():
        val = _analyze_form(ctx, runtime.nth(form, 2))

    return SetBang(
        form=form,
        target=target,
        val=val,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _throw_ast(ctx: AnalyzerContext, form: ISeq) -> Throw:
    assert form.first == SpecialForm.THROW
    with ctx.expr_pos():
        exc = _analyze_form(ctx, runtime.nth(form, 1))
    return Throw(
        form=form, exception=exc, env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _catch_ast(ctx: AnalyzerContext, form: ISeq) -> Catch:
    assert form.first == SpecialForm.CATCH
    nelems = count(form)

    if nelems < 4:
        raise AnalyzerException(
            "catch forms must contain at least 4 elements: (catch class local body*)",
            form=form,
        )

    catch_cls = _analyze_form(ctx, runtime.nth(form, 1))
    if not isinstance(catch_cls, (MaybeClass, MaybeHostForm)):
        raise AnalyzerException(
            "catch forms must name a class type to catch", form=catch_cls
        )

    local_name = runtime.nth(form, 2)
    if not isinstance(local_name, sym.Symbol):
        raise AnalyzerException("catch local must be a symbol", form=local_name)

    with ctx.new_symbol_table("catch"):
        catch_binding = Binding(
            form=local_name,
            name=local_name.name,
            local=LocalType.CATCH,
            env=ctx.get_node_env(),
        )
        ctx.put_new_symbol(local_name, catch_binding)

        catch_body = runtime.nthrest(form, 3)
        catch_statements, catch_ret = _body_ast(ctx, catch_body)
        return Catch(
            form=form,
            class_=catch_cls,
            local=catch_binding,
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
    ctx: AnalyzerContext, form: ISeq
) -> Try:
    assert form.first == SpecialForm.TRY

    try_exprs = []
    catches = []
    finally_: Optional[Do] = None
    for expr in form.rest:
        if isinstance(expr, (llist.List, ISeq)):
            if expr.first == SpecialForm.CATCH:
                if finally_:
                    raise AnalyzerException(
                        "catch forms may not appear after finally forms in a try",
                        form=expr,
                    )
                catches.append(_catch_ast(ctx, expr))
                continue
            elif expr.first == SpecialForm.FINALLY:
                if finally_ is not None:
                    raise AnalyzerException(
                        "try forms may not contain multiple finally forms", form=expr
                    )
                # Finally values are never returned
                with ctx.stmt_pos():
                    *finally_stmts, finally_ret = map(
                        partial(_analyze_form, ctx), expr.rest
                    )
                finally_ = Do(
                    form=expr.rest,
                    statements=vec.vector(finally_stmts),
                    ret=finally_ret,
                    is_body=True,
                    env=ctx.get_node_env(pos=NodeSyntacticPosition.STMT,),
                )
                continue

        lisp_node = _analyze_form(ctx, expr)

        if catches:
            raise AnalyzerException(
                "try body expressions may not appear after catch forms", form=expr
            )
        if finally_:
            raise AnalyzerException(
                "try body expressions may not appear after finally forms", form=expr
            )

        try_exprs.append(lisp_node)

    assert all(
        isinstance(node, Catch) for node in catches
    ), "All catch statements must be catch ops"

    *try_statements, try_ret = try_exprs
    return Try(
        form=form,
        body=Do(
            form=form,
            statements=vec.vector(try_statements),
            ret=try_ret,
            is_body=True,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        ),
        catches=vec.vector(catches),
        finally_=finally_,
        children=vec.v(BODY, CATCHES, FINALLY)
        if finally_ is not None
        else vec.v(BODY, CATCHES),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _var_ast(ctx: AnalyzerContext, form: ISeq) -> VarRef:
    assert form.first == SpecialForm.VAR

    nelems = count(form)
    if nelems != 2:
        raise AnalyzerException(
            "var special forms must contain 2 elements: (var sym)", form=form
        )

    var_sym = runtime.nth(form, 1)
    if not isinstance(var_sym, sym.Symbol):
        raise AnalyzerException("vars may only be resolved for symbols", form=form)

    if var_sym.ns is None:
        var = runtime.resolve_var(sym.symbol(var_sym.name, ctx.current_ns.name))
    else:
        var = runtime.resolve_var(var_sym)

    if var is None:
        raise AnalyzerException(f"cannot resolve var {var_sym}", form=form)

    return VarRef(
        form=var_sym,
        var=var,
        return_var=True,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


SpecialFormHandler = Callable[[AnalyzerContext, ISeq], SpecialFormNode]
_SPECIAL_FORM_HANDLERS: Mapping[sym.Symbol, SpecialFormHandler] = {
    SpecialForm.AWAIT: _await_ast,
    SpecialForm.DEF: _def_ast,
    SpecialForm.DO: _do_ast,
    SpecialForm.DEFTYPE: _deftype_ast,
    SpecialForm.FN: _fn_ast,
    SpecialForm.IF: _if_ast,
    SpecialForm.IMPORT: _import_ast,
    SpecialForm.INTEROP_CALL: _host_interop_ast,
    SpecialForm.LET: _let_ast,
    SpecialForm.LETFN: _letfn_ast,
    SpecialForm.LOOP: _loop_ast,
    SpecialForm.QUOTE: _quote_ast,
    SpecialForm.RECUR: _recur_ast,
    SpecialForm.REIFY: _reify_ast,
    SpecialForm.REQUIRE: _require_ast,
    SpecialForm.SET_BANG: _set_bang_ast,
    SpecialForm.THROW: _throw_ast,
    SpecialForm.TRY: _try_ast,
    SpecialForm.VAR: _var_ast,
}


def _list_node(ctx: AnalyzerContext, form: ISeq) -> Node:
    if ctx.is_quoted:
        return _const_node(ctx, form)

    s = form.first
    if isinstance(s, sym.Symbol):
        handle_special_form = _SPECIAL_FORM_HANDLERS.get(s)
        if handle_special_form is not None:
            return handle_special_form(ctx, form)
        elif s.name.startswith(".-"):
            return _host_prop_ast(ctx, form)
        elif s.name.startswith(".") and s.name != _DOUBLE_DOT_MACRO_NAME:
            return _host_call_ast(ctx, form)

    return _invoke_ast(ctx, form)


def _resolve_nested_symbol(ctx: AnalyzerContext, form: sym.Symbol) -> HostField:
    """Resolve an attribute by recursively accessing the parent object
    as if it were its own namespaced symbol."""
    assert form.ns is not None
    assert "." in form.ns

    parent_ns, parent_name = form.ns.rsplit(".", maxsplit=1)
    parent = sym.symbol(parent_name, ns=parent_ns)
    parent_node = __resolve_namespaced_symbol(ctx, parent)

    return HostField(
        form=form,
        field=form.name,
        target=parent_node,
        is_assignable=True,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def __resolve_namespaced_symbol_in_ns(  # pylint: disable=too-many-branches
    ctx: AnalyzerContext, which_ns: runtime.Namespace, form: sym.Symbol,
) -> Optional[Union[MaybeHostForm, VarRef]]:
    """Resolve the symbol `form` in the context of the Namespace `which_ns`. If
    `allow_fuzzy_macroexpansion_matching` is True and no match is made on existing
    imports, import aliases, or namespace aliases, then attempt to match the
    namespace portion"""
    assert form.ns is not None

    ns_sym = sym.symbol(form.ns)
    if ns_sym in which_ns.imports or ns_sym in which_ns.import_aliases:
        # Fetch the full namespace name for the aliased namespace/module.
        # We don't need this for actually generating the link later, but
        # we _do_ need it for fetching a reference to the module to check
        # for membership.
        if ns_sym in which_ns.import_aliases:
            ns = which_ns.import_aliases[ns_sym]
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
        if safe_name in vars(ns_module):
            return MaybeHostForm(
                form=form,
                class_=munge(ns_sym.name),
                field=safe_name,
                target=vars(ns_module)[safe_name],
                env=ctx.get_node_env(pos=ctx.syntax_position),
            )

        # Then allow builtins
        safe_name = munge(form.name, allow_builtins=True)
        if safe_name not in vars(ns_module):
            raise AnalyzerException("can't identify aliased form", form=form)

        # Aliased imports generate code which uses the import alias, so we
        # don't need to care if this is an import or an alias.
        return MaybeHostForm(
            form=form,
            class_=munge(ns_sym.name),
            field=safe_name,
            target=vars(ns_module)[safe_name],
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )
    elif ns_sym in which_ns.aliases:
        aliased_ns: runtime.Namespace = which_ns.aliases[ns_sym]
        v = Var.find_in_ns(aliased_ns, sym.symbol(form.name))
        if v is None:
            raise AnalyzerException(
                f"unable to resolve symbol '{sym.symbol(form.name, ns_sym.name)}' in this context",
                form=form,
            )
        elif v.meta is not None and v.meta.val_at(SYM_PRIVATE_META_KEY, False):
            raise AnalyzerException(
                f"cannot resolve private Var {form.name} from namespace {form.ns}",
                form=form,
            )
        return VarRef(form=form, var=v, env=ctx.get_node_env(pos=ctx.syntax_position),)

    return None


def __resolve_namespaced_symbol(  # pylint: disable=too-many-branches  # noqa: MC0001
    ctx: AnalyzerContext, form: sym.Symbol
) -> Union[Const, HostField, MaybeClass, MaybeHostForm, VarRef]:
    """Resolve a namespaced symbol into a Python name or Basilisp Var."""
    assert form.ns is not None

    current_ns = ctx.current_ns
    if form.ns == current_ns.name:
        v = current_ns.find(sym.symbol(form.name))
        if v is not None:
            return VarRef(
                form=form, var=v, env=ctx.get_node_env(pos=ctx.syntax_position),
            )
    elif form.ns == _BUILTINS_NS:
        class_ = munge(form.name, allow_builtins=True)
        target = getattr(builtins, class_, None)
        if target is None:
            raise AnalyzerException(
                f"cannot resolve builtin function '{class_}'", form=form
            )
        return MaybeClass(
            form=form,
            class_=class_,
            target=target,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )

    v = Var.find(form)
    if v is not None:
        # Disallow global references to Vars defined with :private metadata
        #
        # Global references to private Vars are allowed in macroexpanded code
        # as long as the macro referencing the private Var is in the same
        # Namespace as the private Var (via `ctx.current_macro_ns`)
        if (
            v.ns != ctx.current_macro_ns
            and v.meta is not None
            and v.meta.val_at(SYM_PRIVATE_META_KEY, False)
        ):
            raise AnalyzerException(
                f"cannot resolve private Var {form.name} from namespace {form.ns}",
                form=form,
            )
        return VarRef(form=form, var=v, env=ctx.get_node_env(pos=ctx.syntax_position))

    if "." in form.name and form.name != _DOUBLE_DOT_MACRO_NAME:
        raise AnalyzerException(
            "symbol names may not contain the '.' operator", form=form
        )

    resolved = __resolve_namespaced_symbol_in_ns(ctx, current_ns, form)
    if resolved is not None:
        return resolved

    if "." in form.ns:
        try:
            return _resolve_nested_symbol(ctx, form)
        except CompilerException:
            raise AnalyzerException(
                f"unable to resolve symbol '{form}' in this context", form=form
            ) from None
    elif ctx.should_allow_unresolved_symbols:
        return _const_node(ctx, form)

    # Imports and requires nested in function definitions, method definitions, and
    # `(do ...)` forms are not statically resolvable, since they haven't necessarily
    # been imported and we want to minimize side-effecting from the compiler. In these
    # cases, we merely verify that we've seen the symbol before and defer to runtime
    # checks by the Python VM to verify that the import or require is legitimate.
    maybe_import_or_require_sym = sym.symbol(form.ns)
    maybe_import_or_require_entry = ctx.symbol_table.find_symbol(
        maybe_import_or_require_sym
    )
    if maybe_import_or_require_entry is not None:
        if maybe_import_or_require_entry.context == LocalType.IMPORT:
            ctx.symbol_table.mark_used(maybe_import_or_require_sym)
            return MaybeHostForm(
                form=form,
                class_=munge(form.ns),
                field=munge(form.name),
                target=None,
                env=ctx.get_node_env(pos=ctx.syntax_position),
            )

    # Static and class methods on types in the current namespace can be referred
    # to as `Type/static-method`. In these cases, we will try to resolve the
    # namespace portion of the symbol as a Var within the current namespace.
    maybe_type_or_class = current_ns.find(sym.symbol(form.ns))
    if maybe_type_or_class is not None:
        safe_name = munge(form.name)
        member = getattr(maybe_type_or_class.value, safe_name, None)

        if member is None:
            raise AnalyzerException(
                f"unable to resolve static or class member '{form}' in this context",
                form=form,
            )

        return HostField(
            form=form,
            field=safe_name,
            target=VarRef(
                form=form,
                var=maybe_type_or_class,
                env=ctx.get_node_env(pos=ctx.syntax_position),
            ),
            is_assignable=False,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )

    raise AnalyzerException(
        f"unable to resolve symbol '{form}' in this context", form=form
    )


def __resolve_bare_symbol(
    ctx: AnalyzerContext, form: sym.Symbol
) -> Union[Const, MaybeClass, VarRef]:
    """Resolve a non-namespaced symbol into a Python name or a local
    Basilisp Var."""
    assert form.ns is None

    # Look up the symbol in the namespace mapping of the current namespace.
    current_ns = ctx.current_ns
    v = current_ns.find(form)
    if v is not None:
        return VarRef(form=form, var=v, env=ctx.get_node_env(pos=ctx.syntax_position),)

    # Look up the symbol in the current macro namespace, if one
    if ctx.current_macro_ns is not None:
        v = ctx.current_macro_ns.find(form)
        if v is not None:
            return VarRef(
                form=form, var=v, env=ctx.get_node_env(pos=ctx.syntax_position),
            )

    if "." in form.name:
        raise AnalyzerException(
            "symbol names may not contain the '.' operator", form=form
        )

    munged = munge(form.name, allow_builtins=True)
    if munged in vars(builtins):
        return MaybeClass(
            form=form,
            class_=munged,
            target=vars(builtins)[munged],
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )

    # Allow users to resolve imported module names directly
    maybe_import = current_ns.get_import(form)
    if maybe_import is not None:
        return MaybeClass(
            form=form,
            class_=munge(form.name),
            target=maybe_import,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )

    if ctx.should_allow_unresolved_symbols:
        return _const_node(ctx, form)

    assert munged not in vars(current_ns.module)
    raise AnalyzerException(
        f"unable to resolve symbol '{form}' in this context", form=form
    )


def _resolve_sym(
    ctx: AnalyzerContext, form: sym.Symbol
) -> Union[Const, HostField, MaybeClass, MaybeHostForm, VarRef]:
    """Resolve a Basilisp symbol as a Var or Python name."""
    # Support special class-name syntax to instantiate new classes
    #   (Classname. *args)
    #   (aliased.Classname. *args)
    #   (fully.qualified.Classname. *args)
    if (
        form.ns is None
        and form.name.endswith(".")
        and form.name != _DOUBLE_DOT_MACRO_NAME
    ):
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
    ctx: AnalyzerContext, form: sym.Symbol
) -> Union[Const, HostField, Local, MaybeClass, MaybeHostForm, VarRef]:
    if ctx.is_quoted:
        return _const_node(ctx, form)

    sym_entry = ctx.symbol_table.find_symbol(form)
    if sym_entry is not None:
        ctx.symbol_table.mark_used(form)
        return Local(
            form=form,
            name=form.name,
            local=sym_entry.context,
            is_assignable=sym_entry.binding.is_assignable,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )

    return _resolve_sym(ctx, form)


def _py_dict_node(ctx: AnalyzerContext, form: dict) -> PyDict:
    keys, vals = [], []
    for k, v in form.items():
        keys.append(_analyze_form(ctx, k))
        vals.append(_analyze_form(ctx, v))

    return PyDict(
        form=form,
        keys=vec.vector(keys),
        vals=vec.vector(vals),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _py_list_node(ctx: AnalyzerContext, form: list) -> PyList:
    return PyList(
        form=form,
        items=vec.vector(map(partial(_analyze_form, ctx), form)),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _py_set_node(ctx: AnalyzerContext, form: set) -> PySet:
    return PySet(
        form=form,
        items=vec.vector(map(partial(_analyze_form, ctx), form)),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _py_tuple_node(ctx: AnalyzerContext, form: tuple) -> PyTuple:
    return PyTuple(
        form=form,
        items=vec.vector(map(partial(_analyze_form, ctx), form)),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


@_with_meta
def _map_node(ctx: AnalyzerContext, form: lmap.Map) -> MapNode:
    keys, vals = [], []
    for k, v in form.items():
        keys.append(_analyze_form(ctx, k))
        vals.append(_analyze_form(ctx, v))

    return MapNode(
        form=form,
        keys=vec.vector(keys),
        vals=vec.vector(vals),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


@_with_meta
def _set_node(ctx: AnalyzerContext, form: lset.Set) -> SetNode:
    return SetNode(
        form=form,
        items=vec.vector(map(partial(_analyze_form, ctx), form)),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


@_with_meta
def _vector_node(ctx: AnalyzerContext, form: vec.Vector) -> VectorNode:
    return VectorNode(
        form=form,
        items=vec.vector(map(partial(_analyze_form, ctx), form)),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


_CONST_NODE_TYPES: Mapping[Type, ConstType] = {
    bool: ConstType.BOOL,
    complex: ConstType.NUMBER,
    datetime: ConstType.INST,
    Decimal: ConstType.DECIMAL,
    dict: ConstType.PY_DICT,
    float: ConstType.NUMBER,
    Fraction: ConstType.FRACTION,
    int: ConstType.NUMBER,
    kw.Keyword: ConstType.KEYWORD,
    list: ConstType.PY_LIST,
    llist.List: ConstType.SEQ,
    lmap.Map: ConstType.MAP,
    lset.Set: ConstType.SET,
    IRecord: ConstType.RECORD,
    ISeq: ConstType.SEQ,
    IType: ConstType.TYPE,
    type(re.compile("")): ConstType.REGEX,
    set: ConstType.PY_SET,
    sym.Symbol: ConstType.SYMBOL,
    str: ConstType.STRING,
    tuple: ConstType.PY_TUPLE,
    type(None): ConstType.NIL,
    uuid.UUID: ConstType.UUID,
    vec.Vector: ConstType.VECTOR,
}


def _const_node(ctx: AnalyzerContext, form: ReaderForm) -> Const:
    assert (
        (
            ctx.is_quoted
            and isinstance(
                form, (sym.Symbol, vec.Vector, llist.List, lmap.Map, lset.Set, ISeq)
            )
        )
        or (ctx.should_allow_unresolved_symbols and isinstance(form, sym.Symbol))
        or (isinstance(form, (llist.List, ISeq)) and form.is_empty)
        or isinstance(
            form,
            (
                bool,
                complex,
                datetime,
                Decimal,
                dict,
                float,
                Fraction,
                int,
                IRecord,
                IType,
                kw.Keyword,
                list,
                Pattern,
                set,
                str,
                tuple,
                type(None),
                uuid.UUID,
            ),
        )
    ), "Constant nodes must be composed of constant values"

    node_type = _CONST_NODE_TYPES.get(type(form), ConstType.UNKNOWN)
    if node_type == ConstType.UNKNOWN:
        if isinstance(form, IRecord):
            node_type = ConstType.RECORD
        elif isinstance(form, ISeq):
            node_type = ConstType.SEQ
        elif isinstance(form, IType):
            node_type = ConstType.TYPE
    assert node_type != ConstType.UNKNOWN, "Only allow known constant types"

    descriptor = Const(
        form=form,
        is_literal=True,
        type=cast(ConstType, node_type),
        val=form,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )

    if hasattr(form, "meta"):
        form_meta = _clean_meta(form.meta)  # type: ignore
        if form_meta is not None:
            meta_ast = _const_node(ctx, form_meta)
            assert isinstance(meta_ast, MapNode) or (
                isinstance(meta_ast, Const) and meta_ast.type == ConstType.MAP
            )
            return descriptor.assoc(meta=meta_ast, children=vec.v(META))

    return descriptor


@_with_loc  # noqa: MC0001
def _analyze_form(  # pylint: disable=too-many-branches
    ctx: AnalyzerContext, form: Union[ReaderForm, ISeq]
) -> Node:
    if isinstance(form, (llist.List, ISeq)):
        # Special case for unquoted empty list
        if form == llist.List.empty():
            with ctx.quoted():
                return _const_node(ctx, form)
        else:
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
            IRecord,
            IType,
            kw.Keyword,
            Pattern,
            sym.Symbol,
            str,
            type(None),
            uuid.UUID,
        ),
    ):
        return _const_node(ctx, form)
    elif isinstance(form, dict):
        if ctx.is_quoted:
            return _const_node(ctx, form)
        return _py_dict_node(ctx, form)
    elif isinstance(form, list):
        if ctx.is_quoted:
            return _const_node(ctx, form)
        return _py_list_node(ctx, form)
    elif isinstance(form, set):
        if ctx.is_quoted:
            return _const_node(ctx, form)
        return _py_set_node(ctx, form)
    elif isinstance(form, tuple):
        if ctx.is_quoted:
            return _const_node(ctx, form)
        return _py_tuple_node(ctx, form)
    else:  # pragma: no cover
        raise AnalyzerException(f"Unexpected form type {type(form)}", form=form)


def analyze_form(ctx: AnalyzerContext, form: ReaderForm) -> Node:
    """Take a Lisp form as an argument and produce a Basilisp syntax
    tree matching the clojure.tools.analyzer AST spec."""
    return _analyze_form(ctx, form).assoc(top_level=True)


def macroexpand_1(form: ReaderForm) -> ReaderForm:
    """Macroexpand form one time. Returns the macroexpanded form. The return
    value may still represent a macro. Does not macroexpand child forms."""
    ctx = AnalyzerContext(
        "<Macroexpand>", should_macroexpand=False, allow_unresolved_symbols=True
    )
    maybe_macro = analyze_form(ctx, form)
    if maybe_macro.op == NodeOp.INVOKE:
        assert isinstance(maybe_macro, Invoke)

        fn = maybe_macro.fn
        if fn.op == NodeOp.VAR and isinstance(fn, VarRef):
            if _is_macro(fn.var):
                assert isinstance(form, ISeq)
                macro_env = ctx.symbol_table.as_env_map()
                return fn.var.value(macro_env, form, *form.rest)
    return maybe_macro.form


def macroexpand(form: ReaderForm) -> ReaderForm:
    """Repeatedly macroexpand form as by macroexpand-1 until form no longer
    represents a macro. Returns the expanded form. Does not macroexpand child
    forms."""
    return analyze_form(
        AnalyzerContext("<Macroexpand>", allow_unresolved_symbols=True), form
    ).form
