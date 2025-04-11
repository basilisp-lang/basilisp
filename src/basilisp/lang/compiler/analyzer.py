# pylint: disable=abstract-class-instantiated,too-many-branches,too-many-lines,too-many-return-statements
import builtins
import collections
import contextlib
import functools
import inspect
import logging
import platform
import re
import sys
import uuid
from collections import defaultdict
from collections.abc import (
    Collection,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSet,
)
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from functools import partial, wraps
from re import Pattern
from typing import Any, Callable, Optional, TypeVar, Union, cast

import attr
from typing_extensions import Literal

from basilisp.lang import keyword as kw
from basilisp.lang import list as llist
from basilisp.lang import map as lmap
from basilisp.lang import queue as lqueue
from basilisp.lang import reader as reader
from basilisp.lang import runtime as runtime
from basilisp.lang import set as lset
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec
from basilisp.lang.compiler.constants import (
    AMPERSAND,
    ARGLISTS_KW,
    COL_KW,
    DEFAULT_COMPILER_FILE_PATH,
    DOC_KW,
    END_COL_KW,
    END_LINE_KW,
    FILE_KW,
    LINE_KW,
    NAME_KW,
    NS_KW,
    REST_KW,
    SYM_ABSTRACT_MEMBERS_META_KEY,
    SYM_ABSTRACT_META_KEY,
    SYM_ASYNC_META_KEY,
    SYM_CLASSMETHOD_META_KEY,
    SYM_DEFAULT_META_KEY,
    SYM_DYNAMIC_META_KEY,
    SYM_GEN_SAFE_PYTHON_PARAM_NAMES_META_KEY,
    SYM_INLINE_META_KW,
    SYM_KWARGS_META_KEY,
    SYM_MACRO_META_KEY,
    SYM_MUTABLE_META_KEY,
    SYM_NO_INLINE_META_KEY,
    SYM_NO_WARN_ON_REDEF_META_KEY,
    SYM_NO_WARN_ON_SHADOW_META_KEY,
    SYM_NO_WARN_ON_VAR_INDIRECTION_META_KEY,
    SYM_NO_WARN_WHEN_UNUSED_META_KEY,
    SYM_PRIVATE_META_KEY,
    SYM_PROPERTY_META_KEY,
    SYM_REDEF_META_KEY,
    SYM_STATICMETHOD_META_KEY,
    SYM_TAG_META_KEY,
    SYM_USE_VAR_INDIRECTION_KEY,
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
    FunctionContextType,
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
)
from basilisp.lang.compiler.nodes import Map as MapNode
from basilisp.lang.compiler.nodes import (
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
)
from basilisp.lang.compiler.nodes import Queue as QueueNode
from basilisp.lang.compiler.nodes import (
    Quote,
    Recur,
    Reify,
    Require,
    RequireAlias,
)
from basilisp.lang.compiler.nodes import Set as SetNode
from basilisp.lang.compiler.nodes import (
    SetBang,
    SpecialFormNode,
    Throw,
    Try,
    VarRef,
)
from basilisp.lang.compiler.nodes import Vector as VectorNode
from basilisp.lang.compiler.nodes import (
    WithMeta,
    Yield,
    deftype_or_reify_python_member_names,
)
from basilisp.lang.interfaces import IMeta, INamed, IRecord, ISeq, IType, IWithMeta
from basilisp.lang.runtime import Var
from basilisp.lang.typing import CompilerOpts, LispForm, ReaderForm
from basilisp.lang.util import OBJECT_DUNDER_METHODS, count, genname, is_abstract, munge
from basilisp.logconfig import TRACE
from basilisp.util import Maybe, partition

# Analyzer logging
logger = logging.getLogger(__name__)

# Analyzer options
GENERATE_AUTO_INLINES = kw.keyword("generate-auto-inlines")
INLINE_FUNCTIONS = kw.keyword("inline-functions")
WARN_ON_ARITY_MISMATCH = kw.keyword("warn-on-arity-mismatch")
WARN_ON_SHADOWED_NAME = kw.keyword("warn-on-shadowed-name")
WARN_ON_SHADOWED_VAR = kw.keyword("warn-on-shadowed-var")
WARN_ON_UNUSED_NAMES = kw.keyword("warn-on-unused-names")
WARN_ON_NON_DYNAMIC_SET = kw.keyword("warn-on-non-dynamic-set")

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


@attr.define
class RecurPoint:
    loop_id: str
    args: Collection[Binding] = ()


@attr.frozen
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


@attr.define
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
            if entry.symbol.name.startswith("_"):
                continue
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

    def _as_env_map(self) -> MutableMapping[sym.Symbol, lmap.PersistentMap]:
        locals_ = {} if self._parent is None else self._parent._as_env_map()
        locals_.update({k: v.binding.to_map() for k, v in self._table.items()})
        return locals_

    def as_env_map(self) -> lmap.PersistentMap:
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
        self._func_ctx: collections.deque[FunctionContext] = collections.deque([])
        self._is_quoted: collections.deque[bool] = collections.deque([])
        self._opts = (
            Maybe(opts).map(lmap.map).or_else_get(lmap.EMPTY)  # type: ignore[arg-type, unused-ignore]
        )
        self._recur_points: collections.deque[RecurPoint] = collections.deque([])
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
    def should_generate_auto_inlines(self) -> bool:
        """If True, generate inline function defs for functions with boolean `^:inline`
        meta keys."""
        return self._opts.val_at(GENERATE_AUTO_INLINES, True)

    @property
    def should_inline_functions(self) -> bool:
        """If True, function calls may be inlined if an inline def is provided."""
        return self._opts.val_at(INLINE_FUNCTIONS, True)

    @property
    def warn_on_arity_mismatch(self) -> bool:
        """If True, warn when a Basilisp function invocation is detected with an
        unsupported number of arguments."""
        return self._opts.val_at(WARN_ON_ARITY_MISMATCH, True)

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
    def warn_on_non_dynamic_set(self) -> bool:
        """If True, warn when attempting to set! a Var not marked as ^:dynamic."""
        return self._opts.val_at(WARN_ON_NON_DYNAMIC_SET, True)

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
        func_ctx = self.func_ctx
        return (
            func_ctx is not None
            and func_ctx.function_type == FunctionContextType.ASYNC_FUNCTION
        )

    @contextlib.contextmanager
    def new_func_ctx(
        self, context_type: FunctionContextType
    ) -> Iterator[FunctionContext]:
        """Context manager which can be used to set a function or method context for
        child nodes to examine. A new function context is pushed onto the stack each
        time the Analyzer finds a new function or method definition, so there may be
        many nested function contexts."""
        func_ctx = FunctionContext(context_type)
        self._func_ctx.append(func_ctx)
        yield func_ctx
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
            name,
            is_context_boundary,
            self.warn_on_unused_names,
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

        If a syntax position is given, it will be included in the environment.
        Otherwise, the position will be set to None."""
        return NodeEnv(
            ns=self.current_ns, file=self.filename, pos=pos, func_ctx=self.func_ctx
        )

    def AnalyzerException(
        self,
        msg: str,
        form: Union[LispForm, None, ISeq] = None,
        lisp_ast: Optional[Node] = None,
    ) -> CompilerException:
        """Return a CompilerException annotated with the current filename and
        :analyzer compiler phase set. The remaining keyword arguments are passed
        directly to the constructor."""
        return CompilerException(
            msg,
            phase=CompilerPhase.ANALYZING,
            filename=self.filename,
            form=form,
            lisp_ast=lisp_ast,
        )


####################
# Private Utilities
####################


BoolMetaGetter = Callable[[Union[IMeta, Var]], bool]
MetaGetter = Callable[[Union[IMeta, Var]], Any]


def _bool_meta_getter(meta_kw: kw.Keyword) -> BoolMetaGetter:
    """Return a function which checks an object with metadata for a boolean
    value by meta_kw."""

    def has_meta_prop(o: Union[IMeta, Var]) -> bool:
        return bool(
            Maybe(o.meta).map(lambda m: m.val_at(meta_kw, None)).or_else_get(False)
        )

    return has_meta_prop


def _meta_getter(meta_kw: kw.Keyword) -> MetaGetter:
    """Return a function which checks an object with metadata for a value by
    meta_kw."""

    def get_meta_prop(o: Union[IMeta, Var]) -> Any:
        return Maybe(o.meta).map(lambda m: m.val_at(meta_kw, None)).value

    return get_meta_prop


_is_artificially_abstract = _bool_meta_getter(SYM_ABSTRACT_META_KEY)
_artificially_abstract_members = _meta_getter(SYM_ABSTRACT_MEMBERS_META_KEY)
_is_async = _bool_meta_getter(SYM_ASYNC_META_KEY)
_is_mutable = _bool_meta_getter(SYM_MUTABLE_META_KEY)
_is_py_classmethod = _bool_meta_getter(SYM_CLASSMETHOD_META_KEY)
_is_py_property = _bool_meta_getter(SYM_PROPERTY_META_KEY)
_is_py_staticmethod = _bool_meta_getter(SYM_STATICMETHOD_META_KEY)
_is_macro = _bool_meta_getter(SYM_MACRO_META_KEY)
_is_no_inline = _bool_meta_getter(SYM_NO_INLINE_META_KEY)
_is_allow_var_indirection = _bool_meta_getter(SYM_NO_WARN_ON_VAR_INDIRECTION_META_KEY)
_is_use_var_indirection = _bool_meta_getter(SYM_USE_VAR_INDIRECTION_KEY)
_inline_meta = _meta_getter(SYM_INLINE_META_KW)
_tag_meta = _meta_getter(SYM_TAG_META_KEY)


T_form = TypeVar("T_form", bound=ReaderForm)
T_node = TypeVar("T_node", bound=Node)
LispAnalyzer = Callable[[T_form, AnalyzerContext], T_node]


def _loc(form: T_form) -> Optional[tuple[int, int, int, int]]:
    """Fetch the location of the form in the original filename from the
    input form, if it has metadata."""
    # Technically, IMeta is sufficient for fetching `form.meta` but the
    # reader only applies line and column metadata to IWithMeta instances
    if isinstance(form, IWithMeta):
        meta = form.meta
        if meta is not None:
            line = meta.get(reader.READER_LINE_KW)
            col = meta.get(reader.READER_COL_KW)
            end_line = meta.get(reader.READER_END_LINE_KW)
            end_col = meta.get(reader.READER_END_COL_KW)
            if (
                isinstance(line, int)
                and isinstance(col, int)
                and isinstance(end_line, int)
                and isinstance(end_col, int)
            ):
                return line, col, end_line, end_col
    return None


def _with_loc(f: LispAnalyzer[T_form, T_node]) -> LispAnalyzer[T_form, T_node]:
    """Attach any available location information from the input form to
    the node environment returned from the parsing function."""

    @wraps(f)
    def _analyze_form(form: T_form, ctx: AnalyzerContext) -> T_node:
        form_loc = _loc(form)
        if form_loc is None:
            return f(form, ctx)
        else:
            return cast(T_node, f(form, ctx).fix_missing_locations(form_loc))

    return _analyze_form


def _clean_meta(meta: Optional[lmap.PersistentMap]) -> Optional[lmap.PersistentMap]:
    """Remove reader metadata from the form's meta map."""
    if meta is None:
        return None
    else:
        new_meta = meta.dissoc(
            reader.READER_LINE_KW,
            reader.READER_COL_KW,
            reader.READER_END_LINE_KW,
            reader.READER_END_COL_KW,
        )
        return None if len(new_meta) == 0 else new_meta


def _body_ast(
    form: Union[llist.PersistentList, ISeq], ctx: AnalyzerContext
) -> tuple[Iterable[Node], Node]:
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
            body_stmts = list(map(lambda form: _analyze_form(form, ctx), stmt_forms))

        with ctx.parent_pos():
            body_expr = _analyze_form(ret_form, ctx)

        body = body_stmts + [body_expr]
    else:
        body = []

    if body:
        *stmts, ret = body
    else:
        stmts, ret = [], _const_node(None, ctx)
    return stmts, ret


def _call_args_ast(
    form: ISeq, ctx: AnalyzerContext
) -> tuple[Iterable[Node], KeywordArgs]:
    """Return a tuple of positional arguments and keyword arguments, splitting at the
    keyword argument marker symbol '**'."""
    with ctx.expr_pos():
        nmarkers = sum(int(e == STAR_STAR) for e in form)
        if nmarkers > 1:
            raise ctx.AnalyzerException(
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

            args = vec.vector(map(lambda form: _analyze_form(form, ctx), pos))
            kw_map = {}
            try:
                for k, v in partition(kws, 2):
                    if isinstance(k, kw.Keyword):
                        munged_k = munge(k.name, allow_builtins=True)
                    elif isinstance(k, str):
                        munged_k = munge(k, allow_builtins=True)
                    else:
                        raise ctx.AnalyzerException(
                            f"keys for keyword arguments must be keywords or strings, not '{type(k)}'",
                            form=k,
                        )

                    if munged_k in kw_map:
                        raise ctx.AnalyzerException(
                            "duplicate keyword argument key in function or method invocation",
                            form=k,
                        )

                    kw_map[munged_k] = _analyze_form(v, ctx)

            except ValueError as e:
                raise ctx.AnalyzerException(
                    "keyword arguments must appear in key/value pairs", form=form
                ) from e
            else:
                kwargs = lmap.map(kw_map)
        else:
            args = vec.vector(map(lambda form: _analyze_form(form, ctx), form))
            kwargs = lmap.EMPTY

        return args, kwargs


def _tag_ast(form: Optional[LispForm], ctx: AnalyzerContext) -> Optional[Node]:
    if form is None:
        return None
    return _analyze_form(form, ctx)


def _with_meta(gen_node: LispAnalyzer[T_form, T_node]) -> LispAnalyzer[T_form, T_node]:
    """Wraps the node generated by gen_node in a :with-meta AST node if the
    original form has meta.

    :with-meta AST nodes are used for non-quoted collection literals and for
    function expressions."""

    @wraps(gen_node)
    def with_meta(form: T_form, ctx: AnalyzerContext) -> T_node:
        assert not ctx.is_quoted, "with-meta nodes are not used in quoted expressions"

        descriptor = gen_node(form, ctx)

        if isinstance(form, IMeta):
            assert isinstance(form.meta, (lmap.PersistentMap, type(None)))
            form_meta = _clean_meta(form.meta)
            if form_meta is not None:
                meta_ast = _analyze_form(form_meta, ctx)
                assert isinstance(meta_ast, MapNode) or (
                    isinstance(meta_ast, Const) and meta_ast.type == ConstType.MAP
                )
                return cast(
                    T_node,
                    WithMeta(
                        form=cast(LispForm, form),
                        meta=meta_ast,
                        expr=descriptor,
                        env=ctx.get_node_env(pos=ctx.syntax_position),
                    ),
                )

        return descriptor

    return with_meta


######################
# Analyzer Entrypoint
######################


@functools.singledispatch
def _analyze_form(form: Union[ReaderForm, ISeq], ctx: AnalyzerContext):
    raise ctx.AnalyzerException(f"Unexpected form type {type(form)}", form=form)  # type: ignore[arg-type]


################
# Special Forms
################


def _await_ast(form: ISeq, ctx: AnalyzerContext) -> Await:
    assert form.first == SpecialForm.AWAIT

    if not ctx.is_async_ctx:
        raise ctx.AnalyzerException(
            "await forms may not appear in non-async context", form=form
        )

    nelems = count(form)
    if nelems != 2:
        raise ctx.AnalyzerException(
            "await forms must contain 2 elements, as in: (await expr)", form=form
        )

    with ctx.expr_pos():
        expr = _analyze_form(runtime.nth(form, 1), ctx)

    return Await(
        form=form,
        expr=expr,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def __should_warn_on_redef(
    current_ns: runtime.Namespace,
    defsym: sym.Symbol,
    def_meta: Optional[lmap.PersistentMap],
) -> bool:
    """Return True if the compiler should emit a warning about this name being redefined."""
    if def_meta is not None and def_meta.val_at(SYM_NO_WARN_ON_REDEF_META_KEY, False):
        return False

    if defsym not in current_ns.interns:
        return False

    var = current_ns.find(defsym)
    assert var is not None, f"Var {defsym} cannot be none here"

    if var.meta is not None and var.meta.val_at(SYM_REDEF_META_KEY):
        return False
    else:
        return bool(var.is_bound)


def _def_ast(  # pylint: disable=too-many-locals,too-many-statements
    form: ISeq, ctx: AnalyzerContext
) -> Def:
    assert form.first == SpecialForm.DEF

    nelems = count(form)
    if nelems not in (2, 3, 4):
        raise ctx.AnalyzerException(
            "def forms must have between 2 and 4 elements, as in: (def name docstring? init?)",
            form=form,
        )

    name = runtime.nth(form, 1)
    if not isinstance(name, sym.Symbol):
        raise ctx.AnalyzerException(
            f"def names must be symbols, not {type(name)}", form=name
        )

    tag_ast = _tag_ast(_tag_meta(name), ctx)

    init_idx: Optional[int]
    children: vec.PersistentVector[kw.Keyword]
    if nelems == 2:
        init_idx = None
        doc = None
        children = vec.EMPTY
    elif nelems == 3:
        init_idx = 2
        doc = None
        children = vec.v(INIT)
    else:
        init_idx = 3
        doc = runtime.nth(form, 2)
        if not isinstance(doc, str):
            raise ctx.AnalyzerException("def docstring must be a string", form=doc)
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
    def_loc = _loc(form) or _loc(name) or (None, None, None, None)
    if def_loc == (None, None, None, None):
        logger.warning(f"def line and column metadata not provided for Var {name}")
    if name.meta is None:
        logger.warning(f"def name symbol has no metadata for Var {name}")
        name = name.with_meta(lmap.EMPTY)
    def_node_env = ctx.get_node_env(pos=ctx.syntax_position)
    def_meta = _clean_meta(
        name.meta.update(  # type: ignore [union-attr]
            lmap.map(
                {
                    COL_KW: def_loc[1],
                    END_COL_KW: def_loc[3],
                    FILE_KW: def_node_env.file,
                    LINE_KW: def_loc[0],
                    END_LINE_KW: def_loc[2],
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
    if isinstance(arglists_meta, llist.PersistentList):
        assert arglists_meta.first == SpecialForm.QUOTE
        var_meta = def_meta.update(  # type: ignore
            {ARGLISTS_KW: runtime.nth(arglists_meta, 1)}
        )
    else:
        var_meta = def_meta

    # Generation fails later if we use the same symbol we received, since
    # its meta may contain values which fail to compile.
    bare_name = sym.symbol(name.name)

    # Warn if this symbol is potentially being redefined (if the Var was
    # previously bound)
    if __should_warn_on_redef(current_ns, bare_name, def_meta):
        logger.warning(
            f"redefining Var '{bare_name}' in namespace {current_ns}:{def_loc[0]}"
        )

    ns_sym = sym.symbol(current_ns.name)
    var = Var.intern_unbound(
        ns_sym,
        bare_name,
        dynamic=def_meta.val_at(SYM_DYNAMIC_META_KEY, False),  # type: ignore
        meta=var_meta,
    )

    # Set the value after interning the Var so the symbol is available for
    # recursive definition.
    if init_idx is not None:
        with ctx.expr_pos():
            init = _analyze_form(runtime.nth(form, init_idx), ctx)

        if isinstance(init, Fn):
            # Attach the automatically generated inline function (if one exists) to the
            # Var and def metadata. We do not need to do this for user-provided inline
            # functions (for which `init.inline_fn` will be None) since those should
            # already be attached to the meta.
            if init.inline_fn is not None:
                assert isinstance(var.meta.val_at(SYM_INLINE_META_KW), bool), (  # type: ignore[union-attr]
                    "Cannot have a user-generated inline function and an automatically "
                    "generated inline function"
                )
                var.alter_meta(lambda m: m.assoc(SYM_INLINE_META_KW, init.inline_fn))  # type: ignore[misc]
                def_meta = def_meta.assoc(SYM_INLINE_META_KW, init.inline_fn.form)  # type: ignore[union-attr]

            if tag_ast is not None and any(
                arity.tag is not None for arity in init.arities
            ):
                raise ctx.AnalyzerException(
                    "def'ed Var :tag conflicts with defined function :tag",
                    form=form,
                )
    else:
        init = None

    descriptor = Def(
        form=form,
        name=bare_name,
        var=var,
        init=init,
        doc=doc,
        children=children,
        env=def_node_env,
        tag=tag_ast,
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
        ctx,
    )

    assert (isinstance(meta_ast, Const) and meta_ast.type == ConstType.MAP) or (
        isinstance(meta_ast, MapNode)
    )
    existing_children = cast(vec.PersistentVector, descriptor.children)
    return descriptor.assoc(
        meta=meta_ast, children=vec.vector(runtime.cons(META, existing_children))
    )


def __deftype_method_param_bindings(
    params: vec.PersistentVector, ctx: AnalyzerContext, special_form: sym.Symbol
) -> tuple[bool, int, list[Binding]]:
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
            raise ctx.AnalyzerException(
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
                raise ctx.AnalyzerException(
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
            raise ctx.AnalyzerException(
                "Expected variadic argument name after '&'", form=params
            ) from None

    return has_vargs, fixed_arity, param_nodes


def __deftype_classmethod(  # pylint: disable=too-many-locals
    form: Union[llist.PersistentList, ISeq],
    ctx: AnalyzerContext,
    method_name: str,
    args: vec.PersistentVector,
    kwarg_support: Optional[KeywordArgSupport] = None,
) -> DefTypeClassMethod:
    """Emit a node for a :classmethod member of a `deftype*` form."""
    with (
        ctx.hide_parent_symbol_table(),
        ctx.new_symbol_table(method_name, is_context_boundary=True),
    ):
        try:
            cls_arg = args[0]
        except IndexError as e:
            raise ctx.AnalyzerException(
                "deftype* class method must include 'cls' argument", form=args
            ) from e
        else:
            if not isinstance(cls_arg, sym.Symbol):
                raise ctx.AnalyzerException(
                    "deftype* class method 'cls' argument must be a symbol",
                    form=args,
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
            params, ctx, SpecialForm.DEFTYPE
        )
        with ctx.new_func_ctx(FunctionContextType.CLASSMETHOD), ctx.expr_pos():
            stmts, ret = _body_ast(runtime.nthrest(form, 2), ctx)
            body = Do(
                form=form.rest,
                statements=vec.vector(stmts),
                ret=ret,
                is_body=True,
                env=ctx.get_node_env(),
            )
        method = DefTypeClassMethod(
            form=form,
            name=method_name,
            params=vec.vector(param_nodes),
            fixed_arity=fixed_arity,
            is_variadic=has_vargs,
            kwarg_support=kwarg_support,
            body=body,
            class_local=cls_binding,
            env=ctx.get_node_env(),
        )
        method.visit(partial(_assert_no_recur, ctx))
        return method


def __deftype_or_reify_method(  # pylint: disable=too-many-arguments,too-many-locals
    form: Union[llist.PersistentList, ISeq],
    ctx: AnalyzerContext,
    method_name: str,
    args: vec.PersistentVector,
    special_form: sym.Symbol,
    kwarg_support: Optional[KeywordArgSupport] = None,
) -> DefTypeMethodArity:
    """Emit a node for a method member of a `deftype*` or `reify*` form."""
    assert special_form in {SpecialForm.DEFTYPE, SpecialForm.REIFY}

    with ctx.new_symbol_table(method_name, is_context_boundary=True):
        try:
            this_arg = args[0]
        except IndexError as e:
            raise ctx.AnalyzerException(
                f"{special_form} method must include 'this' or 'self' argument",
                form=args,
            ) from e
        else:
            if not isinstance(this_arg, sym.Symbol):
                raise ctx.AnalyzerException(
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
            params, ctx, special_form
        )

        loop_id = genname(method_name)
        with ctx.new_recur_point(loop_id, param_nodes):
            with ctx.new_func_ctx(FunctionContextType.METHOD), ctx.expr_pos():
                stmts, ret = _body_ast(runtime.nthrest(form, 2), ctx)
                body = Do(
                    form=form.rest,
                    statements=vec.vector(stmts),
                    ret=ret,
                    is_body=True,
                    env=ctx.get_node_env(),
                )
            method = DefTypeMethodArity(
                form=form,
                name=method_name,
                this_local=this_binding,
                params=vec.vector(param_nodes),
                fixed_arity=fixed_arity,
                is_variadic=has_vargs,
                kwarg_support=kwarg_support,
                body=body,
                loop_id=loop_id,
                env=ctx.get_node_env(),
            )
            method.visit(partial(_assert_recur_is_tail, ctx))
            return method


def __deftype_or_reify_property(
    form: Union[llist.PersistentList, ISeq],
    ctx: AnalyzerContext,
    method_name: str,
    args: vec.PersistentVector,
    special_form: sym.Symbol,
) -> DefTypeProperty:
    """Emit a node for a :property member of a `deftype*` or `reify*` form."""
    assert special_form in {SpecialForm.DEFTYPE, SpecialForm.REIFY}

    with ctx.new_symbol_table(method_name, is_context_boundary=True):
        try:
            this_arg = args[0]
        except IndexError as e:
            raise ctx.AnalyzerException(
                f"{special_form} property must include 'this' or 'self' argument",
                form=args,
            ) from e
        else:
            if not isinstance(this_arg, sym.Symbol):
                raise ctx.AnalyzerException(
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
            params, ctx, special_form
        )

        if len(param_nodes) > 0:
            raise ctx.AnalyzerException(
                f"{special_form} properties may not specify arguments", form=form
            )

        assert not has_vargs, f"{special_form} properties may not have arguments"

        with ctx.new_func_ctx(FunctionContextType.PROPERTY), ctx.expr_pos():
            stmts, ret = _body_ast(runtime.nthrest(form, 2), ctx)
            body = Do(
                form=form.rest,
                statements=vec.vector(stmts),
                ret=ret,
                is_body=True,
                env=ctx.get_node_env(),
            )
        prop = DefTypeProperty(
            form=form,
            name=method_name,
            this_local=this_binding,
            params=vec.vector(param_nodes),
            body=body,
            env=ctx.get_node_env(),
        )
        prop.visit(partial(_assert_no_recur, ctx))
        return prop


def __deftype_staticmethod(
    form: Union[llist.PersistentList, ISeq],
    ctx: AnalyzerContext,
    method_name: str,
    args: vec.PersistentVector,
    kwarg_support: Optional[KeywordArgSupport] = None,
) -> DefTypeStaticMethod:
    """Emit a node for a :staticmethod member of a `deftype*` form."""
    with (
        ctx.hide_parent_symbol_table(),
        ctx.new_symbol_table(method_name, is_context_boundary=True),
    ):
        has_vargs, fixed_arity, param_nodes = __deftype_method_param_bindings(
            args, ctx, SpecialForm.DEFTYPE
        )
        with ctx.new_func_ctx(FunctionContextType.STATICMETHOD), ctx.expr_pos():
            stmts, ret = _body_ast(runtime.nthrest(form, 2), ctx)
            body = Do(
                form=form.rest,
                statements=vec.vector(stmts),
                ret=ret,
                is_body=True,
                env=ctx.get_node_env(),
            )
        method = DefTypeStaticMethod(
            form=form,
            name=method_name,
            params=vec.vector(param_nodes),
            fixed_arity=fixed_arity,
            is_variadic=has_vargs,
            kwarg_support=kwarg_support,
            body=body,
            env=ctx.get_node_env(),
        )
        method.visit(partial(_assert_no_recur, ctx))
        return method


def __deftype_or_reify_prop_or_method_arity(
    form: Union[llist.PersistentList, ISeq],
    ctx: AnalyzerContext,
    special_form: sym.Symbol,
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
        raise ctx.AnalyzerException(
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
        raise ctx.AnalyzerException(
            f"{special_form} does not support classmethod or staticmethod members",
            form=form,
        )

    if sum([is_classmethod, is_property, is_staticmethod]) not in {0, 1}:
        raise ctx.AnalyzerException(
            f"{special_form} member may be only one of: :classmethod, :property, "
            "or :staticmethod",
            form=form,
        )

    args = runtime.nth(form, 1)
    if not isinstance(args, vec.PersistentVector):
        raise ctx.AnalyzerException(
            f"{special_form} member arguments must be vector, not {type(args)}",
            form=args,
        )

    kwarg_meta = __fn_kwargs_support(ctx, form.first) or (
        isinstance(form, IMeta) and __fn_kwargs_support(ctx, form)
    )
    kwarg_support = None if isinstance(kwarg_meta, bool) else kwarg_meta

    if is_classmethod:
        return __deftype_classmethod(
            form, ctx, method_name, args, kwarg_support=kwarg_support
        )
    elif is_property:
        if kwarg_support is not None:
            raise ctx.AnalyzerException(
                f"{special_form} properties may not declare keyword argument support",
                form=form,
            )

        return __deftype_or_reify_property(form, ctx, method_name, args, special_form)
    elif is_staticmethod:
        return __deftype_staticmethod(
            form, ctx, method_name, args, kwarg_support=kwarg_support
        )
    else:
        return __deftype_or_reify_method(
            form, ctx, method_name, args, special_form, kwarg_support=kwarg_support
        )


def __deftype_or_reify_method_node_from_arities(
    form: Union[llist.PersistentList, ISeq],
    ctx: AnalyzerContext,
    arities: list[DefTypeMethodArity],
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
                raise ctx.AnalyzerException(
                    f"{special_form} method may have at most 1 variadic arity",
                    form=arity.form,
                )
            fixed_arity_for_variadic = arity.fixed_arity
            num_variadic += 1
        else:
            if arity.fixed_arity in fixed_arities:
                raise ctx.AnalyzerException(
                    f"{special_form} may not have multiple methods with the same "
                    "fixed arity",
                    form=arity.form,
                )
            fixed_arities.add(arity.fixed_arity)

    if fixed_arity_for_variadic is not None and any(
        fixed_arity_for_variadic < arity for arity in fixed_arities
    ):
        raise ctx.AnalyzerException(
            "variadic arity may not have fewer fixed arity arguments than any other arities",
            form=form,
        )

    assert (
        len({arity.name for arity in arities}) <= 1
    ), "arities must have the same name defined"

    if len(arities) > 1 and any(arity.kwarg_support is not None for arity in arities):
        raise ctx.AnalyzerException(
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


def __deftype_or_reify_impls(  # pylint: disable=too-many-branches,too-many-locals
    form: ISeq, ctx: AnalyzerContext, special_form: sym.Symbol
) -> tuple[list[DefTypeBase], list[DefTypeMember]]:
    """Roll up `deftype*` and `reify*` declared bases and method implementations."""
    assert special_form in {SpecialForm.DEFTYPE, SpecialForm.REIFY}

    if runtime.to_seq(form) is None:
        return [], []

    if not isinstance(form.first, kw.Keyword) or form.first != IMPLEMENTS:
        raise ctx.AnalyzerException(
            f"{special_form} forms must declare which interfaces they implement",
            form=form,
        )

    implements = runtime.nth(form, 1)
    if not isinstance(implements, vec.PersistentVector):
        raise ctx.AnalyzerException(
            f"{special_form} interfaces must be declared as "
            ":implements [Interface1 Interface2 ...]",
            form=implements,
        )

    interface_names: MutableSet[sym.Symbol] = set()
    interfaces = []
    for iface in implements:
        if not isinstance(iface, sym.Symbol):
            raise ctx.AnalyzerException(
                f"{special_form} interfaces must be symbols", form=iface
            )

        if iface in interface_names:
            raise ctx.AnalyzerException(
                f"{special_form} interfaces may only appear once in :implements vector",
                form=iface,
            )
        interface_names.add(iface)

        current_interface = _analyze_form(iface, ctx)
        if not isinstance(current_interface, (MaybeClass, MaybeHostForm, VarRef)):
            raise ctx.AnalyzerException(
                f"{special_form} interface implementation must be an existing interface",
                form=iface,
            )
        interfaces.append(current_interface)

    # Use the insertion-order preserving capabilities of a dictionary with 'True'
    # keys to act as an ordered set of members we've seen. We don't want to register
    # duplicates.
    member_order = {}
    methods: MutableMapping[str, list[DefTypeMethodArity]] = collections.defaultdict(
        list
    )
    py_members: MutableMapping[str, DefTypePythonMember] = {}
    for elem in runtime.nthrest(form, 2):
        if not isinstance(elem, ISeq):
            raise ctx.AnalyzerException(
                f"{special_form} must consist of interface or protocol names and methods",
                form=elem,
            )

        member = __deftype_or_reify_prop_or_method_arity(elem, ctx, special_form)
        member_order[member.name] = True
        if isinstance(
            member, (DefTypeClassMethod, DefTypeProperty, DefTypeStaticMethod)
        ):
            if member.name in py_members:
                raise ctx.AnalyzerException(
                    f"{special_form} class methods, properties, and static methods "
                    "may only have one arity defined",
                    form=elem,
                    lisp_ast=member,
                )
            elif member.name in methods:
                raise ctx.AnalyzerException(
                    f"{special_form} class method, property, or static method name "
                    "already defined as a method",
                    form=elem,
                    lisp_ast=member,
                )
            py_members[member.name] = member
        else:
            if member.name in py_members:
                raise ctx.AnalyzerException(
                    f"{special_form} method name already defined as a class method, "
                    "property, or static method",
                    form=elem,
                    lisp_ast=member,
                )
            methods[member.name].append(member)

    members: list[DefTypeMember] = []
    for member_name in member_order:
        arities = methods.get(member_name)
        if arities is not None:
            members.append(
                __deftype_or_reify_method_node_from_arities(
                    form, ctx, arities, special_form
                )
            )
            continue

        py_member = py_members.get(member_name)
        assert py_member is not None, "Member must be a method or property"
        members.append(py_member)

    return interfaces, members


_var_is_protocol = _bool_meta_getter(VAR_IS_PROTOCOL_META_KEY)


def __is_deftype_member(mem) -> bool:
    """Return True if `mem` names a valid `deftype*` member."""
    return inspect.isroutine(mem) or isinstance(mem, (property, staticmethod))


def __is_reify_member(mem) -> bool:
    """Return True if `mem` names a valid `reify*` member."""
    return inspect.isroutine(mem) or isinstance(mem, property)


if platform.python_implementation() == "CPython":

    def __is_type_weakref(tp: type) -> bool:
        return getattr(tp, "__weakrefoffset__", 0) > 0

else:

    def __is_type_weakref(tp: type) -> bool:  # pylint: disable=unused-argument
        return True


def __get_artificially_abstract_members(
    ctx: AnalyzerContext, special_form: sym.Symbol, interface: DefTypeBase
) -> set[str]:
    if (
        declared_abstract_members := _artificially_abstract_members(
            cast(IMeta, interface.form)
        )
    ) is None:
        return set()

    if (
        not isinstance(declared_abstract_members, lset.PersistentSet)
        or len(declared_abstract_members) == 0
    ):
        raise ctx.AnalyzerException(
            f"{special_form} artificially abstract members must be a set of keywords",
            form=interface.form,
        )

    members = set()
    for mem in declared_abstract_members:
        if isinstance(mem, INamed):
            if mem.ns is not None:
                logger.warning(
                    "Unexpected namespace for artificially abstract member to "
                    f"{special_form}: {mem}"
                )
            members.add(mem.name)
        elif isinstance(mem, str):
            members.add(mem)
        else:
            raise ctx.AnalyzerException(
                f"{special_form} artificially abstract member names must be one of: "
                f"str, keyword, or symbol; got {type(mem)}",
                form=interface.form,
            )
    return members


@attr.define
class _TypeAbstractness:
    """
    :ivar is_statically_verified_as_abstract: a boolean value which, if True,
        indicates that all bases have been statically verified abstract; if False,
        indicates at least one base could not be statically verified
    :ivar artificially_abstract_supertypes: the set of all super-types which have
        been marked as artificially abstract
    :ivar supertype_already_weakref: if True, a supertype is already marked as
        weakref and therefore the resulting type cannot add "__weakref__" to the
        slots list to enable weakref support
    """

    is_statically_verified_as_abstract: bool
    artificially_abstract_supertypes: lset.PersistentSet[DefTypeBase]
    supertype_already_weakref: bool


def __deftype_and_reify_impls_are_all_abstract(  # pylint: disable=too-many-locals,too-many-statements
    ctx: AnalyzerContext,
    special_form: sym.Symbol,
    fields: Iterable[str],
    interfaces: Iterable[DefTypeBase],
    members: Iterable[DefTypeMember],
) -> _TypeAbstractness:
    """Return an object indicating the abstractness of the `deftype*` or `reify*`
    super-types.

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

    supertype_possibly_weakref = []
    unverifiably_abstract = set()
    artificially_abstract: set[DefTypeBase] = set()
    artificially_abstract_base_members: set[str] = set()
    is_member = {
        SpecialForm.DEFTYPE: __is_deftype_member,
        SpecialForm.REIFY: __is_reify_member,
    }[special_form]

    field_names = frozenset(fields)
    member_names = frozenset(deftype_or_reify_python_member_names(members))
    all_member_names = field_names.union(member_names)
    all_interface_methods: set[str] = set()
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
                if isinstance(interface.form, IMeta) and _is_artificially_abstract(
                    interface.form
                ):
                    artificially_abstract.add(interface)
                continue

            # Protocols are defined as maps, with the interface being simply a member
            # of the map, denoted by the keyword `:interface`.
            if _var_is_protocol(interface.var):
                proto_map = interface.var.value
                assert isinstance(proto_map, lmap.PersistentMap)
                interface_type = proto_map.val_at(INTERFACE)
            else:
                interface_type = interface.var.value

        if interface_type is object:
            continue

        if isinstance(interface.form, IMeta) and _is_artificially_abstract(
            interface.form
        ):
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
            # The meta key :abstract-members will give users the escape hatch to force
            # the compiler to recognize those members as abstract members.
            #
            # When dealing with artificially abstract members, it may be the case
            # that members of superclasses of a specific type don't actually declare
            # abstract member methods expected by subclasses. One particular (major)
            # instance is that of `io.IOBase` which does not declare "read" or "write"
            # as abstract members but whose documentation declares both methods should
            # be considered part of the interface.
            artificially_abstract_base_members.update(
                __get_artificially_abstract_members(ctx, special_form, interface)
            )
            supertype_possibly_weakref.append(__is_type_weakref(interface_type))
        elif is_abstract(interface_type):
            interface_names: frozenset[str] = interface_type.__abstractmethods__
            interface_property_names: frozenset[str] = frozenset(
                method
                for method in interface_names
                if isinstance(getattr(interface_type, method), property)
            )
            interface_method_names = interface_names - interface_property_names
            if not interface_method_names.issubset(member_names):
                missing_methods = ", ".join(interface_method_names - member_names)
                raise ctx.AnalyzerException(
                    f"{special_form} definition missing interface members for "
                    f"interface {interface.form}: {missing_methods}",
                    form=interface.form,
                    lisp_ast=interface,
                )
            elif not interface_property_names.issubset(all_member_names):
                missing_fields = ", ".join(interface_property_names - field_names)
                raise ctx.AnalyzerException(
                    f"{special_form} definition missing interface properties for "
                    f"interface {interface.form}: {missing_fields}",
                    form=interface.form,
                    lisp_ast=interface,
                )

            all_interface_methods.update(interface_names)
            supertype_possibly_weakref.append(__is_type_weakref(interface_type))
        else:
            raise ctx.AnalyzerException(
                f"{special_form} interface must be Python abstract class or object",
                form=interface.form,
                lisp_ast=interface,
            )

    # We cannot compute if there are extra methods defined if there are any
    # unverifiably abstract bases, so we just skip this check.
    if unverifiably_abstract:
        logger.warning(
            f"Unable to verify abstractness for {special_form} supertype(s): "
            f"{', '.join(str(e.var) for e in unverifiably_abstract)}"
        )
    else:
        extra_methods = member_names - all_interface_methods - OBJECT_DUNDER_METHODS
        if extra_methods and not extra_methods.issubset(
            artificially_abstract_base_members
        ):
            expected_methods = (
                all_interface_methods | artificially_abstract_base_members
            ) - OBJECT_DUNDER_METHODS
            expected_method_str = ", ".join(expected_methods)
            extra_method_str = ", ".join(extra_methods)
            raise ctx.AnalyzerException(
                f"{special_form} definition for interface includes members not "
                f"part of defined interfaces: {extra_method_str}; expected one of: "
                f"{expected_method_str}"
            )

    return _TypeAbstractness(
        is_statically_verified_as_abstract=not unverifiably_abstract,
        artificially_abstract_supertypes=lset.set(artificially_abstract),
        supertype_already_weakref=any(supertype_possibly_weakref),
    )


__DEFTYPE_DEFAULT_SENTINEL = object()


def _deftype_ast(  # pylint: disable=too-many-locals
    form: ISeq, ctx: AnalyzerContext
) -> DefType:
    assert form.first == SpecialForm.DEFTYPE

    nelems = count(form)
    if nelems < 3:
        raise ctx.AnalyzerException(
            "deftype forms must have 3 or more elements, as in: "
            "(deftype* name fields :implements [bases+impls])",
            form=form,
        )

    name = runtime.nth(form, 1)
    if not isinstance(name, sym.Symbol):
        raise ctx.AnalyzerException(
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
    if not isinstance(fields, vec.PersistentVector):
        raise ctx.AnalyzerException(
            f"deftype* fields must be vector, not {type(fields)}", form=fields
        )

    has_defaults = False
    with ctx.new_symbol_table(name.name):
        is_frozen = True
        param_nodes = []
        for field in fields:
            if not isinstance(field, sym.Symbol):
                raise ctx.AnalyzerException(
                    "deftype* fields must be symbols", form=field
                )

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
                raise ctx.AnalyzerException(
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
                tag=_tag_ast(_tag_meta(field), ctx),
                init=(
                    analyze_form(ctx, field_default)
                    if field_default is not __DEFTYPE_DEFAULT_SENTINEL
                    else None
                ),
            )
            param_nodes.append(binding)
            ctx.put_new_symbol(field, binding, warn_if_unused=False)

        interfaces, members = __deftype_or_reify_impls(
            runtime.nthrest(form, 3), ctx, SpecialForm.DEFTYPE
        )
        type_abstractness = __deftype_and_reify_impls_are_all_abstract(
            ctx, SpecialForm.DEFTYPE, map(lambda f: f.name, fields), interfaces, members
        )
        return DefType(
            form=form,
            name=name.name,
            interfaces=vec.vector(interfaces),
            fields=vec.vector(param_nodes),
            members=vec.vector(members),
            verified_abstract=type_abstractness.is_statically_verified_as_abstract,
            artificially_abstract=type_abstractness.artificially_abstract_supertypes,
            is_frozen=is_frozen,
            use_weakref_slot=not type_abstractness.supertype_already_weakref,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )


def _do_ast(form: ISeq, ctx: AnalyzerContext) -> Do:
    assert form.first == SpecialForm.DO
    statements, ret = _body_ast(form.rest, ctx)
    return Do(
        form=form,
        statements=vec.vector(statements),
        ret=ret,
        use_var_indirection=_is_use_var_indirection(form.first),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def __fn_method_ast(  # pylint: disable=too-many-locals
    form: ISeq,
    ctx: AnalyzerContext,
    fnname: Optional[sym.Symbol] = None,
    is_async: bool = False,
) -> FnArity:
    with ctx.new_symbol_table("fn-method", is_context_boundary=True):
        params = form.first
        if not isinstance(params, vec.PersistentVector):
            raise ctx.AnalyzerException(
                "function arity arguments must be a vector", form=params
            )
        return_tag = _tag_ast(_tag_meta(params), ctx)

        has_vargs, vargs_idx = False, 0
        param_nodes = []
        for i, s in enumerate(params):
            if not isinstance(s, sym.Symbol):
                raise ctx.AnalyzerException(
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
                tag=_tag_ast(_tag_meta(s), ctx),
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
                    raise ctx.AnalyzerException(
                        "function rest parameter name must be a symbol", form=vargs_sym
                    )

                binding = Binding(
                    form=vargs_sym,
                    name=vargs_sym.name,
                    local=LocalType.ARG,
                    tag=_tag_ast(_tag_meta(vargs_sym), ctx),
                    arg_id=vargs_idx + 1,
                    is_variadic=True,
                    env=ctx.get_node_env(),
                )
                param_nodes.append(binding)
                ctx.put_new_symbol(vargs_sym, binding)
            except IndexError:
                raise ctx.AnalyzerException(
                    "Expected variadic argument name after '&'", form=params
                ) from None

        fn_loop_id = genname("fn_arity" if fnname is None else fnname.name)
        with ctx.new_recur_point(fn_loop_id, param_nodes):
            with (
                ctx.new_func_ctx(
                    FunctionContextType.ASYNC_FUNCTION
                    if is_async
                    else FunctionContextType.FUNCTION
                ),
                ctx.expr_pos(),
            ):
                stmts, ret = _body_ast(form.rest, ctx)
                body = Do(
                    form=form.rest,
                    statements=vec.vector(stmts),
                    ret=ret,
                    is_body=True,
                    env=ctx.get_node_env(),
                )
            method = FnArity(
                form=form,
                loop_id=fn_loop_id,
                params=vec.vector(param_nodes),
                tag=return_tag,
                is_variadic=has_vargs,
                fixed_arity=len(param_nodes) - int(has_vargs),
                body=body,
                env=ctx.get_node_env(),
            )
            method.visit(partial(_assert_recur_is_tail, ctx))
            return method


def __fn_kwargs_support(ctx: AnalyzerContext, o: IMeta) -> Optional[KeywordArgSupport]:
    if o.meta is None:
        return None

    kwarg_support = o.meta.val_at(SYM_KWARGS_META_KEY)
    if kwarg_support is None:
        return None

    try:
        return KeywordArgSupport(kwarg_support)
    except ValueError as e:
        raise ctx.AnalyzerException(
            "fn keyword argument support metadata :kwarg must be one of: #{:apply :collect}",
            form=kwarg_support,
        ) from e


InlineMeta = Union[Callable, bool, None]


@functools.singledispatch
def __unquote_args(f: LispForm, _: frozenset[sym.Symbol]):
    return f


@__unquote_args.register(sym.Symbol)
def __unquote_args_sym(f: sym.Symbol, args: frozenset[sym.Symbol]):
    if f in args:
        return llist.l(reader._UNQUOTE, f)
    return f


def _inline_fn_ast(
    ctx: AnalyzerContext,
    form: Union[llist.PersistentList, ISeq],
    name: Optional[Binding],
    arities: vec.PersistentVector[FnArity],
    num_variadic: int,
) -> Optional[Fn]:
    if not ctx.should_generate_auto_inlines:
        return None

    inline_arity = arities[0]

    if num_variadic != 0:
        raise ctx.AnalyzerException(
            "functions with variadic arity may not be inlined",
            form=form,
        )

    if len(arities) != 1:
        raise ctx.AnalyzerException(
            "multi-arity functions cannot be inlined",
            form=form,
        )

    if len(inline_arity.body.statements) > 0:
        raise ctx.AnalyzerException(
            "cannot generate an inline function for functions with more than one "
            "body expression",
            form=form,
        )

    logger.log(
        TRACE, f"Generating inline def for {name.name if name is not None else 'fn'}"
    )
    unquoted_form = reader._postwalk(
        lambda f: __unquote_args(
            f,
            frozenset(binding.form for binding in inline_arity.params),
        ),
        inline_arity.body.ret.form,
    )
    macroed_form = reader.syntax_quote(unquoted_form)
    inline_fn_form = llist.l(
        SpecialForm.FN,
        *([sym.symbol(genname(f"{name.name}-inline"))] if name is not None else []),
        vec.vector(binding.form for binding in inline_arity.params),
        macroed_form,
        meta=lmap.map({SYM_GEN_SAFE_PYTHON_PARAM_NAMES_META_KEY: True}),
    )
    return _fn_ast(inline_fn_form, ctx)


@_with_meta
def _fn_ast(  # pylint: disable=too-many-locals,too-many-statements
    form: Union[llist.PersistentList, ISeq], ctx: AnalyzerContext
) -> Fn:
    assert form.first == SpecialForm.FN

    idx = 1

    with ctx.new_symbol_table("fn", is_context_boundary=True):
        try:
            name = runtime.nth(form, idx)
        except IndexError as e:
            raise ctx.AnalyzerException(
                "fn form must match: (fn* name? [arg*] body*) or (fn* name? method*)",
                form=form,
            ) from e

        name_node: Optional[Binding]
        inline: InlineMeta
        if isinstance(name, sym.Symbol):
            name_node = Binding(
                form=name, name=name.name, local=LocalType.FN, env=ctx.get_node_env()
            )
            is_async = _is_async(name) or isinstance(form, IMeta) and _is_async(form)
            inline = (
                _inline_meta(name) or isinstance(form, IMeta) and _inline_meta(form)
            )
            kwarg_support = (
                __fn_kwargs_support(ctx, name)
                or isinstance(form, IMeta)
                and __fn_kwargs_support(ctx, form)
            )
            ctx.put_new_symbol(name, name_node, warn_if_unused=False)
            idx += 1
        elif isinstance(name, (llist.PersistentList, vec.PersistentVector)):
            name = None
            name_node = None
            is_async = isinstance(form, IMeta) and _is_async(form)
            inline = isinstance(form, IMeta) and _inline_meta(form)
            kwarg_support = isinstance(form, IMeta) and __fn_kwargs_support(ctx, form)
        else:
            raise ctx.AnalyzerException(
                "fn form must match: (fn* name? [arg*] body*) or (fn* name? method*)",
                form=form,
            )

        try:
            arity_or_args = runtime.nth(form, idx)
        except IndexError as e:
            raise ctx.AnalyzerException(
                "fn form expects either multiple arities or a vector of arguments",
                form=form,
            ) from e

        if isinstance(arity_or_args, llist.PersistentList):
            arities = vec.vector(
                map(
                    lambda form: __fn_method_ast(
                        form, ctx, fnname=name, is_async=is_async
                    ),
                    runtime.nthrest(form, idx),
                )
            )
        elif isinstance(arity_or_args, vec.PersistentVector):
            arities = vec.v(
                __fn_method_ast(
                    runtime.nthrest(form, idx), ctx, fnname=name, is_async=is_async
                )
            )
        else:
            raise ctx.AnalyzerException(
                "fn form must match: (fn* name? [arg*] body*) or (fn* name? method*)",
                form=form,
            )

        nmethods = count(arities)
        assert nmethods > 0, "fn must have at least one arity"

        if kwarg_support is not None and nmethods > 1:
            raise ctx.AnalyzerException(
                "multi-arity functions may not declare support for keyword arguments",
                form=form,
            )

        fixed_arities: MutableSet[int] = set()
        fixed_arity_for_variadic: Optional[int] = None
        num_variadic = 0
        for arity in arities:
            if arity.is_variadic:
                if num_variadic > 0:
                    raise ctx.AnalyzerException(
                        "fn may have at most 1 variadic arity", form=arity.form
                    )
                fixed_arity_for_variadic = arity.fixed_arity
                num_variadic += 1
            else:
                if arity.fixed_arity in fixed_arities:
                    raise ctx.AnalyzerException(
                        "fn may not have multiple methods with the same fixed arity",
                        form=arity.form,
                    )
                fixed_arities.add(arity.fixed_arity)

        if fixed_arity_for_variadic is not None and any(
            fixed_arity_for_variadic < arity for arity in fixed_arities
        ):
            raise ctx.AnalyzerException(
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
            # Generate an inlineable (macro) variant of the current function if:
            #  - the metadata for the fn contains a boolean `:inline` meta key
            #  - the function has a single arity, composed of a single expression
            #
            # This node attribute (inline_fn) is not used for holding user provided
            # inline functions. If a user provides an inline function, then that
            # function will be compiled as part of the metadata already and will be
            # found by the compiler on the associated Var without needing this code
            # path at all.
            inline_fn=(
                _inline_fn_ast(ctx, form, name_node, arities, num_variadic)
                if isinstance(inline, bool)
                else None
            ),
        )


def _host_call_ast(form: ISeq, ctx: AnalyzerContext) -> HostCall:
    assert isinstance(form.first, sym.Symbol)

    method = form.first
    assert isinstance(method, sym.Symbol), "host interop field must be a symbol"
    assert method.name.startswith(".")

    if not count(form) >= 2:
        raise ctx.AnalyzerException(
            "host interop calls must be 2 or more elements long", form=form
        )

    args, kwargs = _call_args_ast(runtime.nthrest(form, 2), ctx)
    return HostCall(
        form=form,
        method=method.name[1:],
        target=_analyze_form(runtime.nth(form, 1), ctx),
        args=args,
        kwargs=kwargs,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _host_prop_ast(form: ISeq, ctx: AnalyzerContext) -> HostField:
    assert isinstance(form.first, sym.Symbol)

    field = form.first
    assert isinstance(field, sym.Symbol), "host interop field must be a symbol"

    nelems = count(form)
    assert field.name.startswith(".-")

    if field.name == ".-":
        try:
            field = runtime.nth(form, 2)
        except IndexError as e:
            raise ctx.AnalyzerException(
                "host interop prop must be exactly 3 elems long: (.- target field)",
                form=form,
            ) from e
        else:
            if not isinstance(field, sym.Symbol):
                raise ctx.AnalyzerException(
                    "host interop field must be a symbol", form=form
                )

        if not nelems == 3:
            raise ctx.AnalyzerException(
                "host interop prop must be exactly 3 elems long: (.- target field)",
                form=form,
            )

        return HostField(
            form=form,
            field=field.name,
            target=_analyze_form(runtime.nth(form, 1), ctx),
            is_assignable=True,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )
    else:
        if not nelems == 2:
            raise ctx.AnalyzerException(
                "host interop prop must be exactly 2 elements long: (.-field target)",
                form=form,
            )

        return HostField(
            form=form,
            field=field.name[2:],
            target=_analyze_form(runtime.nth(form, 1), ctx),
            is_assignable=True,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )


def _host_interop_ast(form: ISeq, ctx: AnalyzerContext) -> Union[HostCall, HostField]:
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
                raise ctx.AnalyzerException(
                    "host field accesses must be exactly 3 elements long", form=form
                )

            return HostField(
                form=form,
                field=maybe_m_or_f.name[1:],
                target=_analyze_form(runtime.nth(form, 1), ctx),
                is_assignable=True,
                env=ctx.get_node_env(pos=ctx.syntax_position),
            )
        else:
            args, kwargs = _call_args_ast(runtime.nthrest(form, 3), ctx)
            return HostCall(
                form=form,
                method=maybe_m_or_f.name,
                target=_analyze_form(runtime.nth(form, 1), ctx),
                args=args,
                kwargs=kwargs,
                env=ctx.get_node_env(pos=ctx.syntax_position),
            )
    elif isinstance(maybe_m_or_f, (llist.PersistentList, ISeq)):
        # Likewise, I emit :host-call for forms like (. target (method arg1 ...)).
        method = maybe_m_or_f.first
        if not isinstance(method, sym.Symbol):
            raise ctx.AnalyzerException(
                "host call method must be a symbol", form=method
            )

        args, kwargs = _call_args_ast(maybe_m_or_f.rest, ctx)
        return HostCall(
            form=form,
            method=method.name[1:] if method.name.startswith("-") else method.name,
            target=_analyze_form(runtime.nth(form, 1), ctx),
            args=args,
            kwargs=kwargs,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )
    else:
        raise ctx.AnalyzerException(
            "host interop forms must take the form: "
            "(. instance (method args*)), "
            "(. instance method args*), "
            "(. instance -field), ",
            form=form,
        )


def _if_ast(form: ISeq, ctx: AnalyzerContext) -> If:
    assert form.first == SpecialForm.IF

    nelems = count(form)
    if nelems not in (3, 4):
        raise ctx.AnalyzerException(
            "if forms must have either 3 or 4 elements, as in: (if test then else?)",
            form=form,
        )

    with ctx.expr_pos():
        test_node = _analyze_form(runtime.nth(form, 1), ctx)

    with ctx.parent_pos():
        then_node = _analyze_form(runtime.nth(form, 2), ctx)

        if nelems == 4:
            else_node = _analyze_form(runtime.nth(form, 3), ctx)
        else:
            else_node = _const_node(None, ctx)

    return If(
        form=form,
        test=test_node,
        then=then_node,
        else_=else_node,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


T_alias_node = TypeVar("T_alias_node", ImportAlias, RequireAlias)


def _do_warn_on_import_or_require_name_clash(
    ctx: AnalyzerContext,
    alias_nodes: list[T_alias_node],
    action: Literal["import", "require"],
) -> None:
    assert alias_nodes, "Must have at least one alias"

    # Fetch these locally to avoid triggering more locks than we need to
    current_ns = ctx.current_ns
    aliases, import_aliases, imports = (
        current_ns.aliases,
        current_ns.import_aliases,
        current_ns.imports,
    )

    # Identify duplicates in the import list first
    name_to_nodes = defaultdict(list)
    for node in alias_nodes:
        name_to_nodes[(node.alias or node.name)].append(node)

    for name, nodes in name_to_nodes.items():
        if len(nodes) < 2:
            continue

        logger.warning(f"duplicate name or alias '{name}' in {action}")

    # Now check against names in the namespace
    for name, nodes in name_to_nodes.items():
        name_sym = sym.symbol(name)
        node, *_ = nodes

        if name_sym in aliases:
            logger.warning(
                f"name '{name}' may shadow an existing alias in '{current_ns}'"
            )
        if name_sym in import_aliases:
            logger.warning(
                f"name '{name}' may be shadowed by an existing import alias in "
                f"'{current_ns}'"
            )
        if name_sym in imports:
            logger.warning(
                f"name '{name}' may be shadowed by an existing import in '{current_ns}'"
            )


def _import_ast(form: ISeq, ctx: AnalyzerContext) -> Import:
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
        elif isinstance(f, vec.PersistentVector):
            if len(f) != 3:
                raise ctx.AnalyzerException(
                    "import alias must take the form: [module :as alias]", form=f
                )
            module_name = f.val_at(0)  # type: ignore[assignment]
            if not isinstance(module_name, sym.Symbol):
                raise ctx.AnalyzerException(
                    "Python module name must be a symbol", form=f
                )
            if not AS == f.val_at(1):
                raise ctx.AnalyzerException(
                    "expected :as alias for Python import", form=f
                )
            module_alias_sym = f.val_at(2)
            if not isinstance(module_alias_sym, sym.Symbol):
                raise ctx.AnalyzerException(
                    "Python module alias must be a symbol", form=f
                )
            module_alias = module_alias_sym.name
            if "." in module_alias:
                raise ctx.AnalyzerException(
                    "Python module alias must not contain '.'", form=f
                )

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
            raise ctx.AnalyzerException("symbol or vector expected for import*", form=f)

        aliases.append(
            ImportAlias(
                form=f,
                name=module_name.name,
                alias=module_alias,
                env=ctx.get_node_env(),
            )
        )

    if not aliases:
        raise ctx.AnalyzerException(
            "import forms must name at least one module", form=form
        )

    _do_warn_on_import_or_require_name_clash(ctx, aliases, "import")
    return Import(
        form=form,
        aliases=aliases,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def __handle_macroexpanded_ast(
    original: Union[llist.PersistentList, ISeq],
    expanded: Union[ReaderForm, ISeq],
    ctx: AnalyzerContext,
) -> Node:
    """Prepare the Lisp AST from macroexpanded and inlined code."""
    if isinstance(expanded, IWithMeta) and isinstance(original, IMeta):
        old_meta = expanded.meta
        expanded = expanded.with_meta(
            old_meta.cons(original.meta) if old_meta else original.meta
        )
    with ctx.expr_pos():
        expanded_ast = _analyze_form(expanded, ctx)

    # Verify that macroexpanded code also does not have any non-tail
    # recur forms
    if ctx.recur_point is not None:
        _assert_recur_is_tail(ctx, expanded_ast)

    return expanded_ast.assoc(
        raw_forms=cast(vec.PersistentVector, expanded_ast.raw_forms).cons(original)
    )


def _do_warn_on_arity_mismatch(
    fn: VarRef, form: Union[llist.PersistentList, ISeq], ctx: AnalyzerContext
) -> None:
    if ctx.warn_on_arity_mismatch and getattr(fn.var.value, "_basilisp_fn", False):
        arities: Optional[tuple[Union[int, kw.Keyword]]] = getattr(
            fn.var.value, "arities", None
        )
        if arities is not None:
            has_variadic = REST_KW in arities
            fixed_arities = set(filter(lambda v: v != REST_KW, arities))
            max_fixed_arity = max(fixed_arities) if fixed_arities else None
            # This count could be off by 1 for cases where kwargs are being passed,
            # but only Basilisp functions intended to be called by Python code
            # (e.g. with a :kwargs strategy) should ever be called with kwargs,
            # so this seems unlikely enough.
            num_args = runtime.count(form.rest)
            if has_variadic and (max_fixed_arity is None or num_args > max_fixed_arity):
                return
            if num_args not in fixed_arities:
                report_arities = cast(set[Union[int, str]], set(fixed_arities))
                if has_variadic:
                    report_arities.discard(cast(int, max_fixed_arity))
                    report_arities.add(f"{max_fixed_arity}+")
                loc = (
                    f" ({fn.env.file}:{fn.env.line})"
                    if fn.env.line is not None
                    else f" ({fn.env.file})"
                )
                logger.warning(
                    f"calling function {fn.var}{loc} with {num_args} arguments; "
                    f"expected any of: {', '.join(sorted(map(str, report_arities)))}",
                )


def _invoke_ast(form: Union[llist.PersistentList, ISeq], ctx: AnalyzerContext) -> Node:
    with ctx.expr_pos():
        fn = _analyze_form(form.first, ctx)

    if fn.op == NodeOp.VAR and isinstance(fn, VarRef):
        if _is_macro(fn.var) and ctx.should_macroexpand:
            try:
                macro_env = ctx.symbol_table.as_env_map()
                expanded = fn.var.value(macro_env, form, *form.rest)
                return __handle_macroexpanded_ast(form, expanded, ctx)
            except Exception as e:
                if isinstance(e, CompilerException) and (  # pylint: disable=no-member
                    e.phase in {CompilerPhase.MACROEXPANSION, CompilerPhase.INLINING}
                ):
                    # Do not chain macroexpansion exceptions since they don't
                    # actually add anything of value over the cause exception
                    raise
                raise CompilerException(
                    "error occurred during macroexpansion",
                    filename=ctx.filename,
                    form=form,
                    phase=CompilerPhase.MACROEXPANSION,
                ) from e
        elif (
            ctx.should_inline_functions
            and not (isinstance(form, IMeta) and _is_no_inline(form))
            and fn.var.meta
            and callable(fn.var.meta.get(SYM_INLINE_META_KW))
        ):
            # TODO: also consider whether or not the function(s) inside will be valid
            #       if they are inlined (e.g. if the namespace or module is imported)
            inline_fn = cast(Callable, fn.var.meta.get(SYM_INLINE_META_KW))
            try:
                expanded = inline_fn(*form.rest)
                return __handle_macroexpanded_ast(form, expanded, ctx)
            except Exception as e:
                if isinstance(e, CompilerException) and (  # pylint: disable=no-member
                    e.phase == CompilerPhase.INLINING
                ):
                    raise
                raise CompilerException(
                    "error occurred during inlining",
                    filename=ctx.filename,
                    form=form,
                    phase=CompilerPhase.INLINING,
                ) from e

        _do_warn_on_arity_mismatch(fn, form, ctx)

    args, kwargs = _call_args_ast(form.rest, ctx)
    return Invoke(
        form=form,
        fn=fn,
        args=args,
        kwargs=kwargs,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _let_ast(form: ISeq, ctx: AnalyzerContext) -> Let:
    assert form.first == SpecialForm.LET
    nelems = count(form)

    if nelems < 2:
        raise ctx.AnalyzerException(
            "let forms must have bindings vector and 0 or more body forms", form=form
        )

    bindings = runtime.nth(form, 1)
    if not isinstance(bindings, vec.PersistentVector):
        raise ctx.AnalyzerException("let bindings must be a vector", form=bindings)
    elif len(bindings) % 2 != 0:
        raise ctx.AnalyzerException(
            "let bindings must appear in name-value pairs", form=bindings
        )

    with ctx.new_symbol_table("let"):
        binding_nodes = []
        for name, value in partition(bindings, 2):
            if not isinstance(name, sym.Symbol):
                raise ctx.AnalyzerException(
                    "let binding name must be a symbol", form=name
                )

            binding = Binding(
                form=name,
                name=name.name,
                local=LocalType.LET,
                tag=_tag_ast(_tag_meta(name), ctx),
                init=_analyze_form(value, ctx),
                children=vec.v(INIT),
                env=ctx.get_node_env(),
            )
            binding_nodes.append(binding)
            ctx.put_new_symbol(name, binding)

        let_body = runtime.nthrest(form, 2)
        stmts, ret = _body_ast(let_body, ctx)
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


def __letfn_fn_body(form: ISeq, ctx: AnalyzerContext) -> Fn:
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
        raise ctx.AnalyzerException(
            "letfn function name must be a symbol", form=fn_name
        )

    fn_rest = runtime.nthrest(form, 2)

    fn_body = _analyze_form(
        fn_rest.cons(
            fn_name.with_meta(
                (fn_name.meta or lmap.EMPTY).assoc(SYM_NO_WARN_ON_SHADOW_META_KEY, True)
            )
        ).cons(fn_sym),
        ctx,
    )

    if not isinstance(fn_body, Fn):
        raise ctx.AnalyzerException(
            "letfn bindings must be functions", form=form, lisp_ast=fn_body
        )

    return fn_body


def _letfn_ast(  # pylint: disable=too-many-locals
    form: ISeq, ctx: AnalyzerContext
) -> LetFn:
    assert form.first == SpecialForm.LETFN
    nelems = count(form)

    if nelems < 2:
        raise ctx.AnalyzerException(
            "letfn forms must have bindings vector and 0 or more body forms", form=form
        )

    bindings = runtime.nth(form, 1)
    if not isinstance(bindings, vec.PersistentVector):
        raise ctx.AnalyzerException("letfn bindings must be a vector", form=bindings)
    elif len(bindings) % 2 != 0:
        raise ctx.AnalyzerException(
            "letfn bindings must appear in name-value pairs", form=bindings
        )

    with ctx.new_symbol_table("letfn"):
        # Generate empty Binding nodes to put into the symbol table
        # as forward declarations. All functions in letfn* forms may
        # refer to all other functions regardless of order of definition.
        empty_binding_nodes = []
        for name, value in partition(bindings, 2):
            if not isinstance(name, sym.Symbol):
                raise ctx.AnalyzerException(
                    "letfn binding name must be a symbol", form=name
                )

            if not isinstance(value, llist.PersistentList):
                raise ctx.AnalyzerException(
                    "letfn binding value must be a list", form=value
                )

            binding = Binding(
                form=name,
                name=name.name,
                local=LocalType.LETFN,
                init=_const_node(None, ctx),
                children=vec.v(INIT),
                env=ctx.get_node_env(),
            )
            empty_binding_nodes.append((name, value, binding))
            ctx.put_new_symbol(
                name,
                binding,
            )

        # Once we've generated all of the filler Binding nodes, analyze the
        # function bodies and replace the Binding nodes with full nodes.
        binding_nodes = []
        for fn_name, fn_def, binding in empty_binding_nodes:
            fn_body = __letfn_fn_body(fn_def, ctx)
            new_binding = binding.assoc(init=fn_body)
            binding_nodes.append(new_binding)
            ctx.put_new_symbol(
                fn_name,
                new_binding,
                warn_on_shadowed_name=False,
                warn_on_shadowed_var=False,
            )

        letfn_body = runtime.nthrest(form, 2)
        stmts, ret = _body_ast(letfn_body, ctx)
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


def _loop_ast(form: ISeq, ctx: AnalyzerContext) -> Loop:
    assert form.first == SpecialForm.LOOP
    nelems = count(form)

    if nelems < 2:
        raise ctx.AnalyzerException(
            "loop forms must have bindings vector and 0 or more body forms", form=form
        )

    bindings = runtime.nth(form, 1)
    if not isinstance(bindings, vec.PersistentVector):
        raise ctx.AnalyzerException("loop bindings must be a vector", form=bindings)
    elif len(bindings) % 2 != 0:
        raise ctx.AnalyzerException(
            "loop bindings must appear in name-value pairs", form=bindings
        )

    loop_id = genname("loop")
    with ctx.new_symbol_table(loop_id):
        binding_nodes = []
        for name, value in partition(bindings, 2):
            if not isinstance(name, sym.Symbol):
                raise ctx.AnalyzerException(
                    "loop binding name must be a symbol", form=name
                )

            binding = Binding(
                form=name,
                name=name.name,
                local=LocalType.LOOP,
                init=_analyze_form(value, ctx),
                env=ctx.get_node_env(),
            )
            binding_nodes.append(binding)
            ctx.put_new_symbol(name, binding)

        with ctx.new_recur_point(loop_id, binding_nodes):
            loop_body = runtime.nthrest(form, 2)
            stmts, ret = _body_ast(loop_body, ctx)
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
            loop_node.visit(partial(_assert_recur_is_tail, ctx))
            return loop_node


def _quote_ast(form: ISeq, ctx: AnalyzerContext) -> Quote:
    assert form.first == SpecialForm.QUOTE
    nelems = count(form)

    if nelems != 2:
        raise ctx.AnalyzerException(
            "quote forms must have exactly two elements: (quote form)", form=form
        )

    with ctx.quoted():
        with ctx.expr_pos():
            expr = _analyze_form(runtime.nth(form, 1), ctx)
        assert isinstance(expr, Const), "Quoted expressions must yield :const nodes"
        return Quote(
            form=form,
            expr=expr,
            is_literal=True,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )


def _assert_no_recur(ctx: AnalyzerContext, node: Node) -> None:
    """Assert that `recur` forms do not appear in any position of this or
    child AST nodes."""
    if node.op == NodeOp.RECUR:
        raise ctx.AnalyzerException(
            "recur must appear in tail position", form=node.form, lisp_ast=node
        )
    elif node.op in {NodeOp.FN, NodeOp.LOOP}:
        pass
    else:
        node.visit(partial(_assert_no_recur, ctx))


def _assert_recur_is_tail(ctx: AnalyzerContext, node: Node) -> None:
    """Assert that `recur` forms only appear in the tail position of this
    or child AST nodes.

    `recur` forms may only appear in `do` nodes (both literal and synthetic
    `do` nodes) and in either the :then or :else expression of an `if` node."""
    if node.op == NodeOp.DO:
        assert isinstance(node, Do)
        for child in node.statements:
            _assert_no_recur(ctx, child)
        _assert_recur_is_tail(ctx, node.ret)
    elif node.op in {
        NodeOp.FN,
        NodeOp.FN_ARITY,
        NodeOp.DEFTYPE_METHOD,
        NodeOp.DEFTYPE_METHOD_ARITY,
    }:
        assert isinstance(node, (Fn, FnArity, DefTypeMethod, DefTypeMethodArity))
        node.visit(partial(_assert_recur_is_tail, ctx))
    elif node.op == NodeOp.IF:
        assert isinstance(node, If)
        _assert_no_recur(ctx, node.test)
        _assert_recur_is_tail(ctx, node.then)
        _assert_recur_is_tail(ctx, node.else_)
    elif node.op in {NodeOp.LET, NodeOp.LETFN}:
        assert isinstance(node, (Let, LetFn))
        for binding in node.bindings:
            assert binding.init is not None
            _assert_no_recur(ctx, binding.init)
        _assert_recur_is_tail(ctx, node.body)
    elif node.op == NodeOp.LOOP:
        assert isinstance(node, Loop)
        for binding in node.bindings:
            assert binding.init is not None
            _assert_no_recur(ctx, binding.init)
    elif node.op == NodeOp.RECUR:
        pass
    elif node.op == NodeOp.REIFY:
        assert isinstance(node, Reify)
        for child in node.members:
            _assert_recur_is_tail(ctx, child)
    elif node.op == NodeOp.TRY:
        assert isinstance(node, Try)
        _assert_recur_is_tail(ctx, node.body)
        for catch in node.catches:
            _assert_recur_is_tail(ctx, catch)
        if node.finally_:
            _assert_no_recur(ctx, node.finally_)
    else:
        node.visit(partial(_assert_no_recur, ctx))


def _recur_ast(form: ISeq, ctx: AnalyzerContext) -> Recur:
    assert form.first == SpecialForm.RECUR

    if ctx.recur_point is None:
        if ctx.should_allow_unresolved_symbols:
            loop_id = genname("macroexpand-recur")
        else:
            raise ctx.AnalyzerException("no recur point defined for recur", form=form)
    else:
        if len(ctx.recur_point.args) != count(form.rest):
            raise ctx.AnalyzerException(
                "recur arity does not match last recur point arity", form=form
            )

        loop_id = ctx.recur_point.loop_id

    with ctx.expr_pos():
        exprs = vec.vector(map(lambda form: _analyze_form(form, ctx), form.rest))

    return Recur(form=form, exprs=exprs, loop_id=loop_id, env=ctx.get_node_env())


@_with_meta
def _reify_ast(form: ISeq, ctx: AnalyzerContext) -> Reify:
    assert form.first == SpecialForm.REIFY

    nelems = count(form)
    if nelems < 3:
        raise ctx.AnalyzerException(
            "reify forms must have 3 or more elements, as in: "
            "(reify* :implements [bases+impls])",
            form=form,
        )

    with ctx.new_symbol_table("reify"):
        interfaces, members = __deftype_or_reify_impls(
            runtime.nthrest(form, 1), ctx, SpecialForm.REIFY
        )
        type_abstractness = __deftype_and_reify_impls_are_all_abstract(
            ctx, SpecialForm.REIFY, (), interfaces, members
        )
        return Reify(
            form=form,
            interfaces=vec.vector(interfaces),
            members=vec.vector(members),
            verified_abstract=type_abstractness.is_statically_verified_as_abstract,
            artificially_abstract=type_abstractness.artificially_abstract_supertypes,
            is_frozen=not _is_mutable(form.first),
            use_weakref_slot=not type_abstractness.supertype_already_weakref,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )


def _require_ast(form: ISeq, ctx: AnalyzerContext) -> Require:
    assert form.first == SpecialForm.REQUIRE

    aliases = []
    for f in form.rest:
        if isinstance(f, sym.Symbol):
            module_name = f
            module_alias = None
        elif isinstance(f, vec.PersistentVector):
            if len(f) != 3:
                raise ctx.AnalyzerException(
                    "require alias must take the form: [namespace :as alias]", form=f
                )
            module_name = f.val_at(0)  # type: ignore[assignment]
            if not isinstance(module_name, sym.Symbol):
                raise ctx.AnalyzerException(
                    "Basilisp namespace name must be a symbol", form=f
                )
            if not AS == f.val_at(1):
                raise ctx.AnalyzerException(
                    "expected :as alias for Basilisp alias", form=f
                )
            module_alias_sym = f.val_at(2)
            if not isinstance(module_alias_sym, sym.Symbol):
                raise ctx.AnalyzerException(
                    "Basilisp namespace alias must be a symbol", form=f
                )
            module_alias = module_alias_sym.name
        else:
            raise ctx.AnalyzerException(
                "symbol or vector expected for require*", form=f
            )

        aliases.append(
            RequireAlias(
                form=f,
                name=module_name.name,
                alias=module_alias,
                env=ctx.get_node_env(),
            )
        )

    if not aliases:
        raise ctx.AnalyzerException(
            "require forms must name at least one namespace", form=form
        )

    _do_warn_on_import_or_require_name_clash(ctx, aliases, "require")
    return Require(
        form=form,
        aliases=aliases,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _set_bang_ast(form: ISeq, ctx: AnalyzerContext) -> SetBang:
    assert form.first == SpecialForm.SET_BANG
    nelems = count(form)

    if nelems != 3:
        raise ctx.AnalyzerException(
            "set! forms must contain exactly 3 elements: (set! target value)", form=form
        )

    with ctx.expr_pos():
        target = _analyze_form(runtime.nth(form, 1), ctx)

    if not isinstance(target, Assignable):
        raise ctx.AnalyzerException(
            f"cannot set! targets of type {type(target)}", form=target
        )

    if not target.is_assignable:
        raise ctx.AnalyzerException(
            "cannot set! target which is not assignable",
            form=form,
            lisp_ast=cast(Node, target),
        )

    # Vars may only be set if they are (1) dynamic, and (2) already have a thread
    # binding established via the `binding` macro in Basilisp or by manually pushing
    # thread bindings (e.g. by runtime.push_thread_bindings).
    #
    # We can (generally) establish statically whether a Var is dynamic at compile time,
    # but establishing whether or not a Var will be thread-bound by the time a set!
    # is executed is much more challenging. Given the dynamic nature of the language,
    # it is simply easier to emit warnings for these potentially invalid cases to let
    # users fix the problem.
    if isinstance(target, VarRef):
        if ctx.warn_on_non_dynamic_set and not target.var.dynamic:
            logger.warning(f"set! target {target.var} is not marked as dynamic")
        elif not target.var.is_thread_bound:
            # This case is way more likely to result in noise, so just emit at debug.
            logger.debug(f"set! target {target.var} is not marked as thread-bound")

    with ctx.expr_pos():
        val = _analyze_form(runtime.nth(form, 2), ctx)

    return SetBang(
        form=form,
        target=target,
        val=val,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _throw_ast(form: ISeq, ctx: AnalyzerContext) -> Throw:
    assert form.first == SpecialForm.THROW
    nelems = count(form)

    if nelems < 2 or nelems > 3:
        raise ctx.AnalyzerException(
            "throw forms must contain exactly 2 or 3 elements: (throw exc [cause])",
            form=form,
        )

    with ctx.expr_pos():
        exc = _analyze_form(runtime.nth(form, 1), ctx)

    if nelems == 3:
        with ctx.expr_pos():
            cause = _analyze_form(runtime.nth(form, 2), ctx)
    else:
        cause = None

    return Throw(
        form=form,
        exception=exc,
        cause=cause,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _catch_ast(form: ISeq, ctx: AnalyzerContext) -> Catch:
    assert form.first == SpecialForm.CATCH
    nelems = count(form)

    if nelems < 4:
        raise ctx.AnalyzerException(
            "catch forms must contain at least 4 elements: (catch class local body*)",
            form=form,
        )

    catch_cls = _analyze_form(runtime.nth(form, 1), ctx)
    if not isinstance(catch_cls, (MaybeClass, MaybeHostForm)):
        raise ctx.AnalyzerException(
            "catch forms must name a class type to catch", form=catch_cls
        )

    local_name = runtime.nth(form, 2)
    if not isinstance(local_name, sym.Symbol):
        raise ctx.AnalyzerException("catch local must be a symbol", form=local_name)

    with ctx.new_symbol_table("catch"):
        catch_binding = Binding(
            form=local_name,
            name=local_name.name,
            local=LocalType.CATCH,
            env=ctx.get_node_env(),
        )
        ctx.put_new_symbol(local_name, catch_binding)

        catch_body = runtime.nthrest(form, 3)
        catch_statements, catch_ret = _body_ast(catch_body, ctx)
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


def _try_ast(form: ISeq, ctx: AnalyzerContext) -> Try:
    assert form.first == SpecialForm.TRY

    try_exprs = []
    catches = []
    finally_: Optional[Do] = None
    for expr in form.rest:
        if isinstance(expr, (llist.PersistentList, ISeq)):
            if expr.first == SpecialForm.CATCH:
                if finally_:
                    raise ctx.AnalyzerException(
                        "catch forms may not appear after finally forms in a try",
                        form=expr,
                    )
                catches.append(_catch_ast(expr, ctx))
                continue
            elif expr.first == SpecialForm.FINALLY:
                if finally_ is not None:
                    raise ctx.AnalyzerException(
                        "try forms may not contain multiple finally forms", form=expr
                    )
                # Finally values are never returned
                with ctx.stmt_pos():
                    *finally_stmts, finally_ret = map(
                        lambda form: _analyze_form(form, ctx), expr.rest
                    )
                finally_ = Do(
                    form=expr.rest,
                    statements=vec.vector(finally_stmts),
                    ret=finally_ret,
                    is_body=True,
                    env=ctx.get_node_env(
                        pos=NodeSyntacticPosition.STMT,
                    ),
                )
                continue

        lisp_node = _analyze_form(expr, ctx)

        if catches:
            raise ctx.AnalyzerException(
                "try body expressions may not appear after catch forms", form=expr
            )
        if finally_:
            raise ctx.AnalyzerException(
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
        children=(
            vec.v(BODY, CATCHES, FINALLY)
            if finally_ is not None
            else vec.v(BODY, CATCHES)
        ),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _var_ast(form: ISeq, ctx: AnalyzerContext) -> VarRef:
    assert form.first == SpecialForm.VAR

    nelems = count(form)
    if nelems != 2:
        raise ctx.AnalyzerException(
            "var special forms must contain 2 elements: (var sym)", form=form
        )

    var_sym = runtime.nth(form, 1)
    if not isinstance(var_sym, sym.Symbol):
        raise ctx.AnalyzerException("vars may only be resolved for symbols", form=form)

    if var_sym.ns is None:
        var = runtime.resolve_var(sym.symbol(var_sym.name, ctx.current_ns.name))
    else:
        var = runtime.resolve_var(var_sym)

    if var is None:
        raise ctx.AnalyzerException(f"cannot resolve var {var_sym}", form=form)

    return VarRef(
        form=form,
        var=var,
        return_var=True,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


def _yield_ast(form: ISeq, ctx: AnalyzerContext) -> Yield:
    assert form.first == SpecialForm.YIELD

    if ctx.func_ctx is None:
        raise ctx.AnalyzerException(
            "yield forms may not appear in function context", form=form
        )

    nelems = count(form)
    if nelems not in {1, 2}:
        raise ctx.AnalyzerException(
            "yield forms must contain 1 or 2 elements, as in: (yield [expr])", form=form
        )

    # Indicate that the current function is a generator
    ctx.func_ctx.is_generator = True

    if nelems == 2:
        with ctx.expr_pos():
            expr = _analyze_form(runtime.nth(form, 1), ctx)
        return Yield(
            form=form,
            expr=expr,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )
    else:
        return Yield.expressionless(form, ctx.get_node_env(pos=ctx.syntax_position))


SpecialFormHandler = Callable[[T_form, AnalyzerContext], SpecialFormNode]
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
    SpecialForm.YIELD: _yield_ast,
}


##################
# Data Structures
##################


@_analyze_form.register(llist.PersistentList)
@_analyze_form.register(ISeq)
@_with_loc
def _list_node(form: ISeq, ctx: AnalyzerContext) -> Node:
    if form == llist.EMPTY:
        with ctx.quoted():
            return _const_node(form, ctx)

    if ctx.is_quoted:
        return _const_node(form, ctx)

    s = form.first
    if isinstance(s, sym.Symbol):
        handle_special_form = _SPECIAL_FORM_HANDLERS.get(s)
        if handle_special_form is not None:
            return handle_special_form(form, ctx)
        elif s.name.startswith(".-"):
            return _host_prop_ast(form, ctx)
        elif (
            s.name.startswith(".") and s.ns is None and s.name != _DOUBLE_DOT_MACRO_NAME
        ):
            return _host_call_ast(form, ctx)

    return _invoke_ast(form, ctx)


def __resolve_nested_symbol(ctx: AnalyzerContext, form: sym.Symbol) -> HostField:
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


def __resolve_namespaced_symbol_in_ns(
    ctx: AnalyzerContext,
    which_ns: runtime.Namespace,
    form: sym.Symbol,
) -> Optional[Union[MaybeHostForm, VarRef]]:
    """Resolve the symbol `form` in the context of the Namespace `which_ns`. If
    `allow_fuzzy_macroexpansion_matching` is True and no match is made on existing
    imports, import aliases, or namespace aliases, then attempt to match the
    namespace portion"""
    assert form.ns is not None

    # Import names are always munged by the compiler when they're added to imports,
    # but if the user provides an import alias that is left untouched. Check for
    # the munged symbol in `Namespace.imports` and the unmunged in
    # `Namespace.import_aliases`.
    ns_sym = sym.symbol(form.ns)
    import_sym = sym.symbol(munge(form.ns))
    if import_sym in which_ns.imports or ns_sym in which_ns.import_aliases:
        # Fetch the full namespace name for the aliased namespace/module.
        # We don't need this for actually generating the link later, but
        # we _do_ need it for fetching a reference to the module to check
        # for membership.
        if ns_sym in which_ns.import_aliases:
            ns = which_ns.import_aliases[ns_sym]
            assert ns is not None
            ns_name = ns.name
        else:
            ns_name = import_sym.name

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
            raise ctx.AnalyzerException("can't identify aliased form", form=form)

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
            raise ctx.AnalyzerException(
                f"unable to resolve symbol '{sym.symbol(form.name, ns_sym.name)}' in this context",
                form=form,
            )
        elif v.meta is not None and v.meta.val_at(SYM_PRIVATE_META_KEY, False):
            raise ctx.AnalyzerException(
                f"cannot resolve private Var {form.name} from namespace {form.ns}",
                form=form,
            )
        return VarRef(
            form=form,
            var=v,
            is_allow_var_indirection=_is_allow_var_indirection(form),
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )

    return None


def __resolve_namespaced_symbol(  # pylint: disable=too-many-branches
    ctx: AnalyzerContext, form: sym.Symbol
) -> Union[Const, HostField, MaybeClass, MaybeHostForm, VarRef]:
    """Resolve a namespaced symbol into a Python name or Basilisp Var."""
    # Support Clojure 1.12 qualified method names
    #
    # Does not discriminate between static/class and instance methods (the latter of
    # which must include a leading `.` in Clojure). In Basilisp it was always possible
    # to access object methods (static or otherwise) using the `Classname/field` form
    # because Python objects are essentially just fancy dicts. It is possible to call
    # Python methods by referencing the method directly on the class with the instance
    # as an argument:
    #
    #     "a b c".split() == str.split("a b c")  # => ["a", "b", "c"]
    #
    # Basilisp supported this from the beginning:
    #
    #     (python.str/split "a b c")  ;;=> #py ["a" "b" "c"]
    if form.name.startswith(".") and form.name != _DOUBLE_DOT_MACRO_NAME:
        form = sym.symbol(form.name[1:], ns=form.ns, meta=form.meta)

    assert form.ns is not None

    current_ns = ctx.current_ns
    if form.ns == current_ns.name:
        v = current_ns.find(sym.symbol(form.name))
        if v is not None:
            return VarRef(
                form=form,
                var=v,
                is_allow_var_indirection=_is_allow_var_indirection(form),
                env=ctx.get_node_env(pos=ctx.syntax_position),
            )
    elif form.ns == _BUILTINS_NS:
        class_ = munge(form.name, allow_builtins=True)
        target = getattr(builtins, class_, None)
        if target is None:
            raise ctx.AnalyzerException(
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
        if v.meta is not None and v.meta.val_at(SYM_PRIVATE_META_KEY, False):
            raise ctx.AnalyzerException(
                f"cannot resolve private Var {form.name} from namespace {form.ns}",
                form=form,
            )
        return VarRef(
            form=form,
            var=v,
            is_allow_var_indirection=_is_allow_var_indirection(form),
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )

    if "." in form.name and form.name != _DOUBLE_DOT_MACRO_NAME:
        raise ctx.AnalyzerException(
            "symbol names may not contain the '.' operator", form=form
        )

    resolved = __resolve_namespaced_symbol_in_ns(ctx, current_ns, form)
    if resolved is not None:
        return resolved

    if "." in form.ns:
        try:
            return __resolve_nested_symbol(ctx, form)
        except CompilerException:
            raise ctx.AnalyzerException(
                f"unable to resolve symbol '{form}' in this context", form=form
            ) from None
    elif ctx.should_allow_unresolved_symbols:
        return _const_node(form, ctx)

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
            raise ctx.AnalyzerException(
                f"unable to resolve static or class member '{form}' in this context",
                form=form,
            )

        return HostField(
            form=form,
            field=safe_name,
            target=VarRef(
                form=form,
                var=maybe_type_or_class,
                is_allow_var_indirection=_is_allow_var_indirection(form),
                env=ctx.get_node_env(pos=ctx.syntax_position),
            ),
            is_assignable=False,
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )

    raise ctx.AnalyzerException(
        f"unable to resolve symbol '{form}' in this context", form=form
    )


def __resolve_bare_symbol(
    ctx: AnalyzerContext, form: sym.Symbol
) -> Union[Const, MaybeClass, VarRef]:
    """Resolve a non-namespaced symbol into a Python name or a local Basilisp Var."""
    assert form.ns is None

    # Look up the symbol in the namespace mapping of the current namespace.
    current_ns = ctx.current_ns
    v = current_ns.find(form)
    if v is not None:
        return VarRef(
            form=form,
            var=v,
            is_allow_var_indirection=_is_allow_var_indirection(form),
            env=ctx.get_node_env(pos=ctx.syntax_position),
        )

    if "." in form.name:
        raise ctx.AnalyzerException(
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
        return _const_node(form, ctx)

    assert munged not in vars(current_ns.module)
    raise ctx.AnalyzerException(
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


@_analyze_form.register(sym.Symbol)
@_with_loc
def _symbol_node(
    form: sym.Symbol, ctx: AnalyzerContext
) -> Union[Const, HostField, Local, MaybeClass, MaybeHostForm, VarRef]:
    if ctx.is_quoted:
        return _const_node(form, ctx)

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


@_analyze_form.register(dict)
@_with_loc
def _py_dict_node(form: dict, ctx: AnalyzerContext) -> Union[Const, PyDict]:
    if ctx.is_quoted:
        return _const_node(form, ctx)

    keys, vals = [], []
    for k, v in form.items():
        keys.append(_analyze_form(k, ctx))
        vals.append(_analyze_form(v, ctx))

    return PyDict(
        form=form,
        keys=vec.vector(keys),
        vals=vec.vector(vals),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


@_analyze_form.register(list)
@_with_loc
def _py_list_node(form: list, ctx: AnalyzerContext) -> Union[Const, PyList]:
    if ctx.is_quoted:
        return _const_node(form, ctx)
    return PyList(
        form=form,
        items=vec.vector(map(lambda form: _analyze_form(form, ctx), form)),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


@_analyze_form.register(set)
@_with_loc
def _py_set_node(form: set, ctx: AnalyzerContext) -> Union[Const, PySet]:
    if ctx.is_quoted:
        return _const_node(form, ctx)
    return PySet(
        form=form,
        items=vec.vector(map(lambda form: _analyze_form(form, ctx), form)),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


@_analyze_form.register(tuple)
@_with_loc
def _py_tuple_node(form: tuple, ctx: AnalyzerContext) -> Union[Const, PyTuple]:
    if ctx.is_quoted:
        return _const_node(form, ctx)
    return PyTuple(
        form=form,
        items=vec.vector(map(lambda form: _analyze_form(form, ctx), form)),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


@_with_meta
def _map_node(form: lmap.PersistentMap, ctx: AnalyzerContext) -> MapNode:
    keys, vals = [], []
    for k, v in form.items():
        keys.append(_analyze_form(k, ctx))
        vals.append(_analyze_form(v, ctx))

    return MapNode(
        form=form,
        keys=vec.vector(keys),
        vals=vec.vector(vals),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


@_analyze_form.register(lmap.PersistentMap)
@_with_loc
def _map_node_or_quoted(
    form: lmap.PersistentMap, ctx: AnalyzerContext
) -> Union[Const, MapNode]:
    if ctx.is_quoted:
        return _const_node(form, ctx)
    return _map_node(form, ctx)


@_with_meta
def _queue_node(form: lqueue.PersistentQueue, ctx: AnalyzerContext) -> QueueNode:
    return QueueNode(
        form=form,
        items=vec.vector(map(lambda form: _analyze_form(form, ctx), form)),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


@_analyze_form.register(lqueue.PersistentQueue)
@_with_loc
def _queue_node_or_quoted(
    form: lqueue.PersistentQueue, ctx: AnalyzerContext
) -> Union[Const, QueueNode]:
    if ctx.is_quoted:
        return _const_node(form, ctx)
    return _queue_node(form, ctx)


@_with_meta
def _set_node(form: lset.PersistentSet, ctx: AnalyzerContext) -> SetNode:
    return SetNode(
        form=form,
        items=vec.vector(map(lambda form: _analyze_form(form, ctx), form)),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


@_analyze_form.register(lset.PersistentSet)
@_with_loc
def _set_node_or_quoted(
    form: lset.PersistentSet, ctx: AnalyzerContext
) -> Union[Const, SetNode]:
    if ctx.is_quoted:
        return _const_node(form, ctx)
    return _set_node(form, ctx)


@_with_meta
def _vector_node(form: vec.PersistentVector, ctx: AnalyzerContext) -> VectorNode:
    return VectorNode(
        form=form,
        items=vec.vector(map(lambda form: _analyze_form(form, ctx), form)),
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )


@_analyze_form.register(vec.PersistentVector)
@_with_loc
def _vector_node_or_quoted(
    form: vec.PersistentVector, ctx: AnalyzerContext
) -> Union[Const, VectorNode]:
    if ctx.is_quoted:
        return _const_node(form, ctx)
    return _vector_node(form, ctx)


@functools.singledispatch
def _const_node_type(_: Any) -> ConstType:
    return ConstType.UNKNOWN


for tp, const_type in {
    bool: ConstType.BOOL,
    bytes: ConstType.BYTES,
    complex: ConstType.NUMBER,
    datetime: ConstType.INST,
    Decimal: ConstType.DECIMAL,
    dict: ConstType.PY_DICT,
    float: ConstType.NUMBER,
    Fraction: ConstType.FRACTION,
    int: ConstType.NUMBER,
    kw.Keyword: ConstType.KEYWORD,
    list: ConstType.PY_LIST,
    llist.PersistentList: ConstType.SEQ,
    lmap.PersistentMap: ConstType.MAP,
    lqueue.PersistentQueue: ConstType.QUEUE,
    lset.PersistentSet: ConstType.SET,
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
    vec.PersistentVector: ConstType.VECTOR,
}.items():
    _const_node_type.register(tp, lambda _, default=const_type: default)


@_analyze_form.register(bool)
@_analyze_form.register(bytes)
@_analyze_form.register(complex)
@_analyze_form.register(datetime)
@_analyze_form.register(Decimal)
@_analyze_form.register(float)
@_analyze_form.register(Fraction)
@_analyze_form.register(int)
@_analyze_form.register(IRecord)
@_analyze_form.register(IType)
@_analyze_form.register(kw.Keyword)
@_analyze_form.register(type(re.compile(r"")))
@_analyze_form.register(str)
@_analyze_form.register(type(None))
@_analyze_form.register(uuid.UUID)
@_with_loc
def _const_node(form: ReaderForm, ctx: AnalyzerContext) -> Const:
    assert (
        (
            ctx.is_quoted
            and isinstance(
                form,
                (
                    sym.Symbol,
                    vec.PersistentVector,
                    llist.PersistentList,
                    lmap.PersistentMap,
                    lqueue.PersistentQueue,
                    lset.PersistentSet,
                    ISeq,
                ),
            )
        )
        or (ctx.should_allow_unresolved_symbols and isinstance(form, sym.Symbol))
        or (isinstance(form, (llist.PersistentList, ISeq)) and form.is_empty)
        or isinstance(
            form,
            (
                bool,
                bytes,
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

    node_type = _const_node_type(form)
    assert node_type != ConstType.UNKNOWN, "Only allow known constant types"

    descriptor = Const(
        form=form,
        is_literal=True,
        type=node_type,
        val=form,
        env=ctx.get_node_env(pos=ctx.syntax_position),
    )

    if hasattr(form, "meta"):
        form_meta = _clean_meta(form.meta)  # type: ignore
        if form_meta is not None:
            meta_ast = _const_node(form_meta, ctx)
            assert isinstance(meta_ast, MapNode) or (
                isinstance(meta_ast, Const) and meta_ast.type == ConstType.MAP
            )
            return descriptor.assoc(meta=meta_ast, children=vec.v(META))

    return descriptor


###################
# Public Functions
###################


def analyze_form(ctx: AnalyzerContext, form: ReaderForm) -> Node:
    """Take a Lisp form as an argument and produce a Basilisp syntax
    tree matching the clojure.tools.analyzer AST spec."""
    return _analyze_form(form, ctx).assoc(top_level=True)


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
