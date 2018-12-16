import ast
import collections
import contextlib
import functools
import importlib
import itertools
import logging
import sys
import types
import uuid
from collections import OrderedDict
from datetime import datetime
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from itertools import chain
from typing import (
    Dict,
    Iterable,
    Pattern,
    Tuple,
    Optional,
    List,
    Union,
    Callable,
    NamedTuple,
    cast,
    Deque,
    Any,
)

import astor.code_gen as codegen
from functional import seq

import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.meta as lmeta
import basilisp.lang.reader as reader
import basilisp.lang.runtime as runtime
import basilisp.lang.seq as lseq
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
from basilisp.lang.runtime import Var
from basilisp.lang.typing import LispForm
from basilisp.lang.util import genname, munge
from basilisp.util import Maybe, partition

# Compiler logging
logger = logging.getLogger(__name__)

# Compiler options
USE_VAR_INDIRECTION = "use_var_indirection"
WARN_ON_SHADOWED_NAME = "warn_on_shadowed_name"
WARN_ON_SHADOWED_VAR = "warn_on_shadowed_var"
WARN_ON_UNUSED_NAMES = "warn_on_unused_names"
WARN_ON_VAR_INDIRECTION = "warn_on_var_indirection"

# String constants used in generating code
_BUILTINS_NS = "builtins"
_CORE_NS = "basilisp.core"
_DEFAULT_FN = "__lisp_expr__"
_DO_PREFIX = "lisp_do"
_FN_PREFIX = "lisp_fn"
_IF_PREFIX = "lisp_if"
_IF_TEST_PREFIX = "if_test"
_MULTI_ARITY_ARG_NAME = "multi_arity_args"
_THROW_PREFIX = "lisp_throw"
_TRY_PREFIX = "lisp_try"
_NS_VAR = "__NS"
_LISP_NS_VAR = "*ns*"

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
_SPECIAL_FORMS = lset.s(
    _DEF,
    _DO,
    _FN,
    _IF,
    _IMPORT,
    _INTEROP_CALL,
    _INTEROP_PROP,
    _LET,
    _LOOP,
    _QUOTE,
    _RECUR,
    _THROW,
    _TRY,
    _VAR,
)

_UNQUOTE = sym.symbol("unquote", _CORE_NS)
_UNQUOTE_SPLICING = sym.symbol("unquote-splicing", _CORE_NS)

# Symbols to be ignored for unused symbol warnings
_IGNORED_SYM = sym.symbol("_")
_MACRO_ENV_SYM = sym.symbol("&env")
_MACRO_FORM_SYM = sym.symbol("&form")
_NO_WARN_UNUSED_SYMS = lset.s(_IGNORED_SYM, _MACRO_ENV_SYM, _MACRO_FORM_SYM)

# Symbol table contexts
_SYM_CTX_LOCAL_STARRED = kw.keyword(
    "local-starred", ns="basilisp.lang.compiler.var-context"
)
_SYM_CTX_LOCAL = kw.keyword("local", ns="basilisp.lang.compiler.var-context")
_SYM_CTX_RECUR = kw.keyword("recur", ns="basilisp.lang.compiler.var-context")


class SymbolTableEntry(NamedTuple):
    munged: str
    context: kw.Keyword
    symbol: sym.Symbol
    used: bool = False
    warn_if_unused: bool = True


class SymbolTable:
    CONTEXTS = frozenset([_SYM_CTX_LOCAL, _SYM_CTX_LOCAL_STARRED, _SYM_CTX_RECUR])

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
        self, s: sym.Symbol, munged: str, ctx: kw.Keyword, warn_if_unused: bool = True
    ) -> "SymbolTable":
        assert ctx in SymbolTable.CONTEXTS, f"Symbol context {ctx} must be in CONTEXTS"
        if s in self._table:
            self._table[s] = self._table[s]._replace(
                munged=munged, context=ctx, symbol=s, warn_if_unused=warn_if_unused
            )
        else:
            self._table[s] = SymbolTableEntry(
                munged, ctx, s, warn_if_unused=warn_if_unused
            )
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
            if entry.context != _SYM_CTX_LOCAL:
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


class RecurPoint:
    __slots__ = ("name", "args", "has_recur")

    def __init__(self, name: str, args: vec.Vector) -> None:
        self.name = name
        self.args = args
        self.has_recur = False


class CompilerContext:
    __slots__ = ("_st", "_is_quoted", "_opts", "_recur_points")

    def __init__(self, opts: Dict[str, bool] = None) -> None:
        self._st = collections.deque([SymbolTable("<Top>")])
        self._is_quoted: Deque[bool] = collections.deque([])
        self._opts = Maybe(opts).map(lmap.map).or_else_get(lmap.m())
        self._recur_points: Deque[RecurPoint] = collections.deque([])

        if logger.isEnabledFor(logging.DEBUG):
            for k, v in self._opts:
                logger.debug("Compiler option %s=%s", k, v)

    @property
    def current_ns(self) -> runtime.Namespace:
        return runtime.get_current_ns()

    @property
    def use_var_indirection(self) -> bool:
        """If True, compile all variable references using Var.find indirection."""
        return self._opts.entry(USE_VAR_INDIRECTION, False)

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
    def warn_on_unused_names(self) -> bool:
        """If True, warn when local names are unused."""
        return self._opts.entry(WARN_ON_UNUSED_NAMES, True)

    @property
    def warn_on_var_indirection(self) -> bool:
        """If True, warn when a Var reference cannot be direct linked (iff
        use_var_indirection is False).."""
        return not self.use_var_indirection and self._opts.entry(
            WARN_ON_VAR_INDIRECTION, True
        )

    @property
    def recur_point(self):
        return self._recur_points[-1]

    @contextlib.contextmanager
    def new_recur_point(self, name: str, args: vec.Vector):
        self._recur_points.append(RecurPoint(name, args))
        yield
        self._recur_points.pop()

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

    def add_import(self, imp: sym.Symbol, mod: types.ModuleType, *aliases: sym.Symbol):
        self.current_ns.add_import(imp, mod, *aliases)

    @property
    def imports(self) -> lmap.Map:
        return self.current_ns.imports

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


class CompilerException(Exception):
    pass


def _load_attr(name: str) -> ast.Attribute:
    """Generate recursive Python Attribute AST nodes for resolving nested
    names."""
    attrs = name.split(".")

    def attr_node(node, idx):
        if idx >= len(attrs):
            return node
        return attr_node(
            ast.Attribute(value=node, attr=attrs[idx], ctx=ast.Load()), idx + 1
        )

    return attr_node(ast.Name(id=attrs[0], ctx=ast.Load()), 1)


class ASTNodeType(Enum):
    DEPENDENCY = 1
    NODE = 2


class ASTNode(NamedTuple):
    """ASTNodes are a container for generated Python AST nodes to indicate
    which broadly which type of AST node is being yielded.

    Node types can be either 'dependent' or 'node'. 'dependent' type nodes
    are those which must appear _before_ the generated 'node'(s)."""

    type: ASTNodeType
    node: ast.AST


MixedNode = Union[ASTNode, ast.AST]
MixedNodeStream = Iterable[MixedNode]
ASTStream = Iterable[ASTNode]
PyASTStream = Iterable[ast.AST]
SimpleASTProcessor = Callable[[CompilerContext, LispForm], ast.AST]
ASTProcessor = Callable[[CompilerContext, LispForm], ASTStream]


def _node(node: ast.AST) -> ASTNode:
    """Wrap a Python AST node in a 'node' type Basilisp ASTNode.

    Nodes are the 'real' node(s) generated by any given S-expression. While other
    dependency nodes may be generated as well, the 'node'(s) generated from an
    expression should be considered the primary result of the expression compilation."""
    return ASTNode(ASTNodeType.NODE, node)


def _dependency(node: ast.AST) -> ASTNode:
    """Wrap a Python AST node in a 'dependency' type Basilisp ASTNode.

    Dependency nodes are nodes which much appear before other nodes in order
    for those other nodes to be valid. The most common example is when a Python
    statement must be coerced to an expression. In that case, Basilisp generates
    a function dependency node and then follows up by generating a function call,
    which sits in the expression position for the final expression."""
    return ASTNode(ASTNodeType.DEPENDENCY, node)


def _unwrap_node(n: Optional[MixedNode]) -> ast.AST:
    """Unwrap a possibly wrapped Python AST node into its inner Python AST node type."""
    if isinstance(n, ASTNode):
        return n.node
    elif isinstance(n, ast.AST):
        return n
    else:
        raise CompilerException(
            f"Cannot unwrap object of type {type(n)}: {n}"
        ) from None


def _unwrap_nodes(s: MixedNodeStream) -> List[ast.AST]:
    """Unwrap a stream of Basilisp AST nodes into a list of Python AST nodes."""
    return seq(s).map(_unwrap_node).to_list()


def _nodes_and_exprl(s: ASTStream) -> Tuple[ASTStream, List[ASTNode]]:
    """Split a stream of expressions into the preceding nodes and a list
    containing the final expression."""
    groups: Dict[ASTNodeType, ASTStream] = seq(s).group_by(lambda n: n.type).to_dict()
    inits = groups.get(ASTNodeType.DEPENDENCY, [])
    tail = groups.get(ASTNodeType.NODE, [])
    return inits, list(tail)


def _nodes_and_expr(s: ASTStream) -> Tuple[ASTStream, Optional[ASTNode]]:
    """Split a stream of expressions into the preceding nodes and the
    final expression."""
    inits, tail = _nodes_and_exprl(s)
    try:
        assert len(tail) in [
            0,
            1,
        ], "Use of _nodes_and_expr function with greater than 1 expression"
        return inits, tail[0]
    except IndexError:
        return inits, None


def _statementize(e: ast.AST) -> ast.AST:
    """Transform non-statements into ast.Expr nodes so they can
    stand alone as statements."""
    # noinspection PyPep8
    if isinstance(
        e,
        (
            ast.Assign,
            ast.AnnAssign,  # type: ignore
            ast.AugAssign,
            ast.Raise,
            ast.Assert,
            ast.Pass,
            ast.Import,
            ast.ImportFrom,
            ast.If,
            ast.For,
            ast.While,
            ast.Continue,
            ast.Break,
            ast.Try,
            ast.ExceptHandler,
            ast.With,
            ast.FunctionDef,
            ast.Return,
            ast.Yield,
            ast.YieldFrom,
            ast.Global,
            ast.ClassDef,
            ast.AsyncFunctionDef,
            ast.AsyncFor,
            ast.AsyncWith,
        ),
    ):
        return e
    return ast.Expr(value=e)


def _expressionize(
    body: MixedNodeStream,
    fn_name: str,
    args: Optional[Iterable[ast.arg]] = None,
    vargs: Optional[ast.arg] = None,
) -> ast.FunctionDef:
    """Given a series of expression AST nodes, create a function AST node
    with the given name that can be called and will return the result of
    the final expression in the input body nodes.

    This helps to fix the impedance mismatch of Python, which includes
    statements and expressions, and Lisps, which have only expressions.
    """
    args = Maybe(args).or_else_get([])
    body_nodes: List[ast.AST] = []
    body_list = _unwrap_nodes(body)
    try:
        if len(body_list) > 1:
            body_nodes.extend(seq(body_list).drop_right(1).map(_statementize).to_list())
        body_nodes.append(ast.Return(value=seq(body_list).last()))
    except TypeError:
        body_nodes.append(ast.Return(value=body_list))

    return ast.FunctionDef(
        name=fn_name,
        args=ast.arguments(
            args=args,
            kwarg=None,
            vararg=vargs,
            kwonlyargs=[],
            defaults=[],
            kw_defaults=[],
        ),
        body=body_nodes,
        decorator_list=[],
        returns=None,
    )


_KW_ALIAS = genname("kw")
_LIST_ALIAS = genname("llist")
_MAP_ALIAS = genname("lmap")
_RUNTIME_ALIAS = genname("runtime")
_SET_ALIAS = genname("lset")
_SYM_ALIAS = genname("sym")
_VEC_ALIAS = genname("vec")
_VAR_ALIAS = genname("Var")
_UTIL_ALIAS = genname("langutil")
_NS_VAR_VALUE = f"{_NS_VAR}.value"

_NS_VAR_NAME = _load_attr(f"{_NS_VAR_VALUE}.name")
_NEW_DECIMAL_FN_NAME = _load_attr(f"{_UTIL_ALIAS}.decimal_from_str")
_NEW_FRACTION_FN_NAME = _load_attr(f"{_UTIL_ALIAS}.fraction")
_NEW_INST_FN_NAME = _load_attr(f"{_UTIL_ALIAS}.inst_from_str")
_NEW_KW_FN_NAME = _load_attr(f"{_KW_ALIAS}.keyword")
_NEW_LIST_FN_NAME = _load_attr(f"{_LIST_ALIAS}.list")
_EMPTY_LIST_FN_NAME = _load_attr(f"{_LIST_ALIAS}.List.empty")
_NEW_MAP_FN_NAME = _load_attr(f"{_MAP_ALIAS}.map")
_NEW_REGEX_FN_NAME = _load_attr(f"{_UTIL_ALIAS}.regex_from_str")
_NEW_SET_FN_NAME = _load_attr(f"{_SET_ALIAS}.set")
_NEW_SYM_FN_NAME = _load_attr(f"{_SYM_ALIAS}.symbol")
_NEW_UUID_FN_NAME = _load_attr(f"{_UTIL_ALIAS}.uuid_from_str")
_NEW_VEC_FN_NAME = _load_attr(f"{_VEC_ALIAS}.vector")
_INTERN_VAR_FN_NAME = _load_attr(f"{_VAR_ALIAS}.intern")
_FIND_VAR_FN_NAME = _load_attr(f"{_VAR_ALIAS}.find_safe")
_COLLECT_ARGS_FN_NAME = _load_attr(f"{_RUNTIME_ALIAS}._collect_args")
_COERCE_SEQ_FN_NAME = _load_attr(f"{_RUNTIME_ALIAS}.to_seq")
_TRAMPOLINE_FN_NAME = _load_attr(f"{_RUNTIME_ALIAS}._trampoline")
_TRAMPOLINE_ARGS_FN_NAME = _load_attr(f"{_RUNTIME_ALIAS}._TrampolineArgs")


def _clean_meta(form: lmeta.Meta) -> LispForm:
    """Remove reader metadata from the form's meta map."""
    try:
        meta = form.meta.discard(reader.READER_LINE_KW, reader.READER_COL_KW)
    except AttributeError:
        return None
    if len(meta) == 0:
        return None
    return meta


def _meta_kwargs_ast(  # pylint:disable=inconsistent-return-statements
    ctx: CompilerContext, form: lmeta.Meta
) -> ASTStream:
    if hasattr(form, "meta") and form.meta is not None:
        meta_nodes, meta = _nodes_and_expr(_to_ast(ctx, _clean_meta(form)))
        yield from meta_nodes
        yield _node(ast.keyword(arg="meta", value=_unwrap_node(meta)))
    else:
        return []


_SYM_DYNAMIC_META_KEY = kw.keyword("dynamic")
_SYM_MACRO_META_KEY = kw.keyword("macro")
_SYM_NO_WARN_ON_REDEF_META_KEY = kw.keyword("no-warn-on-redef")
_SYM_NO_WARN_WHEN_UNUSED_META_KEY = kw.keyword("no-warn-when-unused")
_SYM_REDEF_META_KEY = kw.keyword("redef")


def _is_dynamic(v: Var) -> bool:
    """Return True if the Var holds a value which should be compiled to a dynamic
    Var access."""
    return (
        Maybe(v.meta)
        .map(lambda m: m.get(_SYM_DYNAMIC_META_KEY, None))  # type: ignore
        .or_else_get(False)
    )


def _is_macro(v: Var) -> bool:
    """Return True if the Var holds a macro function."""
    return (
        Maybe(v.meta)
        .map(lambda m: m.get(_SYM_MACRO_META_KEY, None))  # type: ignore
        .or_else_get(False)
    )


def _is_redefable(v: Var) -> bool:
    """Return True if the Var can be redefined."""
    return (
        Maybe(v.meta)
        .map(lambda m: m.get(_SYM_REDEF_META_KEY, None))  # type: ignore
        .or_else_get(False)
    )


def _new_symbol(  # pylint: disable=too-many-arguments
    ctx: CompilerContext,
    s: sym.Symbol,
    munged: str,
    sym_ctx: kw.Keyword,
    st: Optional[SymbolTable] = None,
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
    st = Maybe(st).or_else(lambda: ctx.symbol_table)
    if warn_on_shadowed_name and ctx.warn_on_shadowed_name:
        if st.find_symbol(s) is not None:
            logger.warning(f"name '{s}' shadows name from outer scope")
    if (warn_on_shadowed_name or warn_on_shadowed_var) and ctx.warn_on_shadowed_var:
        if ctx.current_ns.find(s) is not None:
            logger.warning(f"name '{s}' shadows def'ed Var from outer scope")
    if s.meta is not None and s.meta.entry(_SYM_NO_WARN_WHEN_UNUSED_META_KEY, None):
        warn_if_unused = False
    st.new_symbol(s, munged, sym_ctx, warn_if_unused=warn_if_unused)


def _def_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Return a Python AST Node for a `def` expression."""
    assert form.first == _DEF
    assert len(form) in range(2, 4)

    ns_name = ast.Call(func=_NEW_SYM_FN_NAME, args=[_NS_VAR_NAME], keywords=[])
    def_name = ast.Call(
        func=_NEW_SYM_FN_NAME, args=[ast.Str(form[1].name)], keywords=[]
    )
    safe_name = munge(form[1].name)

    try:
        def_nodes, def_value = _nodes_and_expr(_to_ast(ctx, form[2]))
    except IndexError:
        def_nodes, def_value = [], None

    meta_nodes, meta = _nodes_and_exprl(_meta_kwargs_ast(ctx, form[1]))

    # If the Var is marked as dynamic, we need to generate a keyword argument
    # for the generated Python code to set the Var as dynamic
    dynamic_kwarg = (
        Maybe(form[1].meta)
        .map(lambda m: m.get(_SYM_DYNAMIC_META_KEY, None))  # type: ignore
        .map(lambda v: [ast.keyword(arg="dynamic", value=ast.NameConstant(v))])
        .or_else_get([])
    )

    yield from meta_nodes
    yield from def_nodes

    if safe_name in ctx.current_ns.module.__dict__ or form[1] in ctx.current_ns.interns:
        no_warn_on_redef = (
            Maybe(form[1].meta)
            .map(lambda m: m.get(_SYM_NO_WARN_ON_REDEF_META_KEY, False))  # type: ignore
            .or_else_get(False)
        )
        if not no_warn_on_redef:
            logger.warning(
                f"redefining local Python name '{safe_name}' in module '{ctx.current_ns.module.__name__}'"
            )

    yield _dependency(
        ast.Assign(
            targets=[ast.Name(id=safe_name, ctx=ast.Store())],
            value=Maybe(def_value)
            .map(_unwrap_node)
            .or_else_get(ast.NameConstant(None)),
        )
    )
    yield _node(
        ast.Call(
            func=_INTERN_VAR_FN_NAME,
            args=[ns_name, def_name, ast.Name(id=safe_name, ctx=ast.Load())],
            keywords=list(chain(dynamic_kwarg, _unwrap_nodes(meta))),  # type: ignore
        )
    )


def _do_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Return a Python AST Node for a `do` expression."""
    assert form.first == _DO

    body = _collection_ast(ctx, form.rest)
    do_fn_name = genname(_DO_PREFIX)

    yield _dependency(_expressionize(body, do_fn_name))
    yield _node(
        ast.Call(func=ast.Name(id=do_fn_name, ctx=ast.Load()), args=[], keywords=[])
    )


FunctionDefDetails = Tuple[List[ast.arg], ASTStream, Optional[ast.arg]]


def _fn_args_body(  # pylint:disable=too-many-locals
    ctx: CompilerContext,
    arg_vec: vec.Vector,
    body_exprs: lseq.Seq,
    warn_on_shadowed_name: bool = True,
    warn_on_shadowed_var: bool = True,
) -> FunctionDefDetails:
    """Generate the Python AST Nodes for a Lisp function argument vector
    and body expressions. Return a tuple of arg nodes and body AST nodes."""
    st = ctx.symbol_table

    vargs, has_vargs, vargs_idx = None, False, 0
    munged = []
    for i, s in enumerate(arg_vec):
        if s == _AMPERSAND:
            has_vargs = True
            vargs_idx = i
            break
        safe = genname(munge(s.name))
        _new_symbol(
            ctx,
            s,
            safe,
            _SYM_CTX_LOCAL,
            st=st,
            warn_on_shadowed_name=warn_on_shadowed_name,
            warn_on_shadowed_var=warn_on_shadowed_var,
        )
        munged.append(safe)

    vargs_body: List[ASTNode] = []
    if has_vargs:
        try:
            vargs_sym = arg_vec[vargs_idx + 1]
            safe = genname(munge(vargs_sym.name))
            safe_local = genname(munge(vargs_sym.name))

            # Collect all variadic arguments together into a seq and
            # reassign them to a different local
            vargs_body.append(
                _dependency(
                    ast.Assign(
                        targets=[ast.Name(id=safe_local, ctx=ast.Store())],
                        value=ast.Call(
                            func=_COLLECT_ARGS_FN_NAME,
                            args=[ast.Name(id=safe, ctx=ast.Load())],
                            keywords=[],
                        ),
                    )
                )
            )

            _new_symbol(ctx, vargs_sym, safe_local, _SYM_CTX_LOCAL, st=st)
            vargs = ast.arg(arg=safe, annotation=None)
        except IndexError:
            raise CompilerException(
                f"Expected variadic argument name after '&'"
            ) from None

    fn_body = list(_collection_ast(ctx, body_exprs))
    if len(fn_body) == 0:
        fn_body = [_node(ast.NameConstant(None))]

    args = [ast.arg(arg=a, annotation=None) for a in munged]
    body = itertools.chain(vargs_body, fn_body)
    return args, cast(ASTStream, body), vargs


FunctionArityDetails = Tuple[int, bool, llist.List]


def _is_sym_macro(ctx: CompilerContext, form: sym.Symbol) -> bool:
    """Determine if the symbol in the current context points to a macro.

    This function is used in asserting that recur only appears in a tail position.
    Since macros expand at compile time, we can skip asserting in the un-expanded
    macro call, since macros are checked after macroexpansion."""
    if form.ns is not None:
        if form.ns == ctx.current_ns.name:
            v = ctx.current_ns.find(sym.symbol(form.name))
            if v is not None:
                return _is_macro(v)
        ns_sym = sym.symbol(form.ns)
        if ns_sym in ctx.current_ns.aliases:
            aliased_ns = ctx.current_ns.aliases[ns_sym]
            v = Var.find(sym.symbol(form.name, ns=aliased_ns.name))
            if v is not None:
                return _is_macro(v)

    v = ctx.current_ns.find(form)
    if v is not None:
        return _is_macro(v)

    return False


def _assert_no_recur(ctx: CompilerContext, form: lseq.Seq) -> None:
    """Assert that the iterable contains no recur special form."""
    for child in form:
        if isinstance(child, lseq.Seqable):
            _assert_no_recur(ctx, child.seq())
        elif isinstance(child, (llist.List, lseq.Seq)):
            if isinstance(child.first, sym.Symbol):
                if _is_sym_macro(ctx, child.first):
                    continue
                elif child.first == _RECUR:
                    raise CompilerException(
                        f"Recur appears outside tail position in {form}"
                    )
                elif child.first == _FN:
                    continue
            _assert_no_recur(ctx, child)


def _assert_recur_is_tail(ctx: CompilerContext, form: lseq.Seq) -> None:  # noqa: C901
    """Assert that recur special forms only appear in tail position in a function."""
    listlen = 0
    first_recur_index = None
    for i, child in enumerate(form):  # pylint:disable=too-many-nested-blocks
        listlen += 1
        if isinstance(child, (llist.List, lseq.Seq)):
            if isinstance(child.first, sym.Symbol) and _is_sym_macro(ctx, child.first):
                continue
            elif child.first == _RECUR:
                if first_recur_index is None:
                    first_recur_index = i
            elif child.first == _DO:
                _assert_recur_is_tail(ctx, child)
            elif child.first == _FN:
                continue
            elif child.first == _IF:
                _assert_no_recur(ctx, lseq.sequence([runtime.nth(child, 1)]))
                _assert_recur_is_tail(ctx, lseq.sequence([runtime.nth(child, 2)]))
                try:
                    _assert_recur_is_tail(ctx, lseq.sequence([runtime.nth(child, 3)]))
                except IndexError:
                    pass
            elif child.first in {_LET, _LOOP}:
                for binding, val in partition(runtime.nth(child, 1), 2):
                    _assert_no_recur(ctx, lseq.sequence([binding]))
                    _assert_no_recur(ctx, lseq.sequence([val]))
                let_body = runtime.nthnext(child, 2)
                if let_body:
                    _assert_recur_is_tail(ctx, let_body)
            elif child.first == _TRY:
                if isinstance(runtime.nth(child, 1), llist.List):
                    _assert_recur_is_tail(ctx, lseq.sequence([runtime.nth(child, 1)]))
                catch_finally = runtime.nthnext(child, 2)
                if catch_finally:
                    for clause in catch_finally:
                        if isinstance(clause, llist.List):
                            if clause.first == _CATCH:
                                _assert_recur_is_tail(
                                    ctx, lseq.sequence([runtime.nthnext(clause, 2)])
                                )
                            elif clause.first == _FINALLY:
                                _assert_no_recur(ctx, clause.rest)
            elif child.first in {
                _DEF,
                _IMPORT,
                _INTEROP_CALL,
                _INTEROP_PROP,
                _THROW,
                _VAR,
            }:
                _assert_no_recur(ctx, child)
            else:
                _assert_no_recur(ctx, child)
        else:
            if isinstance(child, lseq.Seqable):
                _assert_no_recur(ctx, child.seq())

    if first_recur_index is not None:
        if first_recur_index != listlen - 1:
            raise CompilerException("Recur appears outside tail position")


def _fn_arities(
    ctx: CompilerContext, form: llist.List
) -> Iterable[FunctionArityDetails]:
    """Return the arities of a function definition and some additional details about
    the argument vector. Verify that all arities are compatible. In particular, this
    function will throw a CompilerException if any of the following are true:
     - two function definitions have the same number of arguments
     - two function definitions have a rest parameter
     - any function definition has the same number of arguments as a definition
       with a rest parameter

    Given a function such as this:

        (fn a
          ([] :a)
          ([a] a))

    Returns a generator yielding: '(([] :a) ([a] a))

    Single arity functions yield the rest:

        (fn a [] :a) ;=> '(([] :a))"""
    if not all(
        map(
            lambda f: isinstance(f, (llist.List, lseq.Seq))
            and isinstance(f.first, vec.Vector),
            form,
        )
    ):
        assert isinstance(form.first, vec.Vector)
        _assert_recur_is_tail(ctx, form)
        yield len(form.first), False, form
        return

    arg_counts: Dict[int, llist.List] = {}
    has_vargs = False
    vargs_len = None
    for arity in form:
        _assert_recur_is_tail(ctx, arity)

        # Verify each arity is unique
        arg_count = len(arity.first)
        if arg_count in arg_counts:
            raise CompilerException(
                "Each arity in multi-arity fn must be unique",
                [arity, arg_counts[arg_count]],
            )

        # Verify that only one arity contains a rest-param
        is_rest = False
        for arg in arity.first:
            if arg == _AMPERSAND:
                if has_vargs:
                    raise CompilerException(
                        "Only one arity in multi-arity fn may have rest param"
                    )
                is_rest = True
                has_vargs = True
                arg_count -= 1
                vargs_len = arg_count

        # Verify that arities do not exceed rest-param arity
        if vargs_len is not None and any([c >= vargs_len for c in arg_counts]):
            raise CompilerException(
                "No arity in multi-arity fn may exceed the rest param arity"
            )

        # Put this in last so it does not conflict with the above checks
        arg_counts[arg_count] = arity

        yield arg_count, is_rest, arity


def _compose_ifs(
    if_stmts: List[Dict[str, ast.AST]], orelse: List[ast.AST] = None
) -> ast.If:
    """Compose a series of If statements into nested elifs, with
    an optional terminating else."""
    first = if_stmts[0]
    try:
        rest = if_stmts[1:]
        return ast.If(
            test=first["test"],
            body=[first["body"]],
            orelse=[_compose_ifs(rest, orelse=orelse)],
        )
    except IndexError:
        return ast.If(
            test=first["test"],
            body=[first["body"]],
            orelse=Maybe(orelse).or_else_get([]),
        )


def _fn_name(s: Optional[sym.Symbol]) -> str:
    """Generate a safe Python function name from a function name symbol.
    If no symbol is provided, generate a name with a default prefix."""
    return genname("__" + munge(Maybe(s).map(lambda s: s.name).or_else_get(_FN_PREFIX)))


def _single_arity_fn_ast(
    ctx: CompilerContext, name: Optional[sym.Symbol], fndef: llist.List
) -> ASTStream:
    """Generate Python AST nodes for a single-arity function."""
    py_fn_name = _fn_name(name)
    with ctx.new_symbol_table(py_fn_name), ctx.new_recur_point(py_fn_name, fndef.first):
        # Allow named anonymous functions to recursively call themselves
        if name is not None:
            _new_symbol(ctx, name, py_fn_name, _SYM_CTX_RECUR, warn_if_unused=False)

        args, body, vargs = _fn_args_body(ctx, fndef.first, fndef.rest)

        yield _dependency(_expressionize(body, py_fn_name, args=args, vargs=vargs))
        if ctx.recur_point.has_recur:
            yield _node(
                ast.Call(
                    func=_TRAMPOLINE_FN_NAME,
                    args=[ast.Name(id=ctx.recur_point.name, ctx=ast.Load())],
                    keywords=[],
                )
            )
        else:
            yield _node(ast.Name(id=py_fn_name, ctx=ast.Load()))
        return


def _multi_arity_fn_ast(
    ctx: CompilerContext,
    name: Optional[sym.Symbol],
    arities: List[FunctionArityDetails],
) -> ASTStream:
    """Generate Python AST nodes for multi-arity Basilisp function definitions.

    For example, a multi-arity function like this:

        (def f
          (fn f
            ([] (print "No args"))
            ([arg]
              (print arg))
            ([arg & rest]
              (print (concat [arg] rest)))))

    Would yield a function definition in Python code like this:

        def __f_68__arity0():
            return print_('No args')


        def __f_68__arity1(arg_69):
            return print_(arg_69)


        def __f_68__arity_rest(arg_70, *rest_71):
            rest_72 = runtime._collect_args(rest_71)
            return print_(concat(vec.vector([arg_70], meta=None), rest_72))


        def __f_68(*multi_arity_args):
            if len(multi_arity_args) == 0:
                return __f_68__arity0(*multi_arity_args)
            elif len(multi_arity_args) == 1:
                return __f_68__arity1(*multi_arity_args)
            elif len(multi_arity_args) >= 2:
                return __f_68__arity2(*multi_arity_args)


        f = __f_68"""
    py_fn_name = _fn_name(name)
    if_stmts: List[Dict[str, ast.AST]] = []
    multi_arity_args_arg = _load_attr(_MULTI_ARITY_ARG_NAME)
    has_rest = False

    for arg_count, is_rest, arity in arities:
        with ctx.new_recur_point(py_fn_name, arity.first):
            # Allow named anonymous functions to recursively call themselves
            if name is not None:
                _new_symbol(ctx, name, py_fn_name, _SYM_CTX_RECUR, warn_if_unused=False)

            has_rest = any([has_rest, is_rest])
            arity_name = f"{py_fn_name}__arity{'_rest' if is_rest else arg_count}"

            with ctx.new_symbol_table(arity_name):
                # Generate the arity function
                args, body, vargs = _fn_args_body(ctx, arity.first, arity.rest)
                yield _dependency(
                    _expressionize(body, arity_name, args=args, vargs=vargs)
                )

            # If a recur point was established, we generate a trampoline version of the
            # generated function to allow repeated recursive calls without blowing up the
            # stack size.
            if ctx.recur_point.has_recur:
                yield _dependency(
                    ast.Assign(
                        targets=[ast.Name(id=arity_name, ctx=ast.Store())],
                        value=ast.Call(
                            func=_TRAMPOLINE_FN_NAME,
                            args=[ast.Name(id=arity_name, ctx=ast.Load())],
                            keywords=[],
                        ),
                    )
                )

            # Generate an if-statement branch for the arity-dispatch function
            compare_op = ast.GtE() if is_rest else ast.Eq()
            if_stmts.append(
                {
                    "test": ast.Compare(
                        left=ast.Call(
                            func=_load_attr("len"),
                            args=[multi_arity_args_arg],
                            keywords=[],
                        ),
                        ops=[compare_op],
                        comparators=[ast.Num(arg_count)],
                    ),
                    "body": ast.Return(
                        value=ast.Call(
                            func=_load_attr(arity_name),
                            args=[
                                ast.Starred(value=multi_arity_args_arg, ctx=ast.Load())
                            ],
                            keywords=[],
                        )
                    ),
                }
            )

    assert len(if_stmts) == len(arities)

    yield _dependency(
        ast.FunctionDef(
            name=py_fn_name,
            args=ast.arguments(
                args=[],
                kwarg=None,
                vararg=ast.arg(arg=_MULTI_ARITY_ARG_NAME, annotation=None),
                kwonlyargs=[],
                defaults=[],
                kw_defaults=[],
            ),
            body=[
                _compose_ifs(if_stmts),
                ast.Raise(
                    exc=ast.Call(
                        func=_load_attr("basilisp.lang.runtime.RuntimeException"),
                        args=[
                            ast.Str(f"Wrong number of args passed to function: {name}"),
                            ast.Call(
                                func=ast.Name(id="len", ctx=ast.Load()),
                                args=[
                                    ast.Name(id=_MULTI_ARITY_ARG_NAME, ctx=ast.Load())
                                ],
                                keywords=[],
                            ),
                        ],
                        keywords=[],
                    ),
                    cause=None,
                ),
            ],
            decorator_list=[],
            returns=None,
        )
    )

    yield _node(ast.Name(id=py_fn_name, ctx=ast.Load()))


def _fn_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a Python AST Nodes for function definitions."""
    assert form.first == _FN
    has_name = isinstance(form[1], sym.Symbol)
    name = form[1] if has_name else None

    rest_idx = 1 + int(has_name)
    arities = list(_fn_arities(ctx, form[rest_idx:]))
    if len(arities) == 0:
        raise CompilerException("Function def must have argument vector")
    elif len(arities) == 1:
        _, _, fndef = arities[0]
        yield from _single_arity_fn_ast(ctx, name, fndef)
        return
    else:
        yield from _multi_arity_fn_ast(ctx, name, arities)
        return


def _if_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a function call to a utility function which acts as
    an if expression and works around Python's if statement.

    Every expression in Basilisp is true if it is not the literal values nil
    or false. This function compiles direct checks for the test value against
    the Python values None and False to accommodate this behavior.

    Note that the if and else bodies are switched in compilation so that we
    can perform a short-circuit or comparison, rather than exhaustively checking
    for both false and nil each time."""
    assert form.first == _IF
    assert len(form) in range(3, 5)

    test_nodes, test = _nodes_and_expr(_to_ast(ctx, form[1]))
    body_nodes, body = _nodes_and_expr(_to_ast(ctx, form[2]))

    try:
        else_nodes, lelse = _nodes_and_expr(_to_ast(ctx, form[3]))  # type: ignore
    except IndexError:
        else_nodes = []  # type: ignore
        lelse = ast.NameConstant(None)  # type: ignore

    test_name = genname(_IF_TEST_PREFIX)
    test_assign = ast.Assign(
        targets=[ast.Name(id=test_name, ctx=ast.Store())], value=_unwrap_node(test)
    )

    ifstmt = ast.If(
        test=ast.BoolOp(
            op=ast.Or(),
            values=[
                ast.Compare(
                    left=ast.NameConstant(None),
                    ops=[ast.Is()],
                    comparators=[ast.Name(id=test_name, ctx=ast.Load())],
                ),
                ast.Compare(
                    left=ast.NameConstant(False),
                    ops=[ast.Is()],
                    comparators=[ast.Name(id=test_name, ctx=ast.Load())],
                ),
            ],
        ),
        values=[],
        body=[ast.Return(value=_unwrap_node(lelse))],
        orelse=[ast.Return(value=_unwrap_node(body))],
    )

    ifname = genname(_IF_PREFIX)

    yield _dependency(
        ast.FunctionDef(
            name=ifname,
            args=ast.arguments(
                args=[],
                kwarg=None,
                vararg=None,
                kwonlyargs=[],
                defaults=[],
                kw_defaults=[],
            ),
            body=_unwrap_nodes(
                chain(test_nodes, body_nodes, else_nodes, [test_assign, ifstmt])
            ),
            decorator_list=[],
            returns=None,
        )
    )
    yield _node(
        ast.Call(func=ast.Name(id=ifname, ctx=ast.Load()), args=[], keywords=[])
    )


def _import_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Append Import statements into the compiler context nodes."""
    assert form.first == _IMPORT

    last = None
    for f in form.rest:
        if isinstance(f, sym.Symbol):
            module_name = f
            module_alias = module_name.name.split(".", maxsplit=1)[0]
        elif isinstance(f, vec.Vector):
            module_name = f.entry(0)
            assert isinstance(
                module_name, sym.Symbol
            ), "Python module name must be a symbol"
            assert kw.keyword("as") == f.entry(1)
            module_alias = f.entry(2).name
        else:
            raise CompilerException("Symbol or vector expected for import*")

        try:
            module = importlib.import_module(module_name.name)
            if module_name.name != module_alias:
                ctx.add_import(module_name, module, sym.symbol(module_alias))
            else:
                ctx.add_import(module_name, module)
        except ModuleNotFoundError:
            raise ImportError(f"Module '{module_name.name}' not found")

        with ctx.quoted():
            module_alias = munge(module_alias)
            yield _dependency(ast.Global(names=[module_alias]))
            yield _dependency(
                ast.Assign(
                    targets=[ast.Name(id=module_alias, ctx=ast.Store())],
                    value=ast.Call(
                        func=_load_attr("builtins.__import__"),
                        args=[ast.Str(module_name.name)],
                        keywords=[],
                    ),
                )
            )
            last = ast.Name(id=module_alias, ctx=ast.Load())
            yield _dependency(
                ast.Call(
                    func=_load_attr(f"{_NS_VAR_VALUE}.add_import"),
                    args=list(
                        chain(  # type: ignore
                            _unwrap_nodes(_to_ast(ctx, module_name)), [last]
                        )
                    ),
                    keywords=[],
                )
            )

    assert last is not None
    yield _node(last)


def _interop_call_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a Python AST node for Python interop method calls."""
    assert form.first == _INTEROP_CALL
    assert form[1] is not None
    assert isinstance(form[2], sym.Symbol)
    assert form[2] is not None
    assert form[2].ns is None

    target_nodes, target = _nodes_and_expr(_to_ast(ctx, form[1]))
    yield from target_nodes

    call_target = ast.Attribute(
        value=_unwrap_node(target), attr=munge(form[2].name), ctx=ast.Load()
    )

    args: Iterable[ast.AST] = []
    if len(form) > 3:
        nodes, args = _collection_literal_ast(ctx, form[3:])
        yield from nodes

    yield _node(ast.Call(func=call_target, args=list(args), keywords=[]))


def _interop_prop_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a Python AST node for Python interop property access."""
    assert form.first == _INTEROP_PROP
    assert form[1] is not None
    assert isinstance(form[2], sym.Symbol)
    assert form[2].ns is None
    assert len(form) == 3

    target_nodes, target = _nodes_and_expr(_to_ast(ctx, form[1]))
    yield from target_nodes
    yield _node(
        ast.Attribute(
            value=_unwrap_node(target), attr=munge(form[2].name), ctx=ast.Load()
        )
    )


def _let_ast(  # pylint:disable=too-many-locals
    ctx: CompilerContext, form: llist.List
) -> ASTStream:
    """Generate a Python AST node for a let binding.

    Python code for a `let*` binding like this:
        (let* [a 1
               b :length
               c {b a}
               a 4]
          c)

    Should look roughly like this:
        def let_32(a_33, b_34, c_35):
            return c_35

        a_36 = 1
        b_37 = keyword.keyword(\"length\")
        c_38 = lmap.map({b_37: a_36})
        a_36 = 4

        let_32(a_36, b_37, c_38)  #=> {:length 1}
    """
    assert form.first == _LET
    assert isinstance(form[1], vec.Vector)
    assert len(form) >= 3

    # There is a lot going on in this section, so let me break it down.
    # Python doesn't have any syntactic element which can act as a lexical
    # block like `let*`, so we compile the body of a `let*` expression into
    # a Python function.
    #
    # In order to allow `let*` features like re-binding a name, and referencing
    # previously bound names in later expressions, we generate a series of
    # local Python variable assignments with the computed expressions. The
    # semantics of Python's local variables allows us to do this without needing
    # to define special logic to handle that graph-like behavior.
    #
    # In the case above (in the doc-string), we re-bind `a` to 4 at the end of
    # the bindings, but we don't want to assign that value to a NEW Python
    # variable, we really want to mutate the generated `a`. Otherwise, we'll end
    # up sending multiple `a` values into our `let*` closure, which is confusing
    # and messy.
    #
    # Unfortunately, this means we have to keep track of a ton of very similar,
    # but subtly different things as we're generating all of these values. In
    # particular, we have to keep track of the Lisp symbols, which become function
    # parameters; the variable names, which become Python locals in assignments;
    # the variable names, which become Python locals in an expression context (which
    # is a subtly different Python AST node); and the computed expressions.
    with ctx.new_symbol_table(genname("let_st")) as st:
        bindings = list(partition(form[1], 2))

        if not bindings:
            raise CompilerException("Expected at least one binding in 'let*'") from None

        arg_syms: Dict[
            sym.Symbol, str
        ] = OrderedDict()  # Mapping of binding symbols (turned into function parameter names) to munged name  # noqa: E501
        var_names = (
            []
        )  # Names of local Python variables bound to computed expressions prior to the function call
        arg_deps = []  # Argument expression dependency nodes
        arg_exprs = []  # Bound expressions are the expressions a name is bound to
        for s, expr in bindings:
            # Keep track of only the newest symbol and munged name in arg_syms, that way
            # we are only calling the let binding below with the most recent entry.
            munged = genname(munge(s.name))
            arg_syms[s] = munged
            var_names.append(munged)

            expr_deps, expr_node = _nodes_and_expr(_to_ast(ctx, expr))

            # Don't add the new symbol until after we've processed the expression
            _new_symbol(ctx, s, munged, _SYM_CTX_LOCAL, st=st)

            arg_deps.append(expr_deps)
            arg_exprs.append(_unwrap_node(expr_node))

        # Generate an outer function to hold the entire let expression (including bindings).
        # We need to do this to guarantee that no binding expressions are executed as part of
        # an assignment as a dependency node. This eager evaluation could leak out as part of
        # (at least) if statements dependency nodes.
        outer_letname = genname("let")
        let_fn_body: List[ast.AST] = []

        # Generate a function to hold the body of the let expression
        letname = genname("let")

        # Suppress shadowing warnings below since the shadow warnings will be
        # emitted by calling _new_symbol in the loop above
        args, body, vargs = _fn_args_body(
            ctx,
            vec.vector(arg_syms.keys()),
            runtime.nthrest(form, 2),
            warn_on_shadowed_var=False,
            warn_on_shadowed_name=False,
        )
        let_fn_body.append(_expressionize(body, letname, args=args, vargs=vargs))

    # Generate local variable assignments for processing let bindings
    var_names = seq(var_names).map(lambda n: ast.Name(id=n, ctx=ast.Store()))
    for name, deps, expr in zip(var_names, arg_deps, arg_exprs):
        let_fn_body.extend(_unwrap_nodes(deps))
        let_fn_body.append(ast.Assign(targets=[name], value=expr))

    let_fn_body.append(
        ast.Call(
            func=_load_attr(letname),
            args=seq(arg_syms.values())
            .map(lambda n: ast.Name(id=n, ctx=ast.Load()))
            .to_list(),
            keywords=[],
        )
    )

    yield _dependency(_expressionize(let_fn_body, outer_letname))
    yield _node(ast.Call(func=_load_attr(outer_letname), args=[], keywords=[]))


def _loop_ast(  # pylint:disable=too-many-locals
    ctx: CompilerContext, form: llist.List
) -> ASTStream:
    """Generate a Python AST node for a loop special form.

    Python code for a `loop*` binding like this:
        (loop [s   "abc"
               len 0]
          (if (seq s)
            (recur (rest s)
                   (inc len))
            len))

    Should look roughly like this:
        def loop_14():

            def loop_15(s_16, len__17):

                def lisp_if_19():
                    if_test_18 = basilisp.core.seq(s_16)
                    if None is if_test_18 or False is if_test_18:
                        return len__17
                    else:
                        return runtime_5._TrampolineArgs(False, basilisp.core.rest(
                            s_16), basilisp.core.inc(len__17))
                return lisp_if_19()
            s_12 = 'abc'
            len__13 = 0
            return runtime_5._trampoline(loop_15)(s_12, len__13)"""
    assert form.first == _LOOP
    assert isinstance(form[1], vec.Vector)
    assert len(form) >= 3

    # For a better description of what's going on below, peek up at _let_ast.
    with ctx.new_symbol_table(genname("loop_st")) as st:
        bindings = list(partition(form[1], 2))

        arg_syms: Dict[
            sym.Symbol, str
        ] = OrderedDict()  # Mapping of binding symbols (turned into function parameter names) to munged name  # noqa: E501
        var_names = (
            []
        )  # Names of local Python variables bound to computed expressions prior to the function call
        arg_deps = []  # Argument expression dependency nodes
        arg_exprs = []  # Bound expressions are the expressions a name is bound to
        for s, expr in bindings:
            # Keep track of only the newest symbol and munged name in arg_syms, that way
            # we are only calling the loop binding below with the most recent entry.
            munged = genname(munge(s.name))
            arg_syms[s] = munged
            var_names.append(munged)

            expr_deps, expr_node = _nodes_and_expr(_to_ast(ctx, expr))

            # Don't add the new symbol until after we've processed the expression
            _new_symbol(ctx, s, munged, _SYM_CTX_LOCAL, st=st)

            arg_deps.append(expr_deps)
            arg_exprs.append(_unwrap_node(expr_node))

        # Generate an outer function to hold the entire loop expression (including bindings).
        # We need to do this to guarantee that no binding expressions are executed as part of
        # an assignment as a dependency node. This eager evaluation could leak out as part of
        # (at least) if statements dependency nodes.
        outer_loopname = genname("loop")
        loop_fn_body: List[ast.AST] = []

        # Generate a function to hold the body of the loop expression
        loopname = genname("loop")

        with ctx.new_recur_point("loop", vec.vector(arg_syms.keys())):
            # Suppress shadowing warnings below since the shadow warnings will be
            # emitted by calling _new_symbol in the loop above
            args, body, vargs = _fn_args_body(
                ctx,
                vec.vector(arg_syms.keys()),
                runtime.nthrest(form, 2),
                warn_on_shadowed_var=False,
                warn_on_shadowed_name=False,
            )
            loop_fn_body.append(_expressionize(body, loopname, args=args, vargs=vargs))

    # Generate local variable assignments for processing loop bindings
    var_names = seq(var_names).map(lambda n: ast.Name(id=n, ctx=ast.Store()))
    for name, deps, expr in zip(var_names, arg_deps, arg_exprs):
        loop_fn_body.extend(_unwrap_nodes(deps))
        loop_fn_body.append(ast.Assign(targets=[name], value=expr))

    loop_fn_body.append(
        ast.Call(
            func=ast.Call(
                func=_TRAMPOLINE_FN_NAME, args=[_load_attr(loopname)], keywords=[]
            ),
            args=seq(arg_syms.values())
            .map(lambda n: ast.Name(id=n, ctx=ast.Load()))
            .to_list(),
            keywords=[],
        )
    )

    yield _dependency(_expressionize(loop_fn_body, outer_loopname))
    yield _node(ast.Call(func=_load_attr(outer_loopname), args=[], keywords=[]))


def _quote_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a Python AST Node for quoted forms.

    Quoted forms are actually handled in their respective AST processor,
    since only a small number of forms are quoted. This function merely
    marks the `quoted` flag in the compiler context to let any functions
    downstream know that they must quote the form they are compiling."""
    assert form.first == _QUOTE
    assert len(form) == 2
    with ctx.quoted():
        yield from _to_ast(ctx, form[1])


def _recur_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a Python AST Node for recur forms.

    Basilisp recur forms may only appear within function definitions and
    only in tail position.

    Every function definition in Basilisp code establishes a new recur
    point. If a recur form appears, the Basilisp compiler emits a return
    statement returning a basilisp.lang.runtime._TrampolineArgs object,
    which wraps all of the function arguments. Once a recur form is emitted,
    the Basilisp compiler also wraps the function definition in a trampoline,
    which will iteratively (rather than recursively) call the wrapped
    function so long as _TrampolineArgs objects are returned from it. Once
    a non _TrampolineArgs object is returned, that is returned as the final
    return from the function."""
    assert form.first == _RECUR
    assert len(form) >= 1
    try:
        ctx.recur_point.has_recur = True

        expr_deps, exprs = _collection_literal_ast(ctx, form.rest)
        yield from expr_deps

        has_vargs = any([s == _AMPERSAND for s in ctx.recur_point.args])
        yield _node(
            ast.Call(
                func=_TRAMPOLINE_ARGS_FN_NAME,
                args=list(
                    itertools.chain(  # type: ignore
                        [ast.NameConstant(has_vargs)], _unwrap_nodes(exprs)
                    )
                ),
                keywords=[],
            )
        )
    except IndexError:
        raise CompilerException("Attempting to recur without recur point") from None


def _catch_expr_body(body) -> PyASTStream:
    """Given a series of expression AST nodes, create a body of expression
    nodes with a final return node at the end of the list."""
    try:
        if len(body) > 1:
            yield from seq(body).drop_right(1).map(_statementize).to_list()
        yield ast.Return(value=seq(body).last())
    except TypeError:
        yield ast.Return(value=body)


def _catch_ast(ctx: CompilerContext, form: llist.List) -> ast.ExceptHandler:
    """Generate Python AST nodes for `catch` forms."""
    assert form.first == _CATCH
    assert len(form) >= 4
    assert isinstance(form[1], sym.Symbol)
    assert isinstance(form[2], sym.Symbol)

    type_name = form[1].name
    if form[1].ns is not None:
        type_name = f"{form[1].ns}.{type_name}"

    exc_name = munge(form[2].name)
    with ctx.new_symbol_table(genname("catch_block")):
        _new_symbol(ctx, form[2], exc_name, _SYM_CTX_LOCAL)
        body = (
            seq(form[3:])
            .flat_map(lambda f: _to_ast(ctx, f))
            .map(_unwrap_node)
            .to_list()
        )

    return ast.ExceptHandler(
        type=_load_attr(type_name), name=exc_name, body=list(_catch_expr_body(body))
    )


def _finally_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate Python AST nodes for `finally` forms."""
    assert form.first == _FINALLY
    assert len(form) >= 2

    yield from seq(form.rest).flat_map(lambda clause: _to_ast(ctx, clause)).map(
        _unwrap_node
    ).map(_statementize)


def _throw_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a Python AST Node for the `throw` special form."""
    assert form.first == _THROW
    assert len(form) == 2

    deps, expr = _nodes_and_expr(_to_ast(ctx, form[1]))
    yield from deps

    throw_fn = genname(_THROW_PREFIX)
    raise_body = ast.Raise(exc=_unwrap_node(expr), cause=None)

    yield _dependency(
        ast.FunctionDef(
            name=throw_fn,
            args=ast.arguments(
                args=[],
                kwarg=None,
                vararg=None,
                kwonlyargs=[],
                defaults=[],
                kw_defaults=[],
            ),
            body=[raise_body],
            decorator_list=[],
            returns=None,
        )
    )

    yield _node(
        ast.Call(func=ast.Name(id=throw_fn, ctx=ast.Load()), args=[], keywords=[])
    )


def _try_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a Python AST nodes for `try` forms."""
    assert form.first == _TRY
    assert len(form) >= 3

    expr_nodes, expr_v = _nodes_and_expr(_to_ast(ctx, form[1]))
    expr = ast.Return(value=_unwrap_node(expr_v))

    # Split clauses by the first form (which should be either the
    # symbol `catch` or the symbol `finally`). Validate we have 0
    # or 1 finally clauses.
    clauses = seq(form[2:]).group_by(lambda f: f.first.name).to_dict()

    finallys = clauses.get("finally", [])
    if len(finallys) not in [0, 1]:
        raise CompilerException(
            "Only one finally clause may be provided in a try/catch block"
        ) from None

    catch_exprs: List[ast.AST] = seq(clauses.get("catch", [])).map(
        lambda f: _catch_ast(ctx, llist.list(f))
    ).to_list()
    final_exprs: List[ast.AST] = seq(finallys).flat_map(
        lambda f: _finally_ast(ctx, llist.list(f))
    ).to_list()

    # Start building up the try/except block that will be inserted
    # into a function to expressionize it
    try_body = ast.Try(
        body=_unwrap_nodes(chain(expr_nodes, [expr])),
        handlers=catch_exprs,
        orelse=[],
        finalbody=final_exprs,
    )

    # Insert the try/except function into the container
    # nodes vector so it will be available in the calling context
    try_fn_name = genname(_TRY_PREFIX)
    yield _dependency(
        ast.FunctionDef(
            name=try_fn_name,
            args=ast.arguments(
                args=[],
                kwarg=None,
                vararg=None,
                kwonlyargs=[],
                defaults=[],
                kw_defaults=[],
            ),
            body=[try_body],
            decorator_list=[],
            returns=None,
        )
    )

    yield _node(
        ast.Call(func=ast.Name(id=try_fn_name, ctx=ast.Load()), args=[], keywords=[])
    )


def _var_ast(_: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a Python AST Node for the `var` special form."""
    assert form[0] == _VAR
    assert isinstance(form[1], sym.Symbol)

    ns: ast.expr = _NS_VAR_NAME if form[1].ns is None else ast.Str(form[1].ns)

    base_sym = ast.Call(
        func=_NEW_SYM_FN_NAME,
        args=[ast.Str(form[1].name)],
        keywords=[ast.keyword(arg="ns", value=ns)],
    )

    yield _node(ast.Call(func=_FIND_VAR_FN_NAME, args=[base_sym], keywords=[]))


_SPECIAL_FORM_HANDLERS: Dict[
    sym.Symbol, Callable[[CompilerContext, llist.List], ASTStream]
] = {
    _DEF: _def_ast,
    _FN: _fn_ast,
    _IF: _if_ast,
    _IMPORT: _import_ast,
    _INTEROP_CALL: _interop_call_ast,
    _INTEROP_PROP: _interop_prop_ast,
    _DO: _do_ast,
    _LET: _let_ast,
    _LOOP: _loop_ast,
    _QUOTE: _quote_ast,
    _RECUR: _recur_ast,
    _THROW: _throw_ast,
    _TRY: _try_ast,
    _VAR: _var_ast,
}


def _special_form_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a Python AST Node for any Lisp special forms."""
    assert form.first in _SPECIAL_FORMS
    handle_special_form = _SPECIAL_FORM_HANDLERS.get(form.first, None)
    if handle_special_form:
        return handle_special_form(ctx, form)
    raise CompilerException("Special form identified, but not handled") from None


def _resolve_macro_sym(ctx: CompilerContext, form: sym.Symbol) -> Optional[Var]:
    """Determine if a Basilisp symbol refers to a macro and, if so, return the
    Var it points to.

    If the symbol cannot be resolved or does not refer to a macro, then this
    function will return None. _sym_ast will generate the AST for a standard
    function call."""
    if form.ns is not None:
        if form.ns == _BUILTINS_NS:
            return None
        elif form.ns == ctx.current_ns.name:
            return ctx.current_ns.find(sym.symbol(form.name))
        ns_sym = sym.symbol(form.ns)
        if ns_sym in ctx.current_ns.imports:
            # We still import Basilisp code, so we'll want to check if
            # the symbol is referring to a Basilisp Var
            return Var.find(form)
        elif ns_sym in ctx.current_ns.aliases:
            aliased_ns = ctx.current_ns.get_alias(ns_sym)
            if aliased_ns:
                return Var.find(sym.symbol(form.name, ns=aliased_ns.name))
        return None

    return ctx.current_ns.find(form)


def _list_ast(  # pylint: disable=too-many-locals
    ctx: CompilerContext, form: llist.List
) -> ASTStream:
    """Generate a stream of Python AST nodes for a source code list.

    Being the basis of any Lisp language, Lists have a lot of special cases
    which do not apply to other forms returned from the reader.

    First and foremost, lists contain special forms which are fundamental
    pieces of the language which are defined here at the compiler level
    rather than as a macro on top of the language. Forms such as `if`,
    `do`, `def`, and `try` are defined here and must be handled specially
    to generate correct Python code.

    Lists can be quoted, which results in them not being evaluated, but
    returned as a data structure literal, but quoted elements may also
    have _unquoted_ elements contained inside of them, so provision must
    be made to evaluate some elements while leaving others un-evaluated.

    Finally, function and macro calls are also written as lists, so both
    cases must be handled herein."""
    # Empty list
    first = form.first
    if len(form) == 0:
        meta_nodes, meta = _nodes_and_exprl(_meta_kwargs_ast(ctx, form))
        yield from meta_nodes
        yield _node(
            ast.Call(func=_EMPTY_LIST_FN_NAME, args=[], keywords=_unwrap_nodes(meta))
        )
        return

    # Special forms
    if first in _SPECIAL_FORMS and not ctx.is_quoted:
        yield from _special_form_ast(ctx, form)
        return

    # Macros are immediately evaluated so the modified form can be compiled
    if isinstance(first, sym.Symbol):
        v = _resolve_macro_sym(ctx, first)
        if v is not None and _is_macro(v):
            try:
                # Call the macro as (f &form & rest)
                # In Clojure there is &env, which we don't have yet!
                expanded = v.value(form, *form.rest)

                # Verify that macroexpanded code also does not have any
                # non-tail recur forms
                try:
                    if ctx.recur_point.name:
                        _assert_recur_is_tail(ctx, lseq.sequence([expanded]))
                except IndexError:
                    pass

                yield from _to_ast(ctx, expanded)
            except Exception as e:
                raise CompilerException(
                    f"Error occurred during macroexpansion of {form}"
                ) from e
            return

        # Handle interop calls and properties generated dynamically (e.g.
        # by a macro)
        if first.name.startswith(".-"):
            assert first.ns is None, "Interop property symbols may not have a namespace"
            prop_name = sym.symbol(first.name[2:])
            target = runtime.nth(form, 1)
            yield from _interop_prop_ast(ctx, llist.l(_INTEROP_PROP, target, prop_name))
            return
        elif first.name.startswith("."):
            assert first.ns is None, "Interop call symbols may not have a namespace"
            attr_name = sym.symbol(first.name[1:])
            rest = form.rest
            target = rest.first
            args = rest.rest
            interop_form = llist.l(_INTEROP_CALL, target, attr_name, *args)
            yield from _interop_call_ast(ctx, interop_form)
            return

    elems_nodes, elems = _collection_literal_ast(ctx, form)

    # Quoted list
    if ctx.is_quoted:
        meta_nodes, meta = _nodes_and_exprl(_meta_kwargs_ast(ctx, form))
        yield from meta_nodes
        yield _node(
            ast.Call(
                func=_NEW_LIST_FN_NAME,
                args=[ast.List(elems, ast.Load())],
                keywords=_unwrap_nodes(meta),
            )
        )
        return

    yield from elems_nodes
    elems_ast = seq(elems)

    # Function call
    yield _node(
        ast.Call(func=elems_ast.first(), args=elems_ast.drop(1).to_list(), keywords=[])
    )


def _map_ast(ctx: CompilerContext, form: lmap.Map) -> ASTStream:
    key_nodes, keys = _collection_literal_ast(ctx, form.keys())
    val_nodes, vals = _collection_literal_ast(ctx, form.values())
    meta_nodes, meta = _nodes_and_exprl(_meta_kwargs_ast(ctx, form))
    yield from meta_nodes
    yield from key_nodes
    yield from val_nodes
    yield _node(
        ast.Call(
            func=_NEW_MAP_FN_NAME,
            args=[ast.Dict(keys=keys, values=vals)],
            keywords=_unwrap_nodes(meta),
        )
    )


def _set_ast(ctx: CompilerContext, form: lset.Set) -> ASTStream:
    elem_nodes, elems_ast = _collection_literal_ast(ctx, form)
    meta_nodes, meta = _nodes_and_exprl(_meta_kwargs_ast(ctx, form))
    yield from meta_nodes
    yield from elem_nodes
    yield _node(
        ast.Call(
            func=_NEW_SET_FN_NAME,
            args=[ast.List(elems_ast, ast.Load())],
            keywords=_unwrap_nodes(meta),
        )
    )


def _vec_ast(ctx: CompilerContext, form: vec.Vector) -> ASTStream:
    elem_nodes, elems_ast = _collection_literal_ast(ctx, form)
    meta_nodes, meta = _nodes_and_exprl(_meta_kwargs_ast(ctx, form))
    yield from meta_nodes
    yield from elem_nodes
    yield _node(
        ast.Call(
            func=_NEW_VEC_FN_NAME,
            args=[ast.List(elems_ast, ast.Load())],
            keywords=_unwrap_nodes(meta),
        )
    )


def _kw_ast(_: CompilerContext, form: kw.Keyword) -> ASTStream:
    kwarg = (
        Maybe(form.ns)
        .stream()
        .map(lambda ns: ast.keyword(arg="ns", value=ast.Str(form.ns)))
        .to_list()
    )
    yield _node(
        ast.Call(func=_NEW_KW_FN_NAME, args=[ast.Str(form.name)], keywords=kwarg)
    )


def _resolve_sym_var(ctx: CompilerContext, v: Var) -> Optional[str]:
    """Resolve a Basilisp var down to a Python Name (or Attribute).

    If the Var is marked as :dynamic or :redef or the compiler option
    USE_VAR_INDIRECTION is active, do not compile to a direct access.
    If the corresponding function name is not defined in a Python module,
    no direct variable access is possible and Var.find indirection must be
    used."""
    if ctx.use_var_indirection or _is_dynamic(v) or _is_redefable(v):
        return None

    safe_name = munge(v.name.name)
    defined_in_py = safe_name in v.ns.module.__dict__
    if defined_in_py:
        if ctx.current_ns is v.ns:
            return f"{safe_name}"
        else:
            safe_ns = munge(v.ns.name)
            return f"{safe_ns}.{safe_name}"

    if ctx.warn_on_var_indirection:
        logger.warning(f"could not resolve a direct link to Var '{v.name}'")
    return None


def _resolve_sym(ctx: CompilerContext, form: sym.Symbol) -> Optional[str]:  # noqa: C901
    """Resolve a Basilisp symbol down to a Python Name (or Attribute).

    If the symbol cannot be resolved or is specifically marked to prefer Var
    indirection, then this function will return None. _sym_ast will generate a
    Var.find call for runtime resolution."""
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

    # Attempt to resolve any symbol with a namespace to a direct Python call
    if form.ns is not None:
        if form.ns == _BUILTINS_NS:
            return f"{munge(form.name, allow_builtins=True)}"
        elif form.ns == ctx.current_ns.name:
            v = ctx.current_ns.find(sym.symbol(form.name))
            if v is not None:
                return _resolve_sym_var(ctx, v)
        ns_sym = sym.symbol(form.ns)
        if ns_sym in ctx.current_ns.imports or ns_sym in ctx.current_ns.import_aliases:
            # We still import Basilisp code, so we'll want to make sure
            # that the symbol isn't referring to a Basilisp Var first
            v = Var.find(form)
            if v is not None:
                return _resolve_sym_var(ctx, v)

            # Python modules imported using `import*` may be imported
            # with an alias, which sets the local name of the alias to the
            # alias value (much like Python's `from module import member`
            # statement). In this case, we'll want to check for the module
            # using its non-aliased name, but then generate code to access
            # the member using the alias.
            if ns_sym in ctx.current_ns.import_aliases:
                module_name: sym.Symbol = ctx.current_ns.import_aliases[ns_sym]
            else:
                module_name = ns_sym

            # Otherwise, try to direct-link it like a Python variable
            safe_module_name = munge(module_name.name)
            assert (
                safe_module_name in sys.modules
            ), f"Module '{safe_module_name}' is not imported"
            ns_module = sys.modules[safe_module_name]
            safe_ns = munge(form.ns)

            # Try without allowing builtins first
            safe_name = munge(form.name)
            if safe_name in ns_module.__dict__:
                return f"{safe_ns}.{safe_name}"

            # Then allow builtins
            safe_name = munge(form.name, allow_builtins=True)
            if safe_name in ns_module.__dict__:
                return f"{safe_ns}.{safe_name}"

            # If neither resolve, then defer to a Var.find
            if ctx.warn_on_var_indirection:
                logger.warning(
                    f"could not resolve a direct link to Python variable '{form}'"
                )
            return None
        elif ns_sym in ctx.current_ns.aliases:
            aliased_ns: runtime.Namespace = ctx.current_ns.aliases[ns_sym]
            v = Var.find(sym.symbol(form.name, ns=aliased_ns.name))
            if v is not None:
                return _resolve_sym_var(ctx, v)
            if ctx.warn_on_var_indirection:
                logger.warning(f"could not resolve a direct link to Var '{form}'")
            return None

    # Look up the symbol in the namespace mapping of the current namespace.
    # If we do find that mapping, then we can use a Python variable so long
    # as the module defined for this namespace has a Python function backing
    # it. We may have to use a direct namespace reference for imported functions.
    v = ctx.current_ns.find(form)
    if v is not None:
        return _resolve_sym_var(ctx, v)

    if ctx.warn_on_var_indirection:
        logger.warning(f"could not resolve a direct link to Var '{form}'")
    return None


def _sym_ast(ctx: CompilerContext, form: sym.Symbol) -> ASTStream:
    """Return a Python AST node for a Lisp symbol.

    If the symbol is quoted (as determined by the `quoted` kwarg),
    return just the raw symbol.

    If the symbol is not namespaced and the CompilerContext contains a
    SymbolTable (`sym_table` key) and the name of this symbol is
    found in that table, directly generate a Python `Name` node,
    so that we use a raw Python variable.

    Otherwise, generate code to resolve the Var value. If the symbol
    includes a namespace, use that namespace to resolve the Var. If no
    namespace is provided with the symbol, generate code to resolve the
    current namespace at the time of execution."""
    ns: Optional[ast.expr] = None
    if form.ns is None and not ctx.is_quoted:
        ns = _NS_VAR_NAME
    elif form.ns is not None:
        ns = ast.Str(form.ns)

    meta_nodes, meta = _nodes_and_exprl(_meta_kwargs_ast(ctx, form))
    yield from meta_nodes

    sym_kwargs = (
        Maybe(ns).stream().map(lambda v: ast.keyword(arg="ns", value=ns)).to_list()
    )
    sym_kwargs.extend(_unwrap_nodes(meta))
    base_sym = ast.Call(
        func=_NEW_SYM_FN_NAME, args=[ast.Str(form.name)], keywords=sym_kwargs
    )

    if ctx.is_quoted:
        yield _node(base_sym)
        return

    # Look up local symbols (function parameters, let bindings, etc.)
    sym_entry = ctx.symbol_table.find_symbol(form)
    if sym_entry is not None:
        assert (
            sym_entry.munged is not None
        ), f"Lisp symbol '{form}' not found in symbol table"
        assert (
            sym_entry.context != _SYM_CTX_LOCAL_STARRED
        ), "Direct access to varargs forbidden"

        if sym_entry.context in {_SYM_CTX_LOCAL, _SYM_CTX_RECUR}:
            ctx.symbol_table.mark_used(form)
            yield _node(ast.Name(id=sym_entry.munged, ctx=ast.Load()))
            return

    # Resolve def'ed symbols, namespace aliases, imports, etc.
    resolved = _resolve_sym(ctx, form)
    if resolved is not None:
        yield _node(_load_attr(resolved))
        return

    # If we couldn't find the symbol anywhere else, generate a Var.find call
    yield _node(
        ast.Attribute(
            value=ast.Call(func=_FIND_VAR_FN_NAME, args=[base_sym], keywords=[]),
            attr="value",
            ctx=ast.Load(),
        )
    )


def _decimal_ast(_: CompilerContext, form: Decimal) -> ASTStream:
    yield _node(
        ast.Call(func=_NEW_DECIMAL_FN_NAME, args=[ast.Str(str(form))], keywords=[])
    )


def _fraction_ast(_: CompilerContext, form: Fraction) -> ASTStream:
    yield _node(
        ast.Call(
            func=_NEW_FRACTION_FN_NAME,
            args=[ast.Num(form.numerator), ast.Num(form.denominator)],
            keywords=[],
        )
    )


def _inst_ast(_: CompilerContext, form: datetime) -> ASTStream:
    yield _node(
        ast.Call(func=_NEW_INST_FN_NAME, args=[ast.Str(form.isoformat())], keywords=[])
    )


def _regex_ast(_: CompilerContext, form: Pattern) -> ASTStream:
    yield _node(
        ast.Call(func=_NEW_REGEX_FN_NAME, args=[ast.Str(form.pattern)], keywords=[])
    )


def _uuid_ast(_: CompilerContext, form: uuid.UUID) -> ASTStream:
    yield _node(
        ast.Call(func=_NEW_UUID_FN_NAME, args=[ast.Str(str(form))], keywords=[])
    )


def _collection_ast(ctx: CompilerContext, form: Iterable[LispForm]) -> ASTStream:
    """Turn a collection of Lisp forms into Python AST nodes, filtering out
    empty nodes."""
    yield from seq(form).flat_map(lambda x: _to_ast(ctx, x)).filter(
        lambda x: x is not None
    )


def _collection_literal_ast(
    ctx: CompilerContext, form: Iterable[LispForm]
) -> Tuple[ASTStream, PyASTStream]:
    """Turn a collection literal of Lisp forms into Python AST nodes, filtering
    out empty nodes."""
    deps: List[ASTNode] = []
    nodes = []
    for f in form:
        depnodes, exprl = _nodes_and_exprl(_to_ast(ctx, f))
        deps.extend(depnodes)
        nodes.extend(_unwrap_nodes(exprl))

    return deps, nodes


def _with_loc(f: ASTProcessor) -> ASTProcessor:
    """Wrap a reader function in a decorator to supply line and column
    information along with relevant forms."""

    @functools.wraps(f)
    def with_lineno_and_col(ctx: CompilerContext, form: LispForm) -> ASTStream:
        try:
            meta = form.meta  # type: ignore
            line = meta.get(reader.READER_LINE_KW)  # type: ignore
            col = meta.get(reader.READER_COL_KW)  # type: ignore

            for astnode in f(ctx, form):
                astnode.node.lineno = line  # type: ignore
                astnode.node.col_offset = col  # type: ignore
                yield astnode
        except AttributeError:
            yield from f(ctx, form)

    return with_lineno_and_col


@_with_loc  # noqa: C901
def _to_ast(  # pylint: disable=too-many-branches
    ctx: CompilerContext, form: LispForm
) -> ASTStream:
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
    elif isinstance(form, kw.Keyword):
        yield from _kw_ast(ctx, form)
        return
    elif isinstance(form, sym.Symbol):
        yield from _sym_ast(ctx, form)
        return
    elif isinstance(form, str):
        yield _node(ast.Str(form))
        return
    elif isinstance(form, (bool, type(None))):
        yield _node(ast.NameConstant(form))
        return
    elif isinstance(form, (complex, float, int)):
        yield _node(ast.Num(form))
        return
    elif isinstance(form, datetime):
        yield from _inst_ast(ctx, form)
        return
    elif isinstance(form, Decimal):
        yield from _decimal_ast(ctx, form)
        return
    elif isinstance(form, Fraction):
        yield from _fraction_ast(ctx, form)
        return
    elif isinstance(form, uuid.UUID):
        yield from _uuid_ast(ctx, form)
        return
    elif isinstance(form, Pattern):
        yield from _regex_ast(ctx, form)
        return
    else:
        raise TypeError(f"Unexpected form type {type(form)}: {form}")


def _module_imports(ctx: CompilerContext) -> Iterable[ast.Import]:
    """Generate the Python Import AST node for importing all required
    language support modules."""
    aliases = {
        "builtins": None,
        "basilisp.lang.keyword": _KW_ALIAS,
        "basilisp.lang.list": _LIST_ALIAS,
        "basilisp.lang.map": _MAP_ALIAS,
        "basilisp.lang.runtime": _RUNTIME_ALIAS,
        "basilisp.lang.set": _SET_ALIAS,
        "basilisp.lang.symbol": _SYM_ALIAS,
        "basilisp.lang.vector": _VEC_ALIAS,
        "basilisp.lang.util": _UTIL_ALIAS,
    }
    return (
        seq(ctx.imports)
        .map(lambda entry: entry.key.name)
        .map(lambda name: (name, aliases.get(name, None)))
        .map(lambda t: ast.Import(names=[ast.alias(name=t[0], asname=t[1])]))
        .to_list()
    )


def _from_module_import() -> ast.ImportFrom:
    """Generate the Python From ... Import AST node for importing
    language support modules."""
    return ast.ImportFrom(
        module="basilisp.lang.runtime",
        names=[ast.alias(name="Var", asname=_VAR_ALIAS)],
        level=0,
    )


def _ns_var(
    py_ns_var: str = _NS_VAR,
    lisp_ns_var: str = _LISP_NS_VAR,
    lisp_ns_ns: str = _CORE_NS,
) -> ast.Assign:
    """Assign a Python variable named `ns_var` to the value of the current
    namespace."""
    return ast.Assign(
        targets=[ast.Name(id=py_ns_var, ctx=ast.Store())],
        value=ast.Call(
            func=_FIND_VAR_FN_NAME,
            args=[
                ast.Call(
                    func=_NEW_SYM_FN_NAME,
                    args=[ast.Str(lisp_ns_var)],
                    keywords=[ast.keyword(arg="ns", value=ast.Str(lisp_ns_ns))],
                )
            ],
            keywords=[],
        ),
    )


def to_py_source(t: ast.AST, outfile: str) -> None:
    source = codegen.to_source(t)
    with open(outfile, mode="w") as f:
        f.writelines(source)


def to_py_str(t: ast.AST) -> str:
    """Return a string of the Python code which would generate the input
    AST node."""
    return codegen.to_source(t)


BytecodeCollector = Optional[Callable[[types.CodeType], None]]


def compile_and_exec_form(  # pylint: disable= too-many-arguments
    form: LispForm,
    ctx: CompilerContext,
    module: types.ModuleType,
    source_filename: str = "<REPL Input>",
    wrapped_fn_name: str = _DEFAULT_FN,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> Any:
    """Compile and execute the given form. This function will be most useful
    for the REPL and testing purposes. Returns the result of the executed expression.

    Callers may override the wrapped function name, which is used by the
    REPL to evaluate the result of an expression and print it back out."""
    if form is None:
        return None

    if not module.__basilisp_bootstrapped__:  # type: ignore
        _bootstrap_module(ctx, module, source_filename)

    form_ast = seq(_to_ast(ctx, form)).map(_unwrap_node).to_list()
    if form_ast is None:
        return None

    # Split AST nodes into into inits, last group. Only expressionize the last
    # component, and inject the rest of the nodes directly into the module.
    # This will alow the REPL to take advantage of direct Python variable access
    # rather than using Var.find indrection.
    final_wrapped_name = genname(wrapped_fn_name)
    body = _expressionize([form_ast[-1]], final_wrapped_name)
    form_ast = list(
        itertools.chain(map(_statementize, form_ast[:-1]), [body])  # type: ignore
    )

    ast_module = ast.Module(body=form_ast)
    ast.fix_missing_locations(ast_module)

    if runtime.print_generated_python():
        print(to_py_str(ast_module))
    else:
        runtime.add_generated_python(to_py_str(ast_module))

    bytecode = compile(ast_module, source_filename, "exec")
    if collect_bytecode:
        collect_bytecode(bytecode)
    exec(bytecode, module.__dict__)
    return getattr(module, final_wrapped_name)()


def _incremental_compile_module(
    nodes: MixedNodeStream,
    mod: types.ModuleType,
    source_filename: str,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> None:
    """Incrementally compile a stream of AST nodes in module mod.

    The source_filename will be passed to Python's native compile.

    Incremental compilation is an integral part of generating a Python module
    during the same process as macro-expansion."""
    module_body = map(_statementize, _unwrap_nodes(nodes))

    module = ast.Module(body=list(module_body))
    ast.fix_missing_locations(module)

    if runtime.print_generated_python():
        print(to_py_str(module))
    else:
        runtime.add_generated_python(to_py_str(module))

    bytecode = compile(module, source_filename, "exec")
    if collect_bytecode:
        collect_bytecode(bytecode)
    exec(bytecode, mod.__dict__)


def _bootstrap_module(
    ctx: CompilerContext,
    mod: types.ModuleType,
    source_filename: str,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> None:
    """Bootstrap a new module with imports and other boilerplate."""
    preamble: List[ast.AST] = []
    preamble.extend(_module_imports(ctx))
    preamble.append(_from_module_import())
    preamble.append(_ns_var())

    _incremental_compile_module(
        preamble,
        mod,
        source_filename=source_filename,
        collect_bytecode=collect_bytecode,
    )
    mod.__basilisp_bootstrapped__ = True  # type: ignore


def compile_module(
    forms: Iterable[LispForm],
    ctx: CompilerContext,
    module: types.ModuleType,
    source_filename: str,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> None:
    """Compile an entire Basilisp module into Python bytecode which can be
    executed as a Python module.

    This function is designed to generate bytecode which can be used for the
    Basilisp import machinery, to allow callers to import Basilisp modules from
    Python code.
    """
    _bootstrap_module(ctx, module, source_filename)

    for form in forms:
        nodes = [node for node in _to_ast(ctx, form)]
        _incremental_compile_module(
            nodes,
            module,
            source_filename=source_filename,
            collect_bytecode=collect_bytecode,
        )


def compile_bytecode(
    code: List[types.CodeType],
    ctx: CompilerContext,
    module: types.ModuleType,
    source_filename: str,
) -> None:
    """Compile cached bytecode into the given module.

    The Basilisp import hook attempts to cache bytecode while compiling Basilisp
    namespaces. When the cached bytecode is reloaded from disk, it needs to be
    compiled within a bootstrapped module. This function bootstraps the module
    and then proceeds to compile a collection of bytecodes into the module."""
    _bootstrap_module(ctx, module, source_filename)
    for bytecode in code:
        exec(bytecode, module.__dict__)


lrepr = runtime.lrepr
