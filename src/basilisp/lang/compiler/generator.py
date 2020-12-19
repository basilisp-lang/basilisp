import collections
import contextlib
import functools
import logging
import re
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from functools import partial, wraps
from itertools import chain
from typing import (
    Callable,
    Collection,
    Deque,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Pattern,
    Tuple,
    Union,
    cast,
)

import attr

from basilisp import _pyast as ast
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
    DEFAULT_COMPILER_FILE_PATH,
    SYM_DYNAMIC_META_KEY,
    SYM_NO_WARN_ON_REDEF_META_KEY,
    SYM_REDEF_META_KEY,
    VAR_IS_PROTOCOL_META_KEY,
)
from basilisp.lang.compiler.exception import CompilerException, CompilerPhase
from basilisp.lang.compiler.nodes import (
    Await,
    Binding,
    Catch,
    Const,
    Def,
    DefType,
    DefTypeBase,
    DefTypeClassMethod,
    DefTypeMember,
    DefTypeMethod,
    DefTypeMethodArity,
    DefTypeProperty,
    DefTypeStaticMethod,
    Do,
    Fn,
    FnArity,
    HostCall,
    HostField,
    If,
    Import,
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
from basilisp.lang.compiler.nodes import Quote, ReaderLispForm, Recur, Reify, Require
from basilisp.lang.compiler.nodes import Set as SetNode
from basilisp.lang.compiler.nodes import SetBang, Throw, Try, VarRef
from basilisp.lang.compiler.nodes import Vector as VectorNode
from basilisp.lang.compiler.nodes import WithMeta
from basilisp.lang.interfaces import IMeta, IRecord, ISeq, ISeqable, IType
from basilisp.lang.runtime import CORE_NS
from basilisp.lang.runtime import NS_VAR_NAME as LISP_NS_VAR
from basilisp.lang.runtime import BasilispModule, Var
from basilisp.lang.typing import CompilerOpts, LispForm
from basilisp.lang.util import count, genname, munge
from basilisp.util import Maybe

# Generator logging
logger = logging.getLogger(__name__)

# Generator options
USE_VAR_INDIRECTION = kw.keyword("use-var-indirection")
WARN_ON_VAR_INDIRECTION = kw.keyword("warn-on-var-indirection")

# String constants used in generating code
_DEFAULT_FN = "__lisp_expr__"
_DO_PREFIX = "lisp_do"
_FN_PREFIX = "lisp_fn"
_IF_PREFIX = "lisp_if"
_IF_RESULT_PREFIX = "if_result"
_IF_TEST_PREFIX = "if_test"
_LET_RESULT_PREFIX = "let_result"
_LETFN_RESULT_PREFIX = "letfn_result"
_LOOP_RESULT_PREFIX = "loop_result"
_MULTI_ARITY_ARG_NAME = "multi_arity_args"
_SET_BANG_TEMP_PREFIX = "set_bang_val"
_THROW_PREFIX = "lisp_throw"
_TRY_PREFIX = "lisp_try"
_NS_VAR = "_NS"

# Keyword constants used in generating code
_INTERFACE_KW = kw.keyword("interface")
_REST_KW = kw.keyword("rest")


GeneratorException = partial(CompilerException, phase=CompilerPhase.CODE_GENERATION)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SymbolTableEntry:
    context: LocalType
    munged: str
    symbol: sym.Symbol


# pylint: disable=unsupported-membership-test,unsupported-delete-operation,unsupported-assignment-operation
@attr.s(auto_attribs=True, slots=True)
class SymbolTable:
    name: str
    _is_context_boundary: bool = False
    _parent: Optional["SymbolTable"] = None
    _table: MutableMapping[sym.Symbol, SymbolTableEntry] = attr.ib(factory=dict)

    def new_symbol(self, s: sym.Symbol, munged: str, ctx: LocalType) -> "SymbolTable":
        if s in self._table:
            self._table[s] = attr.evolve(
                self._table[s], context=ctx, munged=munged, symbol=s
            )
        else:
            self._table[s] = SymbolTableEntry(ctx, munged, s)
        return self

    def find_symbol(self, s: sym.Symbol) -> Optional[SymbolTableEntry]:
        if s in self._table:
            return self._table[s]
        if self._parent is None:
            return None
        return self._parent.find_symbol(s)

    @contextlib.contextmanager
    def new_frame(self, name: str, is_context_boundary: bool):
        """Context manager for creating a new stack frame."""
        yield SymbolTable(name, is_context_boundary=is_context_boundary, parent=self)

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


class RecurType(Enum):
    FN = kw.keyword("fn")
    METHOD = kw.keyword("method")
    LOOP = kw.keyword("loop")


@attr.s(auto_attribs=True, slots=True)
class RecurPoint:
    loop_id: str
    type: RecurType
    binding_names: Optional[Collection[str]] = None
    is_variadic: Optional[bool] = None
    has_recur: bool = False


class GeneratorContext:
    __slots__ = (
        "_filename",
        "_opts",
        "_recur_points",
        "_st",
        "_this",
        "_var_indirection_override",
    )

    def __init__(
        self, filename: Optional[str] = None, opts: Optional[CompilerOpts] = None
    ) -> None:
        self._filename = Maybe(filename).or_else_get(DEFAULT_COMPILER_FILE_PATH)
        self._opts = Maybe(opts).map(lmap.map).or_else_get(lmap.m())  # type: ignore
        self._recur_points: Deque[RecurPoint] = collections.deque([])
        self._st = collections.deque([SymbolTable("<Top>", is_context_boundary=True)])
        self._this: Deque[sym.Symbol] = collections.deque([])
        self._var_indirection_override: Deque[bool] = collections.deque([])

        if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
            for k, v in self._opts.items():
                logger.debug("Compiler option %s = %s", k, v)

    @property
    def current_ns(self) -> runtime.Namespace:
        return runtime.get_current_ns()

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def use_var_indirection(self) -> bool:
        """If True, compile all variable references using Var.find indirection."""
        return self._opts.val_at(USE_VAR_INDIRECTION, False)

    @property
    def warn_on_var_indirection(self) -> bool:
        """If True, warn when a Var reference cannot be direct linked (iff
        use_var_indirection is False).."""
        return not self.use_var_indirection and self._opts.val_at(
            WARN_ON_VAR_INDIRECTION, True
        )

    @contextlib.contextmanager
    def with_var_indirection_override(self, has_override: bool = True):
        self._var_indirection_override.append(has_override)
        yield
        self._var_indirection_override.pop()

    @property
    def has_var_indirection_override(self) -> bool:
        try:
            return self._var_indirection_override[-1]
        except IndexError:
            return False

    @property
    def recur_point(self):
        return self._recur_points[-1]

    @contextlib.contextmanager
    def new_recur_point(
        self,
        loop_id: str,
        type_: RecurType,
        binding_names: Optional[Collection[str]] = None,
        is_variadic: Optional[bool] = None,
    ):
        self._recur_points.append(
            RecurPoint(
                loop_id, type_, binding_names=binding_names, is_variadic=is_variadic
            )
        )
        yield
        self._recur_points.pop()

    @property
    def symbol_table(self) -> SymbolTable:
        return self._st[-1]

    @contextlib.contextmanager
    def new_symbol_table(self, name: str, is_context_boundary: bool = False):
        old_st = self.symbol_table
        with old_st.new_frame(name, is_context_boundary) as st:
            self._st.append(st)
            yield st
            self._st.pop()

    @property
    def current_this(self):
        return self._this[-1]

    @contextlib.contextmanager
    def new_this(self, this: sym.Symbol):
        self._this.append(this)
        yield
        self._this.pop()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GeneratedPyAST:
    node: ast.AST
    dependencies: Iterable[ast.AST] = ()

    @staticmethod
    def reduce(*genned: "GeneratedPyAST") -> "GeneratedPyAST":
        deps: List[ast.AST] = []
        for n in genned:
            deps.extend(n.dependencies)
            deps.append(n.node)

        return GeneratedPyAST(node=deps[-1], dependencies=deps[:-1])


PyASTStream = Iterable[ast.AST]
SimplePyASTGenerator = Callable[[GeneratorContext, ReaderLispForm], GeneratedPyAST]
PyASTGenerator = Callable[[GeneratorContext, Node], GeneratedPyAST]


####################
# Private Utilities
####################


def _chain_py_ast(
    *genned: GeneratedPyAST,
) -> Tuple[PyASTStream, PyASTStream]:
    """Chain a sequence of generated Python ASTs into a tuple of dependency nodes"""
    deps = chain.from_iterable(map(lambda n: n.dependencies, genned))
    nodes = map(lambda n: n.node, genned)
    return deps, nodes


def _load_attr(name: str, ctx: ast.AST = ast.Load()) -> ast.Attribute:
    """Generate recursive Python Attribute AST nodes for resolving nested
    names."""
    attrs = name.split(".")

    def attr_node(node, idx):
        if idx >= len(attrs):
            node.ctx = ctx
            return node
        return attr_node(
            ast.Attribute(value=node, attr=attrs[idx], ctx=ast.Load()), idx + 1
        )

    return attr_node(ast.Name(id=attrs[0], ctx=ast.Load()), 1)


def _simple_ast_generator(gen_ast):
    """Wrap simpler AST generators to return a GeneratedPyAST."""

    @wraps(gen_ast)
    def wrapped_ast_generator(ctx: GeneratorContext, form: LispForm) -> GeneratedPyAST:
        return GeneratedPyAST(node=gen_ast(ctx, form))

    return wrapped_ast_generator


def _collection_ast(
    ctx: GeneratorContext, form: Iterable[Node]
) -> Tuple[PyASTStream, PyASTStream]:
    """Turn a collection of Lisp forms into Python AST nodes."""
    return _chain_py_ast(*map(partial(gen_py_ast, ctx), form))


# `attrs` throws an a ValueError for certain small class definitions, so
# we want to avoid that.
#
# https://github.com/python-attrs/attrs/issues/589
_ATTR_SLOTS_ON = getattr(attr, "__version_info__", (0,)) >= (19, 4)

_ATTR_CMP_OFF = getattr(attr, "__version_info__", (0,)) >= (19, 2)
_ATTR_CMP_KWARGS = (
    [
        ast.keyword(arg="eq", value=ast.Constant(False)),
        ast.keyword(arg="order", value=ast.Constant(False)),
    ]
    if _ATTR_CMP_OFF
    else [ast.keyword(arg="cmp", value=ast.Constant(False))]
)


def _class_ast(  # pylint: disable=too-many-arguments
    class_name: str,
    body: List[ast.AST],
    bases: Iterable[ast.AST] = (),
    fields: Iterable[str] = (),
    members: Iterable[str] = (),
    verified_abstract: bool = False,
    artificially_abstract_bases: Iterable[ast.AST] = (),
    is_frozen: bool = True,
    use_slots: bool = _ATTR_SLOTS_ON,
) -> ast.ClassDef:
    """Return a Python class definition for `deftype` and `reify` special forms."""
    return ast.ClassDef(
        name=class_name,
        bases=bases,
        keywords=[],
        body=body,
        decorator_list=list(
            chain(
                []
                if verified_abstract
                else [
                    ast.Call(
                        func=_BASILISP_TYPE_FN_NAME,
                        args=[],
                        keywords=[
                            ast.keyword(
                                arg="fields",
                                value=ast.Tuple(
                                    elts=[ast.Constant(e) for e in fields],
                                    ctx=ast.Load(),
                                ),
                            ),
                            ast.keyword(
                                arg="interfaces",
                                value=ast.Tuple(elts=list(bases), ctx=ast.Load()),
                            ),
                            ast.keyword(
                                arg="artificially_abstract_bases",
                                value=ast.Set(elts=list(artificially_abstract_bases)),
                            ),
                            ast.keyword(
                                arg="members",
                                value=ast.Tuple(
                                    elts=[ast.Constant(e) for e in members],
                                    ctx=ast.Load(),
                                ),
                            ),
                        ],
                    )
                ],
                [
                    ast.Call(
                        func=_ATTR_CLASS_DECORATOR_NAME,
                        args=[],
                        keywords=_ATTR_CMP_KWARGS
                        + [
                            ast.keyword(arg="frozen", value=ast.Constant(is_frozen)),
                            ast.keyword(arg="slots", value=ast.Constant(use_slots)),
                        ],
                    ),
                ],
            )
        ),
    )


def _kwargs_ast(
    ctx: GeneratorContext,
    kwargs: KeywordArgs,
) -> Tuple[PyASTStream, PyASTStream]:
    """Return a tuple of dependency nodes and Python `ast.keyword` nodes from a
    Basilisp `KeywordArgs` Node property."""
    kwargs_keys, kwargs_nodes = [], []
    kwargs_deps: List[ast.AST] = []
    for k, v in kwargs.items():
        kwargs_keys.append(k)
        kwarg_ast = gen_py_ast(ctx, v)
        kwargs_nodes.append(kwarg_ast.node)
        kwargs_deps.extend(kwarg_ast.dependencies)
    return (
        kwargs_deps,
        [ast.keyword(arg=k, value=v) for k, v in zip(kwargs_keys, kwargs_nodes)],
    )


def _clean_meta(form: IMeta) -> LispForm:
    """Remove reader metadata from the form's meta map."""
    assert form.meta is not None, "Form must have non-null 'meta' attribute"
    meta = form.meta.dissoc(reader.READER_LINE_KW, reader.READER_COL_KW)
    if len(meta) == 0:
        return None
    return cast(lmap.PersistentMap, meta)


def _ast_with_loc(
    py_ast: GeneratedPyAST, env: NodeEnv, include_dependencies: bool = False
) -> GeneratedPyAST:
    """Hydrate Generated Python AST nodes with line numbers and column offsets
    if they exist in the node environment."""
    if env.line is not None:
        py_ast.node.lineno = env.line

        if include_dependencies:
            for dep in py_ast.dependencies:
                dep.lineno = env.line

    if env.col is not None:
        py_ast.node.col_offset = env.col

        if include_dependencies:
            for dep in py_ast.dependencies:
                dep.col_offset = env.col

    return py_ast


def _with_ast_loc(f):
    """Wrap a generator function in a decorator to supply line and column
    information to the returned Python AST node. Dependency nodes will not
    be hydrated, functions whose returns need dependency nodes to be
    hydrated should use `_with_ast_loc_deps` below."""

    @wraps(f)
    def with_lineno_and_col(
        ctx: GeneratorContext, node: Node, *args, **kwargs
    ) -> GeneratedPyAST:
        py_ast = f(ctx, node, *args, **kwargs)
        return _ast_with_loc(py_ast, node.env)

    return with_lineno_and_col


def _with_ast_loc_deps(f):
    """Wrap a generator function in a decorator to supply line and column
    information to the returned Python AST node and dependency nodes.

    Dependency nodes should likely only be included if they are new nodes
    created in the same function wrapped by this function. Otherwise, dependencies
    returned from e.g. calling `gen_py_ast` should be assumed to already have
    their location information hydrated."""

    @wraps(f)
    def with_lineno_and_col(
        ctx: GeneratorContext, node: Node, *args, **kwargs
    ) -> GeneratedPyAST:
        py_ast = f(ctx, node, *args, **kwargs)
        return _ast_with_loc(py_ast, node.env, include_dependencies=True)

    return with_lineno_and_col


def _is_dynamic(v: Var) -> bool:
    """Return True if the Var holds a value which should be compiled to a dynamic
    Var access."""
    return (
        Maybe(v.meta)
        .map(lambda m: m.get(SYM_DYNAMIC_META_KEY, None))
        .or_else_get(False)
    )


def _is_redefable(v: Var) -> bool:
    """Return True if the Var can be redefined."""
    return (
        Maybe(v.meta).map(lambda m: m.get(SYM_REDEF_META_KEY, None)).or_else_get(False)
    )


def _noop_node() -> ast.AST:
    """Return a Constant node containing the expression `None`.

    The optimizer filters out constant expressions in the AST as standalone
    statements, so for generators which generate code for statement nodes,
    for example, emitting a final `None` expression is a way to effectively
    skirt the requirement for returning an expression node from every
    generator."""
    return ast.Constant(None)


def _var_ns_as_python_sym(name: str) -> str:
    """Return a Var namespace as a valid Python symbol."""
    return munge(name.replace(".", "_"))


#######################
# Aliases & Attributes
#######################


_ATOM_ALIAS = genname("atom")
_COMPILER_ALIAS = genname("compiler")
_CORE_ALIAS = genname("core")
_DELAY_ALIAS = genname("delay")
_EXC_ALIAS = genname("exc")
_FUTURES_ALIAS = genname("futures")
_INTERFACES_ALIAS = genname("interfaces")
_KW_ALIAS = genname("kw")
_LIST_ALIAS = genname("llist")
_MAP_ALIAS = genname("lmap")
_MULTIFN_ALIAS = genname("multifn")
_PROMISE_ALIAS = genname("promise")
_QUEUE_ALIAS = genname("queue")
_READER_ALIAS = genname("reader")
_REDUCED_ALIAS = genname("reduced")
_RUNTIME_ALIAS = genname("runtime")
_SEQ_ALIAS = genname("seq")
_SET_ALIAS = genname("lset")
_SYM_ALIAS = genname("sym")
_VEC_ALIAS = genname("vec")
_VOLATILE_ALIAS = genname("volatile")
_VAR_ALIAS = genname("Var")
_UTIL_ALIAS = genname("langutil")

_MODULE_ALIASES = {
    "basilisp.lang.atom": _ATOM_ALIAS,
    "basilisp.lang.compiler": _COMPILER_ALIAS,
    "basilisp.core": _CORE_ALIAS,
    "basilisp.lang.delay": _DELAY_ALIAS,
    "basilisp.lang.exception": _EXC_ALIAS,
    "basilisp.lang.futures": _FUTURES_ALIAS,
    "basilisp.lang.interfaces": _INTERFACES_ALIAS,
    "basilisp.lang.keyword": _KW_ALIAS,
    "basilisp.lang.list": _LIST_ALIAS,
    "basilisp.lang.map": _MAP_ALIAS,
    "basilisp.lang.multifn": _MULTIFN_ALIAS,
    "basilisp.lang.promise": _PROMISE_ALIAS,
    "basilisp.lang.queue": _QUEUE_ALIAS,
    "basilisp.lang.reader": _READER_ALIAS,
    "basilisp.lang.reduced": _REDUCED_ALIAS,
    "basilisp.lang.runtime": _RUNTIME_ALIAS,
    "basilisp.lang.seq": _SEQ_ALIAS,
    "basilisp.lang.set": _SET_ALIAS,
    "basilisp.lang.symbol": _SYM_ALIAS,
    "basilisp.lang.vector": _VEC_ALIAS,
    "basilisp.lang.volatile": _VOLATILE_ALIAS,
    "basilisp.lang.util": _UTIL_ALIAS,
}

_NS_VAR_VALUE = f"{_NS_VAR}.value"

_NS_VAR_NAME = _load_attr(f"{_NS_VAR_VALUE}.name")
_NEW_DECIMAL_FN_NAME = _load_attr(f"{_UTIL_ALIAS}.decimal_from_str")
_NEW_FRACTION_FN_NAME = _load_attr(f"{_UTIL_ALIAS}.fraction")
_NEW_INST_FN_NAME = _load_attr(f"{_UTIL_ALIAS}.inst_from_str")
_NEW_KW_FN_NAME = _load_attr(f"{_KW_ALIAS}.keyword_from_hash")
_NEW_LIST_FN_NAME = _load_attr(f"{_LIST_ALIAS}.list")
_EMPTY_LIST_FN_NAME = _load_attr(f"{_LIST_ALIAS}.List.empty")
_NEW_QUEUE_FN_NAME = _load_attr(f"{_QUEUE_ALIAS}.queue")
_NEW_MAP_FN_NAME = _load_attr(f"{_MAP_ALIAS}.map")
_NEW_REGEX_FN_NAME = _load_attr(f"{_UTIL_ALIAS}.regex_from_str")
_NEW_SET_FN_NAME = _load_attr(f"{_SET_ALIAS}.set")
_NEW_SYM_FN_NAME = _load_attr(f"{_SYM_ALIAS}.symbol")
_NEW_UUID_FN_NAME = _load_attr(f"{_UTIL_ALIAS}.uuid_from_str")
_NEW_VEC_FN_NAME = _load_attr(f"{_VEC_ALIAS}.vector")
_INTERN_VAR_FN_NAME = _load_attr(f"{_VAR_ALIAS}.intern")
_INTERN_UNBOUND_VAR_FN_NAME = _load_attr(f"{_VAR_ALIAS}.intern_unbound")
_FIND_VAR_FN_NAME = _load_attr(f"{_VAR_ALIAS}.find_safe")
_ATTR_CLASS_DECORATOR_NAME = _load_attr("attr.s")
_ATTRIB_FIELD_FN_NAME = _load_attr("attr.ib")
_COLLECT_ARGS_FN_NAME = _load_attr(f"{_RUNTIME_ALIAS}._collect_args")
_COERCE_SEQ_FN_NAME = _load_attr(f"{_RUNTIME_ALIAS}.to_seq")
_BASILISP_FN_FN_NAME = _load_attr(f"{_RUNTIME_ALIAS}._basilisp_fn")
_FN_WITH_ATTRS_FN_NAME = _load_attr(f"{_RUNTIME_ALIAS}._with_attrs")
_BASILISP_TYPE_FN_NAME = _load_attr(f"{_RUNTIME_ALIAS}._basilisp_type")
_BASILISP_WITH_META_INTERFACE_NAME = _load_attr(f"{_INTERFACES_ALIAS}.IWithMeta")
_BUILTINS_IMPORT_FN_NAME = _load_attr("builtins.__import__")
_IMPORTLIB_IMPORT_MODULE_FN_NAME = _load_attr("importlib.import_module")
_LISP_FN_APPLY_KWARGS_FN_NAME = _load_attr(f"{_RUNTIME_ALIAS}._lisp_fn_apply_kwargs")
_LISP_FN_COLLECT_KWARGS_FN_NAME = _load_attr(
    f"{_RUNTIME_ALIAS}._lisp_fn_collect_kwargs"
)
_PY_CLASSMETHOD_FN_NAME = _load_attr("classmethod")
_PY_PROPERTY_FN_NAME = _load_attr("property")
_PY_STATICMETHOD_FN_NAME = _load_attr("staticmethod")
_TRAMPOLINE_FN_NAME = _load_attr(f"{_RUNTIME_ALIAS}._trampoline")
_TRAMPOLINE_ARGS_FN_NAME = _load_attr(f"{_RUNTIME_ALIAS}._TrampolineArgs")


###################
# Public Utilities
###################


def statementize(e: ast.AST) -> ast.AST:
    """Transform non-statements into ast.Expr nodes so they can
    stand alone as statements."""
    # noinspection PyPep8
    if isinstance(
        e,
        (
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.Expr,
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
            ast.Delete,
        ),
    ):
        return e
    return ast.Expr(value=e)


def expressionize(
    body: GeneratedPyAST,
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
    body_nodes: List[ast.AST] = list(map(statementize, body.dependencies))
    body_nodes.append(ast.Return(value=body.node))

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


#################
# Special Forms
#################


@_with_ast_loc_deps
def _await_to_py_ast(ctx: GeneratorContext, node: Await) -> GeneratedPyAST:
    assert node.op == NodeOp.AWAIT
    expr_ast = gen_py_ast(ctx, node.expr)
    return GeneratedPyAST(
        node=ast.Await(value=expr_ast.node), dependencies=expr_ast.dependencies
    )


def __should_warn_on_redef(
    ctx: GeneratorContext,
    defsym: sym.Symbol,
    safe_name: str,
    def_meta: lmap.PersistentMap,
) -> bool:
    """Return True if the compiler should emit a warning about this name being redefined."""
    no_warn_on_redef = def_meta.val_at(SYM_NO_WARN_ON_REDEF_META_KEY, False)
    if no_warn_on_redef:
        return False

    current_ns = ctx.current_ns
    if defsym in current_ns.interns:
        var = current_ns.find(defsym)
        assert var is not None, f"Var {defsym} cannot be none here"

        if var.meta is not None and var.meta.val_at(SYM_REDEF_META_KEY):
            return False
        else:
            return bool(var.is_bound)
    else:
        return safe_name in current_ns.module.__dict__


@_with_ast_loc
def _def_to_py_ast(  # pylint: disable=too-many-branches
    ctx: GeneratorContext, node: Def
) -> GeneratedPyAST:
    """Return a Python AST Node for a `def` expression."""
    assert node.op == NodeOp.DEF

    defsym = node.name
    ns_name = _load_attr(_NS_VAR_VALUE)
    def_name = ast.Call(
        func=_NEW_SYM_FN_NAME, args=[ast.Constant(defsym.name)], keywords=[]
    )
    safe_name = munge(defsym.name)

    assert node.meta is not None, "Meta should always be attached to Def nodes"
    def_meta = node.meta.form
    assert isinstance(def_meta, lmap.PersistentMap), "Meta should always be a map"

    # If the Var is marked as dynamic, we need to generate a keyword argument
    # for the generated Python code to set the Var as dynamic
    is_dynamic = def_meta.val_at(SYM_DYNAMIC_META_KEY, False)
    dynamic_kwarg = (
        [ast.keyword(arg="dynamic", value=ast.Constant(is_dynamic))]
        if is_dynamic
        else []
    )

    # Warn if this symbol is potentially being redefined (if the Var was
    # previously bound)
    if node.var.is_bound and __should_warn_on_redef(ctx, defsym, safe_name, def_meta):
        logger.warning(
            f"redefining local Python name '{safe_name}' in module "
            f"'{ctx.current_ns.module.__name__}' ({node.env.ns}:{node.env.line})"
        )

    if node.init is not None:
        # Since Python function definitions always take the form `def name(...):`,
        # it is redundant to assign them to the their final name after they have
        # been defined under a private alias. This codepath generates `defn`
        # declarations by directly generating the Python `def` with the correct
        # function name and short-circuiting the default double-declaration.
        assert node.init is not None  # silence MyPy
        if node.init.op == NodeOp.FN:
            assert isinstance(node.init, Fn)
            def_ast = _fn_to_py_ast(  # type: ignore[call-arg]
                ctx, node.init, def_name=defsym.name
            )
            is_defn = True
        elif (
            node.init.op == NodeOp.WITH_META
            and isinstance(node.init, WithMeta)
            and node.init.expr.op == NodeOp.FN
        ):
            assert isinstance(node.init, WithMeta)
            def_ast = _with_meta_to_py_ast(ctx, node.init, def_name=defsym.name)
            is_defn = True
        else:
            def_ast = gen_py_ast(ctx, node.init)
            is_defn = False

        func = _INTERN_VAR_FN_NAME
        extra_args = [ast.Name(id=safe_name, ctx=ast.Load())]
        if is_defn:
            # For defn style def generation, we specifically need to generate
            # the global declaration prior to emitting the Python `def` otherwise
            # the Python compiler will throw an exception during compilation
            # complaining that we assign the value prior to global declaration.
            def_dependencies = list(
                chain(
                    [ast.Global(names=[safe_name])]
                    if node.env.func_ctx is not None
                    else [],
                    def_ast.dependencies,
                )
            )
        else:
            def_dependencies = list(
                chain(
                    def_ast.dependencies,
                    [ast.Global(names=[safe_name])]
                    if node.env.func_ctx is not None
                    else [],
                    [
                        ast.Assign(
                            targets=[ast.Name(id=safe_name, ctx=ast.Store())],
                            value=def_ast.node,
                        )
                    ],
                )
            )
    else:
        # Callers can either `(def v)` to declare an unbound Var or they
        # can redef a bound Var with no init value to fetch a reference
        # to the var. Re-def-ing previously bound Vars without providing
        # a value is essentially a no-op, which should not modify the Var
        # root.
        func = _INTERN_UNBOUND_VAR_FN_NAME
        extra_args = []
        def_dependencies = (
            [ast.Global(names=[safe_name])] if node.env.func_ctx is not None else []
        )

    meta_ast = gen_py_ast(ctx, node.meta)

    return GeneratedPyAST(
        node=ast.Call(
            func=func,
            args=list(chain([ns_name, def_name], extra_args)),
            keywords=list(
                chain(
                    dynamic_kwarg,
                    []
                    if meta_ast is None
                    else [ast.keyword(arg="meta", value=meta_ast.node)],
                )
            ),
        ),
        dependencies=list(
            chain(
                def_dependencies,
                [] if meta_ast is None else meta_ast.dependencies,
            )
        ),
    )


@_with_ast_loc
def __deftype_classmethod_to_py_ast(
    ctx: GeneratorContext,
    node: DefTypeClassMethod,
) -> GeneratedPyAST:
    """Return a Python AST Node for a `deftype*` or `reify*` classmethod."""
    assert node.op == NodeOp.DEFTYPE_CLASSMETHOD

    with ctx.new_symbol_table(node.name, is_context_boundary=True):
        class_name = genname(munge(node.class_local.name))
        class_sym = sym.symbol(node.class_local.name)
        ctx.symbol_table.new_symbol(class_sym, class_name, LocalType.ARG)

        fn_args, varg, fn_body_ast = __fn_args_to_py_ast(ctx, node.params, node.body)
        return GeneratedPyAST(
            node=ast.FunctionDef(
                name=munge(node.name),
                args=ast.arguments(
                    args=list(
                        chain((ast.arg(arg=class_name, annotation=None),), fn_args)
                    ),
                    kwarg=None,
                    vararg=varg,
                    kwonlyargs=[],
                    defaults=[],
                    kw_defaults=[],
                ),
                body=fn_body_ast,
                decorator_list=list(
                    chain([_PY_CLASSMETHOD_FN_NAME], __kwargs_support_decorator(node))
                ),
                returns=None,
            )
        )


@_with_ast_loc
def __deftype_property_to_py_ast(
    ctx: GeneratorContext,
    node: DefTypeProperty,
) -> GeneratedPyAST:
    assert node.op == NodeOp.DEFTYPE_PROPERTY
    method_name = munge(node.name)

    with ctx.new_symbol_table(node.name, is_context_boundary=True):
        this_name = genname(munge(node.this_local.name))
        this_sym = sym.symbol(node.this_local.name)
        ctx.symbol_table.new_symbol(this_sym, this_name, LocalType.THIS)

        with ctx.new_this(this_sym):
            fn_args, varg, fn_body_ast = __fn_args_to_py_ast(
                ctx, node.params, node.body
            )
            return GeneratedPyAST(
                node=ast.FunctionDef(
                    name=method_name,
                    args=ast.arguments(
                        args=list(
                            chain([ast.arg(arg=this_name, annotation=None)], fn_args)
                        ),
                        kwarg=None,
                        vararg=varg,
                        kwonlyargs=[],
                        defaults=[],
                        kw_defaults=[],
                    ),
                    body=fn_body_ast,
                    decorator_list=[_PY_PROPERTY_FN_NAME],
                    returns=None,
                )
            )


def __multi_arity_deftype_dispatch_method(  # pylint: disable=too-many-arguments,too-many-locals
    name: str,
    arity_map: Mapping[int, str],
    default_name: Optional[str] = None,
    max_fixed_arity: Optional[int] = None,
) -> GeneratedPyAST:
    """Return the Python AST nodes for an argument-length dispatch method for
    multi-arity `deftype*` or `reify*` methods.

    The `arity_map` names the mapping of number of arguments to the munged name of the
    method arity handling that method. `default_name` is the name of the default
    handler method if no method is found in the `arity_map` and the number of arguments
    exceeds `max_fixed_arity`. `decorators` are applied to the generated function.

    `instance_or_class_var_name` is used to generate the first argument name and
    prefixes for class methods and standard methods; this name is not used outside the
    generated method and no user code exists here, so it can be a unique name that does
    not match the user's selected 'this' or 'self' name. If no
    `instance_or_class_var_name` is given, then a `class_name` must be given.
    `class_name` is used for static methods to ensure we can dispatch to static method
    arities without a class or self reference. You may *only* provide either a
    `class_name` or `instance_or_class_var_name`. Providing both is an error.

    class DefType:
        def __method_arity_1(self, arg): ...

        def __method_arity_2(self, arg1, arg2): ...

        def method(self, *args):
            nargs = len(args)
            method = {
                1: self.__method_arity_1,
                2: self.__method_arity_2
            }.get(nargs)
            if method:
                return method(*args)
            # Only if default
            if nargs > max_fixed_arity:
                return default(*args)
            raise RuntimeError
    """
    method_prefix = genname("self")

    dispatch_keys, dispatch_vals = [], []
    for k, v in arity_map.items():
        dispatch_keys.append(ast.Constant(k))
        dispatch_vals.append(_load_attr(f"{method_prefix}.{v}"))

    nargs_name = genname("nargs")
    method_name = genname("method")
    body = [
        ast.Assign(
            targets=[ast.Name(id=nargs_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="len", ctx=ast.Load()),
                args=[ast.Name(id=_MULTI_ARITY_ARG_NAME, ctx=ast.Load())],
                keywords=[],
            ),
        ),
        ast.Assign(
            targets=[ast.Name(id=method_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Dict(keys=dispatch_keys, values=dispatch_vals),
                    attr="get",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=nargs_name, ctx=ast.Load())],
                keywords=[],
            ),
        ),
        ast.If(
            test=ast.Compare(
                left=ast.Constant(None),
                ops=[ast.IsNot()],
                comparators=[ast.Name(id=method_name, ctx=ast.Load())],
            ),
            body=[
                ast.Return(
                    value=ast.Call(
                        func=ast.Name(id=method_name, ctx=ast.Load()),
                        args=[
                            ast.Starred(
                                value=ast.Name(
                                    id=_MULTI_ARITY_ARG_NAME, ctx=ast.Load()
                                ),
                                ctx=ast.Load(),
                            )
                        ],
                        keywords=[],
                    )
                )
            ],
            orelse=[]
            if default_name is None
            else [
                ast.If(
                    test=ast.Compare(
                        left=ast.Name(id=nargs_name, ctx=ast.Load()),
                        ops=[ast.GtE()],
                        comparators=[ast.Constant(max_fixed_arity)],
                    ),
                    body=[
                        ast.Return(
                            value=ast.Call(
                                func=_load_attr(f"{method_prefix}.{default_name}"),
                                args=[
                                    ast.Starred(
                                        value=ast.Name(
                                            id=_MULTI_ARITY_ARG_NAME, ctx=ast.Load()
                                        ),
                                        ctx=ast.Load(),
                                    )
                                ],
                                keywords=[],
                            )
                        )
                    ],
                    orelse=[],
                )
            ],
        ),
        ast.Raise(
            exc=ast.Call(
                func=_load_attr("basilisp.lang.runtime.RuntimeException"),
                args=[
                    ast.Constant(f"Wrong number of args passed to method: {name}"),
                    ast.Name(id=nargs_name, ctx=ast.Load()),
                ],
                keywords=[],
            ),
            cause=None,
        ),
    ]

    return GeneratedPyAST(
        # This is a pretty unusual case where we actually want to return the statement
        # node itself, rather than a name or expression. We're injecting all of these
        # nodes directly into the generated class body and Python does not like the
        # empty ast.Name in the class body definition (and also its unnecessary).
        node=ast.FunctionDef(
            name=name,
            args=ast.arguments(
                args=[ast.arg(arg=method_prefix, annotation=None)],
                kwarg=None,
                vararg=ast.arg(arg=_MULTI_ARITY_ARG_NAME, annotation=None),
                kwonlyargs=[],
                defaults=[],
                kw_defaults=[],
            ),
            body=body,
            decorator_list=[],
            returns=None,
        ),
    )


@_with_ast_loc_deps
def __multi_arity_deftype_method_to_py_ast(  # pylint: disable=too-many-arguments,too-many-locals
    ctx: GeneratorContext,
    node: DefTypeMethod,
) -> GeneratedPyAST:
    """Return a Python AST node for a function with multiple arities."""
    arities = node.arities

    assert node.op == NodeOp.DEFTYPE_METHOD and all(
        arity.op == NodeOp.DEFTYPE_METHOD_ARITY for arity in arities
    )

    arity_to_name = {}
    rest_arity_name: Optional[str] = None
    fn_defs = []
    for arity in arities:
        arity_name = arity.python_name
        if arity.is_variadic:
            rest_arity_name = arity_name
        else:
            arity_to_name[arity.fixed_arity] = arity_name

        fn_def = __deftype_method_arity_to_py_ast(ctx, node, arity, arity_name)
        assert (
            not fn_def.dependencies
        ), "deftype* or reify* method arities may not have dependency nodes"
        fn_defs.append(fn_def.node)

    dispatch_fn_ast = __multi_arity_deftype_dispatch_method(
        node.python_name,
        arity_to_name,
        default_name=rest_arity_name,
        max_fixed_arity=node.max_fixed_arity,
    )
    assert (
        not dispatch_fn_ast.dependencies
    ), "dispatch function should have no dependencies"

    return GeneratedPyAST(
        node=dispatch_fn_ast.node,
        dependencies=fn_defs,
    )


def __deftype_method_arity_to_py_ast(
    ctx: GeneratorContext,
    node: DefTypeMethod,
    arity: DefTypeMethodArity,
    method_name: Optional[str] = None,
) -> GeneratedPyAST:
    assert arity.op == NodeOp.DEFTYPE_METHOD_ARITY
    assert node.name == arity.name

    with ctx.new_symbol_table(node.name, is_context_boundary=True), ctx.new_recur_point(
        arity.loop_id, RecurType.METHOD, is_variadic=node.is_variadic
    ):
        this_name = genname(munge(arity.this_local.name))
        this_sym = sym.symbol(arity.this_local.name)
        ctx.symbol_table.new_symbol(this_sym, this_name, LocalType.THIS)

        with ctx.new_this(this_sym):
            fn_args, varg, fn_body_ast = __fn_args_to_py_ast(
                ctx, arity.params, arity.body
            )
            return GeneratedPyAST(
                node=ast.FunctionDef(
                    name=method_name if method_name is not None else munge(arity.name),
                    args=ast.arguments(
                        args=list(
                            chain((ast.arg(arg=this_name, annotation=None),), fn_args)
                        ),
                        kwarg=None,
                        vararg=varg,
                        kwonlyargs=[],
                        defaults=[],
                        kw_defaults=[],
                    ),
                    body=fn_body_ast,
                    decorator_list=list(
                        chain(
                            [_TRAMPOLINE_FN_NAME] if ctx.recur_point.has_recur else [],
                            __kwargs_support_decorator(arity),
                        )
                    ),
                    returns=None,
                )
            )


@_with_ast_loc
def __deftype_method_to_py_ast(
    ctx: GeneratorContext,
    node: DefTypeMethod,
) -> GeneratedPyAST:
    """Return a Python AST Node for a `deftype*` or `reify*` method."""
    assert node.op == NodeOp.DEFTYPE_METHOD

    if len(node.arities) == 1:
        return __deftype_method_arity_to_py_ast(ctx, node, next(iter(node.arities)))
    else:
        return __multi_arity_deftype_method_to_py_ast(ctx, node)


@_with_ast_loc
def __deftype_staticmethod_to_py_ast(
    ctx: GeneratorContext, node: DefTypeStaticMethod
) -> GeneratedPyAST:
    """Return a Python AST Node for a `deftype*` or `reify*` staticmethod."""
    assert node.op == NodeOp.DEFTYPE_STATICMETHOD

    with ctx.new_symbol_table(node.name, is_context_boundary=True):
        fn_args, varg, fn_body_ast = __fn_args_to_py_ast(ctx, node.params, node.body)
        return GeneratedPyAST(
            node=ast.FunctionDef(
                name=munge(node.name),
                args=ast.arguments(
                    args=fn_args,
                    kwarg=None,
                    vararg=varg,
                    kwonlyargs=[],
                    defaults=[],
                    kw_defaults=[],
                ),
                body=fn_body_ast,
                decorator_list=list(
                    chain([_PY_STATICMETHOD_FN_NAME], __kwargs_support_decorator(node))
                ),
                returns=None,
            )
        )


DefTypeASTGenerator = Callable[[GeneratorContext, DefTypeMember], GeneratedPyAST]
_DEFTYPE_MEMBER_HANDLER: Mapping[NodeOp, DefTypeASTGenerator] = {
    NodeOp.DEFTYPE_CLASSMETHOD: __deftype_classmethod_to_py_ast,
    NodeOp.DEFTYPE_METHOD: __deftype_method_to_py_ast,
    NodeOp.DEFTYPE_PROPERTY: __deftype_property_to_py_ast,
    NodeOp.DEFTYPE_STATICMETHOD: __deftype_staticmethod_to_py_ast,
}


def __deftype_member_to_py_ast(
    ctx: GeneratorContext,
    node: DefTypeMember,
) -> GeneratedPyAST:
    member_type = node.op
    handle_deftype_member = _DEFTYPE_MEMBER_HANDLER.get(member_type)
    assert (
        handle_deftype_member is not None
    ), f"Invalid :const AST type handler for {member_type}"
    return handle_deftype_member(ctx, node)


def __deftype_or_reify_bases_to_py_ast(
    ctx: GeneratorContext,
    node: Union[DefType, Reify],
    interfaces: Iterable[DefTypeBase],
) -> List[ast.AST]:
    """Return a list of AST nodes for the base classes for a `deftype*` or `reify*`."""
    assert node.op in {NodeOp.DEFTYPE, NodeOp.REIFY}

    bases: List[ast.AST] = []
    for base in interfaces:
        base_node = gen_py_ast(ctx, base)
        assert (
            count(base_node.dependencies) == 0
        ), "Class and host form nodes do not have dependencies"

        # Protocols are defined as Maps
        if (
            isinstance(base, VarRef)
            and base.var.meta is not None
            and base.var.meta.val_at(VAR_IS_PROTOCOL_META_KEY)
        ):
            bases.append(
                ast.Call(
                    func=ast.Attribute(
                        value=base_node.node, attr="val_at", ctx=ast.Load()
                    ),
                    args=[
                        ast.Call(
                            func=_NEW_KW_FN_NAME,
                            args=[
                                ast.Constant(hash(_INTERFACE_KW)),
                                ast.Constant("interface"),
                            ],
                            keywords=[],
                        )
                    ],
                    keywords=[],
                )
            )
        else:
            bases.append(base_node.node)

    return bases


@_with_ast_loc
def _deftype_to_py_ast(  # pylint: disable=too-many-branches,too-many-locals
    ctx: GeneratorContext, node: DefType
) -> GeneratedPyAST:
    """Return a Python AST Node for a `deftype*` expression."""
    assert node.op == NodeOp.DEFTYPE
    type_name = munge(node.name)
    ctx.symbol_table.new_symbol(sym.symbol(node.name), type_name, LocalType.DEFTYPE)

    bases = __deftype_or_reify_bases_to_py_ast(ctx, node, node.interfaces)
    artificially_abstract_bases = __deftype_or_reify_bases_to_py_ast(
        ctx, node, node.artificially_abstract
    )

    with ctx.new_symbol_table(node.name):
        fields = []
        type_nodes: List[ast.AST] = []
        type_deps: List[ast.AST] = []
        for field in node.fields:
            safe_field = munge(field.name)

            if field.init is not None:
                default_nodes = gen_py_ast(ctx, field.init)
                type_deps.extend(default_nodes.dependencies)
                attr_default_kws = [
                    ast.keyword(arg="default", value=default_nodes.node)
                ]
            else:
                attr_default_kws = []

            type_nodes.append(
                ast.Assign(
                    targets=[ast.Name(id=safe_field, ctx=ast.Store())],
                    value=ast.Call(
                        func=_ATTRIB_FIELD_FN_NAME, args=[], keywords=attr_default_kws
                    ),
                )
            )
            ctx.symbol_table.new_symbol(sym.symbol(field.name), safe_field, field.local)
            fields.append(safe_field)

        for member in node.members:
            type_ast = __deftype_member_to_py_ast(ctx, member)
            type_nodes.append(type_ast.node)
            type_nodes.extend(type_ast.dependencies)

        return GeneratedPyAST(
            node=ast.Name(id=type_name, ctx=ast.Load()),
            dependencies=list(
                chain(
                    type_deps,
                    [
                        _class_ast(
                            type_name,
                            type_nodes or [ast.Pass()],
                            bases=bases,
                            fields=fields,
                            members=node.python_member_names,
                            verified_abstract=node.verified_abstract,
                            artificially_abstract_bases=artificially_abstract_bases,
                            is_frozen=node.is_frozen,
                            use_slots=True,
                        ),
                        ast.Call(
                            func=_INTERN_VAR_FN_NAME,
                            args=[
                                _load_attr(_NS_VAR_VALUE),
                                ast.Call(
                                    func=_NEW_SYM_FN_NAME,
                                    args=[ast.Constant(type_name)],
                                    keywords=[],
                                ),
                                ast.Name(id=type_name, ctx=ast.Load()),
                            ],
                            keywords=[],
                        ),
                    ],
                )
            ),
        )


def _wrap_override_var_indirection(f: PyASTGenerator) -> PyASTGenerator:
    """
    Wrap a Node generator to apply a special override requiring Var indirection
    for any Var accesses generated within top-level `do` blocks.

    This is a bit of a hack to account for the `ns` macro, which is the first
    form in most standard Namespaces. When Basilisp `require`s a Namespace, it
    (like in Clojure) simply loads the file and lets that Namespace's `ns` macro
    create the new Namespace and perform any setup. However, the Basilisp
    compiler desperately tries to emit "smarter" Python code which avoids using
    `Var.find` whenever the resolved symbol can be safely called directly from
    the generated Pythom module. Without this hack, the compiler will emit code
    during macroexpansion to access `basilisp.core` functions used in the `ns`
    macro directly, even though they will not be available yet in the target
    Namespace module.
    """

    @wraps(f)
    def _wrapped_do(
        ctx: GeneratorContext, node: Node, *args, **kwargs
    ) -> GeneratedPyAST:
        if isinstance(node, Do) and node.top_level:
            with ctx.with_var_indirection_override():
                return f(ctx, node, *args, **kwargs)  # type: ignore[call-arg]
        else:
            with ctx.with_var_indirection_override(False):
                return f(ctx, node, *args, **kwargs)  # type: ignore[call-arg]

    return _wrapped_do


@_wrap_override_var_indirection
@_with_ast_loc_deps
def _do_to_py_ast(ctx: GeneratorContext, node: Do) -> GeneratedPyAST:
    """Return a Python AST Node for a `do` expression."""
    assert node.op == NodeOp.DO
    assert not node.is_body

    body_ast = GeneratedPyAST.reduce(
        *map(partial(gen_py_ast, ctx), chain(node.statements, [node.ret]))
    )

    fn_body_ast: List[ast.AST] = []
    do_result_name = genname(_DO_PREFIX)
    fn_body_ast.extend(map(statementize, body_ast.dependencies))
    fn_body_ast.append(
        ast.Assign(
            targets=[ast.Name(id=do_result_name, ctx=ast.Store())], value=body_ast.node
        )
    )

    return GeneratedPyAST(
        node=ast.Name(id=do_result_name, ctx=ast.Load()), dependencies=fn_body_ast
    )


@_with_ast_loc
def _synthetic_do_to_py_ast(ctx: GeneratorContext, node: Do) -> GeneratedPyAST:
    """Return AST elements generated from reducing a synthetic Lisp :do node
    (e.g. a :do node which acts as a body for another node)."""
    assert node.op == NodeOp.DO
    assert node.is_body

    # TODO: investigate how to handle recur in node.ret

    return GeneratedPyAST.reduce(
        *map(partial(gen_py_ast, ctx), chain(node.statements, [node.ret]))
    )


MetaNode = Union[Const, MapNode]


def __fn_name(s: Optional[str]) -> str:
    """Generate a safe Python function name from a function name symbol.
    If no symbol is provided, generate a name with a default prefix."""
    return genname("__" + munge(Maybe(s).or_else_get(_FN_PREFIX)))


def __fn_args_to_py_ast(
    ctx: GeneratorContext, params: Iterable[Binding], body: Do
) -> Tuple[List[ast.arg], Optional[ast.arg], List[ast.AST]]:
    """Generate a list of Python AST nodes from function method parameters."""
    fn_args, varg = [], None
    fn_body_ast: List[ast.AST] = []
    for binding in params:
        assert binding.init is None, ":fn nodes cannot have bindint :inits"
        assert varg is None, "Must have at most one variadic arg"
        arg_name = genname(munge(binding.name))

        if not binding.is_variadic:
            fn_args.append(ast.arg(arg=arg_name, annotation=None))
            ctx.symbol_table.new_symbol(
                sym.symbol(binding.name), arg_name, LocalType.ARG
            )
        else:
            varg = ast.arg(arg=arg_name, annotation=None)
            safe_local = genname(munge(binding.name))
            fn_body_ast.append(
                ast.Assign(
                    targets=[ast.Name(id=safe_local, ctx=ast.Store())],
                    value=ast.Call(
                        func=_COLLECT_ARGS_FN_NAME,
                        args=[ast.Name(id=arg_name, ctx=ast.Load())],
                        keywords=[],
                    ),
                )
            )
            ctx.symbol_table.new_symbol(
                sym.symbol(binding.name), safe_local, LocalType.ARG
            )

    body_ast = _synthetic_do_to_py_ast(ctx, body)
    fn_body_ast.extend(map(statementize, body_ast.dependencies))
    fn_body_ast.append(ast.Return(value=body_ast.node))

    return fn_args, varg, fn_body_ast


def __fn_decorator(
    arities: Iterable[int],
    has_rest_arg: bool = False,
) -> ast.Call:
    return ast.Call(
        func=_BASILISP_FN_FN_NAME,
        args=[],
        keywords=[
            ast.keyword(
                arg="arities",
                value=ast.Tuple(
                    elts=list(
                        chain(
                            map(ast.Constant, arities),
                            [
                                ast.Call(
                                    func=_NEW_KW_FN_NAME,
                                    args=[
                                        ast.Constant(hash(_REST_KW)),
                                        ast.Constant("rest"),
                                    ],
                                    keywords=[],
                                )
                            ]
                            if has_rest_arg
                            else [],
                        )
                    ),
                    ctx=ast.Load(),
                ),
            )
        ],
    )


def __fn_meta(
    ctx: GeneratorContext, meta_node: Optional[MetaNode] = None
) -> Tuple[Iterable[ast.AST], Iterable[ast.AST]]:
    if meta_node is not None:
        meta_ast = gen_py_ast(ctx, meta_node)
        return (
            meta_ast.dependencies,
            [
                ast.Call(
                    func=_FN_WITH_ATTRS_FN_NAME,
                    args=[],
                    keywords=[ast.keyword(arg="meta", value=meta_ast.node)],
                )
            ],
        )
    else:
        return (), ()


def __kwargs_support_decorator(
    node: Union[Fn, DefTypeMethodArity, DefTypeClassMethod, DefTypeStaticMethod]
) -> Iterable[ast.AST]:
    if node.kwarg_support is None:
        return

    yield {
        KeywordArgSupport.APPLY_KWARGS: _LISP_FN_APPLY_KWARGS_FN_NAME,
        KeywordArgSupport.COLLECT_KWARGS: _LISP_FN_COLLECT_KWARGS_FN_NAME,
    }[node.kwarg_support]


@_with_ast_loc_deps
def __single_arity_fn_to_py_ast(
    ctx: GeneratorContext,
    node: Fn,
    method: FnArity,
    def_name: Optional[str] = None,
    meta_node: Optional[MetaNode] = None,
) -> GeneratedPyAST:
    """Return a Python AST node for a function with a single arity."""
    assert node.op == NodeOp.FN
    assert method.op == NodeOp.FN_ARITY

    lisp_fn_name = node.local.name if node.local is not None else None
    py_fn_name = __fn_name(lisp_fn_name) if def_name is None else munge(def_name)
    py_fn_node = ast.AsyncFunctionDef if node.is_async else ast.FunctionDef
    with ctx.new_symbol_table(
        py_fn_name, is_context_boundary=True
    ), ctx.new_recur_point(method.loop_id, RecurType.FN, is_variadic=node.is_variadic):
        # Allow named anonymous functions to recursively call themselves
        if lisp_fn_name is not None:
            ctx.symbol_table.new_symbol(
                sym.symbol(lisp_fn_name), py_fn_name, LocalType.FN
            )

        fn_args, varg, fn_body_ast = __fn_args_to_py_ast(
            ctx, method.params, method.body
        )
        meta_deps, meta_decorators = __fn_meta(ctx, meta_node)
        return GeneratedPyAST(
            node=ast.Name(id=py_fn_name, ctx=ast.Load()),
            dependencies=list(
                chain(
                    meta_deps,
                    [
                        py_fn_node(
                            name=py_fn_name,
                            args=ast.arguments(
                                args=fn_args,
                                kwarg=None,
                                vararg=varg,
                                kwonlyargs=[],
                                defaults=[],
                                kw_defaults=[],
                            ),
                            body=fn_body_ast,
                            decorator_list=list(
                                chain(
                                    __kwargs_support_decorator(node),
                                    meta_decorators,
                                    [
                                        __fn_decorator(
                                            (len(fn_args),)
                                            if not method.is_variadic
                                            else (),
                                            has_rest_arg=method.is_variadic,
                                        )
                                    ],
                                    [_TRAMPOLINE_FN_NAME]
                                    if ctx.recur_point.has_recur
                                    else [],
                                )
                            ),
                            returns=None,
                        )
                    ],
                )
            ),
        )


def __handle_async_return(node: ast.AST) -> ast.Return:
    return ast.Return(value=ast.Await(value=node))


def __handle_return(node: ast.AST) -> ast.Return:
    return ast.Return(value=node)


def __multi_arity_dispatch_fn(  # pylint: disable=too-many-arguments,too-many-locals
    ctx: GeneratorContext,
    name: str,
    arity_map: Mapping[int, str],
    default_name: Optional[str] = None,
    max_fixed_arity: Optional[int] = None,
    meta_node: Optional[MetaNode] = None,
    is_async: bool = False,
) -> GeneratedPyAST:
    """Return the Python AST nodes for a argument-length dispatch function
    for multi-arity functions.

    def fn(*args):
        nargs = len(args)
        arity = __fn_dispatch_map.get(nargs)
        if arity:
            return arity(*args)
        # Only if default
        if nargs > max_fixed_arity:
            return default(*args)
        raise RuntimeError
    """
    dispatch_map_name = f"{name}_dispatch_map"

    dispatch_keys, dispatch_vals = [], []
    for k, v in arity_map.items():
        dispatch_keys.append(ast.Constant(k))
        dispatch_vals.append(ast.Name(id=v, ctx=ast.Load()))

    # Async functions should return await, otherwise just return
    handle_return = __handle_async_return if is_async else __handle_return

    nargs_name = genname("nargs")
    arity_name = genname("arity")
    body = [
        ast.Assign(
            targets=[ast.Name(id=nargs_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="len", ctx=ast.Load()),
                args=[ast.Name(id=_MULTI_ARITY_ARG_NAME, ctx=ast.Load())],
                keywords=[],
            ),
        ),
        ast.Assign(
            targets=[ast.Name(id=arity_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=dispatch_map_name, ctx=ast.Load()),
                    attr="get",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=nargs_name, ctx=ast.Load())],
                keywords=[],
            ),
        ),
        ast.If(
            test=ast.Compare(
                left=ast.Constant(None),
                ops=[ast.IsNot()],
                comparators=[ast.Name(id=arity_name, ctx=ast.Load())],
            ),
            body=[
                handle_return(
                    ast.Call(
                        func=ast.Name(id=arity_name, ctx=ast.Load()),
                        args=[
                            ast.Starred(
                                value=ast.Name(
                                    id=_MULTI_ARITY_ARG_NAME, ctx=ast.Load()
                                ),
                                ctx=ast.Load(),
                            )
                        ],
                        keywords=[],
                    )
                )
            ],
            orelse=[]
            if default_name is None
            else [
                ast.If(
                    test=ast.Compare(
                        left=ast.Name(id=nargs_name, ctx=ast.Load()),
                        ops=[ast.GtE()],
                        comparators=[ast.Constant(max_fixed_arity)],
                    ),
                    body=[
                        handle_return(
                            ast.Call(
                                func=ast.Name(id=default_name, ctx=ast.Load()),
                                args=[
                                    ast.Starred(
                                        value=ast.Name(
                                            id=_MULTI_ARITY_ARG_NAME, ctx=ast.Load()
                                        ),
                                        ctx=ast.Load(),
                                    )
                                ],
                                keywords=[],
                            )
                        )
                    ],
                    orelse=[],
                )
            ],
        ),
        ast.Raise(
            exc=ast.Call(
                func=_load_attr("basilisp.lang.runtime.RuntimeException"),
                args=[
                    ast.Constant(f"Wrong number of args passed to function: {name}"),
                    ast.Name(id=nargs_name, ctx=ast.Load()),
                ],
                keywords=[],
            ),
            cause=None,
        ),
    ]

    py_fn_node = ast.AsyncFunctionDef if is_async else ast.FunctionDef
    meta_deps, meta_decorators = __fn_meta(ctx, meta_node)
    return GeneratedPyAST(
        node=ast.Name(id=name, ctx=ast.Load()),
        dependencies=chain(
            [
                ast.Assign(
                    targets=[ast.Name(id=dispatch_map_name, ctx=ast.Store())],
                    value=ast.Dict(keys=dispatch_keys, values=dispatch_vals),
                )
            ],
            meta_deps,
            [
                py_fn_node(
                    name=name,
                    args=ast.arguments(
                        args=[],
                        kwarg=None,
                        vararg=ast.arg(arg=_MULTI_ARITY_ARG_NAME, annotation=None),
                        kwonlyargs=[],
                        defaults=[],
                        kw_defaults=[],
                    ),
                    body=body,
                    decorator_list=list(
                        chain(
                            meta_decorators,
                            [
                                __fn_decorator(
                                    arity_map.keys(),
                                    has_rest_arg=default_name is not None,
                                )
                            ],
                        )
                    ),
                    returns=None,
                )
            ],
        ),
    )


@_with_ast_loc_deps
def __multi_arity_fn_to_py_ast(  # pylint: disable=too-many-locals
    ctx: GeneratorContext,
    node: Fn,
    arities: Collection[FnArity],
    def_name: Optional[str] = None,
    meta_node: Optional[MetaNode] = None,
) -> GeneratedPyAST:
    """Return a Python AST node for a function with multiple arities."""
    assert node.op == NodeOp.FN
    assert all(arity.op == NodeOp.FN_ARITY for arity in arities)
    assert node.kwarg_support is None, "multi-arity functions do not support kwargs"

    lisp_fn_name = node.local.name if node.local is not None else None
    py_fn_name = __fn_name(lisp_fn_name) if def_name is None else munge(def_name)

    py_fn_node = ast.AsyncFunctionDef if node.is_async else ast.FunctionDef

    arity_to_name = {}
    rest_arity_name: Optional[str] = None
    fn_defs = []
    for arity in arities:
        arity_name = (
            f"{py_fn_name}__arity{'_rest' if arity.is_variadic else arity.fixed_arity}"
        )
        if arity.is_variadic:
            rest_arity_name = arity_name
        else:
            arity_to_name[arity.fixed_arity] = arity_name

        with ctx.new_symbol_table(
            arity_name, is_context_boundary=True
        ), ctx.new_recur_point(
            arity.loop_id, RecurType.FN, is_variadic=node.is_variadic
        ):
            # Allow named anonymous functions to recursively call themselves
            if lisp_fn_name is not None:
                ctx.symbol_table.new_symbol(
                    sym.symbol(lisp_fn_name), py_fn_name, LocalType.FN
                )

            fn_args, varg, fn_body_ast = __fn_args_to_py_ast(
                ctx, arity.params, arity.body
            )
            fn_defs.append(
                py_fn_node(
                    name=arity_name,
                    args=ast.arguments(
                        args=fn_args,
                        kwarg=None,
                        vararg=varg,
                        kwonlyargs=[],
                        defaults=[],
                        kw_defaults=[],
                    ),
                    body=fn_body_ast,
                    decorator_list=[_TRAMPOLINE_FN_NAME]
                    if ctx.recur_point.has_recur
                    else [],
                    returns=None,
                )
            )

    dispatch_fn_ast = __multi_arity_dispatch_fn(
        ctx,
        py_fn_name,
        arity_to_name,
        default_name=rest_arity_name,
        max_fixed_arity=node.max_fixed_arity,
        meta_node=meta_node,
        is_async=node.is_async,
    )

    return GeneratedPyAST(
        node=dispatch_fn_ast.node,
        dependencies=list(chain(fn_defs, dispatch_fn_ast.dependencies)),
    )


@_wrap_override_var_indirection
@_with_ast_loc
def _fn_to_py_ast(
    ctx: GeneratorContext,
    node: Fn,
    def_name: Optional[str] = None,
    meta_node: Optional[MetaNode] = None,
) -> GeneratedPyAST:
    """Return a Python AST Node for a `fn` expression."""
    assert node.op == NodeOp.FN
    if len(node.arities) == 1:
        return __single_arity_fn_to_py_ast(
            ctx, node, next(iter(node.arities)), def_name=def_name, meta_node=meta_node
        )
    else:
        return __multi_arity_fn_to_py_ast(
            ctx, node, node.arities, def_name=def_name, meta_node=meta_node
        )


@_with_ast_loc_deps
def __if_body_to_py_ast(
    ctx: GeneratorContext, node: Node, result_name: Optional[str]
) -> GeneratedPyAST:
    """Generate custom `if` nodes to handle `recur` bodies.

    Recur nodes can appear in the then and else expressions of `if` forms.
    Recur nodes generate Python `continue` statements, which we would otherwise
    attempt to insert directly into an expression. Python will complain if
    it finds a statement in an expression AST slot, so we special case the
    recur handling here.

    If `result_name` is None, then the `if` form is syntactically in a statement
    position, so there is no need to emit an assignment."""
    if node.op == NodeOp.RECUR and ctx.recur_point.type == RecurType.LOOP:
        assert isinstance(node, Recur)
        return _recur_to_py_ast(ctx, node)
    elif node.op == NodeOp.DO:
        assert isinstance(node, Do)
        if_body = _synthetic_do_to_py_ast(ctx, node.assoc(is_body=True))
        return GeneratedPyAST(
            node=ast.Assign(
                targets=[ast.Name(id=result_name, ctx=ast.Store())], value=if_body.node
            )
            if result_name is not None
            else if_body.node,
            dependencies=list(map(statementize, if_body.dependencies)),
        )
    else:
        py_ast = gen_py_ast(ctx, node)
        return GeneratedPyAST(
            node=ast.Assign(
                targets=[ast.Name(id=result_name, ctx=ast.Store())], value=py_ast.node
            )
            if result_name is not None
            else py_ast.node,
            dependencies=py_ast.dependencies,
        )


@_with_ast_loc_deps
def _if_to_py_ast(ctx: GeneratorContext, node: If) -> GeneratedPyAST:
    """Generate an intermediate if statement which assigns to a temporary
    variable, which is returned as the expression value at the end of
    evaluation.

    Every expression in Basilisp is true if it is not the literal values nil
    or false. This function compiles direct checks for the test value against
    the Python values None and False to accommodate this behavior.

    Note that the if and else bodies are switched in compilation so that we
    can perform a short-circuit or comparison, rather than exhaustively checking
    for both false and nil each time."""
    assert node.op == NodeOp.IF

    test_ast = gen_py_ast(ctx, node.test)
    result_name = (
        genname(_IF_RESULT_PREFIX)
        if node.env.pos == NodeSyntacticPosition.EXPR
        else None
    )

    then_ast = __if_body_to_py_ast(ctx, node.then, result_name)
    else_ast = __if_body_to_py_ast(ctx, node.else_, result_name)

    test_name = genname(_IF_TEST_PREFIX)
    test_assign = ast.Assign(
        targets=[ast.Name(id=test_name, ctx=ast.Store())], value=test_ast.node
    )

    ifstmt = ast.If(
        test=ast.BoolOp(
            op=ast.Or(),
            values=[
                ast.Compare(
                    left=ast.Constant(None),
                    ops=[ast.Is()],
                    comparators=[ast.Name(id=test_name, ctx=ast.Load())],
                ),
                ast.Compare(
                    left=ast.Constant(False),
                    ops=[ast.Is()],
                    comparators=[ast.Name(id=test_name, ctx=ast.Load())],
                ),
            ],
        ),
        values=[],
        body=list(map(statementize, chain(else_ast.dependencies, [else_ast.node]))),
        orelse=list(map(statementize, chain(then_ast.dependencies, [then_ast.node]))),
    )

    return GeneratedPyAST(
        node=ast.Name(id=result_name, ctx=ast.Load())
        if result_name is not None
        else _noop_node(),
        dependencies=list(chain(test_ast.dependencies, [test_assign, ifstmt])),
    )


@_with_ast_loc_deps
def _import_to_py_ast(ctx: GeneratorContext, node: Import) -> GeneratedPyAST:
    """Return a Python AST node for a Basilisp `import*` expression."""
    assert node.op == NodeOp.IMPORT

    last = None
    deps: List[ast.AST] = []
    for alias in node.aliases:
        safe_name = munge(alias.name)

        # Always use builtins.__import__ and assign to the first name component
        # if there's no alias. Otherwise, we could potentially overwrite a parent
        # import if parent and child are both imported:
        #   (import* collections collections.abc)
        if alias.alias is not None:
            py_import_alias = munge(alias.alias)
            import_func = _IMPORTLIB_IMPORT_MODULE_FN_NAME
        else:
            py_import_alias = safe_name.split(".", maxsplit=1)[0]
            import_func = _BUILTINS_IMPORT_FN_NAME

        ctx.symbol_table.context_boundary.new_symbol(
            sym.symbol(alias.alias or alias.name), py_import_alias, LocalType.IMPORT
        )

        if node.env.func_ctx is not None:
            deps.append(ast.Global(names=[py_import_alias]))
        deps.append(
            ast.Assign(
                targets=[ast.Name(id=py_import_alias, ctx=ast.Store())],
                value=ast.Call(
                    func=import_func,
                    args=[ast.Constant(safe_name)],
                    keywords=[],
                ),
            )
        )
        last = ast.Name(id=py_import_alias, ctx=ast.Load())

        deps.append(
            ast.Call(
                func=_load_attr(f"{_NS_VAR_VALUE}.add_import"),
                args=list(
                    chain(
                        [
                            ast.Call(
                                func=_NEW_SYM_FN_NAME,
                                args=[ast.Constant(safe_name)],
                                keywords=[],
                            ),
                            last,
                        ],
                        [
                            ast.Call(
                                func=_NEW_SYM_FN_NAME,
                                args=[ast.Constant(alias.alias)],
                                keywords=[],
                            )
                        ]
                        if alias.alias is not None
                        else [],
                    )
                ),
                keywords=[],
            )
        )

    assert last is not None, "import* node must have at least one import"
    return GeneratedPyAST(node=last, dependencies=deps)


@_with_ast_loc
def _invoke_to_py_ast(ctx: GeneratorContext, node: Invoke) -> GeneratedPyAST:
    """Return a Python AST Node for a Basilisp function invocation."""
    assert node.op == NodeOp.INVOKE

    fn_ast = gen_py_ast(ctx, node.fn)
    args_deps, args_nodes = _collection_ast(ctx, node.args)
    kwargs_deps, kwargs_nodes = _kwargs_ast(ctx, node.kwargs)

    return GeneratedPyAST(
        node=ast.Call(
            func=fn_ast.node,
            args=list(args_nodes),
            keywords=list(kwargs_nodes),
        ),
        dependencies=list(chain(fn_ast.dependencies, args_deps, kwargs_deps)),
    )


@_with_ast_loc_deps
def _let_to_py_ast(ctx: GeneratorContext, node: Let) -> GeneratedPyAST:
    """Return a Python AST Node for a `let*` expression."""
    assert node.op == NodeOp.LET

    with ctx.new_symbol_table("let"):
        let_body_ast: List[ast.AST] = []
        for binding in node.bindings:
            init_node = binding.init
            assert init_node is not None
            init_ast = gen_py_ast(ctx, init_node)
            binding_name = genname(munge(binding.name))
            let_body_ast.extend(init_ast.dependencies)
            let_body_ast.append(
                ast.Assign(
                    targets=[ast.Name(id=binding_name, ctx=ast.Store())],
                    value=init_ast.node,
                )
            )
            ctx.symbol_table.new_symbol(
                sym.symbol(binding.name), binding_name, LocalType.LET
            )

        let_result_name = genname(_LET_RESULT_PREFIX)
        body_ast = _synthetic_do_to_py_ast(ctx, node.body)
        let_body_ast.extend(map(statementize, body_ast.dependencies))

        if node.env.pos == NodeSyntacticPosition.EXPR:
            let_body_ast.append(
                ast.Assign(
                    targets=[ast.Name(id=let_result_name, ctx=ast.Store())],
                    value=body_ast.node,
                )
            )
            return GeneratedPyAST(
                node=ast.Name(id=let_result_name, ctx=ast.Load()),
                dependencies=let_body_ast,
            )
        else:
            let_body_ast.append(body_ast.node)
            return GeneratedPyAST(node=_noop_node(), dependencies=let_body_ast)


@_with_ast_loc_deps
def _letfn_to_py_ast(ctx: GeneratorContext, node: LetFn) -> GeneratedPyAST:
    """Return a Python AST Node for a `letfn*` expression."""
    assert node.op == NodeOp.LETFN

    with ctx.new_symbol_table("letfn"):
        binding_names = []
        for binding in node.bindings:
            binding_name = genname(munge(binding.name))
            ctx.symbol_table.new_symbol(
                sym.symbol(binding.name), binding_name, LocalType.LET
            )
            binding_names.append((binding_name, binding))

        letfn_body_ast: List[ast.AST] = []
        for binding_name, binding in binding_names:
            init_node = binding.init
            assert init_node is not None
            init_ast = gen_py_ast(ctx, init_node)
            letfn_body_ast.extend(init_ast.dependencies)
            letfn_body_ast.append(
                ast.Assign(
                    targets=[ast.Name(id=binding_name, ctx=ast.Store())],
                    value=init_ast.node,
                )
            )

        letfn_result_name = genname(_LETFN_RESULT_PREFIX)
        body_ast = _synthetic_do_to_py_ast(ctx, node.body)
        letfn_body_ast.extend(map(statementize, body_ast.dependencies))

        if node.env.pos == NodeSyntacticPosition.EXPR:
            letfn_body_ast.append(
                ast.Assign(
                    targets=[ast.Name(id=letfn_result_name, ctx=ast.Store())],
                    value=body_ast.node,
                )
            )
            return GeneratedPyAST(
                node=ast.Name(id=letfn_result_name, ctx=ast.Load()),
                dependencies=letfn_body_ast,
            )
        else:
            letfn_body_ast.append(body_ast.node)
            return GeneratedPyAST(node=_noop_node(), dependencies=letfn_body_ast)


@_with_ast_loc_deps
def _loop_to_py_ast(ctx: GeneratorContext, node: Loop) -> GeneratedPyAST:
    """Return a Python AST Node for a `loop*` expression."""
    assert node.op == NodeOp.LOOP

    with ctx.new_symbol_table("loop"):
        binding_names = []
        init_bindings: List[ast.AST] = []
        for binding in node.bindings:
            init_node = binding.init
            assert init_node is not None
            init_ast = gen_py_ast(ctx, init_node)
            init_bindings.extend(init_ast.dependencies)
            binding_name = genname(munge(binding.name))
            binding_names.append(binding_name)
            init_bindings.append(
                ast.Assign(
                    targets=[ast.Name(id=binding_name, ctx=ast.Store())],
                    value=init_ast.node,
                )
            )
            ctx.symbol_table.new_symbol(
                sym.symbol(binding.name), binding_name, LocalType.LOOP
            )

        loop_result_name = genname(_LOOP_RESULT_PREFIX)
        with ctx.new_recur_point(
            node.loop_id, RecurType.LOOP, binding_names=binding_names
        ):
            loop_body_ast: List[ast.AST] = []
            body_ast = _synthetic_do_to_py_ast(ctx, node.body)
            loop_body_ast.extend(map(statementize, body_ast.dependencies))
            loop_body_ast.append(
                ast.Assign(
                    targets=[ast.Name(id=loop_result_name, ctx=ast.Store())],
                    value=body_ast.node,
                )
            )
            loop_body_ast.append(ast.Break())

            return GeneratedPyAST(
                node=_load_attr(loop_result_name),
                dependencies=list(
                    chain(
                        [
                            ast.Assign(
                                targets=[
                                    ast.Name(id=loop_result_name, ctx=ast.Store())
                                ],
                                value=ast.Constant(None),
                            )
                        ],
                        init_bindings,
                        [
                            ast.While(
                                test=ast.Constant(True), body=loop_body_ast, orelse=[]
                            )
                        ],
                    )
                ),
            )


@_with_ast_loc
def _quote_to_py_ast(ctx: GeneratorContext, node: Quote) -> GeneratedPyAST:
    """Return a Python AST Node for a `quote` expression."""
    assert node.op == NodeOp.QUOTE
    return _const_node_to_py_ast(ctx, node.expr)


@_with_ast_loc
def __fn_recur_to_py_ast(ctx: GeneratorContext, node: Recur) -> GeneratedPyAST:
    """Return a Python AST node for `recur` occurring inside a `fn*`."""
    assert node.op == NodeOp.RECUR
    assert ctx.recur_point.is_variadic is not None
    recur_nodes: List[ast.AST] = []
    recur_deps: List[ast.AST] = []
    for expr in node.exprs:
        expr_ast = gen_py_ast(ctx, expr)
        recur_nodes.append(expr_ast.node)
        recur_deps.extend(expr_ast.dependencies)

    return GeneratedPyAST(
        node=ast.Call(
            func=_TRAMPOLINE_ARGS_FN_NAME,
            args=list(chain([ast.Constant(ctx.recur_point.is_variadic)], recur_nodes)),
            keywords=[],
        ),
        dependencies=recur_deps,
    )


@_with_ast_loc
def __deftype_method_recur_to_py_ast(
    ctx: GeneratorContext, node: Recur
) -> GeneratedPyAST:
    """Return a Python AST node for `recur` occurring inside a `deftype*` method."""
    assert node.op == NodeOp.RECUR
    recur_nodes: List[ast.AST] = []
    recur_deps: List[ast.AST] = []
    for expr in node.exprs:
        expr_ast = gen_py_ast(ctx, expr)
        recur_nodes.append(expr_ast.node)
        recur_deps.extend(expr_ast.dependencies)

    this_entry = ctx.symbol_table.find_symbol(ctx.current_this)
    assert this_entry is not None, "Field type local must have this"

    return GeneratedPyAST(
        node=ast.Call(
            func=_TRAMPOLINE_ARGS_FN_NAME,
            args=list(
                chain(
                    [
                        ast.Constant(ctx.recur_point.is_variadic),
                        ast.Name(id=this_entry.munged, ctx=ast.Load()),
                    ],
                    recur_nodes,
                )
            ),
            keywords=[],
        ),
        dependencies=recur_deps,
    )


@_with_ast_loc_deps
def __loop_recur_to_py_ast(ctx: GeneratorContext, node: Recur) -> GeneratedPyAST:
    """Return a Python AST node for `recur` occurring inside a `loop`."""
    assert node.op == NodeOp.RECUR

    recur_deps: List[ast.AST] = []
    recur_targets: List[ast.Name] = []
    recur_exprs: List[ast.AST] = []
    for name, expr in zip(ctx.recur_point.binding_names, node.exprs):
        expr_ast = gen_py_ast(ctx, expr)
        recur_deps.extend(expr_ast.dependencies)
        recur_targets.append(ast.Name(id=name, ctx=ast.Store()))
        recur_exprs.append(expr_ast.node)

    if len(recur_targets) == 1:
        assert len(recur_exprs) == 1
        recur_deps.append(ast.Assign(targets=recur_targets, value=recur_exprs[0]))
    else:
        recur_deps.append(
            ast.Assign(
                targets=[ast.Tuple(elts=recur_targets, ctx=ast.Store())],
                value=ast.Tuple(elts=recur_exprs, ctx=ast.Load()),
            )
        )
    recur_deps.append(ast.Continue())

    return GeneratedPyAST(node=ast.Constant(None), dependencies=recur_deps)


_RECUR_TYPE_HANDLER = {
    RecurType.FN: __fn_recur_to_py_ast,
    RecurType.METHOD: __deftype_method_recur_to_py_ast,
    RecurType.LOOP: __loop_recur_to_py_ast,
}


def _recur_to_py_ast(ctx: GeneratorContext, node: Recur) -> GeneratedPyAST:
    """Return a Python AST Node for a `recur` expression.

    Note that `recur` nodes can only legally appear in two AST locations:
      (1) in :then or :else expressions in :if nodes, and
      (2) in :ret expressions in :do nodes

    As such, both of these handlers special case the recur construct, as it
    is the only case in which the code generator emits a statement rather than
    an expression."""
    assert node.op == NodeOp.RECUR
    assert ctx.recur_point is not None, "Must have set a recur point to recur"
    handle_recur = _RECUR_TYPE_HANDLER.get(ctx.recur_point.type)
    assert (
        handle_recur is not None
    ), f"No recur point handler defined for {ctx.recur_point.type}"
    ctx.recur_point.has_recur = True
    return handle_recur(ctx, node)


@_with_ast_loc
def _reify_to_py_ast(
    ctx: GeneratorContext, node: Reify, meta_node: Optional[MetaNode] = None
) -> GeneratedPyAST:
    """Return a Python AST Node for a `reify*` expression."""
    assert node.op == NodeOp.REIFY

    meta_ast: Optional[GeneratedPyAST]
    if meta_node is not None:
        meta_ast = gen_py_ast(ctx, meta_node)
    else:
        meta_ast = None

    bases: List[ast.AST] = [
        _BASILISP_WITH_META_INTERFACE_NAME,
        *__deftype_or_reify_bases_to_py_ast(ctx, node, node.interfaces),
    ]
    artificially_abstract_bases = __deftype_or_reify_bases_to_py_ast(
        ctx, node, node.artificially_abstract
    )
    type_name = munge(genname("ReifiedType"))

    with ctx.new_symbol_table("reify"):
        type_nodes: List[ast.AST] = [
            ast.Assign(
                targets=[ast.Name(id="_meta", ctx=ast.Store())],
                value=ast.Call(
                    func=_ATTRIB_FIELD_FN_NAME,
                    args=[],
                    keywords=[ast.keyword(arg="default", value=ast.Constant(None))],
                ),
            ),
            ast.FunctionDef(
                name="meta",
                args=ast.arguments(
                    args=[
                        ast.arg(arg="self", annotation=None),
                    ],
                    kwarg=None,
                    vararg=None,
                    kwonlyargs=[],
                    defaults=[],
                    kw_defaults=[],
                ),
                body=[ast.Return(value=_load_attr("self._meta"))],
                decorator_list=[],
                returns=None,
            ),
            ast.FunctionDef(
                name="with_meta",
                args=ast.arguments(
                    args=[
                        ast.arg(arg="self", annotation=None),
                        ast.arg(arg="new_meta", annotation=None),
                    ],
                    kwarg=None,
                    vararg=None,
                    kwonlyargs=[],
                    defaults=[],
                    kw_defaults=[],
                ),
                body=[
                    ast.Return(
                        value=ast.Call(
                            func=ast.Name(id=type_name, ctx=ast.Load()),
                            args=[ast.Name(id="new_meta", ctx=ast.Load())],
                            keywords=[],
                        )
                    )
                ],
                decorator_list=[],
                returns=None,
            ),
        ]
        type_deps: List[ast.AST] = []

        for member in node.members:
            type_ast = __deftype_member_to_py_ast(ctx, member)
            type_nodes.append(type_ast.node)
            type_nodes.extend(type_ast.dependencies)

        return GeneratedPyAST(
            node=ast.Call(
                func=ast.Name(id=type_name, ctx=ast.Load()),
                args=[] if meta_ast is None else [meta_ast.node],
                keywords=[],
            ),
            dependencies=list(
                chain(
                    type_deps,
                    [] if meta_ast is None else meta_ast.dependencies,
                    [
                        _class_ast(
                            type_name,
                            type_nodes,
                            bases=bases,
                            members=chain(
                                ["meta", "with_meta"], node.python_member_names
                            ),
                            verified_abstract=node.verified_abstract,
                            artificially_abstract_bases=artificially_abstract_bases,
                            is_frozen=True,
                            use_slots=_ATTR_SLOTS_ON,
                        )
                    ],
                )
            ),
        )


@_with_ast_loc_deps
def _require_to_py_ast(_: GeneratorContext, node: Require) -> GeneratedPyAST:
    """Return a Python AST node for a Basilisp `require*` expression.

    In Clojure, `require` simply loads the file corresponding to the required
    Namespace directly. At the time that Namespace is loaded, `*ns*` is set as
    the _requiring_ Namespace. The `ns` macro, switches `*ns*` using `in-ns` and
    then begins requiring additional Namespaces (like a depth-first search).
    All of the Namespace setup work is done by the _required_ Namespace's `ns`
    macro, rather than any compiler or runtime mandated machinery.

    Prior to the addition of this form, Basilisp worked much more similarly to
    Python. Most of the import mechanisms were _imposed_ on the required Namespace
    and Namespaces were set up primarily by the compiler and runtime. However, in
    order to support all of the features Clojure supports, it was not practical
    to continue in that direction. The `require*` special form was split from
    `import*` to allow special semantics for required Namespaces (which are still
    compiled with Python modules under the hood).

    `require*` delegates to the current Namespace's `require` method to import
    and alias the required module(s) and manage their own state. Special care is
    taken to ensure that the value of `*ns*` is preserved between requires, since
    the Namespaces themselves are otherwise responsible determining its value."""
    assert node.op == NodeOp.REQUIRE

    last = None
    requiring_ns_name = genname("requiring_ns")
    deps: List[ast.AST] = [
        # Fetch the requiring namespace first prior to initiating requires
        # in order to ensure the require is only made into that namespace
        # (in case the required module changes the value of *ns*)
        ast.Assign(
            targets=[ast.Name(id=requiring_ns_name, ctx=ast.Store())],
            value=_load_attr(_NS_VAR_VALUE),
        )
    ]
    for alias in node.aliases:
        py_require_alias = _var_ns_as_python_sym(alias.name)
        last = ast.Name(id=py_require_alias, ctx=ast.Load())

        deps.append(
            ast.Try(
                body=[
                    ast.Assign(
                        targets=[ast.Name(id=py_require_alias, ctx=ast.Store())],
                        value=ast.Call(
                            func=_load_attr(f"{requiring_ns_name}.require"),
                            args=list(
                                chain(
                                    [ast.Constant(alias.name)],
                                    [
                                        ast.Call(
                                            func=_NEW_SYM_FN_NAME,
                                            args=[ast.Constant(alias.alias)],
                                            keywords=[],
                                        )
                                    ]
                                    if alias.alias is not None
                                    else [],
                                )
                            ),
                            keywords=[],
                        ),
                    )
                ],
                handlers=[],
                orelse=[],
                finalbody=[
                    # Restore the original namespace after each require to ensure that the
                    # following require starts with a clean slate
                    ast.Assign(
                        targets=[_load_attr(_NS_VAR_VALUE, ctx=ast.Store())],
                        value=ast.Name(id=requiring_ns_name, ctx=ast.Load()),
                    ),
                ],
            )
        )

    deps.append(
        ast.Delete(targets=[ast.Name(id=requiring_ns_name, ctx=ast.Del())]),
    )

    assert last is not None, "require* node must have at least one import"
    return GeneratedPyAST(node=last, dependencies=deps)


@_with_ast_loc
def _set_bang_to_py_ast(ctx: GeneratorContext, node: SetBang) -> GeneratedPyAST:
    """Return a Python AST Node for a `set!` expression."""
    assert node.op == NodeOp.SET_BANG

    val_ast = gen_py_ast(ctx, node.val)

    target = node.target
    assert isinstance(
        target, (HostField, Local, VarRef)
    ), f"invalid set! target type {type(target)}"

    if isinstance(target, HostField):
        target_ast = _interop_prop_to_py_ast(ctx, target, is_assigning=True)
    elif isinstance(target, VarRef):
        target_ast = _var_sym_to_py_ast(ctx, target, is_assigning=True)
    elif isinstance(target, Local):
        target_ast = _local_sym_to_py_ast(ctx, target, is_assigning=True)
    else:  # pragma: no cover
        raise GeneratorException(
            f"invalid set! target type {type(target)}", lisp_ast=target
        )

    if node.env.pos == NodeSyntacticPosition.EXPR:
        val_temp_name = genname(_SET_BANG_TEMP_PREFIX)
        return GeneratedPyAST(
            node=ast.Name(id=val_temp_name, ctx=ast.Load()),
            dependencies=list(
                chain(
                    val_ast.dependencies,
                    [
                        ast.Assign(
                            targets=[ast.Name(id=val_temp_name, ctx=ast.Store())],
                            value=val_ast.node,
                        )
                    ],
                    target_ast.dependencies,
                    [ast.Assign(targets=[target_ast.node], value=val_ast.node)],
                )
            ),
        )
    else:
        return GeneratedPyAST(
            node=_noop_node(),
            dependencies=list(
                chain(
                    val_ast.dependencies,
                    target_ast.dependencies,
                    [ast.Assign(targets=[target_ast.node], value=val_ast.node)],
                )
            ),
        )


@_with_ast_loc_deps
def _throw_to_py_ast(ctx: GeneratorContext, node: Throw) -> GeneratedPyAST:
    """Return a Python AST Node for a `throw` expression."""
    assert node.op == NodeOp.THROW

    exc_ast = gen_py_ast(ctx, node.exception)
    raise_body = ast.Raise(exc=exc_ast.node, cause=None)

    return GeneratedPyAST(
        node=_noop_node(),
        dependencies=list(chain(exc_ast.dependencies, [raise_body])),
    )


def __catch_to_py_ast(
    ctx: GeneratorContext, catch: Catch, *, try_expr_name: str
) -> ast.ExceptHandler:
    assert catch.class_.op in {NodeOp.MAYBE_CLASS, NodeOp.MAYBE_HOST_FORM}

    exc_type = gen_py_ast(ctx, catch.class_)
    assert (
        count(exc_type.dependencies) == 0
    ), ":maybe-class and :maybe-host-form node cannot have dependency nodes"

    exc_binding = catch.local
    assert (
        exc_binding.local == LocalType.CATCH
    ), ":local of :binding node must be :catch for Catch node"

    with ctx.new_symbol_table("catch"):
        catch_exc_name = genname(munge(exc_binding.name))
        ctx.symbol_table.new_symbol(
            sym.symbol(exc_binding.name), catch_exc_name, LocalType.CATCH
        )
        catch_ast = _synthetic_do_to_py_ast(ctx, catch.body)
        return ast.ExceptHandler(
            type=exc_type.node,
            name=catch_exc_name,
            body=list(
                chain(
                    map(statementize, catch_ast.dependencies),
                    [
                        ast.Assign(
                            targets=[ast.Name(id=try_expr_name, ctx=ast.Store())],
                            value=catch_ast.node,
                        )
                    ],
                )
            ),
        )


@_with_ast_loc_deps
def _try_to_py_ast(ctx: GeneratorContext, node: Try) -> GeneratedPyAST:
    """Return a Python AST Node for a `try` expression."""
    assert node.op == NodeOp.TRY

    try_expr_name = genname("try_expr")

    body_ast = _synthetic_do_to_py_ast(ctx, node.body)
    catch_handlers = list(
        map(partial(__catch_to_py_ast, ctx, try_expr_name=try_expr_name), node.catches)
    )

    finallys: List[ast.AST] = []
    if node.finally_ is not None:
        finally_ast = _synthetic_do_to_py_ast(ctx, node.finally_)
        finallys.extend(map(statementize, finally_ast.dependencies))
        finallys.append(statementize(finally_ast.node))

    return GeneratedPyAST(
        node=ast.Name(id=try_expr_name, ctx=ast.Load()),
        dependencies=[
            ast.Try(
                body=list(
                    chain(
                        body_ast.dependencies,
                        [
                            ast.Assign(
                                targets=[ast.Name(id=try_expr_name, ctx=ast.Store())],
                                value=body_ast.node,
                            )
                        ],
                    )
                ),
                handlers=catch_handlers,
                orelse=[],
                finalbody=finallys,
            )
        ],
    )


##########
# Symbols
##########


@_with_ast_loc
def _local_sym_to_py_ast(
    ctx: GeneratorContext, node: Local, is_assigning: bool = False
) -> GeneratedPyAST:
    """Generate a Python AST node for accessing a locally defined Python variable."""
    assert node.op == NodeOp.LOCAL

    sym_entry = ctx.symbol_table.find_symbol(sym.symbol(node.name))
    assert sym_entry is not None

    if node.local == LocalType.FIELD:
        this_entry = ctx.symbol_table.find_symbol(ctx.current_this)
        assert this_entry is not None, "Field type local must have this"

        return GeneratedPyAST(
            node=_load_attr(
                f"{this_entry.munged}.{sym_entry.munged}",
                ctx=ast.Store() if is_assigning else ast.Load(),
            )
        )
    else:
        return GeneratedPyAST(
            node=ast.Name(
                id=sym_entry.munged, ctx=ast.Store() if is_assigning else ast.Load()
            )
        )


def __name_in_module(name: str, module: BasilispModule) -> Optional[str]:
    """Resolve the name inside of module. If the munged name can be found inside the
    module, return the munged name. Return None otherwise."""
    safe_name = munge(name)
    if safe_name not in module.__dict__:
        safe_name = munge(name, allow_builtins=True)

    return safe_name if safe_name in module.__dict__ else None


def __var_direct_link_to_py_ast(
    current_ns: runtime.Namespace, var: runtime.Var, py_var_ctx: ast.AST
) -> Optional[GeneratedPyAST]:
    """Attempt to directly link a Var reference to a Python variable in the module of
    the current Namespace.

    We can direct link a Var if and only if a munged version of the Var name can be
    found in the Var namespace module."""
    var_ns = var.ns
    var_name = var.name.name

    safe_name = __name_in_module(var_name, var_ns.module)
    if safe_name is not None:
        if var_ns is current_ns:
            return GeneratedPyAST(node=ast.Name(id=safe_name, ctx=py_var_ctx))

        safe_ns = _var_ns_as_python_sym(var_ns.name)
        aliased_ns_name = __name_in_module(safe_ns, current_ns.module)
        if aliased_ns_name is not None:
            return GeneratedPyAST(
                node=_load_attr(
                    f"{aliased_ns_name}.{safe_name}",
                    ctx=py_var_ctx,
                )
            )
    return None


def __var_find_to_py_ast(
    var_name: str, ns_name: str, py_var_ctx: ast.AST
) -> GeneratedPyAST:
    """Generate Var.find calls for the named symbol."""
    return GeneratedPyAST(
        node=ast.Attribute(
            value=ast.Call(
                func=_FIND_VAR_FN_NAME,
                args=[
                    ast.Call(
                        func=_NEW_SYM_FN_NAME,
                        args=[ast.Constant(var_name)],
                        keywords=[ast.keyword(arg="ns", value=ast.Constant(ns_name))],
                    )
                ],
                keywords=[],
            ),
            attr="value",
            ctx=py_var_ctx,
        )
    )


@_with_ast_loc
def _var_sym_to_py_ast(  # pylint: disable=too-many-branches
    ctx: GeneratorContext, node: VarRef, is_assigning: bool = False
) -> GeneratedPyAST:
    """Generate a Python AST node for accessing a Var.

    If the Var is marked as :dynamic or :redef or the compiler option
    USE_VAR_INDIRECTION is active, do not compile to a direct access.
    If the corresponding function name is not defined in a Python module,
    no direct variable access is possible and Var.find indirection must be
    used."""
    assert node.op == NodeOp.VAR

    var = node.var
    var_ns = var.ns
    var_ns_name = var_ns.name
    var_name = var.name.name
    py_var_ctx = ast.Store() if is_assigning else ast.Load()

    # Return the actual Var, rather than its value if requested
    if node.return_var:
        return GeneratedPyAST(
            node=ast.Call(
                func=_FIND_VAR_FN_NAME,
                args=[
                    ast.Call(
                        func=_NEW_SYM_FN_NAME,
                        args=[ast.Constant(var_name)],
                        keywords=[
                            ast.keyword(arg="ns", value=ast.Constant(var_ns_name))
                        ],
                    )
                ],
                keywords=[],
            )
        )

    # Check if we should use Var indirection
    if (
        ctx.has_var_indirection_override
        or ctx.use_var_indirection
        or _is_dynamic(var)
        or _is_redefable(var)
    ):
        return __var_find_to_py_ast(var_name, var_ns_name, py_var_ctx)

    # Otherwise, try to direct-link it like a Python variable
    direct_link = __var_direct_link_to_py_ast(ctx.current_ns, var, py_var_ctx)
    if direct_link is not None:
        return direct_link

    if ctx.warn_on_var_indirection:
        logger.warning(
            f"could not resolve a direct link to Var '{var_name}' "
            f"({node.env.ns}:{node.env.line})"
        )

    # If we failed to direct link, we can always fall back on Var indirection
    return __var_find_to_py_ast(var_name, var_ns_name, py_var_ctx)


#################
# Python Interop
#################


@_with_ast_loc
def _interop_call_to_py_ast(ctx: GeneratorContext, node: HostCall) -> GeneratedPyAST:
    """Generate a Python AST node for Python interop method calls."""
    assert node.op == NodeOp.HOST_CALL

    target_ast = gen_py_ast(ctx, node.target)
    args_deps, args_nodes = _collection_ast(ctx, node.args)
    kwargs_deps, kwargs_nodes = _kwargs_ast(ctx, node.kwargs)

    return GeneratedPyAST(
        node=ast.Call(
            func=ast.Attribute(
                value=target_ast.node,
                attr=munge(node.method, allow_builtins=True),
                ctx=ast.Load(),
            ),
            args=list(args_nodes),
            keywords=list(kwargs_nodes),
        ),
        dependencies=list(chain(target_ast.dependencies, args_deps, kwargs_deps)),
    )


@_with_ast_loc
def _interop_prop_to_py_ast(
    ctx: GeneratorContext, node: HostField, is_assigning: bool = False
) -> GeneratedPyAST:
    """Generate a Python AST node for Python interop property access."""
    assert node.op == NodeOp.HOST_FIELD

    target_ast = gen_py_ast(ctx, node.target)

    return GeneratedPyAST(
        node=ast.Attribute(
            value=target_ast.node,
            attr=munge(node.field),
            ctx=ast.Store() if is_assigning else ast.Load(),
        ),
        dependencies=target_ast.dependencies,
    )


@_with_ast_loc
def _maybe_class_to_py_ast(_: GeneratorContext, node: MaybeClass) -> GeneratedPyAST:
    """Generate a Python AST node for accessing a potential Python module
    variable name."""
    assert node.op == NodeOp.MAYBE_CLASS
    return GeneratedPyAST(
        node=ast.Name(id=_MODULE_ALIASES.get(node.class_, node.class_), ctx=ast.Load())
    )


@_with_ast_loc
def _maybe_host_form_to_py_ast(
    _: GeneratorContext, node: MaybeHostForm
) -> GeneratedPyAST:
    """Generate a Python AST node for accessing a potential Python module
    variable name with a namespace."""
    assert node.op == NodeOp.MAYBE_HOST_FORM
    return GeneratedPyAST(
        node=_load_attr(f"{_MODULE_ALIASES.get(node.class_, node.class_)}.{node.field}")
    )


#########################
# Non-Quoted Collections
#########################


@_with_ast_loc
def _map_to_py_ast(
    ctx: GeneratorContext, node: MapNode, meta_node: Optional[MetaNode] = None
) -> GeneratedPyAST:
    assert node.op == NodeOp.MAP

    meta_ast: Optional[GeneratedPyAST]
    if meta_node is not None:
        meta_ast = gen_py_ast(ctx, meta_node)
    else:
        meta_ast = None

    key_deps, keys = _chain_py_ast(*map(partial(gen_py_ast, ctx), node.keys))
    val_deps, vals = _chain_py_ast(*map(partial(gen_py_ast, ctx), node.vals))
    return GeneratedPyAST(
        node=ast.Call(
            func=_NEW_MAP_FN_NAME,
            args=[ast.Dict(keys=list(keys), values=list(vals))],
            keywords=Maybe(meta_ast)
            .map(lambda p: [ast.keyword(arg="meta", value=p.node)])
            .or_else_get([]),
        ),
        dependencies=list(
            chain(
                Maybe(meta_ast).map(lambda p: p.dependencies).or_else_get([]),
                key_deps,
                val_deps,
            )
        ),
    )


@_with_ast_loc
def _queue_to_py_ast(
    ctx: GeneratorContext, node: QueueNode, meta_node: Optional[MetaNode] = None
) -> GeneratedPyAST:
    assert node.op == NodeOp.QUEUE

    meta_ast: Optional[GeneratedPyAST]
    if meta_node is not None:
        meta_ast = gen_py_ast(ctx, meta_node)
    else:
        meta_ast = None

    elem_deps, elems = _chain_py_ast(*map(partial(gen_py_ast, ctx), node.items))
    return GeneratedPyAST(
        node=ast.Call(
            func=_NEW_QUEUE_FN_NAME,
            args=[ast.List(list(elems), ast.Load())],
            keywords=Maybe(meta_ast)
            .map(lambda p: [ast.keyword(arg="meta", value=p.node)])
            .or_else_get([]),
        ),
        dependencies=list(
            chain(
                Maybe(meta_ast).map(lambda p: p.dependencies).or_else_get([]), elem_deps
            )
        ),
    )


@_with_ast_loc
def _set_to_py_ast(
    ctx: GeneratorContext, node: SetNode, meta_node: Optional[MetaNode] = None
) -> GeneratedPyAST:
    assert node.op == NodeOp.SET

    meta_ast: Optional[GeneratedPyAST]
    if meta_node is not None:
        meta_ast = gen_py_ast(ctx, meta_node)
    else:
        meta_ast = None

    elem_deps, elems = _chain_py_ast(*map(partial(gen_py_ast, ctx), node.items))
    return GeneratedPyAST(
        node=ast.Call(
            func=_NEW_SET_FN_NAME,
            args=[ast.List(list(elems), ast.Load())],
            keywords=Maybe(meta_ast)
            .map(lambda p: [ast.keyword(arg="meta", value=p.node)])
            .or_else_get([]),
        ),
        dependencies=list(
            chain(
                Maybe(meta_ast).map(lambda p: p.dependencies).or_else_get([]), elem_deps
            )
        ),
    )


@_with_ast_loc
def _vec_to_py_ast(
    ctx: GeneratorContext, node: VectorNode, meta_node: Optional[MetaNode] = None
) -> GeneratedPyAST:
    assert node.op == NodeOp.VECTOR

    meta_ast: Optional[GeneratedPyAST]
    if meta_node is not None:
        meta_ast = gen_py_ast(ctx, meta_node)
    else:
        meta_ast = None

    elem_deps, elems = _chain_py_ast(*map(partial(gen_py_ast, ctx), node.items))
    return GeneratedPyAST(
        node=ast.Call(
            func=_NEW_VEC_FN_NAME,
            args=[ast.List(list(elems), ast.Load())],
            keywords=Maybe(meta_ast)
            .map(lambda p: [ast.keyword(arg="meta", value=p.node)])
            .or_else_get([]),
        ),
        dependencies=list(
            chain(
                Maybe(meta_ast).map(lambda p: list(p.dependencies)).or_else_get([]),
                elem_deps,
            )
        ),
    )


#####################
# Python Collections
#####################


@_with_ast_loc
def _py_dict_to_py_ast(ctx: GeneratorContext, node: PyDict) -> GeneratedPyAST:
    assert node.op == NodeOp.PY_DICT

    key_deps, keys = _chain_py_ast(*map(partial(gen_py_ast, ctx), node.keys))
    val_deps, vals = _chain_py_ast(*map(partial(gen_py_ast, ctx), node.vals))
    return GeneratedPyAST(
        node=ast.Dict(keys=list(keys), values=list(vals)),
        dependencies=list(chain(key_deps, val_deps)),
    )


@_with_ast_loc
def _py_list_to_py_ast(ctx: GeneratorContext, node: PyList) -> GeneratedPyAST:
    assert node.op == NodeOp.PY_LIST

    elem_deps, elems = _chain_py_ast(*map(partial(gen_py_ast, ctx), node.items))
    return GeneratedPyAST(
        node=ast.List(elts=list(elems), ctx=ast.Load()), dependencies=list(elem_deps)
    )


@_with_ast_loc
def _py_set_to_py_ast(ctx: GeneratorContext, node: PySet) -> GeneratedPyAST:
    assert node.op == NodeOp.PY_SET

    elem_deps, elems = _chain_py_ast(*map(partial(gen_py_ast, ctx), node.items))
    return GeneratedPyAST(node=ast.Set(elts=list(elems)), dependencies=list(elem_deps))


@_with_ast_loc
def _py_tuple_to_py_ast(ctx: GeneratorContext, node: PyTuple) -> GeneratedPyAST:
    assert node.op == NodeOp.PY_TUPLE

    elem_deps, elems = _chain_py_ast(*map(partial(gen_py_ast, ctx), node.items))
    return GeneratedPyAST(
        node=ast.Tuple(elts=list(elems), ctx=ast.Load()), dependencies=list(elem_deps)
    )


############
# With Meta
############


_WITH_META_EXPR_HANDLER = {
    NodeOp.FN: _fn_to_py_ast,
    NodeOp.MAP: _map_to_py_ast,
    NodeOp.QUEUE: _queue_to_py_ast,
    NodeOp.REIFY: _reify_to_py_ast,
    NodeOp.SET: _set_to_py_ast,
    NodeOp.VECTOR: _vec_to_py_ast,
}


def _with_meta_to_py_ast(
    ctx: GeneratorContext, node: WithMeta, **kwargs
) -> GeneratedPyAST:
    """Generate a Python AST node for Python interop method calls."""
    assert node.op == NodeOp.WITH_META

    handle_expr = _WITH_META_EXPR_HANDLER.get(node.expr.op)
    assert (
        handle_expr is not None
    ), "No expression handler for with-meta child node type"
    return handle_expr(ctx, node.expr, meta_node=node.meta, **kwargs)


#################
# Constant Nodes
#################


@functools.singledispatch
def _const_val_to_py_ast(form: LispForm, _: GeneratorContext) -> GeneratedPyAST:
    """Generate Python AST nodes for constant Lisp forms.

    Nested values in collections for :const nodes are not analyzed, so recursive
    structures need to call into this function to generate Python AST nodes for
    nested elements. For top-level :const Lisp AST nodes, see
    `_const_node_to_py_ast`."""
    raise GeneratorException(f"No constant handler is defined for type {type(form)}")


def _collection_literal_to_py_ast(
    ctx: GeneratorContext, form: Iterable[LispForm]
) -> Iterable[GeneratedPyAST]:
    """Turn a quoted collection literal of Lisp forms into Python AST nodes.

    This function can only handle constant values. It does not call back into
    the generic AST generators, so only constant values will be generated down
    this path."""
    yield from map(lambda form: _const_val_to_py_ast(form, ctx), form)


def _const_meta_kwargs_ast(  # pylint:disable=inconsistent-return-statements
    ctx: GeneratorContext, form: LispForm
) -> Optional[GeneratedPyAST]:
    if isinstance(form, IMeta) and form.meta is not None:
        genned = _const_val_to_py_ast(_clean_meta(form), ctx)
        return GeneratedPyAST(
            node=ast.keyword(arg="meta", value=genned.node),
            dependencies=genned.dependencies,
        )
    else:
        return None


@_const_val_to_py_ast.register(bool)
@_const_val_to_py_ast.register(type(None))
@_const_val_to_py_ast.register(complex)
@_const_val_to_py_ast.register(float)
@_const_val_to_py_ast.register(int)
@_const_val_to_py_ast.register(str)
@_simple_ast_generator
def _py_const_to_py_ast(form: Union[bool, None], _: GeneratorContext) -> ast.AST:
    return ast.Constant(form)


@_const_val_to_py_ast.register(sym.Symbol)
def _const_sym_to_py_ast(form: sym.Symbol, ctx: GeneratorContext) -> GeneratedPyAST:
    meta = _const_meta_kwargs_ast(ctx, form)

    sym_kwarg = (
        Maybe(form.ns)
        .map(lambda v: [ast.keyword(arg="ns", value=ast.Constant(v))])
        .or_else(list)
    )
    meta_kwarg = Maybe(meta).map(lambda p: [p.node]).or_else(list)
    base_sym = ast.Call(
        func=_NEW_SYM_FN_NAME,
        args=[ast.Constant(form.name)],
        keywords=list(chain(sym_kwarg, meta_kwarg)),
    )

    return GeneratedPyAST(
        node=base_sym,
        dependencies=Maybe(meta).map(lambda p: p.dependencies).or_else_get([]),
    )


@_const_val_to_py_ast.register(kw.Keyword)
@_simple_ast_generator
def _kw_to_py_ast(form: kw.Keyword, _: GeneratorContext) -> ast.AST:
    kwarg = (
        Maybe(form.ns)
        .map(lambda ns: [ast.keyword(arg="ns", value=ast.Constant(form.ns))])
        .or_else(list)
    )
    return ast.Call(
        func=_NEW_KW_FN_NAME,
        args=[ast.Constant(hash(form)), ast.Constant(form.name)],
        keywords=kwarg,
    )


@_const_val_to_py_ast.register(Decimal)
@_simple_ast_generator
def _decimal_to_py_ast(form: Decimal, _: GeneratorContext) -> ast.AST:
    return ast.Call(
        func=_NEW_DECIMAL_FN_NAME, args=[ast.Constant(str(form))], keywords=[]
    )


@_const_val_to_py_ast.register(Fraction)
@_simple_ast_generator
def _fraction_to_py_ast(form: Fraction, _: GeneratorContext) -> ast.AST:
    return ast.Call(
        func=_NEW_FRACTION_FN_NAME,
        args=[ast.Constant(form.numerator), ast.Constant(form.denominator)],
        keywords=[],
    )


@_const_val_to_py_ast.register(datetime)
@_simple_ast_generator
def _inst_to_py_ast(form: datetime, _: GeneratorContext) -> ast.AST:
    return ast.Call(
        func=_NEW_INST_FN_NAME, args=[ast.Constant(form.isoformat())], keywords=[]
    )


@_const_val_to_py_ast.register(type(re.compile(r"")))
@_simple_ast_generator
def _regex_to_py_ast(form: Pattern, _: GeneratorContext) -> ast.AST:
    return ast.Call(
        func=_NEW_REGEX_FN_NAME, args=[ast.Constant(form.pattern)], keywords=[]
    )


@_const_val_to_py_ast.register(uuid.UUID)
@_simple_ast_generator
def _uuid_to_py_ast(form: uuid.UUID, _: GeneratorContext) -> ast.AST:
    return ast.Call(func=_NEW_UUID_FN_NAME, args=[ast.Constant(str(form))], keywords=[])


@_const_val_to_py_ast.register(dict)
def _const_py_dict_to_py_ast(node: dict, ctx: GeneratorContext) -> GeneratedPyAST:
    key_deps, keys = _chain_py_ast(*_collection_literal_to_py_ast(ctx, node.keys()))
    val_deps, vals = _chain_py_ast(*_collection_literal_to_py_ast(ctx, node.values()))
    return GeneratedPyAST(
        node=ast.Dict(keys=list(keys), values=list(vals)),
        dependencies=list(chain(key_deps, val_deps)),
    )


@_const_val_to_py_ast.register(list)
def _const_py_list_to_py_ast(node: list, ctx: GeneratorContext) -> GeneratedPyAST:
    elem_deps, elems = _chain_py_ast(*_collection_literal_to_py_ast(ctx, node))
    return GeneratedPyAST(
        node=ast.List(elts=list(elems), ctx=ast.Load()), dependencies=list(elem_deps)
    )


@_const_val_to_py_ast.register(set)
def _const_py_set_to_py_ast(node: set, ctx: GeneratorContext) -> GeneratedPyAST:
    elem_deps, elems = _chain_py_ast(*_collection_literal_to_py_ast(ctx, node))
    return GeneratedPyAST(node=ast.Set(elts=list(elems)), dependencies=list(elem_deps))


@_const_val_to_py_ast.register(tuple)
def _const_py_tuple_to_py_ast(node: tuple, ctx: GeneratorContext) -> GeneratedPyAST:
    elem_deps, elems = _chain_py_ast(*_collection_literal_to_py_ast(ctx, node))
    return GeneratedPyAST(
        node=ast.Tuple(elts=list(elems), ctx=ast.Load()), dependencies=list(elem_deps)
    )


@_const_val_to_py_ast.register(lmap.PersistentMap)
def _const_map_to_py_ast(
    form: lmap.PersistentMap, ctx: GeneratorContext
) -> GeneratedPyAST:
    key_deps, keys = _chain_py_ast(*_collection_literal_to_py_ast(ctx, form.keys()))
    val_deps, vals = _chain_py_ast(*_collection_literal_to_py_ast(ctx, form.values()))
    meta = _const_meta_kwargs_ast(ctx, form)
    return GeneratedPyAST(
        node=ast.Call(
            func=_NEW_MAP_FN_NAME,
            args=[ast.Dict(keys=list(keys), values=list(vals))],
            keywords=Maybe(meta).map(lambda p: [p.node]).or_else_get([]),
        ),
        dependencies=list(
            chain(
                key_deps,
                val_deps,
                Maybe(meta).map(lambda p: p.dependencies).or_else_get([]),
            )
        ),
    )


@_const_val_to_py_ast.register(lqueue.PersistentQueue)
def _const_queue_to_py_ast(
    form: lqueue.PersistentQueue, ctx: GeneratorContext
) -> GeneratedPyAST:
    elem_deps, elems = _chain_py_ast(*_collection_literal_to_py_ast(ctx, form))
    meta = _const_meta_kwargs_ast(ctx, form)
    return GeneratedPyAST(
        node=ast.Call(
            func=_NEW_QUEUE_FN_NAME,
            args=[ast.List(list(elems), ast.Load())],
            keywords=Maybe(meta).map(lambda p: [p.node]).or_else_get([]),
        ),
        dependencies=list(
            chain(elem_deps, Maybe(meta).map(lambda p: p.dependencies).or_else_get([]))
        ),
    )


@_const_val_to_py_ast.register(lset.PersistentSet)
def _const_set_to_py_ast(
    form: lset.PersistentSet, ctx: GeneratorContext
) -> GeneratedPyAST:
    elem_deps, elems = _chain_py_ast(*_collection_literal_to_py_ast(ctx, form))
    meta = _const_meta_kwargs_ast(ctx, form)
    return GeneratedPyAST(
        node=ast.Call(
            func=_NEW_SET_FN_NAME,
            args=[ast.List(list(elems), ast.Load())],
            keywords=Maybe(meta).map(lambda p: [p.node]).or_else_get([]),
        ),
        dependencies=list(
            chain(elem_deps, Maybe(meta).map(lambda p: p.dependencies).or_else_get([]))
        ),
    )


@_const_val_to_py_ast.register(IRecord)
def _const_record_to_py_ast(form: IRecord, ctx: GeneratorContext) -> GeneratedPyAST:
    assert isinstance(form, IRecord) and isinstance(
        form, ISeqable
    ), "IRecord types should also be ISeq"

    tp = type(form)
    assert hasattr(tp, "create") and callable(
        tp.create
    ), "IRecord and IType must declare a .create class method"

    form_seq = runtime.to_seq(form)
    assert form_seq is not None, "IRecord types must be iterable"

    keys, vals = [], []
    vals_deps: List[ast.AST] = []
    for k, v in form_seq:
        assert isinstance(k, kw.Keyword), "Record key in seq must be keyword"
        key_nodes = _kw_to_py_ast(k, ctx)
        keys.append(key_nodes.node)
        assert (
            len(key_nodes.dependencies) == 0  # type: ignore[arg-type]
        ), "Simple AST generators must emit no dependencies"

        val_nodes = _const_val_to_py_ast(v, ctx)
        vals.append(val_nodes.node)
        vals_deps.extend(val_nodes.dependencies)

    return GeneratedPyAST(
        node=ast.Call(
            func=_load_attr(f"{tp.__qualname__}.create"),
            args=[
                ast.Call(
                    func=_NEW_MAP_FN_NAME,
                    args=[ast.Dict(keys=keys, values=vals)],
                    keywords=[],
                )
            ],
            keywords=[],
        ),
        dependencies=vals_deps,
    )


@_const_val_to_py_ast.register(llist.PersistentList)
@_const_val_to_py_ast.register(ISeq)
def _const_seq_to_py_ast(
    form: Union[llist.PersistentList, ISeq], ctx: GeneratorContext
) -> GeneratedPyAST:
    elem_deps, elems = _chain_py_ast(*_collection_literal_to_py_ast(ctx, form))

    if isinstance(form, llist.PersistentList):
        meta = _const_meta_kwargs_ast(ctx, form)
    else:
        meta = None

    return GeneratedPyAST(
        node=ast.Call(
            func=_NEW_LIST_FN_NAME,
            args=[ast.List(list(elems), ast.Load())],
            keywords=Maybe(meta).map(lambda p: [p.node]).or_else_get([]),
        ),
        dependencies=list(
            chain(elem_deps, Maybe(meta).map(lambda p: p.dependencies).or_else_get([]))
        ),
    )


@_const_val_to_py_ast.register(IType)
def _const_type_to_py_ast(form: IType, ctx: GeneratorContext) -> GeneratedPyAST:
    tp = type(form)

    ctor_args = []
    ctor_arg_deps: List[ast.AST] = []
    for field in attr.fields(tp):
        field_nodes = _const_val_to_py_ast(getattr(form, field.name, None), ctx)
        ctor_args.append(field_nodes.node)
        ctor_args.extend(field_nodes.dependencies)

    return GeneratedPyAST(
        node=ast.Call(func=_load_attr(tp.__qualname__), args=ctor_args, keywords=[]),
        dependencies=ctor_arg_deps,
    )


@_const_val_to_py_ast.register(vec.PersistentVector)
def _const_vec_to_py_ast(
    form: vec.PersistentVector, ctx: GeneratorContext
) -> GeneratedPyAST:
    elem_deps, elems = _chain_py_ast(*_collection_literal_to_py_ast(ctx, form))
    meta = _const_meta_kwargs_ast(ctx, form)
    return GeneratedPyAST(
        node=ast.Call(
            func=_NEW_VEC_FN_NAME,
            args=[ast.List(list(elems), ast.Load())],
            keywords=Maybe(meta).map(lambda p: [p.node]).or_else_get([]),
        ),
        dependencies=list(
            chain(
                elem_deps,
                Maybe(meta).map(lambda p: list(p.dependencies)).or_else_get([]),
            )
        ),
    )


@_with_ast_loc
def _const_node_to_py_ast(ctx: GeneratorContext, lisp_ast: Const) -> GeneratedPyAST:
    """Generate Python AST nodes for a :const Lisp AST node.

    Nested values in collections for :const nodes are not analyzed. Consequently,
    this function cannot be called recursively for those nested values. Instead,
    call `_const_val_to_py_ast` on nested values."""
    assert lisp_ast.op == NodeOp.CONST
    return _const_val_to_py_ast(lisp_ast.val, ctx)


_NODE_HANDLERS: Mapping[NodeOp, PyASTGenerator] = {
    NodeOp.AWAIT: _await_to_py_ast,
    NodeOp.CONST: _const_node_to_py_ast,
    NodeOp.DEF: _def_to_py_ast,
    NodeOp.DEFTYPE: _deftype_to_py_ast,
    NodeOp.DO: _do_to_py_ast,
    NodeOp.FN: _fn_to_py_ast,
    NodeOp.HOST_CALL: _interop_call_to_py_ast,
    NodeOp.HOST_FIELD: _interop_prop_to_py_ast,
    NodeOp.IF: _if_to_py_ast,
    NodeOp.IMPORT: _import_to_py_ast,
    NodeOp.INVOKE: _invoke_to_py_ast,
    NodeOp.LET: _let_to_py_ast,
    NodeOp.LETFN: _letfn_to_py_ast,
    NodeOp.LOCAL: _local_sym_to_py_ast,
    NodeOp.LOOP: _loop_to_py_ast,
    NodeOp.MAP: _map_to_py_ast,
    NodeOp.MAYBE_CLASS: _maybe_class_to_py_ast,
    NodeOp.MAYBE_HOST_FORM: _maybe_host_form_to_py_ast,
    NodeOp.PY_DICT: _py_dict_to_py_ast,
    NodeOp.PY_LIST: _py_list_to_py_ast,
    NodeOp.PY_SET: _py_set_to_py_ast,
    NodeOp.PY_TUPLE: _py_tuple_to_py_ast,
    NodeOp.QUEUE: _queue_to_py_ast,
    NodeOp.QUOTE: _quote_to_py_ast,
    NodeOp.RECUR: _recur_to_py_ast,  # type: ignore
    NodeOp.REIFY: _reify_to_py_ast,
    NodeOp.REQUIRE: _require_to_py_ast,
    NodeOp.SET: _set_to_py_ast,
    NodeOp.SET_BANG: _set_bang_to_py_ast,
    NodeOp.THROW: _throw_to_py_ast,
    NodeOp.TRY: _try_to_py_ast,
    NodeOp.VAR: _var_sym_to_py_ast,
    NodeOp.VECTOR: _vec_to_py_ast,
    NodeOp.WITH_META: _with_meta_to_py_ast,  # type: ignore
}


###################
# Public Functions
###################


def gen_py_ast(ctx: GeneratorContext, lisp_ast: Node) -> GeneratedPyAST:
    """Take a Lisp AST node as an argument and produce zero or more Python
    AST nodes.

    This is the primary entrypoint for generating AST nodes from Lisp
    syntax. It may be called recursively to compile child forms."""
    op: NodeOp = lisp_ast.op
    assert op is not None, "Lisp AST nodes must have an :op key"
    handle_node = _NODE_HANDLERS.get(op)
    assert (
        handle_node is not None
    ), f"Lisp AST nodes :op has no handler defined for op {op}"
    return handle_node(ctx, lisp_ast)


#############################
# Bootstrap Basilisp Modules
#############################


def _module_imports(ns: runtime.Namespace) -> Iterable[ast.Import]:
    """Generate the Python Import AST node for importing all required
    language support modules."""
    # Yield `import basilisp` so code attempting to call fully qualified
    # `basilisp.lang...` modules don't result in compiler errors
    yield ast.Import(names=[ast.alias(name="basilisp", asname=None)])
    for s in sorted(ns.imports.keys(), key=lambda s: s.name):
        name = s.name
        alias = _MODULE_ALIASES.get(name, None)
        yield ast.Import(names=[ast.alias(name=name, asname=alias)])


def _from_module_import() -> ast.ImportFrom:
    """Generate the Python From ... Import AST node for importing
    language support modules."""
    return ast.ImportFrom(
        module="basilisp.lang.runtime",
        names=[
            ast.alias(name="Var", asname=_VAR_ALIAS),
        ],
        level=0,
    )


def _ns_var(
    py_ns_var: str = _NS_VAR, lisp_ns_var: str = LISP_NS_VAR, lisp_ns_ns: str = CORE_NS
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
                    args=[ast.Constant(lisp_ns_var)],
                    keywords=[ast.keyword(arg="ns", value=ast.Constant(lisp_ns_ns))],
                )
            ],
            keywords=[],
        ),
    )


def py_module_preamble(
    ns: runtime.Namespace,
) -> GeneratedPyAST:
    """Bootstrap a new module with imports and other boilerplate."""
    preamble: List[ast.AST] = []
    preamble.extend(_module_imports(ns))
    preamble.append(_from_module_import())
    preamble.append(_ns_var())
    return GeneratedPyAST(node=ast.Constant(None), dependencies=preamble)
