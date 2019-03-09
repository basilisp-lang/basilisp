import ast
import collections
import contextlib
import importlib
import logging
import re
import types
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from functools import wraps, partial
from itertools import chain
from typing import (
    Iterable,
    Pattern,
    Optional,
    List,
    Union,
    Deque,
    Dict,
    Callable,
    Tuple,
    Type,
    Collection,
)

import attr

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
from basilisp.lang.compiler.constants import (
    DEFAULT_COMPILER_FILE_PATH,
    SYM_DYNAMIC_META_KEY,
    SYM_NO_WARN_ON_REDEF_META_KEY,
    SYM_REDEF_META_KEY,
)
from basilisp.lang.compiler.exception import CompilerException, CompilerPhase
from basilisp.lang.compiler.nodes import (
    Node,
    NodeOp,
    ConstType,
    Const,
    WithMeta,
    Def,
    Do,
    If,
    VarRef,
    HostCall,
    HostField,
    MaybeClass,
    MaybeHostForm,
    Map as MapNode,
    Set as SetNode,
    Vector as VectorNode,
    Quote,
    ReaderLispForm,
    Invoke,
    Throw,
    Try,
    LocalType,
    SetBang,
    Local,
    Let,
    Loop,
    Recur,
    Fn,
    Import,
    FnMethod,
    Binding,
    NodeEnv,
    Catch,
)
from basilisp.lang.runtime import Var
from basilisp.lang.typing import LispForm
from basilisp.lang.util import count, genname, munge
from basilisp.util import Maybe

# Generator logging
logger = logging.getLogger(__name__)

# Generator options
USE_VAR_INDIRECTION = "use_var_indirection"
WARN_ON_VAR_INDIRECTION = "warn_on_var_indirection"

# Lisp AST node keywords
INIT = kw.keyword("init")

# String constants used in generating code
_BUILTINS_NS = "builtins"
_CORE_NS = "basilisp.core"
_DEFAULT_FN = "__lisp_expr__"
_DO_PREFIX = "lisp_do"
_FN_PREFIX = "lisp_fn"
_IF_PREFIX = "lisp_if"
_IF_RESULT_PREFIX = "if_result"
_IF_TEST_PREFIX = "if_test"
_MULTI_ARITY_ARG_NAME = "multi_arity_args"
_THROW_PREFIX = "lisp_throw"
_TRY_PREFIX = "lisp_try"
_NS_VAR = "__NS"
_LISP_NS_VAR = "*ns*"


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
    _parent: Optional["SymbolTable"] = None
    _table: Dict[sym.Symbol, SymbolTableEntry] = attr.ib(factory=dict)
    _children: Dict[str, "SymbolTable"] = attr.ib(factory=dict)

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

    def append_frame(self, name: str, parent: "SymbolTable" = None) -> "SymbolTable":
        new_frame = SymbolTable(name, parent=parent)
        self._children[name] = new_frame
        return new_frame

    def pop_frame(self, name: str) -> None:
        del self._children[name]

    @contextlib.contextmanager
    def new_frame(self, name):
        """Context manager for creating a new stack frame."""
        new_frame = self.append_frame(name, parent=self)
        yield new_frame
        self.pop_frame(name)


class RecurType(Enum):
    FN = kw.keyword("fn")
    LOOP = kw.keyword("loop")


@attr.s(auto_attribs=True, slots=True)
class RecurPoint:
    loop_id: str
    type: RecurType
    binding_names: Optional[Collection[str]] = None
    is_variadic: Optional[bool] = None
    has_recur: bool = False


class GeneratorContext:
    __slots__ = ("_filename", "_opts", "_recur_points", "_st")

    def __init__(
        self, filename: Optional[str] = None, opts: Optional[Dict[str, bool]] = None
    ) -> None:
        self._filename = Maybe(filename).or_else_get(DEFAULT_COMPILER_FILE_PATH)
        self._opts = Maybe(opts).map(lmap.map).or_else_get(lmap.m())
        self._recur_points: Deque[RecurPoint] = collections.deque([])
        self._st = collections.deque([SymbolTable("<Top>")])

        if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
            for k, v in self._opts:
                logger.debug("Compiler option %s=%s", k, v)

    @property
    def current_ns(self) -> runtime.Namespace:
        return runtime.get_current_ns()

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def use_var_indirection(self) -> bool:
        """If True, compile all variable references using Var.find indirection."""
        return self._opts.entry(USE_VAR_INDIRECTION, False)

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
    def new_symbol_table(self, name):
        old_st = self.symbol_table
        with old_st.new_frame(name) as st:
            self._st.append(st)
            yield st
            self._st.pop()

    def add_import(self, imp: sym.Symbol, mod: types.ModuleType, *aliases: sym.Symbol):
        self.current_ns.add_import(imp, mod, *aliases)

    @property
    def imports(self) -> lmap.Map:
        return self.current_ns.imports


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


def _chain_py_ast(*genned: GeneratedPyAST,) -> Tuple[PyASTStream, PyASTStream]:
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
            return node
        return attr_node(ast.Attribute(value=node, attr=attrs[idx], ctx=ctx), idx + 1)

    return attr_node(ast.Name(id=attrs[0], ctx=ctx), 1)


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


def _clean_meta(form: lmeta.Meta) -> LispForm:
    """Remove reader metadata from the form's meta map."""
    meta = form.meta.discard(reader.READER_LINE_KW, reader.READER_COL_KW)
    if len(meta) == 0:
        return None
    return meta


def _ast_with_loc(
    py_ast: GeneratedPyAST, env: NodeEnv, include_dependencies: bool = False
) -> GeneratedPyAST:
    """Hydrate Generated Python AST nodes with line numbers and column offsets
    if they exist in the node environment."""
    if env.line is not None:
        py_ast.node.lineno = env.line  # type: ignore

        if include_dependencies:
            for dep in py_ast.dependencies:
                dep.lineno = env.line

    if env.col is not None:
        py_ast.node.col_offset = env.col  # type: ignore

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
        .map(lambda m: m.get(SYM_DYNAMIC_META_KEY, None))  # type: ignore
        .or_else_get(False)
    )


def _is_redefable(v: Var) -> bool:
    """Return True if the Var can be redefined."""
    return (
        Maybe(v.meta)
        .map(lambda m: m.get(SYM_REDEF_META_KEY, None))  # type: ignore
        .or_else_get(False)
    )


#######################
# Aliases & Attributes
#######################


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
            ast.AnnAssign,  # type: ignore
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


def __should_warn_on_redef(
    ctx: GeneratorContext, defsym: sym.Symbol, safe_name: str, def_meta: lmap.Map
) -> bool:
    """Return True if the compiler should emit a warning about this name being redefined."""
    no_warn_on_redef = def_meta.entry(SYM_NO_WARN_ON_REDEF_META_KEY, False)
    if no_warn_on_redef:
        return False
    elif safe_name in ctx.current_ns.module.__dict__:
        return True
    elif defsym in ctx.current_ns.interns:
        var = ctx.current_ns.find(defsym)
        assert var is not None, f"Var {defsym} cannot be none here"

        if var.meta is not None and var.meta.entry(SYM_REDEF_META_KEY):
            return False
        elif var.is_bound:
            return True
        else:
            return False
    else:
        return False


@_with_ast_loc
def _def_to_py_ast(  # pylint: disable=too-many-branches
    ctx: GeneratorContext, node: Def
) -> GeneratedPyAST:
    """Return a Python AST Node for a `def` expression."""
    assert node.op == NodeOp.DEF

    defsym = node.name
    is_defn = False

    if INIT in node.children:
        # Since Python function definitions always take the form `def name(...):`,
        # it is redundant to assign them to the their final name after they have
        # been defined under a private alias. This codepath generates `defn`
        # declarations by directly generating the Python `def` with the correct
        # function name and short-circuiting the default double-declaration.
        assert node.init is not None, "Def init must be defined"
        if node.init.op == NodeOp.FN:
            assert isinstance(node.init, Fn)
            def_ast = _fn_to_py_ast(ctx, node.init, def_name=defsym.name)
            is_defn = True
        else:
            def_ast = gen_py_ast(ctx, node.init)
    else:
        def_ast = GeneratedPyAST(node=ast.NameConstant(None))

    ns_name = ast.Call(func=_NEW_SYM_FN_NAME, args=[_NS_VAR_NAME], keywords=[])
    def_name = ast.Call(func=_NEW_SYM_FN_NAME, args=[ast.Str(defsym.name)], keywords=[])
    safe_name = munge(defsym.name)

    assert node.meta is not None, "Meta should always be attached to Def nodes"
    def_meta = node.meta.form
    assert isinstance(def_meta, lmap.Map), "Meta should always be a map"

    # If the Var is marked as dynamic, we need to generate a keyword argument
    # for the generated Python code to set the Var as dynamic
    is_dynamic = def_meta.entry(SYM_DYNAMIC_META_KEY, False)
    dynamic_kwarg = (
        [ast.keyword(arg="dynamic", value=ast.NameConstant(is_dynamic))]
        if is_dynamic
        else []
    )

    # Warn if this symbol is potentially being redefined
    if __should_warn_on_redef(ctx, defsym, safe_name, def_meta):
        logger.warning(
            f"redefining local Python name '{safe_name}' in module '{ctx.current_ns.module.__name__}'"
        )

    meta_ast = gen_py_ast(ctx, node.meta)

    # For defn style def generation, we specifically need to generate the
    # global declaration prior to emitting the Python `def` otherwise the
    # Python compiler will throw an exception during compilation
    # complaining that we assign the value prior to global declaration.
    if is_defn:
        def_dependencies = list(
            chain(
                [] if node.top_level else [ast.Global(names=[safe_name])],
                def_ast.dependencies,
                [] if meta_ast is None else meta_ast.dependencies,
            )
        )
    else:
        def_dependencies = list(
            chain(
                def_ast.dependencies,
                [] if node.top_level else [ast.Global(names=[safe_name])],
                [
                    ast.Assign(
                        targets=[ast.Name(id=safe_name, ctx=ast.Store())],
                        value=def_ast.node,
                    )
                ],
                [] if meta_ast is None else meta_ast.dependencies,
            )
        )

    return GeneratedPyAST(
        node=ast.Call(
            func=_INTERN_VAR_FN_NAME,
            args=[ns_name, def_name, ast.Name(id=safe_name, ctx=ast.Load())],
            keywords=list(
                chain(
                    dynamic_kwarg,
                    []
                    if meta_ast is None
                    else [ast.keyword(arg="meta", value=meta_ast.node)],
                )
            ),  # type: ignore
        ),
        dependencies=def_dependencies,
    )


@_with_ast_loc_deps
def _do_to_py_ast(ctx: GeneratorContext, node: Do) -> GeneratedPyAST:
    """Return a Python AST Node for a `do` expression."""
    assert node.op == NodeOp.DO
    assert not node.is_body

    body_ast = GeneratedPyAST.reduce(
        *chain(
            map(partial(gen_py_ast, ctx), node.statements), [gen_py_ast(ctx, node.ret)]
        )
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
        *chain(
            map(partial(gen_py_ast, ctx), node.statements), [gen_py_ast(ctx, node.ret)]
        )
    )


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


@_with_ast_loc_deps
def __single_arity_fn_to_py_ast(
    ctx: GeneratorContext, node: Fn, method: FnMethod, def_name: Optional[str] = None
) -> GeneratedPyAST:
    """Return a Python AST node for a function with a single arity."""
    assert node.op == NodeOp.FN
    assert method.op == NodeOp.FN_METHOD

    lisp_fn_name = node.local.name if node.local is not None else None
    py_fn_name = __fn_name(lisp_fn_name) if def_name is None else munge(def_name)
    with ctx.new_symbol_table(py_fn_name), ctx.new_recur_point(
        method.loop_id, RecurType.FN, is_variadic=node.is_variadic
    ):
        # Allow named anonymous functions to recursively call themselves
        if lisp_fn_name is not None:
            ctx.symbol_table.new_symbol(
                sym.symbol(lisp_fn_name), py_fn_name, LocalType.FN
            )

        fn_args, varg, fn_body_ast = __fn_args_to_py_ast(
            ctx, method.params, method.body
        )
        return GeneratedPyAST(
            node=ast.Name(id=py_fn_name, ctx=ast.Load()),
            dependencies=[
                ast.FunctionDef(
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
                    decorator_list=[_TRAMPOLINE_FN_NAME]
                    if ctx.recur_point.has_recur
                    else [],
                    returns=None,
                )
            ],
        )


def __multi_arity_dispatch_fn(
    name: str,
    arity_map: Dict[int, str],
    default_name: Optional[str] = None,
    max_fixed_arity: Optional[int] = None,
) -> GeneratedPyAST:
    """Return the Python AST nodes for a argument-length dispatch function
    for multi-arity functions.

    def fn(*args):
        nargs = len(args)
        method = __fn_dispatch_map.get(nargs)
        if method:
            return method(*args)
        # Only if default
        if nargs > max_fixed_arity:
            return default(*args)
        raise RuntimeError
    """
    dispatch_map_name = f"{name}_dispatch_map"

    dispatch_keys, dispatch_vals = [], []
    for k, v in arity_map.items():
        dispatch_keys.append(ast.Num(k))
        dispatch_vals.append(ast.Name(id=v, ctx=ast.Load()))

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
                left=ast.NameConstant(None),
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
                        comparators=[ast.Num(max_fixed_arity)],
                    ),
                    body=[
                        ast.Return(
                            value=ast.Call(
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
                    ast.Str(f"Wrong number of args passed to function: {name}"),
                    ast.Name(id=nargs_name, ctx=ast.Load()),
                ],
                keywords=[],
            ),
            cause=None,
        ),
    ]

    return GeneratedPyAST(
        node=ast.Name(id=name, ctx=ast.Load()),
        dependencies=[
            ast.Assign(
                targets=[ast.Name(id=dispatch_map_name, ctx=ast.Store())],
                value=ast.Dict(keys=dispatch_keys, values=dispatch_vals),
            ),
            ast.FunctionDef(
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
                decorator_list=[],
                returns=None,
            ),
        ],
    )


@_with_ast_loc_deps
def __multi_arity_fn_to_py_ast(
    ctx: GeneratorContext,
    node: Fn,
    methods: Collection[FnMethod],
    def_name: Optional[str] = None,
) -> GeneratedPyAST:
    """Return a Python AST node for a function with multiple arities."""
    assert node.op == NodeOp.FN
    assert all([method.op == NodeOp.FN_METHOD for method in methods])

    lisp_fn_name = node.local.name if node.local is not None else None
    py_fn_name = __fn_name(lisp_fn_name) if def_name is None else munge(def_name)

    arity_to_name = {}
    rest_arity_name: Optional[str] = None
    fn_defs = []
    for method in methods:
        arity_name = f"{py_fn_name}__arity{'_rest' if method.is_variadic else method.fixed_arity}"
        if method.is_variadic:
            rest_arity_name = arity_name
        else:
            arity_to_name[method.fixed_arity] = arity_name

        with ctx.new_symbol_table(arity_name), ctx.new_recur_point(
            method.loop_id, RecurType.FN, is_variadic=node.is_variadic
        ):
            # Allow named anonymous functions to recursively call themselves
            if lisp_fn_name is not None:
                ctx.symbol_table.new_symbol(
                    sym.symbol(lisp_fn_name), py_fn_name, LocalType.FN
                )

            fn_args, varg, fn_body_ast = __fn_args_to_py_ast(
                ctx, method.params, method.body
            )
            fn_defs.append(
                ast.FunctionDef(
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
        py_fn_name,
        arity_to_name,
        default_name=rest_arity_name,
        max_fixed_arity=node.max_fixed_arity,
    )

    return GeneratedPyAST(
        node=dispatch_fn_ast.node,
        dependencies=list(chain(fn_defs, dispatch_fn_ast.dependencies)),
    )


@_with_ast_loc
def _fn_to_py_ast(
    ctx: GeneratorContext, node: Fn, def_name: Optional[str] = None
) -> GeneratedPyAST:
    """Return a Python AST Node for a `fn` expression."""
    assert node.op == NodeOp.FN
    if len(node.methods) == 1:
        return __single_arity_fn_to_py_ast(
            ctx, node, next(iter(node.methods)), def_name=def_name
        )
    else:
        return __multi_arity_fn_to_py_ast(ctx, node, node.methods, def_name=def_name)


@_with_ast_loc_deps
def __if_body_to_py_ast(
    ctx: GeneratorContext, node: Node, result_name: str
) -> GeneratedPyAST:
    """Generate custom `if` nodes to handle `recur` bodies.

    Recur nodes can appear in the then and else expressions of `if` forms.
    Recur nodes generate Python `continue` statements, which we would otherwise
    attempt to insert directly into an expression. Python will complain if
    it finds a statement in an expression AST slot, so we special case the
    recur handling here."""
    if node.op == NodeOp.RECUR and ctx.recur_point.type == RecurType.LOOP:
        assert isinstance(node, Recur)
        return _recur_to_py_ast(ctx, node)
    elif node.op == NodeOp.DO:
        assert isinstance(node, Do)
        if_body = _synthetic_do_to_py_ast(ctx, node.assoc(is_body=True))
        return GeneratedPyAST(
            node=ast.Assign(
                targets=[ast.Name(id=result_name, ctx=ast.Store())], value=if_body.node
            ),
            dependencies=list(map(statementize, if_body.dependencies)),
        )
    else:
        py_ast = gen_py_ast(ctx, node)
        return GeneratedPyAST(
            node=ast.Assign(
                targets=[ast.Name(id=result_name, ctx=ast.Store())], value=py_ast.node
            ),
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
    result_name = genname(_IF_RESULT_PREFIX)

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
        body=list(map(statementize, chain(else_ast.dependencies, [else_ast.node]))),
        orelse=list(map(statementize, chain(then_ast.dependencies, [then_ast.node]))),
    )

    return GeneratedPyAST(
        node=ast.Name(id=result_name, ctx=ast.Load()),
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
        safe_alias = munge(alias.alias)

        try:
            module = importlib.import_module(safe_name)
            if alias.name != alias.alias:
                ctx.add_import(sym.symbol(alias.name), module, sym.symbol(alias.alias))
            else:
                ctx.add_import(sym.symbol(alias.name), module)
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Python module '{alias.name}' not found", node.form, node
            ) from e

        deps.append(
            ast.Assign(
                targets=[ast.Name(id=safe_alias, ctx=ast.Store())],
                value=ast.Call(
                    func=_load_attr("builtins.__import__"),
                    args=[ast.Str(safe_name)],
                    keywords=[],
                ),
            )
        )
        last = ast.Name(id=safe_alias, ctx=ast.Load())

        # Note that we add this import to the live running system in the above
        # calls to `ctx.add_import`, however, since we compile and cache Python
        # bytecode, we need to generate calls to `add_import` for the running
        # namespace so when this code is reloaded from the cache, the runtime
        # is correctly configured.
        deps.append(
            ast.Call(
                func=_load_attr(f"{_NS_VAR_VALUE}.add_import"),
                args=[
                    ast.Call(
                        func=_NEW_SYM_FN_NAME, args=[ast.Str(safe_name)], keywords=[]
                    ),
                    last,
                ],
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

    return GeneratedPyAST(
        node=ast.Call(func=fn_ast.node, args=list(args_nodes), keywords=[]),
        dependencies=list(chain(fn_ast.dependencies, args_deps)),
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

        let_result_name = genname("let_result")
        body_ast = _synthetic_do_to_py_ast(ctx, node.body)
        let_body_ast.extend(map(statementize, body_ast.dependencies))
        let_body_ast.append(
            ast.Assign(
                targets=[ast.Name(id=let_result_name, ctx=ast.Store())],
                value=body_ast.node,
            )
        )

        return GeneratedPyAST(
            node=ast.Name(id=let_result_name, ctx=ast.Load()), dependencies=let_body_ast
        )


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

        loop_result_name = genname("loop")
        with ctx.new_recur_point(
            node.loop_id, RecurType.LOOP, binding_names=binding_names
        ):
            loop_body_ast: List[ast.AST] = []
            body_ast = _synthetic_do_to_py_ast(ctx, node.body)
            loop_body_ast.extend(body_ast.dependencies)
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
                                value=ast.NameConstant(None),
                            )
                        ],
                        init_bindings,
                        [
                            ast.While(
                                test=ast.NameConstant(True),
                                body=loop_body_ast,
                                orelse=[],
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
            args=list(
                chain(
                    [ast.NameConstant(ctx.recur_point.is_variadic)], recur_nodes
                )  # type: ignore
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

    return GeneratedPyAST(node=ast.NameConstant(None), dependencies=recur_deps)


_RECUR_TYPE_HANDLER = {
    RecurType.FN: __fn_recur_to_py_ast,
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
def _set_bang_to_py_ast(ctx: GeneratorContext, node: SetBang) -> GeneratedPyAST:
    """Return a Python AST Node for a `set!` expression."""
    assert node.op == NodeOp.SET_BANG

    val_temp_name = genname("set_bang_val")
    val_ast = gen_py_ast(ctx, node.val)

    target = node.target
    assert isinstance(
        target, (HostField, Local, VarRef)
    ), f"invalid set! target type {type(target)}"

    if isinstance(target, HostField):
        target_ast = _interop_prop_to_py_ast(ctx, target, is_assigning=True)
    elif isinstance(target, VarRef):
        target_ast = _var_sym_to_py_ast(ctx, target, is_assigning=True)
    elif isinstance(target, Local):  # pragma: no cover
        # Local nodes cannot be assigned by any existing code, but the
        # clojure.tools.analyzer.jvm AST adds additional specialized Java
        # nodes to the base clojure.tools.analyzer AST spec which include
        # assignable locals (such as fields in deftype forms). I'm going
        # to keep this branch here for now since I suspect I'll eventually
        # enrich the AST to include assignable locals.
        safe_name = munge(target.name)
        target_ast = GeneratedPyAST(node=ast.Name(id=safe_name, ctx=ast.Store()))

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


@_with_ast_loc_deps
def _throw_to_py_ast(ctx: GeneratorContext, node: Throw) -> GeneratedPyAST:
    """Return a Python AST Node for a `throw` expression."""
    assert node.op == NodeOp.THROW

    throw_fn = genname(_THROW_PREFIX)
    exc_ast = gen_py_ast(ctx, node.exception)
    raise_body = ast.Raise(exc=exc_ast.node, cause=None)

    return GeneratedPyAST(
        node=ast.Call(func=ast.Name(id=throw_fn, ctx=ast.Load()), args=[], keywords=[]),
        dependencies=[
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
                body=list(chain(exc_ast.dependencies, [raise_body])),
                decorator_list=[],
                returns=None,
            )
        ],
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

    return GeneratedPyAST(
        node=ast.Name(
            id=sym_entry.munged, ctx=ast.Store() if is_assigning else ast.Load()
        )
    )


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
                        args=[ast.Str(var_name)],
                        keywords=[ast.keyword(arg="ns", value=ast.Str(ns_name))],
                    )
                ],
                keywords=[],
            ),
            attr="value",
            ctx=py_var_ctx,
        )
    )


@_with_ast_loc
def _var_sym_to_py_ast(
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
    ns = var.ns
    ns_name = ns.name
    ns_module = ns.module
    safe_ns = munge(ns_name)
    var_name = var.name.name
    py_var_ctx = ast.Store() if is_assigning else ast.Load()

    # Return the actual var, rather than its value if requested
    if node.return_var:
        return GeneratedPyAST(
            node=ast.Call(
                func=_FIND_VAR_FN_NAME,
                args=[
                    ast.Call(
                        func=_NEW_SYM_FN_NAME,
                        args=[ast.Str(var_name)],
                        keywords=[ast.keyword(arg="ns", value=ast.Str(ns_name))],
                    )
                ],
                keywords=[],
            )
        )

    # Check if we should use Var indirection
    if ctx.use_var_indirection or _is_dynamic(var) or _is_redefable(var):
        return __var_find_to_py_ast(var_name, ns_name, py_var_ctx)

    # Otherwise, try to direct-link it like a Python variable
    # Try without allowing builtins first
    safe_name = munge(var_name)
    if safe_name not in ns_module.__dict__:
        # Try allowing builtins
        safe_name = munge(var_name, allow_builtins=True)

    if safe_name in ns_module.__dict__:
        if ns is ctx.current_ns:
            return GeneratedPyAST(node=ast.Name(id=safe_name, ctx=py_var_ctx))
        return GeneratedPyAST(node=_load_attr(f"{safe_ns}.{safe_name}", ctx=py_var_ctx))

    if ctx.warn_on_var_indirection:
        logger.warning(f"could not resolve a direct link to Var '{var_name}'")

    return __var_find_to_py_ast(var_name, ns_name, py_var_ctx)


#################
# Python Interop
#################


@_with_ast_loc
def _interop_call_to_py_ast(ctx: GeneratorContext, node: HostCall) -> GeneratedPyAST:
    """Generate a Python AST node for Python interop method calls."""
    assert node.op == NodeOp.HOST_CALL

    target_ast = gen_py_ast(ctx, node.target)
    args_deps, args_nodes = _collection_ast(ctx, node.args)

    return GeneratedPyAST(
        node=ast.Call(
            func=ast.Attribute(
                value=target_ast.node,
                attr=munge(node.method, allow_builtins=True),
                ctx=ast.Load(),
            ),
            args=list(args_nodes),
            keywords=[],
        ),
        dependencies=list(chain(target_ast.dependencies, args_deps)),
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
    return GeneratedPyAST(node=ast.Name(id=node.class_, ctx=ast.Load()))


@_with_ast_loc
def _maybe_host_form_to_py_ast(
    _: GeneratorContext, node: MaybeHostForm
) -> GeneratedPyAST:
    """Generate a Python AST node for accessing a potential Python module
    variable name with a namespace."""
    assert node.op == NodeOp.MAYBE_HOST_FORM
    return GeneratedPyAST(node=_load_attr(f"{node.class_}.{node.field}"))


#########################
# Non-Quoted Collections
#########################


MetaNode = Union[Const, MapNode]


@_with_ast_loc
def _map_to_py_ast(
    ctx: GeneratorContext, node: MapNode, meta_node: Optional[MetaNode] = None
) -> GeneratedPyAST:
    assert node.op == NodeOp.MAP

    if meta_node is not None:
        meta_ast: Optional[GeneratedPyAST] = gen_py_ast(ctx, meta_node)
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
def _set_to_py_ast(
    ctx: GeneratorContext, node: SetNode, meta_node: Optional[MetaNode] = None
) -> GeneratedPyAST:
    assert node.op == NodeOp.SET

    if meta_node is not None:
        meta_ast: Optional[GeneratedPyAST] = gen_py_ast(ctx, meta_node)
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

    if meta_node is not None:
        meta_ast: Optional[GeneratedPyAST] = gen_py_ast(ctx, meta_node)
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


############
# With Meta
############


_WITH_META_EXPR_HANDLER = {  # type: ignore
    NodeOp.FN: None,  # TODO: function with meta
    NodeOp.MAP: _map_to_py_ast,
    NodeOp.SET: _set_to_py_ast,
    NodeOp.VECTOR: _vec_to_py_ast,
}


def _with_meta_to_py_ast(ctx: GeneratorContext, node: WithMeta) -> GeneratedPyAST:
    """Generate a Python AST node for Python interop method calls."""
    assert node.op == NodeOp.WITH_META

    handle_expr = _WITH_META_EXPR_HANDLER.get(node.expr.op)
    assert (
        handle_expr is not None
    ), "No expression handler for with-meta child node type"
    return handle_expr(ctx, node.expr, meta_node=node.meta)  # type: ignore


#################
# Constant Nodes
#################


def _const_meta_kwargs_ast(  # pylint:disable=inconsistent-return-statements
    ctx: GeneratorContext, form: lmeta.Meta
) -> Optional[GeneratedPyAST]:
    if hasattr(form, "meta") and form.meta is not None:
        genned = _const_val_to_py_ast(ctx, _clean_meta(form))
        return GeneratedPyAST(
            node=ast.keyword(arg="meta", value=genned.node),
            dependencies=genned.dependencies,
        )
    else:
        return None


@_simple_ast_generator
def _name_const_to_py_ast(_: GeneratorContext, form: Union[bool, None]) -> ast.AST:
    return ast.NameConstant(form)


@_simple_ast_generator
def _num_to_py_ast(_: GeneratorContext, form: Union[complex, float, int]) -> ast.AST:
    return ast.Num(form)


@_simple_ast_generator
def _str_to_py_ast(_: GeneratorContext, form: str) -> ast.AST:
    return ast.Str(form)


def _const_sym_to_py_ast(ctx: GeneratorContext, form: sym.Symbol) -> GeneratedPyAST:
    meta = _const_meta_kwargs_ast(ctx, form)

    sym_kwargs = (
        Maybe(form.ns)
        .stream()
        .map(lambda v: ast.keyword(arg="ns", value=ast.Str(v)))
        .to_list()
    )
    sym_kwargs.extend(Maybe(meta).map(lambda p: [p.node]).or_else_get([]))
    base_sym = ast.Call(
        func=_NEW_SYM_FN_NAME, args=[ast.Str(form.name)], keywords=sym_kwargs
    )

    return GeneratedPyAST(
        node=base_sym,
        dependencies=Maybe(meta).map(lambda p: p.dependencies).or_else_get([]),
    )


@_simple_ast_generator
def _kw_to_py_ast(_: GeneratorContext, form: kw.Keyword) -> ast.AST:
    kwarg = (
        Maybe(form.ns)
        .stream()
        .map(lambda ns: ast.keyword(arg="ns", value=ast.Str(form.ns)))
        .to_list()
    )
    return ast.Call(func=_NEW_KW_FN_NAME, args=[ast.Str(form.name)], keywords=kwarg)


@_simple_ast_generator
def _decimal_to_py_ast(_: GeneratorContext, form: Decimal) -> ast.AST:
    return ast.Call(func=_NEW_DECIMAL_FN_NAME, args=[ast.Str(str(form))], keywords=[])


@_simple_ast_generator
def _fraction_to_py_ast(_: GeneratorContext, form: Fraction) -> ast.AST:
    return ast.Call(
        func=_NEW_FRACTION_FN_NAME,
        args=[ast.Num(form.numerator), ast.Num(form.denominator)],
        keywords=[],
    )


@_simple_ast_generator
def _inst_to_py_ast(_: GeneratorContext, form: datetime) -> ast.AST:
    return ast.Call(
        func=_NEW_INST_FN_NAME, args=[ast.Str(form.isoformat())], keywords=[]
    )


@_simple_ast_generator
def _regex_to_py_ast(_: GeneratorContext, form: Pattern) -> ast.AST:
    return ast.Call(func=_NEW_REGEX_FN_NAME, args=[ast.Str(form.pattern)], keywords=[])


@_simple_ast_generator
def _uuid_to_py_ast(_: GeneratorContext, form: uuid.UUID) -> ast.AST:
    return ast.Call(func=_NEW_UUID_FN_NAME, args=[ast.Str(str(form))], keywords=[])


def _const_map_to_py_ast(ctx: GeneratorContext, form: lmap.Map) -> GeneratedPyAST:
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


def _const_set_to_py_ast(ctx: GeneratorContext, form: lset.Set) -> GeneratedPyAST:
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


def _const_seq_to_py_ast(
    ctx: GeneratorContext, form: Union[llist.List, lseq.Seq]
) -> GeneratedPyAST:
    elem_deps, elems = _chain_py_ast(*_collection_literal_to_py_ast(ctx, form))

    if isinstance(form, llist.List):
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


def _const_vec_to_py_ast(ctx: GeneratorContext, form: vec.Vector) -> GeneratedPyAST:
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


_CONST_VALUE_HANDLERS: Dict[Type, SimplePyASTGenerator] = {  # type: ignore
    bool: _name_const_to_py_ast,
    complex: _num_to_py_ast,
    datetime: _inst_to_py_ast,
    Decimal: _decimal_to_py_ast,
    float: _num_to_py_ast,
    Fraction: _fraction_to_py_ast,
    int: _num_to_py_ast,
    kw.Keyword: _kw_to_py_ast,
    llist.List: _const_seq_to_py_ast,
    lmap.Map: _const_map_to_py_ast,
    lset.Set: _const_set_to_py_ast,
    lseq.Seq: _const_seq_to_py_ast,
    type(re.compile("")): _regex_to_py_ast,
    sym.Symbol: _const_sym_to_py_ast,
    str: _str_to_py_ast,
    type(None): _name_const_to_py_ast,
    uuid.UUID: _uuid_to_py_ast,
    vec.Vector: _const_vec_to_py_ast,
}


def _const_val_to_py_ast(ctx: GeneratorContext, form: LispForm) -> GeneratedPyAST:
    """Generate Python AST nodes for constant Lisp forms.

    Nested values in collections for :const nodes are not parsed, so recursive
    structures need to call into this function to generate Python AST nodes for
    nested elements. For top-level :const Lisp AST nodes, see
    `_const_node_to_py_ast`."""
    handle_value = _CONST_VALUE_HANDLERS.get(type(form))
    if handle_value is None and isinstance(form, lseq.Seq):
        handle_value = _const_seq_to_py_ast  # type: ignore
    assert handle_value is not None, "A type handler must be defined for constants"
    return handle_value(ctx, form)


def _collection_literal_to_py_ast(
    ctx: GeneratorContext, form: Iterable[LispForm]
) -> Iterable[GeneratedPyAST]:
    """Turn a quoted collection literal of Lisp forms into Python AST nodes.

    This function can only handle constant values. It does not call back into
    the generic AST generators, so only constant values will be generated down
    this path."""
    yield from map(partial(_const_val_to_py_ast, ctx), form)


_CONSTANT_HANDLER: Dict[ConstType, SimplePyASTGenerator] = {  # type: ignore
    ConstType.BOOL: _name_const_to_py_ast,
    ConstType.INST: _inst_to_py_ast,
    ConstType.NUMBER: _num_to_py_ast,
    ConstType.DECIMAL: _decimal_to_py_ast,
    ConstType.FRACTION: _fraction_to_py_ast,
    ConstType.KEYWORD: _kw_to_py_ast,
    ConstType.MAP: _const_map_to_py_ast,
    ConstType.SET: _const_set_to_py_ast,
    ConstType.SEQ: _const_seq_to_py_ast,
    ConstType.REGEX: _regex_to_py_ast,
    ConstType.SYMBOL: _const_sym_to_py_ast,
    ConstType.STRING: _str_to_py_ast,
    ConstType.NIL: _name_const_to_py_ast,
    ConstType.UUID: _uuid_to_py_ast,
    ConstType.VECTOR: _const_vec_to_py_ast,
}


@_with_ast_loc
def _const_node_to_py_ast(ctx: GeneratorContext, lisp_ast: Const) -> GeneratedPyAST:
    """Generate Python AST nodes for a :const Lisp AST node.

    Nested values in collections for :const nodes are not parsed. Consequently,
    this function cannot be called recursively for those nested values. Instead,
    call `_const_val_to_py_ast` on nested values."""
    assert lisp_ast.op == NodeOp.CONST
    node_type = lisp_ast.type
    handle_const_node = _CONSTANT_HANDLER.get(node_type)
    assert handle_const_node is not None, f"No :const AST type handler for {node_type}"
    node_val = lisp_ast.val
    return handle_const_node(ctx, node_val)


_NODE_HANDLERS: Dict[NodeOp, PyASTGenerator] = {  # type: ignore
    NodeOp.CONST: _const_node_to_py_ast,
    NodeOp.DEF: _def_to_py_ast,
    NodeOp.DO: _do_to_py_ast,
    NodeOp.FN: _fn_to_py_ast,
    NodeOp.HOST_CALL: _interop_call_to_py_ast,
    NodeOp.HOST_FIELD: _interop_prop_to_py_ast,
    NodeOp.IF: _if_to_py_ast,
    NodeOp.IMPORT: _import_to_py_ast,
    NodeOp.INVOKE: _invoke_to_py_ast,
    NodeOp.LET: _let_to_py_ast,
    NodeOp.LETFN: None,
    NodeOp.LOCAL: _local_sym_to_py_ast,
    NodeOp.LOOP: _loop_to_py_ast,
    NodeOp.MAP: _map_to_py_ast,
    NodeOp.MAYBE_CLASS: _maybe_class_to_py_ast,
    NodeOp.MAYBE_HOST_FORM: _maybe_host_form_to_py_ast,
    NodeOp.QUOTE: _quote_to_py_ast,
    NodeOp.RECUR: _recur_to_py_ast,
    NodeOp.SET: _set_to_py_ast,
    NodeOp.SET_BANG: _set_bang_to_py_ast,
    NodeOp.THROW: _throw_to_py_ast,
    NodeOp.TRY: _try_to_py_ast,
    NodeOp.VAR: _var_sym_to_py_ast,
    NodeOp.VECTOR: _vec_to_py_ast,
    NodeOp.WITH_META: _with_meta_to_py_ast,
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


_MODULE_ALIASES = {
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


def _module_imports(ctx: GeneratorContext) -> Iterable[ast.Import]:
    """Generate the Python Import AST node for importing all required
    language support modules."""
    for imp in ctx.imports:
        name = imp.key.name
        alias = _MODULE_ALIASES.get(name, None)
        yield ast.Import(names=[ast.alias(name=name, asname=alias)])


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


def py_module_preamble(ctx: GeneratorContext,) -> GeneratedPyAST:
    """Bootstrap a new module with imports and other boilerplate."""
    preamble: List[ast.AST] = []
    preamble.extend(_module_imports(ctx))
    preamble.append(_from_module_import())
    preamble.append(_ns_var())
    return GeneratedPyAST(node=ast.NameConstant(None), dependencies=preamble)
