import ast
import collections
import contextlib
import importlib
import logging
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
from basilisp.lang.compyler.constants import (
    DEFAULT_COMPILER_FILE_PATH,
    SYM_DYNAMIC_META_KEY,
    SYM_NO_WARN_ON_REDEF_META_KEY,
    SYM_REDEF_META_KEY,
)
from basilisp.lang.compyler.exception import CompilerException, CompilerPhase
from basilisp.lang.compyler.nodes import (
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
    HostInterop,
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

        if logger.isEnabledFor(logging.DEBUG):
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
    try:
        meta = form.meta.discard(reader.READER_LINE_KW, reader.READER_COL_KW)
    except AttributeError:
        return None
    if len(meta) == 0:
        return None
    return meta


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


def _def_to_py_ast(ctx: GeneratorContext, node: Def) -> GeneratedPyAST:
    """Return a Python AST Node for a `def` expression."""
    assert node.op == NodeOp.DEF

    defsym = node.name

    if INIT in node.children:
        assert node.init is not None, "Def init must be defined"
        def_ast = gen_py_ast(ctx, node.init)
    else:
        def_ast = GeneratedPyAST(node=ast.NameConstant(None))

    ns_name = ast.Call(func=_NEW_SYM_FN_NAME, args=[_NS_VAR_NAME], keywords=[])
    def_name = ast.Call(func=_NEW_SYM_FN_NAME, args=[ast.Str(defsym.name)], keywords=[])
    safe_name = munge(defsym.name)

    # TODO: compiler meta

    # If the Var is marked as dynamic, we need to generate a keyword argument
    # for the generated Python code to set the Var as dynamic
    dynamic_kwarg = (
        Maybe(defsym.meta)
        .map(lambda m: m.get(SYM_DYNAMIC_META_KEY, None))  # type: ignore
        .map(lambda v: [ast.keyword(arg="dynamic", value=ast.NameConstant(v))])
        .or_else_get([])
    )

    meta_ast = gen_py_ast(ctx, node.meta) if node.meta is not None else None

    # Warn if this symbol is potentially being redefined
    if safe_name in ctx.current_ns.module.__dict__ or defsym in ctx.current_ns.interns:
        no_warn_on_redef = (
            Maybe(defsym.meta)
            .map(lambda m: m.get(SYM_NO_WARN_ON_REDEF_META_KEY, False))  # type: ignore
            .or_else_get(False)
        )
        if not no_warn_on_redef:
            logger.warning(
                f"redefining local Python name '{safe_name}' in module '{ctx.current_ns.module.__name__}'"
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
        dependencies=chain(
            def_ast.dependencies,
            [] if node.top_level else [ast.Global(names=[safe_name])],
            [
                ast.Assign(
                    targets=[ast.Name(id=safe_name, ctx=ast.Store())],
                    value=def_ast.node,
                )
            ],
            [] if meta_ast is None else meta_ast.dependencies,
        ),
    )


def _do_to_py_ast(ctx: GeneratorContext, node: Do) -> GeneratedPyAST:
    """Return a Python AST Node for a `do` expression."""
    assert node.op == NodeOp.DO
    assert not node.is_body

    do_fn_name = genname(_DO_PREFIX)

    return GeneratedPyAST(
        node=ast.Call(
            func=ast.Name(id=do_fn_name, ctx=ast.Load()), args=[], keywords=[]
        ),
        dependencies=[
            expressionize(
                GeneratedPyAST.reduce(
                    *chain(
                        map(partial(gen_py_ast, ctx), node.statements),
                        [gen_py_ast(ctx, node.ret)],
                    )
                ),
                do_fn_name,
            )
        ],
    )


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


def __single_arity_fn_to_py_ast(
    ctx: GeneratorContext, node: Fn, method: FnMethod
) -> GeneratedPyAST:
    """Return a Python AST node for a function with a single arity."""
    assert node.op == NodeOp.FN
    assert method.op == NodeOp.FN_METHOD

    lisp_fn_name = node.local.name if node.local is not None else None
    py_fn_name = __fn_name(lisp_fn_name)
    with ctx.new_symbol_table(py_fn_name), ctx.new_recur_point(
        method.loop_id, RecurType.FN, is_variadic=node.is_variadic
    ):
        # Allow named anonymous functions to recursively call themselves
        if lisp_fn_name is not None:
            ctx.symbol_table.new_symbol(
                sym.symbol(lisp_fn_name), munge(lisp_fn_name), LocalType.FN
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


def __multi_arity_fn_to_py_ast(
    ctx: GeneratorContext, node: Fn, methods: Collection[FnMethod]
) -> GeneratedPyAST:
    """Return a Python AST node for a function with multiple arities."""
    assert node.op == NodeOp.FN
    assert all([method.op == NodeOp.FN_METHOD for method in methods])

    lisp_fn_name = node.local.name if node.local is not None else None
    py_fn_name = __fn_name(lisp_fn_name)

    arity_to_name = {}
    rest_arity_name: Optional[str] = None
    fn_defs = []
    for method in methods:
        arity_name = f"{py_fn_name}__arity{'_rest' if method.is_variadic else method.fixed_arity}"
        arity_to_name[method.fixed_arity] = arity_name
        if method.is_variadic:
            rest_arity_name = arity_name

        with ctx.new_symbol_table(arity_name), ctx.new_recur_point(
            method.loop_id, RecurType.FN, is_variadic=node.is_variadic
        ):
            # Allow named anonymous functions to recursively call themselves
            if lisp_fn_name is not None:
                ctx.symbol_table.new_symbol(
                    sym.symbol(lisp_fn_name), munge(lisp_fn_name), LocalType.FN
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


def _fn_to_py_ast(ctx: GeneratorContext, node: Fn) -> GeneratedPyAST:
    """Return a Python AST Node for a `fn` expression."""
    assert node.op == NodeOp.FN
    if len(node.methods) == 1:
        return __single_arity_fn_to_py_ast(ctx, node, next(iter(node.methods)))
    else:
        return __multi_arity_fn_to_py_ast(ctx, node, node.methods)


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
    else:
        then_ast = gen_py_ast(ctx, node)
        return GeneratedPyAST(
            node=ast.Assign(
                targets=[ast.Name(id=result_name, ctx=ast.Store())], value=then_ast.node
            ),
            dependencies=then_ast.dependencies,
        )


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
        body=list(chain(else_ast.dependencies, [else_ast.node])),
        orelse=list(chain(then_ast.dependencies, [then_ast.node])),
    )

    return GeneratedPyAST(
        node=ast.Name(id=result_name, ctx=ast.Load()),
        dependencies=list(chain(test_ast.dependencies, [test_assign, ifstmt])),
    )


def _import_to_py_ast(ctx: GeneratorContext, node: Import) -> GeneratedPyAST:
    """Return a Python AST node for a Basilisp `import*` expression."""
    assert node.op == NodeOp.IMPORT

    last = None
    deps: List[ast.AST] = []
    for alias in node.aliases:
        try:
            module = importlib.import_module(alias.name)
            if alias.name != alias.alias:
                ctx.add_import(sym.symbol(alias.name), module, sym.symbol(alias.alias))
            else:
                ctx.add_import(sym.symbol(alias.name), module)
        except ModuleNotFoundError as e:
            raise GeneratorException(
                f"Python module '{alias.name}' not found", form=node.form, lisp_ast=node
            ) from e

        if not node.top_level:
            deps.append(ast.Global(names=[alias.alias]))

        deps.append(
            ast.Assign(
                targets=[ast.Name(id=alias.alias, ctx=ast.Store())],
                value=ast.Call(
                    func=_load_attr("builtins.__import__"),
                    args=[ast.Str(alias.name)],
                    keywords=[],
                ),
            )
        )
        last = ast.Name(id=alias.alias, ctx=ast.Load())

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
                        func=_NEW_SYM_FN_NAME, args=[ast.Str(alias.name)], keywords=[]
                    ),
                    last,
                ],
                keywords=[],
            )
        )

    assert last is not None, "import* node must have at least one import"
    return GeneratedPyAST(node=last, dependencies=deps)


def _invoke_to_py_ast(ctx: GeneratorContext, node: Invoke) -> GeneratedPyAST:
    """Return a Python AST Node for a Basilisp function invocation."""
    assert node.op == NodeOp.INVOKE

    fn_ast = gen_py_ast(ctx, node.fn)
    args_deps, args_nodes = _collection_ast(ctx, node.args)

    return GeneratedPyAST(
        node=ast.Call(func=fn_ast.node, args=list(args_nodes), keywords=[]),
        dependencies=chain(fn_ast.dependencies, args_deps),
    )


def _let_to_py_ast(ctx: GeneratorContext, node: Let) -> GeneratedPyAST:
    """Return a Python AST Node for a `let*` expression."""
    assert node.op == NodeOp.LET

    with ctx.new_symbol_table("let"):
        fn_body_ast: List[ast.AST] = []
        for binding in node.bindings:
            init_node = binding.init
            assert init_node is not None
            init_ast = gen_py_ast(ctx, init_node)
            binding_name = genname(munge(binding.name))
            fn_body_ast.extend(init_ast.dependencies)
            fn_body_ast.append(
                ast.Assign(
                    targets=[ast.Name(id=binding_name, ctx=ast.Store())],
                    value=init_ast.node,
                )
            )
            ctx.symbol_table.new_symbol(
                sym.symbol(binding.name), binding_name, LocalType.LET
            )

        body_ast = _synthetic_do_to_py_ast(ctx, node.body)
        fn_body_ast.extend(map(statementize, body_ast.dependencies))
        fn_body_ast.append(ast.Return(value=body_ast.node))

        let_fn_name = genname("let")
        return GeneratedPyAST(
            node=ast.Call(func=_load_attr(let_fn_name), args=[], keywords=[]),
            dependencies=[
                ast.FunctionDef(
                    name=let_fn_name,
                    args=ast.arguments(
                        args=[],
                        kwarg=None,
                        vararg=None,
                        kwonlyargs=[],
                        defaults=[],
                        kw_defaults=[],
                    ),
                    body=fn_body_ast,
                    decorator_list=[],
                    returns=None,
                )
            ],
        )


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

        loop_fn_name = genname("loop")
        with ctx.new_recur_point(
            node.loop_id, RecurType.LOOP, binding_names=binding_names
        ):
            loop_body_ast: List[ast.AST] = []
            body_ast = _synthetic_do_to_py_ast(ctx, node.body)
            loop_body_ast.extend(body_ast.dependencies)
            loop_body_ast.append(ast.Return(value=body_ast.node))

            return GeneratedPyAST(
                node=ast.Call(func=_load_attr(loop_fn_name), args=[], keywords=[]),
                dependencies=[
                    ast.FunctionDef(
                        name=loop_fn_name,
                        args=ast.arguments(
                            args=[],
                            kwarg=None,
                            vararg=None,
                            kwonlyargs=[],
                            defaults=[],
                            kw_defaults=[],
                        ),
                        body=list(
                            chain(
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
                        decorator_list=[],
                        returns=None,
                    )
                ],
            )


def _quote_to_py_ast(ctx: GeneratorContext, node: Quote) -> GeneratedPyAST:
    """Return a Python AST Node for a `quote` expression."""
    assert node.op == NodeOp.QUOTE
    return _const_node_to_py_ast(ctx, node.expr)


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


def __loop_recur_to_py_ast(ctx: GeneratorContext, node: Recur) -> GeneratedPyAST:
    """Return a Python AST node for `recur` occurring inside a `loop`."""
    assert node.op == NodeOp.RECUR
    recur_deps: List[ast.AST] = []
    for name, expr in zip(ctx.recur_point.binding_names, node.exprs):
        expr_ast = gen_py_ast(ctx, expr)
        recur_deps.extend(expr_ast.dependencies)
        recur_deps.append(
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())], value=expr_ast.node
            )
        )

    return GeneratedPyAST(node=ast.Continue(), dependencies=recur_deps)


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


def _set_bang_to_py_ast(ctx: GeneratorContext, node: SetBang) -> GeneratedPyAST:
    """Return a Python AST Node for a `set!` expression."""
    assert node.op == NodeOp.SET_BANG

    val_temp_name = genname("set_bang_val")
    val_ast = gen_py_ast(ctx, node.val)

    target = node.target
    if isinstance(target, Local):
        safe_name = munge(target.name)
        target_ast = GeneratedPyAST(node=ast.Name(id=safe_name, ctx=ast.Store()))
    elif isinstance(target, HostField):
        target_ast = _interop_prop_to_py_ast(ctx, target, is_assigning=True)
    elif isinstance(target, HostInterop):
        target_ast = _interop_to_py_ast(ctx, target, is_assigning=True)
    elif isinstance(target, VarRef):
        target_ast = _var_sym_to_py_ast(ctx, target, is_assigning=True)
    else:
        raise GeneratorException(
            f"invalid set! target type {type(target)}", lisp_ast=node
        )

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


def _try_to_py_ast(ctx: GeneratorContext, node: Try) -> GeneratedPyAST:
    """Return a Python AST Node for a `try` expression."""
    assert node.op == NodeOp.TRY

    try_expr_name = genname("try_expr")

    body_ast = _synthetic_do_to_py_ast(ctx, node.body)

    catch_handlers = []
    for catch in node.catches:
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
            catch_handlers.append(
                ast.ExceptHandler(
                    type=exc_type.node,
                    name=munge(exc_binding.name),
                    body=list(
                        chain(
                            map(statementize, catch_ast.dependencies),
                            [
                                ast.Assign(
                                    targets=[
                                        ast.Name(id=try_expr_name, ctx=ast.Store())
                                    ],
                                    value=catch_ast.node,
                                )
                            ],
                        )
                    ),
                )
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
    var_name: str, ns_name: str, py_var_ctx: Union[ast.Load, ast.Store]
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
        return GeneratedPyAST(
            node=ast.Name(id=f"{safe_ns}.{safe_name}", ctx=py_var_ctx)
        )

    return __var_find_to_py_ast(var_name, ns_name, py_var_ctx)


#################
# Python Interop
#################


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


def _interop_to_py_ast(
    ctx: GeneratorContext, node: HostInterop, is_assigning: bool = False
) -> GeneratedPyAST:
    """Generate a Python AST node for Python property or field access."""
    assert node.op == NodeOp.HOST_INTEROP

    target_ast = gen_py_ast(ctx, node.target)

    return GeneratedPyAST(
        node=ast.Attribute(
            value=target_ast.node,
            attr=munge(node.m_or_f),
            ctx=ast.Store() if is_assigning else ast.Load(),
        ),
        dependencies=target_ast.dependencies,
    )


def _maybe_class_to_py_ast(_: GeneratorContext, node: MaybeClass) -> GeneratedPyAST:
    """Generate a Python AST node for accessing a potential Python module
    variable name."""
    assert node.op == NodeOp.MAYBE_CLASS
    return GeneratedPyAST(node=ast.Name(id=munge(node.class_), ctx=ast.Load()))


def _maybe_host_form_to_py_ast(
    _: GeneratorContext, node: MaybeHostForm
) -> GeneratedPyAST:
    """Generate a Python AST node for accessing a potential Python module
    variable name with a namespace."""
    assert node.op == NodeOp.MAYBE_HOST_FORM

    if node.class_ == _BUILTINS_NS:
        return GeneratedPyAST(
            node=ast.Name(
                id=f"{munge(node.field, allow_builtins=True)}", ctx=ast.Load()
            )
        )

    return GeneratedPyAST(node=_load_attr(f"{munge(node.class_)}.{munge(node.field)}"))


#########################
# Non-Quoted Collections
#########################


MetaNode = Union[Const, MapNode]


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
            chain(keys, vals, Maybe(meta).map(lambda p: p.dependencies).or_else_get([]))
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
            chain(elems, Maybe(meta).map(lambda p: p.dependencies).or_else_get([]))
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
            chain(elems, Maybe(meta).map(lambda p: p.dependencies).or_else_get([]))
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
        dependencies=Maybe(meta).map(lambda p: list(p.dependencies)).or_else_get([]),
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
    Pattern: _regex_to_py_ast,
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
    NodeOp.HOST_INTEROP: _interop_to_py_ast,
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
