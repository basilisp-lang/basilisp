import ast
import collections
import contextlib
import itertools
import logging
import types
import uuid
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from functools import wraps, partial
from typing import (
    Iterable,
    Pattern,
    Optional,
    List,
    Union,
    Deque,
    Dict,
    Callable,
    NamedTuple,
    Tuple,
    Type,
)

from functional import seq

import basilisp.lang.map as lmap
import basilisp.lang.meta as lmeta
import basilisp.lang.reader as reader
import basilisp.lang.runtime as runtime
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
from basilisp.lang.compyler.constants import *
from basilisp.lang.typing import LispForm
from basilisp.lang.util import genname
from basilisp.util import Maybe

# Compiler logging
logger = logging.getLogger(__name__)

DEFAULT_COMPILER_FILE_PATH = "NO_SOURCE_PATH"

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


class RecurPoint:
    __slots__ = ("name", "args", "has_recur")

    def __init__(self, name: str, args: vec.Vector) -> None:
        self.name = name
        self.args = args
        self.has_recur = False


class GeneratorContext:
    __slots__ = ("_filename", "_opts", "_recur_points")

    def __init__(
        self, filename: Optional[str] = None, opts: Optional[Dict[str, bool]] = None
    ) -> None:
        self._filename = Maybe(filename).or_else_get(DEFAULT_COMPILER_FILE_PATH)
        self._opts = Maybe(opts).map(lmap.map).or_else_get(lmap.m())
        self._recur_points: Deque[RecurPoint] = collections.deque([])

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
    def recur_point(self):
        return self._recur_points[-1]

    @contextlib.contextmanager
    def new_recur_point(self, name: str, args: vec.Vector):
        self._recur_points.append(RecurPoint(name, args))
        yield
        self._recur_points.pop()

    def add_import(self, imp: sym.Symbol, mod: types.ModuleType, *aliases: sym.Symbol):
        self.current_ns.add_import(imp, mod, *aliases)

    @property
    def imports(self) -> lmap.Map:
        return self.current_ns.imports


class GeneratedPyAST(NamedTuple):
    node: ast.AST
    dependencies: Iterable[ast.AST] = ()

    def as_dep(self, node: ast.AST) -> "GeneratedPyAST":
        return GeneratedPyAST(
            node=node, dependencies=itertools.chain(self.dependencies, (self.node,))
        )

    @staticmethod
    def reduce(*genned: "GeneratedPyAST") -> "GeneratedPyAST":
        deps: List[ast.AST] = []
        for n in genned:
            deps.extend(n.dependencies)
            deps.append(n.node)

        return GeneratedPyAST(node=deps[-1], dependencies=deps[:-1])


LispAST = lmap.Map
PyASTStream = Iterable[ast.AST]
SimplePyASTGenerator = Callable[[GeneratorContext, LispForm], GeneratedPyAST]
PyASTGenerator = Callable[[GeneratorContext, lmap.Map], GeneratedPyAST]


def _chain_py_ast(*genned: GeneratedPyAST,) -> Tuple[PyASTStream, PyASTStream]:
    """Chain a sequence of generated Python ASTs into a tuple of dependency nodes"""
    deps = itertools.chain.from_iterable(map(lambda n: n.dependencies, genned))
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


def _collection_ast(
    ctx: GeneratorContext, form: Iterable[LispForm]
) -> Tuple[PyASTStream, PyASTStream]:
    """Turn a collection of Lisp forms into Python AST nodes."""
    return _chain_py_ast(*map(partial(gen_py_ast, ctx), form))


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
    body_nodes: List[ast.AST] = list(body.dependencies)
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


def _clean_meta(form: lmeta.Meta) -> LispForm:
    """Remove reader metadata from the form's meta map."""
    try:
        meta = form.meta.discard(reader.READER_LINE_KW, reader.READER_COL_KW)
    except AttributeError:
        return None
    if len(meta) == 0:
        return None
    return meta


#################
# Special Forms
#################


def _do_to_py_ast(ctx: GeneratorContext, form: lmap.Map) -> GeneratedPyAST:
    """Return a Python AST Node for a `do` expression."""
    assert form.entry(OP) == DO
    assert not form.entry(BODY_Q)

    do_fn_name = genname(_DO_PREFIX)
    body = form.entry(STATEMENTS)
    ret = form.entry(RET)

    return GeneratedPyAST(
        node=ast.Call(
            func=ast.Name(id=do_fn_name, ctx=ast.Load()), args=[], keywords=[]
        ),
        dependencies=[
            expressionize(
                GeneratedPyAST.reduce(
                    *itertools.chain(
                        map(partial(gen_py_ast, ctx), body), [gen_py_ast(ctx, ret)]
                    )
                ),
                do_fn_name,
            )
        ],
    )


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
    ns: Optional[ast.expr] = None
    if form.ns is None:
        ns = _NS_VAR_NAME
    else:
        ns = ast.Str(form.ns)

    meta = _const_meta_kwargs_ast(ctx, form)

    sym_kwargs = (
        Maybe(ns).stream().map(lambda v: ast.keyword(arg="ns", value=ns)).to_list()
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
            itertools.chain(
                keys, vals, Maybe(meta).map(lambda p: p.dependencies).or_else_get([])
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
            itertools.chain(
                elems, Maybe(meta).map(lambda p: p.dependencies).or_else_get([])
            )
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
    lmap.Map: _const_map_to_py_ast,
    lset.Set: _const_set_to_py_ast,
    Pattern: _regex_to_py_ast,
    sym.Symbol: _const_sym_to_py_ast,
    str: _str_to_py_ast,
    type(None): _name_const_to_py_ast,
    uuid.UUID: _uuid_to_py_ast,
    vec.Vector: _const_vec_to_py_ast,
}


def _const_val_to_py_ast(ctx: GeneratorContext, form: LispForm) -> GeneratedPyAST:
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


_CONSTANT_HANDLER: Dict[kw.Keyword, SimplePyASTGenerator] = {  # type: ignore
    BOOL: _name_const_to_py_ast,
    INST: _inst_to_py_ast,
    NUMBER: _num_to_py_ast,
    DECIMAL: _decimal_to_py_ast,
    FRACTION: _fraction_to_py_ast,
    KEYWORD: _kw_to_py_ast,
    MAP: _const_map_to_py_ast,
    SET: _const_set_to_py_ast,
    REGEX: _regex_to_py_ast,
    SYMBOL: None,
    STRING: _str_to_py_ast,
    NIL: _name_const_to_py_ast,
    UUID: _uuid_to_py_ast,
    VECTOR: _const_vec_to_py_ast,
}


def _const_node_to_py_ast(ctx: GeneratorContext, lisp_ast: lmap.Map) -> GeneratedPyAST:
    """Generate Python AST nodes for a :const Lisp AST node.

    Nested values in collections for :const nodes are not parsed. Consequently,
    this function cannot be called recursively for those nested values. Instead,
    call `_const_val_to_py_ast` on nested values."""
    assert lisp_ast.entry(OP) == CONST
    node_type: kw.Keyword = lisp_ast.entry(TYPE)
    handle_node = _CONSTANT_HANDLER.get(node_type)
    assert handle_node is not None
    node_val: LispForm = lisp_ast.entry(VAL)
    return handle_node(ctx, node_val)


_NODE_HANDLERS: Dict[kw.Keyword, PyASTGenerator] = {  # type: ignore
    CONST: _const_node_to_py_ast,
    DEF: None,
    DO: _do_to_py_ast,
    FN: None,
    HOST_CALL: None,
    HOST_FIELD: None,
    HOST_INTEROP: None,
    IF: None,
    INVOKE: None,
    LET: None,
    LETFN: None,
    LOOP: None,
    MAP: None,
    MAYBE_CLASS: None,
    MAYBE_HOST_FORM: None,
    NEW: None,
    QUOTE: None,
    RECUR: None,
    SET: None,
    SET_BANG: None,
    THROW: None,
    TRY: None,
    VAR: None,
    VECTOR: None,
    WITH_META: None,
}


###################
# Public Functions
###################


def gen_py_ast(ctx: GeneratorContext, lisp_ast: lmap.Map) -> GeneratedPyAST:
    """Take a Lisp AST node as an argument and produce zero or more Python
    AST nodes.

    This is the primary entrypoint for generating AST nodes from Lisp
    syntax. It may be called recursively to compile child forms."""
    op: kw.Keyword = lisp_ast.entry(OP)
    assert op is not None, "Lisp AST nodes must have an :op key"
    handle_node = _NODE_HANDLERS.get(op)
    assert handle_node is not None, "Lisp AST nodes :op has no handler defined"
    return handle_node(ctx, lisp_ast)


#############################
# Bootstrap Basilisp Modules
#############################


def _module_imports(ctx: GeneratorContext) -> Iterable[ast.Import]:
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


def py_module_preamble(ctx: GeneratorContext,) -> GeneratedPyAST:
    """Bootstrap a new module with imports and other boilerplate."""
    preamble: List[ast.AST] = []
    preamble.extend(_module_imports(ctx))
    preamble.append(_from_module_import())
    preamble.append(_ns_var())
    return GeneratedPyAST(node=ast.NameConstant(None), dependencies=preamble)
