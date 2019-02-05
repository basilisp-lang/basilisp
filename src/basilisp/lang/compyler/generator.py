import ast
import collections
import contextlib
import logging
import types
import uuid
from datetime import datetime
from decimal import Decimal
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
    Any,
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
)
from basilisp.lang.typing import LispForm
from basilisp.lang.util import genname, munge
from basilisp.util import Maybe

# Compiler logging
logger = logging.getLogger(__name__)

DEFAULT_COMPILER_FILE_PATH = "NO_SOURCE_PATH"

# Lisp AST node keywords
INIT = kw.keyword("init")

# Symbol meta keys
SYM_DYNAMIC_META_KEY = kw.keyword("dynamic")
SYM_NO_WARN_ON_REDEF_META_KEY = kw.keyword("no-warn-on-redef")

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


def count(seq: Iterable) -> int:
    return sum([1 for _ in seq])


GeneratorException = partial(CompilerException, phase=CompilerPhase.CODE_GENERATION)


class RecurType(Enum):
    FN = kw.keyword("fn")
    LOOP = kw.keyword("loop")


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RecurPoint:
    loop_id: str
    binding_names: Collection[str]
    type: RecurType


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
    ctx: GeneratorContext, form: Iterable[Node]
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


def _def_to_py_ast(ctx: GeneratorContext, node: Def) -> GeneratedPyAST:
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
            keywords=list(chain(dynamic_kwarg)),  # type: ignore
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
    """Return AST elements generated from reducing a
    synthetic (e.g. a :do node which acts as a body for another node) Lisp :do
    node."""
    assert node.op == NodeOp.DO
    assert node.is_body

    return GeneratedPyAST.reduce(
        *chain(
            map(partial(gen_py_ast, ctx), node.statements), [gen_py_ast(ctx, node.ret)]
        )
    )


def _if_to_py_ast(ctx: GeneratorContext, node: If) -> GeneratedPyAST:
    """Generate a function call to a utility function which acts as
    an if expression and works around Python's if statement.

    Every expression in Basilisp is true if it is not the literal values nil
    or false. This function compiles direct checks for the test value against
    the Python values None and False to accommodate this behavior.

    Note that the if and else bodies are switched in compilation so that we
    can perform a short-circuit or comparison, rather than exhaustively checking
    for both false and nil each time."""
    assert node.op == NodeOp.IF

    test_ast = gen_py_ast(ctx, node.test)
    then_ast = gen_py_ast(ctx, node.then)
    else_ast = gen_py_ast(ctx, node.else_)

    test_name = genname(_IF_TEST_PREFIX)
    result_name = genname(_IF_RESULT_PREFIX)
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
        body=list(
            chain(
                else_ast.dependencies,
                [
                    ast.Assign(
                        targets=[ast.Name(id=result_name, ctx=ast.Store())],
                        value=else_ast.node,
                    )
                ],
            )
        ),
        orelse=list(
            chain(
                then_ast.dependencies,
                [
                    ast.Assign(
                        targets=[ast.Name(id=result_name, ctx=ast.Store())],
                        value=then_ast.node,
                    )
                ],
            )
        ),
    )

    return GeneratedPyAST(
        node=ast.Name(id=result_name, ctx=ast.Load()),
        dependencies=[test_assign, ifstmt],
    )


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

    fn_body_ast: List[ast.AST] = []
    for binding in node.bindings:
        init_node = binding.init
        assert init_node is not None
        init_ast = gen_py_ast(ctx, init_node)
        fn_body_ast.extend(init_ast.dependencies)
        fn_body_ast.append(
            ast.Assign(
                targets=[ast.Name(id=munge(binding.name.name), ctx=ast.Store())],
                value=init_ast.node,
            )
        )

    body_ast = _synthetic_do_to_py_ast(ctx, node.body)
    fn_body_ast.extend(body_ast.dependencies)
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

    init_bindings: List[ast.AST] = []
    for binding in node.bindings:
        init_node = binding.init
        assert init_node is not None
        init_ast = gen_py_ast(ctx, init_node)
        init_bindings.extend(init_ast.dependencies)
        init_bindings.append(
            ast.Assign(
                targets=[ast.Name(id=munge(binding.name.name), ctx=ast.Store())],
                value=init_ast.node,
            )
        )

    loop_body_ast: List[ast.AST] = []
    body_ast = _synthetic_do_to_py_ast(ctx, node.body)
    loop_body_ast.extend(body_ast.dependencies)
    loop_body_ast.append(ast.Return(value=body_ast.node))

    loop_fn_name = genname("loop")
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


def _set_bang_to_py_ast(ctx: GeneratorContext, node: SetBang) -> GeneratedPyAST:
    """Return a Python AST Node for a `quote` expression."""
    assert node.op == NodeOp.SET_BANG

    val_temp_name = genname("set_bang_val")
    val_ast = gen_py_ast(ctx, node.val)

    target = node.target
    if isinstance(target, Local):
        safe_name = munge(target.name.name)
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

        catch_ast = _synthetic_do_to_py_ast(ctx, catch.body)
        catch_handlers.append(
            ast.ExceptHandler(
                type=exc_type.node,
                name=munge(exc_binding.name.name),
                body=list(
                    chain(
                        catch_ast.dependencies,
                        [
                            ast.Assign(
                                targets=[ast.Name(id=try_expr_name, ctx=ast.Store())],
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
        finallys.extend(finally_ast.dependencies)
        finallys.append(finally_ast.node)

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
    _: GeneratorContext, node: Local, is_assigning: bool = False
) -> GeneratedPyAST:
    """Generate a Python AST node for accessing a locally defined Python variable."""
    assert node.op == NodeOp.LOCAL

    return GeneratedPyAST(
        node=ast.Name(
            id=munge(node.name.name), ctx=ast.Store() if is_assigning else ast.Load()
        )
    )


def _var_sym_to_py_ast(
    _: GeneratorContext, node: VarRef, is_assigning: bool = False
) -> GeneratedPyAST:
    """Generate a Python AST node for accessing a Var."""
    assert node.op == NodeOp.VAR

    # TODO: direct link to Python variable, if possible

    var: runtime.Var = node.var

    return GeneratedPyAST(
        node=ast.Attribute(
            value=ast.Call(
                func=_FIND_VAR_FN_NAME,
                args=[
                    ast.Call(
                        func=_NEW_SYM_FN_NAME,
                        args=[ast.Str(var.name.name)],
                        keywords=[ast.keyword(arg="ns", value=ast.Str(var.ns.name))],
                    )
                ],
                keywords=[],
            ),
            attr="value",
            ctx=ast.Store() if is_assigning else ast.Load(),
        )
    )


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
                attr=munge(node.method.name, allow_builtins=True),
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
            attr=munge(node.field.name),
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
            attr=munge(node.m_or_f.name),
            ctx=ast.Store() if is_assigning else ast.Load(),
        ),
        dependencies=target_ast.dependencies,
    )


def _maybe_class_to_py_ast(_: GeneratorContext, node: MaybeClass) -> GeneratedPyAST:
    """Generate a Python AST node for accessing a potential Python module
    variable name."""
    assert node.op == NodeOp.MAYBE_CLASS

    class_ = node.class_
    assert class_.ns is None

    return GeneratedPyAST(node=ast.Name(id=munge(class_.name), ctx=ast.Load()))


def _maybe_host_form_to_py_ast(
    _: GeneratorContext, node: MaybeHostForm
) -> GeneratedPyAST:
    """Generate a Python AST node for accessing a potential Python module
    variable name with a namespace."""
    assert node.op == NodeOp.MAYBE_HOST_FORM

    ns = node.class_

    if ns.name == _BUILTINS_NS:
        return GeneratedPyAST(
            node=ast.Name(
                id=f"{munge(node.field.name, allow_builtins=True)}", ctx=ast.Load()
            )
        )

    return GeneratedPyAST(node=_load_attr(f"{munge(ns.name)}.{munge(node.field.name)}"))


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
                keys,
                vals,
                Maybe(meta_ast).map(lambda p: p.dependencies).or_else_get([]),
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
            chain(elems, Maybe(meta_ast).map(lambda p: p.dependencies).or_else_get([]))
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
        dependencies=Maybe(meta_ast)
        .map(lambda p: list(p.dependencies))
        .or_else_get([]),
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
    NodeOp.FN: None,
    NodeOp.HOST_CALL: _interop_call_to_py_ast,
    NodeOp.HOST_FIELD: _interop_prop_to_py_ast,
    NodeOp.HOST_INTEROP: _interop_to_py_ast,
    NodeOp.IF: _if_to_py_ast,
    NodeOp.INVOKE: _invoke_to_py_ast,
    NodeOp.LET: _let_to_py_ast,
    NodeOp.LETFN: None,
    NodeOp.LOCAL: _local_sym_to_py_ast,
    NodeOp.LOOP: _loop_to_py_ast,
    NodeOp.MAP: _map_to_py_ast,
    NodeOp.MAYBE_CLASS: _maybe_class_to_py_ast,
    NodeOp.MAYBE_HOST_FORM: _maybe_host_form_to_py_ast,
    NodeOp.QUOTE: _quote_to_py_ast,
    NodeOp.RECUR: None,
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
