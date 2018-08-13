import ast
import collections
import contextlib
import functools
import itertools
import types
import uuid
from collections import OrderedDict
from datetime import datetime
from enum import Enum
from itertools import chain
from typing import (Dict, Iterable, Pattern, Tuple, Optional, List, Union, Callable, Mapping, NamedTuple, cast, Deque,
                    Any)

import astor.code_gen as codegen
from functional import seq

import basilisp.lang.atom as atom
import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.meta as meta
import basilisp.lang.runtime as runtime
import basilisp.lang.seq as lseq
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.util
import basilisp.lang.vector as vec
import basilisp.reader as reader
from basilisp.lang.runtime import Var
from basilisp.lang.typing import LispForm
from basilisp.lang.util import genname, munge
from basilisp.util import Maybe

USE_VAR_INDIRECTION = 'use_var_indirection'

_BUILTINS_NS = 'builtins'
_CORE_NS = 'basilisp.core'
_DEFAULT_FN = '__lisp_expr__'
_DO_PREFIX = 'lisp_do'
_FN_PREFIX = 'lisp_fn'
_IF_PREFIX = 'lisp_if'
_THROW_PREFIX = 'lisp_throw'
_TRY_PREFIX = 'lisp_try'
_NS_VAR = '__NS'
_LISP_NS_VAR = '*ns*'

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
_QUOTE = sym.symbol("quote")
_THROW = sym.symbol("throw")
_TRY = sym.symbol("try")
_VAR = sym.symbol("var")
_SPECIAL_FORMS = lset.s(_DEF, _DO, _FN, _IF, _IMPORT, _INTEROP_CALL,
                        _INTEROP_PROP, _LET, _QUOTE, _THROW, _TRY, _VAR)

_UNQUOTE = sym.symbol("unquote", _CORE_NS)
_UNQUOTE_SPLICING = sym.symbol("unquote-splicing", _CORE_NS)

_SYM_CTX_LOCAL_STARRED = kw.keyword(
    'local-starred', ns='basilisp.compiler.var-context')
_SYM_CTX_LOCAL = kw.keyword('local', ns='basilisp.compiler.var-context')

SymbolTableEntry = Tuple[str, kw.Keyword, sym.Symbol]


class SymbolTable:
    CONTEXTS = frozenset([_SYM_CTX_LOCAL, _SYM_CTX_LOCAL_STARRED])

    __slots__ = ('_name', '_parent', '_table', '_children')

    def __init__(self,
                 name: str,
                 parent: 'SymbolTable' = None,
                 table: Dict[sym.Symbol, SymbolTableEntry] = None,
                 children: Dict[str, 'SymbolTable'] = None) -> None:
        self._name = name
        self._parent = parent
        self._table = {} if table is None else table
        self._children = {} if children is None else children

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self):
        return (f"SymbolTable({self._name}, parent={repr(self._parent.name)}, "
                f"table={repr(self._table)}, children={len(self._children)})")

    def new_symbol(self, s: sym.Symbol, munged: str,
                   ctx: kw.Keyword) -> 'SymbolTable':
        if ctx not in SymbolTable.CONTEXTS:
            raise TypeError(f"Context {ctx} not a valid Symbol Context")
        self._table[s] = (munged, ctx, s)
        return self

    def find_symbol(self, s: sym.Symbol) -> Optional[SymbolTableEntry]:
        if s in self._table:
            return self._table[s]
        if self._parent is None:
            return None
        return self._parent.find_symbol(s)

    def append_frame(self, name: str,
                     parent: 'SymbolTable' = None) -> 'SymbolTable':
        new_frame = SymbolTable(name, parent=parent)
        self._children[name] = new_frame
        return new_frame

    def pop_frame(self, name: str) -> None:
        del self._children[name]

    @contextlib.contextmanager
    def new_frame(self, name):
        new_frame = self.append_frame(name, parent=self)
        yield new_frame
        self.pop_frame(name)


class CompilerContext:
    __slots__ = ('_st', '_is_quoted', '_opts')

    def __init__(self, opts: Dict[str, bool] = None) -> None:
        self._st = collections.deque([SymbolTable('<Top>')])
        self._is_quoted: Deque[bool] = collections.deque([])
        self._opts = Maybe(opts).map(lmap.map).or_else_get(lmap.m())

    @property
    def current_ns(self) -> runtime.Namespace:
        return runtime.get_current_ns()

    @property
    def opts(self) -> Mapping[str, bool]:
        return self._opts  # type: ignore

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

    @contextlib.contextmanager
    def unquoted(self):
        self._is_quoted.append(False)
        yield
        self._is_quoted.pop()

    def add_import(self, imp: sym.Symbol):
        self.current_ns.add_import(imp)

    @property
    def imports(self):
        return self.current_ns.imports

    @property
    def symbol_table(self) -> SymbolTable:
        return self._st[-1]

    @contextlib.contextmanager
    def new_symbol_table(self, name):
        old_st = self.symbol_table
        with old_st.new_frame(name) as st:
            self._st.append(st)
            yield
            self._st.pop()


class CompilerException(Exception):
    pass


def _assertl(c: bool, msg=None):
    if not c:
        raise CompilerException(msg)


def _load_attr(name: str) -> ast.Attribute:
    """Generate recursive Python Attribute AST nodes for resolving nested
    names."""
    attrs = name.split('.')

    def attr_node(node, idx):
        if idx >= len(attrs):
            return node
        return attr_node(
            ast.Attribute(value=node, attr=attrs[idx], ctx=ast.Load()),
            idx + 1)

    return attr_node(ast.Name(id=attrs[0], ctx=ast.Load()), 1)


def _is_py_module(name: str) -> bool:
    """Determine if a namespace is Python module."""
    try:
        __import__(name)
        return True
    except ModuleNotFoundError:
        return False


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
        raise CompilerException(f"Cannot unwrap object of type {type(n)}: {n}") from None


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
        assert len(tail) in [0, 1], "Use of _nodes_and_expr function with greater than 1 expression"
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
            ast.AsyncWith)):
        return e
    return ast.Expr(value=e)


def _expressionize(body: MixedNodeStream,
                   fn_name: str,
                   args: Iterable[ast.arg] = None,
                   vargs: ast.arg = None) -> ast.FunctionDef:
    """Given a series of expression AST nodes, create a function AST node
    with the given name that can be called and will return the result of
    the final expression in the input body nodes.

    This helps to fix the impedance mismatch of Python, which includes
    statements and expressions, and Lisps, which have only expressions.
    """
    args = [] if args is None else args
    body_nodes: List[ast.AST] = []
    body_list = _unwrap_nodes(body)
    try:
        if len(body_list) > 1:
            body_nodes.extend(
                seq(body_list).drop_right(1).map(_statementize).to_list())
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
            kw_defaults=[]),
        body=body_nodes,
        decorator_list=[],
        returns=None)


_KW_ALIAS = 'kw'
_LIST_ALIAS = 'llist'
_MAP_ALIAS = 'lmap'
_RUNTIME_ALIAS = 'runtime'
_SET_ALIAS = 'lset'
_SYM_ALIAS = 'sym'
_VEC_ALIAS = 'vec'
_VAR_ALIAS = 'Var'
_UTIL_ALIAS = 'langutil'
_NS_VAR_VALUE = f'{_NS_VAR}.value'

_NS_VAR_NAME = _load_attr(f'{_NS_VAR_VALUE}.name')
_NEW_INST_FN_NAME = _load_attr(f'{_UTIL_ALIAS}.inst_from_str')
_NEW_KW_FN_NAME = _load_attr(f'{_KW_ALIAS}.keyword')
_NEW_LIST_FN_NAME = _load_attr(f'{_LIST_ALIAS}.list')
_EMPTY_LIST_FN_NAME = _load_attr(f'{_LIST_ALIAS}.List.empty')
_NEW_MAP_FN_NAME = _load_attr(f'{_MAP_ALIAS}.map')
_NEW_REGEX_FN_NAME = _load_attr(f'{_UTIL_ALIAS}.regex_from_str')
_NEW_SET_FN_NAME = _load_attr(f'{_SET_ALIAS}.set')
_NEW_SYM_FN_NAME = _load_attr(f'{_SYM_ALIAS}.symbol')
_NEW_UUID_FN_NAME = _load_attr(f'{_UTIL_ALIAS}.uuid_from_str')
_NEW_VEC_FN_NAME = _load_attr(f'{_VEC_ALIAS}.vector')
_INTERN_VAR_FN_NAME = _load_attr(f'{_VAR_ALIAS}.intern')
_FIND_VAR_FN_NAME = _load_attr(f'{_VAR_ALIAS}.find')
_COLLECT_ARGS_FN_NAME = _load_attr(f'{_RUNTIME_ALIAS}._collect_args')
_COERCE_SEQ_FN_NAME = _load_attr(f'{_RUNTIME_ALIAS}.to_seq')


def _clean_meta(form: meta.Meta) -> LispForm:
    """Remove reader metadata from the form's meta map."""
    try:
        meta = form.meta.discard(reader._READER_LINE_KW, reader._READER_COL_KW)
    except AttributeError:
        return None
    if len(meta) == 0:
        return None
    return meta


def _meta_kwargs_ast(ctx: CompilerContext,
                     form: meta.Meta) -> ASTStream:
    if hasattr(form, 'meta') and form.meta is not None:
        meta_nodes, meta = _nodes_and_expr(_to_ast(ctx, _clean_meta(form)))
        yield from meta_nodes
        yield _node(ast.keyword(arg='meta', value=_unwrap_node(meta)))
    else:
        return []


_SYM_MACRO_META_KEY = kw.keyword("macro")


def _is_macro(v: Var) -> bool:
    """Return True if the Var holds a macro function."""
    try:
        return Maybe(v.meta).map(
            lambda m: m.get(_SYM_MACRO_META_KEY, None)  # type: ignore
        ).or_else_get(
            False)
    except (KeyError, AttributeError):
        return False


def _def_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Return a Python AST Node for a `def` expression."""
    assert form.first == _DEF
    assert len(form) in range(2, 4)

    ns_name = ast.Call(func=_NEW_SYM_FN_NAME, args=[_NS_VAR_NAME], keywords=[])
    def_name = ast.Call(
        func=_NEW_SYM_FN_NAME, args=[ast.Str(form[1].name)], keywords=[])
    safe_name = munge(form[1].name)

    try:
        def_nodes, def_value = _nodes_and_expr(_to_ast(ctx, form[2]))
    except KeyError:
        def_nodes, def_value = [], None

    meta_nodes, meta = _nodes_and_exprl(_meta_kwargs_ast(ctx, form[1]))

    yield from meta_nodes
    yield from def_nodes
    yield _dependency(ast.Assign(targets=[ast.Name(id=safe_name, ctx=ast.Store())],
                                 value=Maybe(def_value).map(_unwrap_node).or_else_get(ast.NameConstant(None))))
    yield _node(ast.Call(
        func=_INTERN_VAR_FN_NAME,
        args=[ns_name, def_name, ast.Name(id=safe_name, ctx=ast.Load())],
        keywords=_unwrap_nodes(meta)))


def _do_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Return a Python AST Node for a `do` expression."""
    assert form.first == _DO

    body = _collection_ast(ctx, form.rest)
    do_fn_name = genname(_DO_PREFIX)

    yield _dependency(_expressionize(body, do_fn_name))
    yield _node(ast.Call(
        func=ast.Name(id=do_fn_name, ctx=ast.Load()), args=[], keywords=[]))


def _fn_args_body(ctx: CompilerContext, arg_vec: vec.Vector,
                  body_exprs: llist.List
                  ) -> Tuple[List[ast.arg], ASTStream, Optional[ast.arg]]:
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
        st.new_symbol(s, safe, _SYM_CTX_LOCAL)
        munged.append(safe)

    vargs_body: List[ASTNode] = []
    if has_vargs:
        try:
            vargs_sym = arg_vec[vargs_idx + 1]
            safe = genname(munge(vargs_sym.name))
            safe_local = genname(munge(vargs_sym.name))

            # Collect all variadic arguments together into a seq and
            # reassign them to a different local
            vargs_body.append(_dependency(ast.Assign(targets=[ast.Name(id=safe_local, ctx=ast.Store())],
                                                     value=ast.Call(func=_COLLECT_ARGS_FN_NAME,
                                                                    args=[ast.Name(id=safe, ctx=ast.Load())],
                                                                    keywords=[]))))

            st.new_symbol(vargs_sym, safe_local, _SYM_CTX_LOCAL)
            vargs = ast.arg(arg=safe, annotation=None)
        except IndexError:
            raise CompilerException(
                f"Expected variadic argument name after '&'") from None

    args = [ast.arg(arg=a, annotation=None) for a in munged]
    body = itertools.chain(vargs_body, _collection_ast(ctx, body_exprs))
    return args, cast(ASTStream, body), vargs


def _fn_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a Python AST Node for an anonymous function."""
    assert form.first == _FN
    has_name = isinstance(form[1], sym.Symbol)
    name = genname("__" + (munge(form[1].name) if has_name else _FN_PREFIX))

    arg_idx = 1 + int(has_name)
    body_idx = 2 + int(has_name)

    assert isinstance(form[arg_idx], vec.Vector)

    with ctx.new_symbol_table(name):
        args, body, vargs = _fn_args_body(ctx, form[arg_idx], form[body_idx:])

        yield _dependency(_expressionize(body, name, args=args, vargs=vargs))
        yield _node(ast.Name(id=name, ctx=ast.Load()))


def _if_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a function call to a utility function which acts as
    an if expression and works around Python's if statement."""
    assert form.first == _IF
    assert len(form) in range(3, 5)

    test_nodes, test = _nodes_and_expr(_to_ast(ctx, form[1]))
    body_nodes, body = _nodes_and_expr(_to_ast(ctx, form[2]))

    try:
        else_nodes, lelse = _nodes_and_expr(_to_ast(ctx, form[3]))  # type: ignore
    except IndexError:
        else_nodes = []  # type: ignore
        lelse = ast.NameConstant(None)  # type: ignore

    ifstmt = ast.If(
        test=_unwrap_node(test),
        body=[ast.Return(value=_unwrap_node(body))],
        orelse=[ast.Return(value=_unwrap_node(lelse))])

    ifname = genname(_IF_PREFIX)

    yield _dependency(ast.FunctionDef(
        name=ifname,
        args=ast.arguments(
            args=[],
            kwarg=None,
            vararg=None,
            kwonlyargs=[],
            defaults=[],
            kw_defaults=[]),
        body=_unwrap_nodes(chain(test_nodes, body_nodes, else_nodes, [ifstmt])),
        decorator_list=[],
        returns=None))
    yield _node(ast.Call(
        func=ast.Name(id=ifname, ctx=ast.Load()), args=[], keywords=[]))


def _import_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Append Import statements into the compiler context nodes."""
    assert form[0] == _IMPORT
    assert all([isinstance(f, sym.Symbol) for f in form.rest])

    import_names = []
    for s in form.rest:
        if not _is_py_module(s.name):
            raise ImportError(f"Module '{s.name}' not found")
        ctx.add_import(s)
        with ctx.quoted():
            yield _dependency(ast.Call(
                func=_load_attr(f'{_NS_VAR_VALUE}.add_import'),
                args=_unwrap_nodes(_to_ast(ctx, s)),
                keywords=[]))
        import_names.append(ast.alias(name=s.name, asname=None))
    yield _dependency(ast.Import(names=import_names))
    yield _node(ast.NameConstant(None))


def _interop_call_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a Python AST node for Python interop method calls."""
    assert form[0] == _INTEROP_CALL
    assert form[1] is not None
    assert isinstance(form[2], sym.Symbol)
    assert form[2] is not None
    assert form[2].ns is None

    target_nodes, target = _nodes_and_expr(_to_ast(ctx, form[1]))
    yield from target_nodes

    call_target = ast.Attribute(
        value=_unwrap_node(target), attr=munge(form[2].name), ctx=ast.Load())

    args: Iterable[ast.AST] = []
    if len(form) > 3:
        nodes, args = _collection_literal_ast(ctx, form[3:])
        yield from nodes

    yield _node(ast.Call(func=call_target, args=list(args), keywords=[]))


def _interop_prop_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a Python AST node for Python interop property access."""
    assert form[0] == _INTEROP_PROP
    assert form[1] is not None
    assert isinstance(form[2], sym.Symbol)
    assert form[2].ns is None
    assert len(form) == 3

    target_nodes, target = _nodes_and_expr(_to_ast(ctx, form[1]))
    yield from target_nodes
    yield _node(ast.Attribute(
        value=_unwrap_node(target), attr=munge(form[2].name), ctx=ast.Load()))


def _let_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
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
    assert form[0] == _LET
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
    st = ctx.symbol_table
    bindings = seq(form[1]).grouped(2)

    if bindings.empty():
        raise CompilerException("Expected at least one binding in 'let*'") from None

    arg_syms: Dict[sym.Symbol, str] = OrderedDict()  # Mapping of binding symbols (turned into function parameter names) to munged name
    var_names = []  # Names of local Python variables bound to computed expressions prior to the function call
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
        st.new_symbol(s, munged, _SYM_CTX_LOCAL)

        arg_deps.append(expr_deps)
        arg_exprs.append(_unwrap_node(expr_node))

    # Generate a function to hold the body of the let expression
    letname = genname('let')
    with ctx.new_symbol_table(letname):
        args, body, vargs = _fn_args_body(ctx, vec.vector(arg_syms.keys()), form[2:])
        yield _dependency(_expressionize(body, letname, args=args, vargs=vargs))

    # Generate local variable assignments for processing let bindings
    var_names = seq(var_names).map(lambda n: ast.Name(id=n, ctx=ast.Store()))
    for name, deps, expr in zip(var_names, arg_deps, arg_exprs):
        yield from deps
        yield _dependency(ast.Assign(targets=[name], value=expr))

    yield _node(ast.Call(func=_load_attr(letname),
                         args=seq(arg_syms.values()).map(lambda n: ast.Name(id=n, ctx=ast.Load())).to_list(),
                         keywords=[]))


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

    body = seq(form[3:]).flat_map(lambda f: _to_ast(ctx, f)).map(_unwrap_node).to_list()
    return ast.ExceptHandler(
        type=ast.Name(id=form[1].name, ctx=ast.Load()),
        name=munge(form[2].name),
        body=list(_catch_expr_body(body)))


def _finally_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate Python AST nodes for `finally` forms."""
    assert isinstance(form, llist.List)
    assert form.first == _FINALLY
    assert len(form) >= 2

    yield from seq(form.rest) \
        .flat_map(lambda clause: _to_ast(ctx, clause)) \
        .map(_unwrap_node) \
        .map(lambda node: _statementize(node))


def _throw_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a Python AST Node for the `throw` special form."""
    assert form.first == _THROW
    assert len(form) == 2

    deps, expr = _nodes_and_expr(_to_ast(ctx, form[1]))
    yield from deps

    throw_fn = genname(_THROW_PREFIX)
    raise_body = ast.Raise(exc=_unwrap_node(expr), cause=None)

    yield _dependency(ast.FunctionDef(
        name=throw_fn,
        args=ast.arguments(
            args=[],
            kwarg=None,
            vararg=None,
            kwonlyargs=[],
            defaults=[],
            kw_defaults=[]),
        body=[raise_body],
        decorator_list=[],
        returns=None))

    yield _node(ast.Call(func=ast.Name(id=throw_fn, ctx=ast.Load()), args=[], keywords=[]))


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
        raise CompilerException("Only one finally clause may be provided in a try/catch block") from None

    catch_exprs: List[ast.AST] = seq(clauses.get("catch", [])).map(lambda f: _catch_ast(ctx, f)).to_list()
    final_exprs: List[ast.AST] = seq(finallys).flat_map(lambda f: _finally_ast(ctx, f)).to_list()

    # Start building up the try/except block that will be inserted
    # into a function to expressionize it
    try_body = ast.Try(
        body=_unwrap_nodes(chain(expr_nodes, [expr])),
        handlers=catch_exprs,
        orelse=[],
        finalbody=final_exprs)

    # Insert the try/except function into the container
    # nodes vector so it will be available in the calling context
    try_fn_name = genname(_TRY_PREFIX)
    yield _dependency(ast.FunctionDef(
        name=try_fn_name,
        args=ast.arguments(
            args=[],
            kwarg=None,
            vararg=None,
            kwonlyargs=[],
            defaults=[],
            kw_defaults=[]),
        body=[try_body],
        decorator_list=[],
        returns=None))

    yield _node(ast.Call(
        func=ast.Name(id=try_fn_name, ctx=ast.Load()),
        args=[],
        keywords=[]))


def _var_ast(_: CompilerContext, form: llist.List) -> ASTStream:
    """Generate a Python AST Node for the `var` special form."""
    assert form[0] == _VAR
    assert isinstance(form[1], sym.Symbol)

    ns: ast.expr = _NS_VAR_NAME if form[1].ns is None else ast.Str(form[1].ns)

    base_sym = ast.Call(
        func=_NEW_SYM_FN_NAME,
        args=[ast.Str(form[1].name)],
        keywords=[ast.keyword(arg='ns', value=ns)])

    yield _node(ast.Call(func=_FIND_VAR_FN_NAME, args=[base_sym], keywords=[]))


def _special_form_ast(ctx: CompilerContext,
                      form: llist.List) -> ASTStream:
    """Generate a Python AST Node for any Lisp special forms."""
    assert form.first in _SPECIAL_FORMS
    which = form.first
    if which == _DEF:
        yield from _def_ast(ctx, form)
        return
    elif which == _FN:
        yield from _fn_ast(ctx, form)
        return
    elif which == _IF:
        yield from _if_ast(ctx, form)
        return
    elif which == _IMPORT:
        yield from _import_ast(ctx, form)  # type: ignore
        return
    elif which == _INTEROP_CALL:
        yield from _interop_call_ast(ctx, form)
        return
    elif which == _INTEROP_PROP:
        yield from _interop_prop_ast(ctx, form)
        return
    elif which == _DO:
        yield from _do_ast(ctx, form)
        return
    elif which == _LET:
        yield from _let_ast(ctx, form)
        return
    elif which == _QUOTE:
        yield from _quote_ast(ctx, form)
        return
    elif which == _THROW:
        yield from _throw_ast(ctx, form)
        return
    elif which == _TRY:
        yield from _try_ast(ctx, form)
        return
    elif which == _VAR:
        yield from _var_ast(ctx, form)
        return
    raise CompilerException("Special form identified, but not handled") from None


def _list_ast(ctx: CompilerContext, form: llist.List) -> ASTStream:
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
        yield _node(ast.Call(
            func=_EMPTY_LIST_FN_NAME,
            args=[],
            keywords=_unwrap_nodes(meta)))
        return

    # Special forms
    if first in _SPECIAL_FORMS and not ctx.is_quoted:
        yield from _special_form_ast(ctx, form)
        return

    # Macros are immediately evaluated so the modified form can be compiled
    if isinstance(first, sym.Symbol):
        if first.ns is not None:
            v = Var.find(first)
        else:
            ns_sym = sym.symbol(first.name, ns=ctx.current_ns.name)
            v = Var.find(ns_sym)

        if v is not None and _is_macro(v):
            try:
                # Call the macro as (f &form & rest)
                # In Clojure there is &env, which we don't have yet!
                expanded = v.value(form, *form.rest)
                yield from _to_ast(ctx, expanded)
            except Exception as e:
                raise CompilerException(f"Error occurred during macroexpansion of {first}") from e
            return

    elems_nodes, elems = _collection_literal_ast(ctx, form)

    # Quoted list
    if ctx.is_quoted:
        meta_nodes, meta = _nodes_and_exprl(_meta_kwargs_ast(ctx, form))
        yield from meta_nodes
        yield _node(ast.Call(
            func=_NEW_LIST_FN_NAME,
            args=[ast.List(elems, ast.Load())],
            keywords=_unwrap_nodes(meta)))
        return

    yield from elems_nodes
    elems_ast = seq(elems)

    # Function call
    yield _node(ast.Call(
        func=elems_ast.first(), args=elems_ast.drop(1).to_list(), keywords=[]))


def _map_ast(ctx: CompilerContext, form: lmap.Map) -> ASTStream:
    key_nodes, keys = _collection_literal_ast(ctx, form.keys())
    val_nodes, vals = _collection_literal_ast(ctx, form.values())
    meta_nodes, meta = _nodes_and_exprl(_meta_kwargs_ast(ctx, form))
    yield from meta_nodes
    yield from key_nodes
    yield from val_nodes
    yield _node(ast.Call(
        func=_NEW_MAP_FN_NAME,
        args=[ast.Dict(keys=keys, values=vals)],
        keywords=_unwrap_nodes(meta)))


def _set_ast(ctx: CompilerContext, form: lset.Set) -> ASTStream:
    elem_nodes, elems_ast = _collection_literal_ast(ctx, form)
    meta_nodes, meta = _nodes_and_exprl(_meta_kwargs_ast(ctx, form))
    yield from meta_nodes
    yield from elem_nodes
    yield _node(ast.Call(
        func=_NEW_SET_FN_NAME,
        args=[ast.List(elems_ast, ast.Load())],
        keywords=_unwrap_nodes(meta)))


def _vec_ast(ctx: CompilerContext, form: vec.Vector) -> ASTStream:
    elem_nodes, elems_ast = _collection_literal_ast(ctx, form)
    meta_nodes, meta = _nodes_and_exprl(_meta_kwargs_ast(ctx, form))
    yield from meta_nodes
    yield from elem_nodes
    yield _node(ast.Call(
        func=_NEW_VEC_FN_NAME,
        args=[ast.List(elems_ast, ast.Load())],
        keywords=_unwrap_nodes(meta)))


def _kw_ast(_: CompilerContext, form: kw.Keyword) -> ASTStream:
    kwarg = Maybe(form.ns) \
        .stream() \
        .map(lambda ns: ast.keyword(arg='ns', value=ast.Str(form.ns))) \
        .to_list()
    yield _node(ast.Call(
        func=_NEW_KW_FN_NAME, args=[ast.Str(form.name)], keywords=kwarg))


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

    sym_kwargs = Maybe(ns).stream() \
        .map(lambda v: ast.keyword(arg='ns', value=ns)) \
        .to_list()
    sym_kwargs.extend(_unwrap_nodes(meta))
    base_sym = ast.Call(
        func=_NEW_SYM_FN_NAME, args=[ast.Str(form.name)], keywords=sym_kwargs)

    if ctx.is_quoted:
        yield _node(base_sym)
        return

    # Look up local symbols (function parameters, let bindings, etc.)
    st = ctx.symbol_table
    st_sym = st.find_symbol(form)

    if st_sym is not None:
        munged, sym_ctx, _ = st_sym
        if munged is None:
            raise ValueError(f"Lisp symbol '{form}' not found in {repr(st)}")

        if sym_ctx == _SYM_CTX_LOCAL:
            yield _node(ast.Name(id=munged, ctx=ast.Load()))
            return
        elif sym_ctx == _SYM_CTX_LOCAL_STARRED:
            raise CompilerException("Direct access to varargs forbidden")

    # Attempt to resolve any symbol with a namespace to a direct Python call
    if form.ns is not None:
        if form.ns == _BUILTINS_NS:
            yield _node(_load_attr(f"{munge(form.name)}"))
            return
        ns_sym = sym.symbol(form.ns)
        if ns_sym in ctx.current_ns.imports:
            safe_ns = munge(form.ns)
            safe_name = munge(form.name)
            yield _node(_load_attr(f"{safe_ns}.{safe_name}"))
            return
        elif ns_sym in ctx.current_ns.aliases:
            aliased_ns = ctx.current_ns.aliases[ns_sym]
            safe_ns = munge(aliased_ns)
            safe_name = munge(form.name)
            yield _node(_load_attr(f"{safe_ns}.{safe_name}"))
            return
    else:
        if not USE_VAR_INDIRECTION in ctx.opts:
            # Look up the symbol in the namespace mapping of the current namespace.
            # If we do find that mapping, then we can use a Python variable so long
            # as the module defined for this namespace has a Python function backing
            # it. We may have to use a direct namespace reference for imported functions.
            v = ctx.current_ns.find(form)
            if v is not None:
                safe_name = munge(form.name)
                defined_in_py = safe_name in v.ns.module.__dict__
                if defined_in_py:
                    if ctx.current_ns is v.ns:
                        yield _node(_load_attr(f"{safe_name}"))
                        return
                    else:
                        safe_ns = munge(v.ns.name)
                        yield _node(_load_attr(f"{safe_ns}.{safe_name}"))
                        return

    # If we couldn't find the symbol anywhere else, generate a Var.find call
    yield _node(ast.Attribute(
        value=ast.Call(func=_FIND_VAR_FN_NAME, args=[base_sym], keywords=[]),
        attr='value',
        ctx=ast.Load()))


def _regex_ast(_: CompilerContext, form: Pattern) -> ASTStream:
    yield _node(ast.Call(
        func=_NEW_REGEX_FN_NAME, args=[ast.Str(form.pattern)], keywords=[]))


def _inst_ast(_: CompilerContext, form: datetime) -> ASTStream:
    yield _node(ast.Call(
        func=_NEW_INST_FN_NAME, args=[ast.Str(form.isoformat())], keywords=[]))


def _uuid_ast(_: CompilerContext, form: uuid.UUID) -> ASTStream:
    yield _node(ast.Call(
        func=_NEW_UUID_FN_NAME, args=[ast.Str(str(form))], keywords=[]))


def _collection_ast(ctx: CompilerContext,
                    form: Iterable[LispForm]) -> ASTStream:
    """Turn a collection of Lisp forms into Python AST nodes, filtering out
    empty nodes."""
    yield from seq(form) \
        .flat_map(lambda x: _to_ast(ctx, x)) \
        .filter(lambda x: x is not None)


def _collection_literal_ast(ctx: CompilerContext,
                            form: Iterable[LispForm]) -> Tuple[ASTStream, PyASTStream]:
    """Turn a collection literal of Lisp forms into Python AST nodes, filtering
    out empty nodes."""
    orig = seq(form) \
        .map(lambda x: _to_ast(ctx, x)) \
        .map(lambda x: _nodes_and_exprl(x))

    return (orig.flat_map(lambda x: x[0]).to_list(),
            orig.flat_map(lambda x: x[1]).map(_unwrap_node).to_list())


def _with_loc(f: ASTProcessor) -> ASTProcessor:
    """Wrap a reader function in a decorator to supply line and column
    information along with relevant forms."""

    @functools.wraps(f)
    def with_lineno_and_col(ctx: CompilerContext, form: LispForm) -> ASTStream:
        try:
            meta = form.meta  # type: ignore
            line = meta.get(reader._READER_LINE_KW)  # type: ignore
            col = meta.get(reader._READER_COL_KW)  # type: ignore

            for astnode in f(ctx, form):
                astnode.node.lineno = line
                astnode.node.col_offset = col
                yield astnode
        except AttributeError:
            yield from f(ctx, form)

    return with_lineno_and_col


@_with_loc
def _to_ast(ctx: CompilerContext, form: LispForm) -> ASTStream:
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
    elif isinstance(form, bool):
        yield _node(ast.NameConstant(form))
        return
    elif isinstance(form, type(None)):
        yield _node(ast.NameConstant(None))
        return
    elif isinstance(form, float):
        yield _node(ast.Num(form))
        return
    elif isinstance(form, int):
        yield _node(ast.Num(form))
        return
    elif isinstance(form, datetime):
        yield from _inst_ast(ctx, form)
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
    aliases = {'builtins': None,
               'basilisp.lang.keyword': _KW_ALIAS,
               'basilisp.lang.list': _LIST_ALIAS,
               'basilisp.lang.map': _MAP_ALIAS,
               'basilisp.lang.runtime': _RUNTIME_ALIAS,
               'basilisp.lang.set': _SET_ALIAS,
               'basilisp.lang.symbol': _SYM_ALIAS,
               'basilisp.lang.vector': _VEC_ALIAS,
               'basilisp.lang.util': _UTIL_ALIAS}
    return seq(ctx.imports) \
        .map(lambda s: s.name) \
        .map(lambda name: (name, aliases.get(name, None))) \
        .map(lambda t: ast.Import(names=[ast.alias(name=t[0], asname=t[1])])) \
        .to_list()


def _from_module_import() -> ast.ImportFrom:
    """Generate the Python From ... Import AST node for importing
    language support modules."""
    return ast.ImportFrom(
        module='basilisp.lang.runtime',
        names=[ast.alias(name='Var', asname=None)],
        level=0)


def _ns_var(py_ns_var: str = _NS_VAR,
            lisp_ns_var: str = _LISP_NS_VAR,
            lisp_ns_ns: str = _CORE_NS) -> ast.Assign:
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
                    keywords=[
                        ast.keyword(arg='ns', value=ast.Str(lisp_ns_ns))
                    ])
            ],
            keywords=[]))


def to_py_source(t: ast.AST, outfile: str) -> None:
    source = codegen.to_source(t)
    with open(outfile, mode='w') as f:
        f.writelines(source)


def to_py_str(t: ast.AST) -> str:
    """Return a string of the Python code which would generate the input
    AST node."""
    return codegen.to_source(t)


def compile_and_exec_form(form: LispForm,
                          ctx: CompilerContext,
                          module: types.ModuleType,
                          source_filename: str = '<REPL Input>',
                          wrapped_fn_name: str = _DEFAULT_FN) -> Any:
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
    form_ast = list(itertools.chain(map(_statementize, form_ast[:-1]), [body]))

    ast_module = ast.Module(body=form_ast)
    ast.fix_missing_locations(ast_module)

    if runtime.print_generated_python():
        print(to_py_str(ast_module))

    bytecode = compile(ast_module, source_filename, 'exec')
    exec(bytecode, module.__dict__)
    return getattr(module, final_wrapped_name)()


def _incremental_compile_module(nodes: MixedNodeStream,
                                mod: types.ModuleType,
                                source_filename: str) -> None:
    """Incrementally compile a stream of AST nodes in module mod.

    The source_filename will be passed to Python's native compile.

    Incremental compilation is an integral part of generating a Python module
    during the same process as macro-expansion."""
    module_body = map(_statementize, _unwrap_nodes(nodes))

    module = ast.Module(body=list(module_body))
    ast.fix_missing_locations(module)

    if runtime.print_generated_python():
        print(to_py_str(module))

    bytecode = compile(module, source_filename, 'exec')
    exec(bytecode, mod.__dict__)


def _bootstrap_module(ctx: CompilerContext, mod: types.ModuleType, source_filename: str) -> None:
    """Bootstrap a new module with imports and other boilerplate."""
    preamble: List[ast.AST] = []
    preamble.extend(_module_imports(ctx))
    preamble.append(_from_module_import())
    preamble.append(_ns_var())

    _incremental_compile_module(preamble, mod, source_filename=source_filename)
    mod.__basilisp_bootstrapped__ = True  # type: ignore


def compile_module(forms: Iterable[LispForm],
                   ctx: CompilerContext,
                   module: types.ModuleType,
                   source_filename: str) -> None:
    """Compile an entire Basilisp module into Python bytecode which can be
    executed as a Python module.

    This function is designed to generate bytecode which can be used for the
    Basilisp import machinery, to allow callers to import Basilisp modules from
    Python code.
    """
    _bootstrap_module(ctx, module, source_filename)

    for form in forms:
        nodes = [node for node in _to_ast(ctx, form)]
        _incremental_compile_module(nodes, module, source_filename=source_filename)


lrepr = basilisp.lang.util.lrepr
