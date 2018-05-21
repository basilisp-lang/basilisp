import ast
import collections
import contextlib
import types
import uuid
from datetime import datetime
from typing import Dict, Iterable, Pattern, Tuple, Optional, Collection, List, Union, Any

import astor.code_gen as codegen
from functional import seq

import basilisp.lang.atom as atom
import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.meta as meta
import basilisp.lang.runtime as runtime
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.util
import basilisp.lang.vector as vec
import basilisp.reader as reader
import basilisp.walker as walk
from basilisp.util import Maybe

_CORE_NS = 'basilisp.core'
_DEFAULT_FN = '__lisp_expr__'
_DO_PREFIX = 'lisp_do'
_FN_PREFIX = 'lisp_fn'
_IF_PREFIX = 'lisp_if'
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
_QUOTE = sym.symbol("quote")
_TRY = sym.symbol("try")
_VAR = sym.symbol("var")
_SPECIAL_FORMS = lset.s(_DEF, _DO, _FN, _IF, _IMPORT, _INTEROP_CALL,
                        _INTEROP_PROP, _QUOTE, _TRY, _VAR)

_MUNGE_REPLACEMENTS = {
    '+': '__PLUS__',
    '-': '_',
    '*': '__STAR__',
    '/': '__DIV__',
    '>': '__GT__',
    '<': '__LT__',
    '!': '__BANG__',
    '=': '__EQ__',
    '?': '__Q__',
    '\\': '__IDIV__',
    '&': '__AMP__'
}

# Use an atomically incremented integer as a suffix for all
# user-defined function and variable names compiled into Python
# code so no conflicts occur
_NAME_COUNTER = atom.Atom(1)

_SYM_CTX_LOCAL_STARRED = kw.keyword(
    'local-starred', ns='basilisp.compiler.var-context')
_SYM_CTX_LOCAL = kw.keyword('local', ns='basilisp.compiler.var-context')
_SYM_CTX_NS = kw.keyword('namespace', ns='basilisp.compiler.var-context')
_SYM_CTX_IMPORT = kw.keyword('import', ns='basilisp.compiler.var-context')


class SymbolTable:
    CONTEXTS = frozenset(
        [_SYM_CTX_LOCAL, _SYM_CTX_LOCAL_STARRED, _SYM_CTX_NS, _SYM_CTX_IMPORT])

    __slots__ = ('_name', '_parent', '_table', '_children')

    def __init__(self,
                 name: str,
                 parent: 'SymbolTable' = None,
                 table: Dict[sym.Symbol, Tuple[str, kw.Keyword]] = None,
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
        self._table[s] = (munged, ctx)
        return self

    def find_symbol(self, s: sym.Symbol) -> Optional[Tuple[str, kw.Keyword]]:
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
    __slots__ = ('_st', '_nodes', '_is_quoted')

    def __init__(self):
        self._st = collections.deque([SymbolTable('<Top>')])
        self._nodes = collections.deque([atom.Atom(vec.v())])
        self._is_quoted = collections.deque([])

    @property
    def current_ns(self):
        return runtime.get_current_ns().value

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

    def append_node(self, node) -> None:
        nodes = self.nodes
        nodes.swap(lambda v: v.conj(node))

    @property
    def nodes(self):
        return self._nodes[-1]

    @contextlib.contextmanager
    def new_nodes(self):
        parent_nodes = self.nodes
        new_nodes = atom.Atom(vec.v())
        self._nodes.append(new_nodes)
        yield new_nodes, parent_nodes
        self._nodes.pop()

    def clear_nodes(self) -> None:
        self._nodes.clear()
        self._nodes = collections.deque([atom.Atom(vec.v())])

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


def genname(prefix: str) -> str:
    """Generate a unique function name with the given prefix."""
    i = _NAME_COUNTER.swap(lambda x: x + 1)
    return f"{prefix}_{i}"


def _is_py_module(name: str) -> bool:
    """Determine if a namespace is Python module."""
    try:
        __import__(name)
        return True
    except ModuleNotFoundError:
        return False


def _statementize(e: ast.AST) -> ast.AST:
    """Transform non-statements into ast.Expr nodes so they can
    stand alone as statements."""
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


def _expressionize(body: Collection[ast.AST],
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
    try:
        if len(body) > 1:
            body_nodes.extend(
                seq(body).drop_right(1).map(_statementize).to_list())
        body_nodes.append(ast.Return(value=seq(body).last()))
    except TypeError:
        body_nodes.append(ast.Return(value=body))

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


def munge(s: str) -> str:
    """Replace characters which are not valid in Python symbols
    with valid replacement strings."""
    new_str = []
    for c in s:
        new_str.append(_MUNGE_REPLACEMENTS.get(c, c))

    return ''.join(new_str)


_KW_ALIAS = 'kw'
_LIST_ALIAS = 'llist'
_MAP_ALIAS = 'lmap'
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
_NEW_MAP_FN_NAME = _load_attr(f'{_MAP_ALIAS}.map')
_NEW_REGEX_FN_NAME = _load_attr(f'{_UTIL_ALIAS}.regex_from_str')
_NEW_SET_FN_NAME = _load_attr(f'{_SET_ALIAS}.set')
_NEW_SYM_FN_NAME = _load_attr(f'{_SYM_ALIAS}.symbol')
_NEW_UUID_FN_NAME = _load_attr(f'{_UTIL_ALIAS}.uuid_from_str')
_NEW_VEC_FN_NAME = _load_attr(f'{_VEC_ALIAS}.vector')
_INTERN_VAR_FN_NAME = _load_attr(f'{_VAR_ALIAS}.intern')
_FIND_VAR_FN_NAME = _load_attr(f'{_VAR_ALIAS}.find')

LispFormAST = Union[bool, datetime, int, float, kw.Keyword, llist.List,
                    lmap.Map, None, Pattern, lset.Set, str, sym.Symbol,
                    vec.Vector, uuid.UUID]
LispSymbolAST = Union[ast.Attribute, ast.Call, ast.Name, ast.Starred]


def _meta_kwargs_ast(ctx: CompilerContext,
                     form: meta.Meta) -> Iterable[ast.keyword]:
    if hasattr(form, 'meta') and form.meta is not None:
        return [ast.keyword(arg='meta', value=_to_ast(ctx, form.meta))]
    return []


def _def_ast(ctx: CompilerContext, form: llist.List) -> ast.Call:
    """Return a Python AST Node for a `def` expression."""
    assert form.first == _DEF
    assert len(form) in range(2, 4)

    ns_name = ast.Call(func=_NEW_SYM_FN_NAME, args=[_NS_VAR_NAME], keywords=[])
    def_name = ast.Call(
        func=_NEW_SYM_FN_NAME, args=[ast.Str(form[1].name)], keywords=[])

    # Put this def'ed name into the symbol table as a namespace var
    ctx.symbol_table.new_symbol(form[1], genname(munge(form[1].name)),
                                _SYM_CTX_NS)

    try:
        def_value: Optional[ast.AST] = _to_ast(ctx, form[2])
    except KeyError:
        def_value = None
    return ast.Call(
        func=_INTERN_VAR_FN_NAME,
        args=[ns_name, def_name, def_value],
        keywords=_meta_kwargs_ast(ctx, form[1]))


def _do_ast(ctx: CompilerContext, form: llist.List) -> ast.Call:
    """Return a Python AST Node for a `do` expression."""
    assert form.first == _DO

    with ctx.new_nodes() as (inner_nodes, nodes):
        gen_body = _collection_ast(ctx, form.rest)
        body: List[ast.AST] = []

        found_inner_nodes = inner_nodes.deref()
        if len(found_inner_nodes) > 0:
            body.extend(found_inner_nodes)
        body.extend(gen_body)

        do_fn_name = genname(_DO_PREFIX)
        do_fn = _expressionize(body, do_fn_name)
        nodes.swap(lambda v: v.conj(do_fn))

        return ast.Call(
            func=ast.Name(id=do_fn_name, ctx=ast.Load()), args=[], keywords=[])


def _fn_args_body(ctx: CompilerContext, arg_vec, body_exprs
                  ) -> Tuple[List[ast.arg], List[ast.AST], Optional[ast.arg]]:
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

    if has_vargs:
        try:
            vargs_sym = arg_vec[vargs_idx + 1]
            safe = genname(munge(vargs_sym.name))
            st.new_symbol(vargs_sym, safe, _SYM_CTX_LOCAL_STARRED)
            vargs = ast.arg(arg=safe, annotation=None)
        except IndexError:
            raise CompilerException(
                f"Expected variadic argument name after '&'")

    args = [ast.arg(arg=a, annotation=None) for a in munged]
    body = _collection_ast(ctx, body_exprs)
    return args, body, vargs


def _fn_ast(ctx: CompilerContext, form: llist.List) -> ast.Name:
    """Generate a Python AST Node for an anonymous function."""
    assert form.first == _FN
    has_name = isinstance(form[1], sym.Symbol)
    name = munge(form[1].name) if has_name else genname(_FN_PREFIX)

    arg_idx = 1 + int(has_name)
    body_idx = 2 + int(has_name)

    assert isinstance(form[arg_idx], vec.Vector)

    with ctx.new_symbol_table(name):
        args, body, vargs = _fn_args_body(ctx, form[arg_idx], form[body_idx:])

        fn = _expressionize(body, name, args=args, vargs=vargs)
        ctx.append_node(fn)
        return ast.Name(id=name, ctx=ast.Load())


def _if_ast(ctx: CompilerContext, form: llist.List) -> ast.Call:
    """Generate a function call to a utility function which acts as
    an if expression and works around Python's if statement."""
    assert form.first == _IF
    assert len(form) in range(3, 5)

    with ctx.new_nodes() as (inner_nodes, nodes):
        body: List[ast.AST] = []
        ifstmt = ast.If(
            test=_to_ast(ctx, form[1]),
            body=[ast.Return(value=_to_ast(ctx, form[2]))],
            orelse=[ast.Return(value=ast.NameConstant(None))]
            if len(form) < 4 else [ast.Return(value=_to_ast(ctx, form[3]))])

        found_inner_nodes = inner_nodes.deref()
        if len(found_inner_nodes) > 0:
            body.extend(found_inner_nodes)
        body.append(ifstmt)

        ifname = genname(_IF_PREFIX)
        if_fn = ast.FunctionDef(
            name=ifname,
            args=ast.arguments(
                args=[],
                kwarg=None,
                vararg=None,
                kwonlyargs=[],
                defaults=[],
                kw_defaults=[]),
            body=body,
            decorator_list=[],
            returns=None)
        nodes.swap(lambda v: v.conj(if_fn))

        return ast.Call(
            func=ast.Name(id=ifname, ctx=ast.Load()), args=[], keywords=[])


def _import_ast(ctx: CompilerContext, form: llist.List) -> None:
    """Append Import statements into the compiler context nodes."""
    assert form[0] == _IMPORT
    assert all([isinstance(f, sym.Symbol) for f in form.rest])

    import_names = []
    for s in form.rest:
        if not _is_py_module(s.name):
            raise ImportError(f"Module '{s.name}' not found")
        ctx.symbol_table.new_symbol(s, munge(s.name), _SYM_CTX_IMPORT)
        with ctx.quoted():
            ctx.append_node(
                ast.Call(
                    func=_load_attr(f'{_NS_VAR_VALUE}.add_import'),
                    args=[_to_ast(ctx, s)],
                    keywords=[]))
        import_names.append(ast.alias(name=s.name, asname=None))
    ctx.append_node(ast.Import(names=import_names))

    return None


def _interop_call_ast(ctx: CompilerContext, form: llist.List) -> ast.Call:
    """Generate a Python AST node for Python interop method calls."""
    assert form[0] == _INTEROP_CALL
    assert form[1] is not None
    assert isinstance(form[2], sym.Symbol)
    assert form[2] is not None
    assert form[2].ns is None

    call_target = ast.Attribute(
        value=_to_ast(ctx, form[1]), attr=munge(form[2].name), ctx=ast.Load())

    args: List[ast.AST] = []
    if len(form) > 3:
        args = _collection_ast(ctx, form[3:])

    return ast.Call(func=call_target, args=args, keywords=[])


def _interop_prop_ast(ctx: CompilerContext, form: llist.List) -> ast.Attribute:
    """Generate a Python AST node for Python interop property access."""
    assert form[0] == _INTEROP_PROP
    assert form[1] is not None
    assert isinstance(form[2], sym.Symbol)
    assert form[2].ns is None
    assert len(form) == 3

    return ast.Attribute(
        value=_to_ast(ctx, form[1]), attr=munge(form[2].name), ctx=ast.Load())


def _quote_ast(ctx: CompilerContext, form: llist.List) -> Optional[ast.AST]:
    """Generate a Python AST Node for quoted forms."""
    assert form[0] == _QUOTE
    assert len(form) == 2
    with ctx.quoted():
        return _to_ast(ctx, form[1])


def _catch_expr_body(body) -> Iterable[ast.AST]:
    """Given a series of expression AST nodes, create a body of expression
    nodes with a final return node at the end of the list."""
    body_nodes: List[ast.AST] = []
    try:
        if len(body) > 1:
            body_nodes.extend(
                seq(body).drop_right(1).map(_statementize).to_list())
        body_nodes.append(ast.Return(value=seq(body).last()))
    except TypeError:
        body_nodes.append(ast.Return(value=body))
    return body_nodes


def _catch_ast(ctx: CompilerContext, form: llist.List) -> ast.ExceptHandler:
    """Generate Python AST nodes for `catch` forms."""
    assert form[0] == _CATCH
    assert len(form) >= 4
    assert isinstance(form[1], sym.Symbol)
    assert isinstance(form[2], sym.Symbol)

    body = [_to_ast(ctx, f) for f in form[3:]]
    return ast.ExceptHandler(
        type=ast.Name(id=form[1].name, ctx=ast.Load()),
        name=munge(form[2].name),
        body=_catch_expr_body(body))


def _try_ast(ctx: CompilerContext, form: llist.List) -> ast.Call:
    """Generate a Python AST nodes for `try` forms."""
    assert form[0] == _TRY
    assert len(form) >= 3

    with ctx.new_nodes() as (inner_nodes, nodes):
        expr = ast.Return(value=_to_ast(ctx, form[1]))

        # Generate Python exception handlers
        catches = form[2:-1]
        if len(catches) > 0:
            catches = [_catch_ast(ctx, f) for f in form[2:-1]]
        else:
            catches = []

        # Determine if a finally clause was provided or if all of
        # the final clauses are catch clauses
        final_clause = form[-1]
        assert isinstance(final_clause, llist.List)
        final_exprs: List[ast.AST] = []
        if final_clause.first == _FINALLY:
            final_exprs = seq(final_clause.rest) \
                .map(lambda clause: _to_ast(ctx, clause)) \
                .map(lambda node: _statementize(node)) \
                .to_list()
        else:
            catches.append(_catch_ast(ctx, final_clause))

        # Start building up the try/except block that will be inserted
        # into a function to expressionize it
        try_body: List[ast.AST] = []

        found_inner_nodes = inner_nodes.deref()
        if len(found_inner_nodes) > 0:
            try_body.extend(found_inner_nodes)

        try_body.append(
            ast.Try(
                body=[expr],
                handlers=catches,
                orelse=[],
                finalbody=final_exprs))

        # Insert the try/except function into the container
        # nodes vector so it will be available in the calling context
        try_fn_name = genname(_TRY_PREFIX)
        nodes.swap(lambda v: v.conj(ast.FunctionDef(
            name=try_fn_name,
            args=ast.arguments(
                args=[],
                kwarg=None,
                vararg=None,
                kwonlyargs=[],
                defaults=[],
                kw_defaults=[]),
            body=try_body,
            decorator_list=[],
            returns=None)))

        return ast.Call(
            func=ast.Name(id=try_fn_name, ctx=ast.Load()),
            args=[],
            keywords=[])


def _var_ast(form: llist.List) -> ast.Call:
    """Generate a Python AST Node for the `var` special form."""
    assert form[0] == _VAR
    assert isinstance(form[1], sym.Symbol)

    ns: ast.expr = _NS_VAR_NAME if form[1].ns is None else ast.Str(form[1].ns)

    base_sym = ast.Call(
        func=_NEW_SYM_FN_NAME,
        args=[ast.Str(form[1].name)],
        keywords=[ast.keyword(arg='ns', value=ns)])

    return ast.Call(func=_FIND_VAR_FN_NAME, args=[base_sym], keywords=[])


def _special_form_ast(ctx: CompilerContext,
                      form: llist.List) -> Optional[ast.AST]:
    """Generate a Python AST Node for any Lisp special forms."""
    assert form.first in _SPECIAL_FORMS
    which = form.first
    if which == _DEF:
        return _def_ast(ctx, form)
    elif which == _FN:
        return _fn_ast(ctx, form)
    elif which == _IF:
        return _if_ast(ctx, form)
    elif which == _IMPORT:
        return _import_ast(ctx, form)  # type: ignore
    elif which == _INTEROP_CALL:
        return _interop_call_ast(ctx, form)
    elif which == _INTEROP_PROP:
        return _interop_prop_ast(ctx, form)
    elif which == _DO:
        return _do_ast(ctx, form)
    elif which == _QUOTE:
        return _quote_ast(ctx, form)
    elif which == _TRY:
        return _try_ast(ctx, form)
    elif which == _VAR:
        return _var_ast(form)
    assert False, "Special form identified, but not handled"


def _list_ast(ctx: CompilerContext, form: llist.List) -> Optional[ast.AST]:
    # Special forms
    if form.first in _SPECIAL_FORMS and not ctx.is_quoted:
        return _special_form_ast(ctx, form)

    elems_ast = _collection_ast(ctx, form)

    # Quoted list
    if ctx.is_quoted:
        return ast.Call(
            func=_NEW_LIST_FN_NAME,
            args=[ast.List(elems_ast, ast.Load())],
            keywords=_meta_kwargs_ast(ctx, form))

    # Function call
    return ast.Call(
        func=elems_ast[0], args=[e for e in elems_ast[1:]], keywords=[])


def _map_ast(ctx: CompilerContext, form: lmap.Map) -> ast.Call:
    keys = _collection_ast(ctx, form.keys())
    vals = _collection_ast(ctx, form.values())

    return ast.Call(
        func=_NEW_MAP_FN_NAME,
        args=[ast.Dict(keys=keys, values=vals)],
        keywords=_meta_kwargs_ast(ctx, form))


def _set_ast(ctx: CompilerContext, form: lset.Set) -> ast.Call:
    elems_ast = _collection_ast(ctx, form)
    return ast.Call(
        func=_NEW_SET_FN_NAME,
        args=[ast.List(elems_ast, ast.Load())],
        keywords=_meta_kwargs_ast(ctx, form))


def _vec_ast(ctx: CompilerContext, form: vec.Vector) -> ast.Call:
    elems_ast = _collection_ast(ctx, form)
    return ast.Call(
        func=_NEW_VEC_FN_NAME,
        args=[ast.List(elems_ast, ast.Load())],
        keywords=_meta_kwargs_ast(ctx, form))


def _kw_ast(form: kw.Keyword) -> ast.Call:
    kwarg = Maybe(form.ns) \
        .stream() \
        .map(lambda ns: ast.keyword(arg='ns', value=ast.Str(form.ns))) \
        .to_list()
    return ast.Call(
        func=_NEW_KW_FN_NAME, args=[ast.Str(form.name)], keywords=kwarg)


def _sym_ast(ctx: CompilerContext, form: sym.Symbol) -> LispSymbolAST:
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

    sym_kwargs = Maybe(ns).stream().map(
        lambda v: ast.keyword(arg='ns', value=ns)).to_list()
    sym_kwargs.extend(_meta_kwargs_ast(ctx, form))
    base_sym = ast.Call(
        func=_NEW_SYM_FN_NAME, args=[ast.Str(form.name)], keywords=sym_kwargs)

    if ctx.is_quoted:
        return base_sym

    st = ctx.symbol_table
    st_sym = st.find_symbol(form)

    if st_sym is not None:
        munged, sym_ctx = st_sym
        if munged is None:
            raise ValueError(f"Lisp symbol '{form}' not found in {repr(st)}")

        if sym_ctx == _SYM_CTX_LOCAL:
            return ast.Name(id=munged, ctx=ast.Load())
        elif sym_ctx == _SYM_CTX_LOCAL_STARRED:
            return ast.Starred(
                value=ast.Name(id=munged, ctx=ast.Load()), ctx=ast.Load())

    if form.ns is not None:
        ns_sym = sym.symbol(form.ns)
        ns_sym_info = st.find_symbol(ns_sym)
        if ns_sym_info is not None:
            _, sym_ctx = ns_sym_info
            if sym_ctx == _SYM_CTX_IMPORT:
                return _load_attr(f"{form.ns}.{form.name}")
        if ctx.current_ns.get_import(ns_sym):
            safe_ns = munge(form.ns)
            safe_name = munge(form.name)
            return _load_attr(f"{safe_ns}.{safe_name}")

    return ast.Attribute(
        value=ast.Call(func=_FIND_VAR_FN_NAME, args=[base_sym], keywords=[]),
        attr='value',
        ctx=ast.Load())


def _regex_ast(form: Pattern) -> ast.Call:
    return ast.Call(
        func=_NEW_REGEX_FN_NAME, args=[ast.Str(form.pattern)], keywords=[])


def _inst_ast(form: datetime) -> ast.Call:
    return ast.Call(
        func=_NEW_INST_FN_NAME, args=[ast.Str(form.isoformat())], keywords=[])


def _uuid_ast(form: uuid.UUID) -> ast.Call:
    return ast.Call(
        func=_NEW_UUID_FN_NAME, args=[ast.Str(str(form))], keywords=[])


def _collection_ast(ctx: CompilerContext,
                    form: Iterable[LispFormAST]) -> List[ast.AST]:
    """Turn a collection of Lisp forms into Python AST nodes, filtering out """
    return seq(form) \
        .map(lambda x: _to_ast(ctx, x)) \
        .filter(lambda x: x is not None) \
        .to_list()


def _to_ast(ctx: CompilerContext, form: LispFormAST) -> Optional[ast.AST]:
    """Take a Lisp form as an argument and produce a Python AST node.

    This function passes along keyword arguments to any eligible
    downstream functions.

    The `nodes` argument is an Atom containing a List, which is used
    to accumulate nodes which cannot be directly replaced into the
    tree and which will need to be injected to the generated Python
    module at the very end."""
    if isinstance(form, llist.List):
        return _list_ast(ctx, form)
    elif isinstance(form, vec.Vector):
        return _vec_ast(ctx, form)
    elif isinstance(form, lmap.Map):
        return _map_ast(ctx, form)
    elif isinstance(form, lset.Set):
        return _set_ast(ctx, form)
    elif isinstance(form, kw.Keyword):
        return _kw_ast(form)
    elif isinstance(form, sym.Symbol):
        return _sym_ast(ctx, form)
    elif isinstance(form, str):
        return ast.Str(form)
    elif isinstance(form, bool):
        return ast.NameConstant(form)
    elif isinstance(form, type(None)):
        return ast.NameConstant(None)
    elif isinstance(form, float):
        return ast.Num(form)
    elif isinstance(form, int):
        return ast.Num(form)
    elif isinstance(form, datetime):
        return _inst_ast(form)
    elif isinstance(form, uuid.UUID):
        return _uuid_ast(form)
    elif isinstance(form, Pattern):
        return _regex_ast(form)
    else:
        raise TypeError(f"Unexpected form type {type(form)}")


def _module_imports() -> ast.Import:
    """Generate the Python Import AST node for importing all required
    language support modules."""
    return ast.Import(names=[
        ast.alias(name='builtins', asname=None),
        ast.alias(name='basilisp.lang.keyword', asname=_KW_ALIAS),
        ast.alias(name='basilisp.lang.list', asname=_LIST_ALIAS),
        ast.alias(name='basilisp.lang.map', asname=_MAP_ALIAS),
        ast.alias(name='basilisp.lang.set', asname=_SET_ALIAS),
        ast.alias(name='basilisp.lang.symbol', asname=_SYM_ALIAS),
        ast.alias(name='basilisp.lang.vector', asname=_VEC_ALIAS),
        ast.alias(name='basilisp.lang.util', asname=_UTIL_ALIAS)
    ])


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


def _to_py_source(ast: ast.AST, outfile: str) -> None:
    source = codegen.to_source(ast)
    with open(outfile, mode='w') as f:
        f.writelines(source)


def _to_py_str(ast: ast.AST) -> str:
    """Return a string of the Python code which would generate the input
    AST node."""
    return codegen.to_source(ast)


def _exec_ast(ast: ast.Module,
              module_name: str = 'REPL',
              expr_fn: str = _DEFAULT_FN,
              source_filename: str = '<REPL Input>'):
    """Execute a Python AST node generated from one of the compile functions
    provided in this module. Return the result of the executed module code."""
    global_scope: Dict[str, Any] = {}
    mod = types.ModuleType(module_name)
    bytecode = compile(ast, source_filename, 'exec')
    exec(bytecode, global_scope, mod.__dict__)
    return getattr(mod, expr_fn)()


def _compile_forms(forms: LispFormAST,
                   wrapped_fn_name: str = _DEFAULT_FN) -> Optional[ast.Module]:
    """Compile the given forms into final module which can be compiled into
    valid Python code. Returns a Python module AST node.

    Callers may override the wrapped function name, which is used by the
    REPL to evaluate the result of an expression and print it back out."""
    if forms is None:
        return None

    ctx = CompilerContext()

    def compile_form(form: LispFormAST):
        expr_body = [
            _module_imports(),
            _from_module_import(),
            _ns_var(),
        ]

        ctx.clear_nodes()
        form_ast = walk.prewalk(lambda f: _to_ast(ctx, f), form)
        if form_ast is None:
            return None
        expr_body.extend(list(ctx.nodes.deref()))
        expr_body.append(form_ast)

        body = _expressionize(expr_body, wrapped_fn_name)

        module = ast.Module(body=[body])
        ast.fix_missing_locations(module)

        if runtime.print_generated_python():
            print(_to_py_str(module))

        return _exec_ast(module, expr_fn=wrapped_fn_name)

    return seq(forms) \
        .map(compile_form) \
        .sequence[-1]


def compile_file(filename: str,
                 wrapped_fn_name: str = _DEFAULT_FN) -> Optional[ast.Module]:
    """Compile a file with the given name into a Python module AST node."""
    forms = reader.read_file(filename)
    return _compile_forms(forms, wrapped_fn_name)


def compile_str(s: str,
                wrapped_fn_name: str = _DEFAULT_FN) -> Optional[ast.Module]:
    """Compile the forms in a string into a Python module AST node."""
    forms = reader.read_str(s)
    return _compile_forms(forms, wrapped_fn_name)


lrepr = basilisp.lang.util.lrepr
