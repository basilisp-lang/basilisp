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
    OBJECT_DUNDER_METHODS,
    SYM_ASYNC_META_KEY,
    SYM_CLASSMETHOD_META_KEY,
    SYM_DEFAULT_META_KEY,
    SYM_DYNAMIC_META_KEY,
    SYM_MACRO_META_KEY,
    SYM_MUTABLE_META_KEY,
    SYM_NO_WARN_WHEN_UNUSED_META_KEY,
    SYM_PROPERTY_META_KEY,
    SYM_STATICMETHOD_META_KEY,
    SpecialForm,
)
from basilisp.lang.compiler.exception import CompilerException, CompilerPhase
from basilisp.lang.compiler.nodes import (
    Assignable,
    Await,
    Binding,
    Catch,
    ClassMethod,
    Const,
    ConstType,
    Def,
    DefType,
    DefTypeBase,
    DefTypeMember,
    Do,
    Fn,
    FnMethod,
    HostCall,
    HostField,
    If,
    Import,
    ImportAlias,
    Invoke,
    Let,
    LetFn,
    Local,
    LocalType,
    Loop,
    Map as MapNode,
    MaybeClass,
    MaybeHostForm,
    Method,
    Node,
    NodeEnv,
    NodeOp,
    PropertyMethod,
    PyDict,
    PyList,
    PySet,
    PyTuple,
    Quote,
    Recur,
    Set as SetNode,
    SetBang,
    SpecialFormNode,
    StaticMethod,
    Throw,
    Try,
    VarRef,
    Vector as VectorNode,
    WithMeta,
)
from basilisp.lang.interfaces import IMeta, IRecord, ISeq, IType
from basilisp.lang.runtime import Var
from basilisp.lang.typing import LispForm, ReaderForm
from basilisp.lang.util import count, genname, munge
from basilisp.util import Maybe, partition

# Analyzer logging
logger = logging.getLogger(__name__)

# Analyzer options
WARN_ON_SHADOWED_NAME = "warn_on_shadowed_name"
WARN_ON_SHADOWED_VAR = "warn_on_shadowed_var"
WARN_ON_UNUSED_NAMES = "warn_on_unused_names"

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
    _parent: Optional["SymbolTable"] = None
    _table: MutableMapping[sym.Symbol, SymbolTableEntry] = attr.ib(factory=dict)
    _children: MutableMapping[str, "SymbolTable"] = attr.ib(factory=dict)

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
                    .map(
                        lambda m: f": {m.val_at(reader.READER_LINE_KW)}"  # type: ignore
                    )
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

    def _as_env_map(self) -> MutableMapping[sym.Symbol, lmap.Map]:
        locals_ = {} if self._parent is None else self._parent._as_env_map()
        locals_.update({k: v.binding.to_map() for k, v in self._table.items()})
        return locals_

    def as_env_map(self) -> lmap.Map:
        """Return a map of symbols to the local binding objects in the
        local symbol table as of this call."""
        return lmap.map(self._as_env_map())


class AnalyzerContext:
    __slots__ = (
        "_filename",
        "_func_ctx",
        "_is_quoted",
        "_opts",
        "_recur_points",
        "_should_macroexpand",
        "_st",
    )

    def __init__(
        self,
        filename: Optional[str] = None,
        opts: Optional[Mapping[str, bool]] = None,
        should_macroexpand: bool = True,
    ) -> None:
        self._filename = Maybe(filename).or_else_get(DEFAULT_COMPILER_FILE_PATH)
        self._func_ctx: Deque[bool] = collections.deque([])
        self._is_quoted: Deque[bool] = collections.deque([])
        self._opts = (
            Maybe(opts).map(lmap.map).or_else_get(lmap.Map.empty())  # type: ignore
        )
        self._recur_points: Deque[RecurPoint] = collections.deque([])
        self._should_macroexpand = should_macroexpand
        self._st = collections.deque([SymbolTable("<Top>")])

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
    def should_macroexpand(self) -> bool:
        return self._should_macroexpand

    @property
    def is_async_ctx(self) -> bool:
        try:
            return self._func_ctx[-1] is True
        except IndexError:
            return False

    @contextlib.contextmanager
    def new_func_ctx(self, is_async: bool = False):
        self._func_ctx.append(is_async)
        yield
        self._func_ctx.pop()

    @property
    def recur_point(self) -> Optional[RecurPoint]:
        try:
            return self._recur_points[-1]
        except IndexError:
            return None

    @contextlib.contextmanager
    def new_recur_point(self, loop_id: str, args: Collection[Any] = ()):
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
        st = self.symbol_table
        if warn_on_shadowed_name and self.warn_on_shadowed_name:
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
    def new_symbol_table(self, name):
        old_st = self.symbol_table
        with old_st.new_frame(name, self.warn_on_unused_names) as st:
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

    def get_node_env(self):
        return NodeEnv(ns=self.current_ns, file=self.filename)


MetaGetter = Callable[[Union[IMeta, Var]], bool]
AnalyzeFunction = Callable[[AnalyzerContext, Union[LispForm, ISeq]], Node]


def _meta_getter(meta_kw: kw.Keyword) -> MetaGetter:
    """Return a function which checks an object with metadata for a boolean
    value by meta_kw."""

    def has_meta_prop(o: Union[IMeta, Var]) -> bool:
        return (  # type: ignore
            Maybe(o.meta).map(lambda m: m.val_at(meta_kw, None)).or_else_get(False)
        )

    return has_meta_prop


_is_async = _meta_getter(SYM_ASYNC_META_KEY)
_is_mutable = _meta_getter(SYM_MUTABLE_META_KEY)
_is_py_classmethod = _meta_getter(SYM_CLASSMETHOD_META_KEY)
_is_py_property = _meta_getter(SYM_PROPERTY_META_KEY)
_is_py_staticmethod = _meta_getter(SYM_STATICMETHOD_META_KEY)
_is_macro = _meta_getter(SYM_MACRO_META_KEY)


def _loc(form: Union[LispForm, ISeq]) -> Optional[Tuple[int, int]]:
    """Fetch the location of the form in the original filename from the
    input form, if it has metadata."""
    try:
        meta = form.meta  # type: ignore
        line = meta.get(reader.READER_LINE_KW)  # type: ignore
        col = meta.get(reader.READER_COL_KW)  # type: ignore
    except AttributeError:
        return None
    else:
        assert isinstance(line, int) and isinstance(col, int)
        return line, col


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

    If the body is empty, return a constant node containing nil."""
    body = list(map(partial(_analyze_form, ctx), form))
    if body:
        *stmts, ret = body
    else:
        stmts, ret = [], _const_node(ctx, None)
    return stmts, ret


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
                    form=form, meta=meta_ast, expr=descriptor, env=ctx.get_node_env()
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

    expr = _analyze_form(ctx, runtime.nth(form, 1))
    return Await(form=form, expr=expr, env=ctx.get_node_env())


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

    if nelems == 2:
        init = None
        doc = None
        children: vec.Vector[kw.Keyword] = vec.Vector.empty()
    elif nelems == 3:
        init = _analyze_form(ctx, runtime.nth(form, 2))
        doc = None
        children = vec.v(INIT)
    else:
        init = _analyze_form(ctx, runtime.nth(form, 3))
        doc = runtime.nth(form, 2)
        if not isinstance(doc, str):
            raise AnalyzerException("def docstring must be a string", form=doc)
        children = vec.v(INIT)

    # Attach metadata relevant for the current process below.
    def_loc = _loc(form)
    def_node_env = ctx.get_node_env()
    def_meta = _clean_meta(
        name.meta.update(
            lmap.map(
                {
                    COL_KW: def_loc[1] if def_loc is not None else None,
                    FILE_KW: def_node_env.file,
                    LINE_KW: def_loc[0] if def_loc is not None else None,
                    NAME_KW: name,
                    NS_KW: ctx.current_ns,
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

    ns_sym = sym.symbol(ctx.current_ns.name)
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
    #        :ns   ((.- basilisp.lang.runtime/Namespace get) 'user)}
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
                    llist.l(SpecialForm.QUOTE, sym.symbol(ctx.current_ns.name)),
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
    ctx: AnalyzerContext, params: vec.Vector
) -> Tuple[bool, List[Binding]]:
    """Generate parameter bindings for deftype* methods.

    Special cases for class and static methods must be handled by their
    respective handlers. This method will only produce vanilla ARG type
    bindings."""
    has_vargs, vargs_idx = False, 0
    param_nodes = []
    for i, s in enumerate(params):
        if not isinstance(s, sym.Symbol):
            raise AnalyzerException(
                "deftype* method parameter name must be a symbol", form=s
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
                    "deftype* method rest parameter name must be a symbol",
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

    return has_vargs, param_nodes


def __deftype_classmethod(
    ctx: AnalyzerContext,
    form: Union[llist.List, ISeq],
    method_name: str,
    args: vec.Vector,
) -> ClassMethod:
    """Emit a node for a :classmethod member of a deftype* form."""
    with ctx.hide_parent_symbol_table(), ctx.new_symbol_table(method_name):
        try:
            cls_arg = args[0]
        except IndexError:
            raise AnalyzerException(
                f"deftype* class method must include 'cls' argument", form=args
            )
        else:
            if not isinstance(cls_arg, sym.Symbol):
                raise AnalyzerException(
                    f"deftype* method 'cls' argument must be a symbol", form=args
                )
            cls_binding = Binding(
                form=cls_arg,
                name=cls_arg.name,
                local=LocalType.ARG,
                env=ctx.get_node_env(),
            )
            ctx.put_new_symbol(cls_arg, cls_binding)

        params = args[1:]
        has_vargs, param_nodes = __deftype_method_param_bindings(ctx, params)
        stmts, ret = _body_ast(ctx, runtime.nthrest(form, 2))
        method = ClassMethod(
            form=form,
            name=method_name,
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
            class_local=cls_binding,
            is_variadic=has_vargs,
        )
        method.visit(_assert_no_recur)
        return method


def __deftype_method(
    ctx: AnalyzerContext,
    form: Union[llist.List, ISeq],
    method_name: str,
    args: vec.Vector,
) -> Method:
    """Emit a node for a method member of a deftype* form."""
    with ctx.new_symbol_table(method_name):
        try:
            this_arg = args[0]
        except IndexError:
            raise AnalyzerException(
                f"deftype* method must include 'this' or 'self' argument", form=args
            )
        else:
            if not isinstance(this_arg, sym.Symbol):
                raise AnalyzerException(
                    f"deftype* method 'this' argument must be a symbol", form=args
                )
            this_binding = Binding(
                form=this_arg,
                name=this_arg.name,
                local=LocalType.THIS,
                env=ctx.get_node_env(),
            )
            ctx.put_new_symbol(this_arg, this_binding, warn_if_unused=False)

        params = args[1:]
        has_vargs, param_nodes = __deftype_method_param_bindings(ctx, params)

        loop_id = genname(method_name)
        with ctx.new_recur_point(loop_id, param_nodes):
            stmts, ret = _body_ast(ctx, runtime.nthrest(form, 2))
            method = Method(
                form=form,
                name=method_name,
                this_local=this_binding,
                params=vec.vector(param_nodes),
                is_variadic=has_vargs,
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


def __deftype_property(
    ctx: AnalyzerContext,
    form: Union[llist.List, ISeq],
    method_name: str,
    args: vec.Vector,
) -> PropertyMethod:
    """Emit a node for a :property member of a deftype* form."""
    with ctx.new_symbol_table(method_name):
        try:
            this_arg = args[0]
        except IndexError:
            raise AnalyzerException(
                f"deftype* method must include 'this' or 'self' argument", form=args
            )
        else:
            if not isinstance(this_arg, sym.Symbol):
                raise AnalyzerException(
                    f"deftype* method 'this' argument must be a symbol", form=args
                )
            this_binding = Binding(
                form=this_arg,
                name=this_arg.name,
                local=LocalType.THIS,
                env=ctx.get_node_env(),
            )
            ctx.put_new_symbol(this_arg, this_binding, warn_if_unused=False)

        params = args[1:]
        has_vargs, param_nodes = __deftype_method_param_bindings(ctx, params)

        if len(param_nodes) > 0:
            raise AnalyzerException(
                "deftype* properties may not specify arguments", form=form
            )

        assert not has_vargs, "deftype* properties may not have arguments"

        stmts, ret = _body_ast(ctx, runtime.nthrest(form, 2))
        prop = PropertyMethod(
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
) -> StaticMethod:
    """Emit a node for a :staticmethod member of a deftype* form."""
    with ctx.hide_parent_symbol_table(), ctx.new_symbol_table(method_name):
        has_vargs, param_nodes = __deftype_method_param_bindings(ctx, args)
        stmts, ret = _body_ast(ctx, runtime.nthrest(form, 2))
        method = StaticMethod(
            form=form,
            name=method_name,
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
            is_variadic=has_vargs,
        )
        method.visit(_assert_no_recur)
        return method


def __deftype_member(
    ctx: AnalyzerContext, form: Union[llist.List, ISeq]
) -> DefTypeMember:
    """Emit a member node for a deftype* form.

    Member nodes are determined by the presence or absence of certain
    metadata elements on the input form (or the form's first member,
    typically a symbol naming that member)."""
    if not isinstance(form.first, sym.Symbol):
        raise AnalyzerException(
            "deftype* method must be named by symbol: (name [& args] & body)",
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

    if not sum([is_classmethod, is_property, is_staticmethod]) in {0, 1}:
        raise AnalyzerException(
            "deftype* member may be only one of: :classmethod, :property, or :staticmethod",
            form=form,
        )

    args = runtime.nth(form, 1)
    if not isinstance(args, vec.Vector):
        raise AnalyzerException(
            f"deftype* member arguments must be vector, not {type(args)}", form=args
        )

    if is_classmethod:
        return __deftype_classmethod(ctx, form, method_name, args)
    elif is_property:
        return __deftype_property(ctx, form, method_name, args)
    elif is_staticmethod:
        return __deftype_staticmethod(ctx, form, method_name, args)
    else:
        return __deftype_method(ctx, form, method_name, args)


def __deftype_impls(  # pylint: disable=too-many-branches
    ctx: AnalyzerContext, form: ISeq
) -> Tuple[List[DefTypeBase], List[DefTypeMember]]:
    """Roll up deftype* declared bases and method implementations."""
    interface_names: MutableSet[sym.Symbol] = set()
    interfaces = []
    methods: List[DefTypeMember] = []

    if runtime.to_seq(form) is None:
        return [], []

    if not isinstance(form.first, kw.Keyword) or form.first != IMPLEMENTS:
        raise AnalyzerException(
            f"deftype* forms must declare which interfaces they implement", form=form
        )

    implements = runtime.nth(form, 1)
    if not isinstance(implements, vec.Vector):
        raise AnalyzerException(
            "deftype* interfaces must be declared as :implements [Interface1 Interface2 ...]",
            form=implements,
        )

    for iface in implements:
        if not isinstance(iface, sym.Symbol):
            raise AnalyzerException("deftype* interfaces must be symbols", form=iface)

        if iface in interface_names:
            raise AnalyzerException(
                "deftype* interfaces may only appear once in :implements vector",
                form=iface,
            )
        interface_names.add(iface)

        current_interface = _analyze_form(ctx, iface)
        if not isinstance(current_interface, (MaybeClass, MaybeHostForm, VarRef)):
            raise AnalyzerException(
                f"deftype* interface implementation must be an existing interface",
                form=iface,
            )
        interfaces.append(current_interface)

    for elem in runtime.nthrest(form, 2):
        if isinstance(elem, ISeq):
            methods.append(__deftype_member(ctx, elem))
        else:
            raise AnalyzerException(
                f"deftype* must consist of interface or protocol names and methods",
                form=elem,
            )

    return interfaces, list(methods)


def __is_abstract(tp: Type) -> bool:
    """Return True if tp is an abstract class.

    The builtin inspect.isabstract returns False for marker abstract classes
    which do not define any abstract members."""
    if inspect.isabstract(tp):
        return True
    elif (
        inspect.isclass(tp)
        and hasattr(tp, "__abstractmethods__")
        and tp.__abstractmethods__ == frozenset()
    ):
        return True
    else:
        return False


def __assert_deftype_impls_are_abstract(  # pylint: disable=too-many-branches,too-many-locals
    fields: Iterable[str],
    interfaces: Iterable[DefTypeBase],
    members: Iterable[DefTypeMember],
) -> None:
    field_names = frozenset(fields)
    member_names = frozenset(munge(member.name) for member in members)
    all_member_names = field_names.union(member_names)
    all_interface_methods: Set[str] = set()
    for interface in interfaces:
        if isinstance(interface, (MaybeClass, MaybeHostForm)):
            interface_type = interface.target
        elif isinstance(interface, VarRef):
            interface_type = interface.var.value
        else:  # pragma: no cover
            assert False, "Interface must be MaybeClass, MaybeHostForm, or VarRef"

        if interface_type is object:
            continue

        if not __is_abstract(interface_type):
            raise AnalyzerException(
                "deftype* interface must be Python abstract class or object",
                form=interface.form,
                lisp_ast=interface,
            )

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
                "deftype* definition missing interface members for interface "
                f"{interface.form}: {missing_methods}"
            )
        elif not interface_property_names.issubset(all_member_names):
            missing_fields = ", ".join(interface_property_names - field_names)
            raise AnalyzerException(
                "deftype* definition missing interface properties for interface "
                f"{interface.form}: {missing_fields}"
            )

        all_interface_methods.update(interface_names)

    extra_methods = member_names - all_interface_methods - OBJECT_DUNDER_METHODS
    if extra_methods:
        extra_method_str = ", ".join(extra_methods)
        raise AnalyzerException(
            "deftype* definition for interface includes members not part of "
            f"defined interfaces: {extra_method_str}"
        )


__DEFTYPE_DEFAULT_SENTINEL = object()


def _deftype_ast(  # pylint: disable=too-many-branches
    ctx: AnalyzerContext, form: ISeq
) -> DefType:
    assert form.first == SpecialForm.DEFTYPE

    nelems = count(form)
    if nelems < 3:
        raise AnalyzerException(
            f"deftype forms must have 3 or more elements, as in: (deftype* name fields [bases+impls])",
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
                raise AnalyzerException(f"deftype* fields must be symbols", form=field)

            field_default = (
                Maybe(field.meta)
                .map(
                    lambda m: m.val_at(  # type: ignore
                        SYM_DEFAULT_META_KEY, __DEFTYPE_DEFAULT_SENTINEL
                    )
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

        interfaces, members = __deftype_impls(ctx, runtime.nthrest(form, 3))
        __assert_deftype_impls_are_abstract(
            map(lambda f: f.name, fields), interfaces, members
        )
        return DefType(
            form=form,
            name=name.name,
            interfaces=vec.vector(interfaces),
            fields=vec.vector(param_nodes),
            members=vec.vector(members),
            is_frozen=is_frozen,
            env=ctx.get_node_env(),
        )


def _do_ast(ctx: AnalyzerContext, form: ISeq) -> Do:
    assert form.first == SpecialForm.DO
    *statements, ret = map(partial(_analyze_form, ctx), form.rest)
    return Do(
        form=form, statements=vec.vector(statements), ret=ret, env=ctx.get_node_env()
    )


def __fn_method_ast(  # pylint: disable=too-many-branches,too-many-locals
    ctx: AnalyzerContext, form: ISeq, fnname: Optional[sym.Symbol] = None
) -> FnMethod:
    with ctx.new_symbol_table("fn-method"):
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
            stmts, ret = _body_ast(ctx, form.rest)
            method = FnMethod(
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


@_with_meta  # noqa: MC0001
def _fn_ast(  # pylint: disable=too-many-branches
    ctx: AnalyzerContext, form: Union[llist.List, ISeq]
) -> Fn:
    assert form.first == SpecialForm.FN

    idx = 1

    with ctx.new_symbol_table("fn"):
        try:
            name = runtime.nth(form, idx)
        except IndexError:
            raise AnalyzerException(
                "fn form must match: (fn* name? [arg*] body*) or (fn* name? method*)",
                form=form,
            )

        if isinstance(name, sym.Symbol):
            name_node: Optional[Binding] = Binding(
                form=name, name=name.name, local=LocalType.FN, env=ctx.get_node_env()
            )
            assert name_node is not None
            is_async = _is_async(name) or isinstance(form, IMeta) and _is_async(form)
            ctx.put_new_symbol(name, name_node, warn_if_unused=False)
            idx += 1
        elif isinstance(name, (llist.List, vec.Vector)):
            name = None
            name_node = None
            is_async = isinstance(form, IMeta) and _is_async(form)
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

        with ctx.new_func_ctx(is_async=is_async):
            if isinstance(arity_or_args, llist.List):
                methods = vec.vector(
                    map(
                        partial(__fn_method_ast, ctx, fnname=name),
                        runtime.nthrest(form, idx),
                    )
                )
            elif isinstance(arity_or_args, vec.Vector):
                methods = vec.v(
                    __fn_method_ast(ctx, runtime.nthrest(form, idx), fnname=name)
                )
            else:
                raise AnalyzerException(
                    "fn form must match: (fn* name? [arg*] body*) or (fn* name? method*)",
                    form=form,
                )

        assert count(methods) > 0, "fn must have at least one arity"

        fixed_arities: MutableSet[int] = set()
        fixed_arity_for_variadic: Optional[int] = None
        num_variadic = 0
        for method in methods:
            if fixed_arity_for_variadic is not None:
                if method.fixed_arity >= fixed_arity_for_variadic:
                    raise AnalyzerException(
                        "fn may not have a method with fixed arity greater than "
                        "fixed arity of variadic function",
                        form=method.form,
                    )
            if method.is_variadic:
                if num_variadic > 0:
                    raise AnalyzerException(
                        "fn may have at most 1 variadic arity", form=method.form
                    )
                fixed_arity_for_variadic = method.fixed_arity
                num_variadic += 1
            else:
                if method.fixed_arity in fixed_arities:
                    raise AnalyzerException(
                        "fn may not have multiple methods with the same fixed arity",
                        form=method.form,
                    )
                fixed_arities.add(method.fixed_arity)

        if fixed_arity_for_variadic is not None and any(
            [fixed_arity_for_variadic < arity for arity in fixed_arities]
        ):
            raise AnalyzerException(
                "variadic arity may not have fewer fixed arity arguments than any other arities",
                form=form,
            )

        return Fn(
            form=form,
            is_variadic=num_variadic == 1,
            max_fixed_arity=max([node.fixed_arity for node in methods]),
            methods=methods,
            local=name_node,
            env=ctx.get_node_env(),
            is_async=is_async,
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

    return HostCall(
        form=form,
        method=method.name[1:],
        target=_analyze_form(ctx, runtime.nth(form, 1)),
        args=vec.vector(map(partial(_analyze_form, ctx), runtime.nthrest(form, 2))),
        env=ctx.get_node_env(),
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
            env=ctx.get_node_env(),
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
            env=ctx.get_node_env(),
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
                env=ctx.get_node_env(),
            )
        else:
            return HostCall(
                form=form,
                method=maybe_m_or_f.name,
                target=_analyze_form(ctx, runtime.nth(form, 1)),
                args=vec.vector(
                    map(partial(_analyze_form, ctx), runtime.nthrest(form, 3))
                ),
                env=ctx.get_node_env(),
            )
    elif isinstance(maybe_m_or_f, (llist.List, ISeq)):
        # Likewise, I emit :host-call for forms like (. target (method arg1 ...)).
        method = maybe_m_or_f.first
        if not isinstance(method, sym.Symbol):
            raise AnalyzerException("host call method must be a symbol", form=method)

        return HostCall(
            form=form,
            method=method.name[1:] if method.name.startswith("-") else method.name,
            target=_analyze_form(ctx, runtime.nth(form, 1)),
            args=vec.vector(map(partial(_analyze_form, ctx), maybe_m_or_f.rest)),
            env=ctx.get_node_env(),
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

    if nelems == 4:
        else_node = _analyze_form(ctx, runtime.nth(form, 3))
    else:
        else_node = _const_node(ctx, None)

    return If(
        form=form,
        test=_analyze_form(ctx, runtime.nth(form, 1)),
        then=_analyze_form(ctx, runtime.nth(form, 2)),
        else_=else_node,
        env=ctx.get_node_env(),
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

    return Import(form=form, aliases=aliases, env=ctx.get_node_env())


def _invoke_ast(ctx: AnalyzerContext, form: Union[llist.List, ISeq]) -> Node:
    fn = _analyze_form(ctx, form.first)

    if fn.op == NodeOp.VAR and isinstance(fn, VarRef):
        if _is_macro(fn.var):
            if ctx.should_macroexpand:
                try:
                    macro_env = ctx.symbol_table.as_env_map()
                    expanded = fn.var.value(macro_env, form, *form.rest)
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

    return Invoke(
        form=form,
        fn=fn,
        args=vec.vector(map(partial(_analyze_form, ctx), form.rest)),
        env=ctx.get_node_env(),
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
            env=ctx.get_node_env(),
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
                env=ctx.get_node_env(),
            )
            loop_node.visit(_assert_recur_is_tail)
            return loop_node


def _quote_ast(ctx: AnalyzerContext, form: ISeq) -> Quote:
    assert form.first == SpecialForm.QUOTE

    with ctx.quoted():
        expr = _analyze_form(ctx, runtime.nth(form, 1))
        assert isinstance(expr, Const), "Quoted expressions must yield :const nodes"
        return Quote(form=form, expr=expr, is_literal=True, env=ctx.get_node_env())


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
    elif node.op in {NodeOp.FN, NodeOp.FN_METHOD, NodeOp.METHOD}:
        assert isinstance(node, (Fn, FnMethod, Method))
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

    exprs = vec.vector(map(partial(_analyze_form, ctx), form.rest))
    return Recur(
        form=form, exprs=exprs, loop_id=ctx.recur_point.loop_id, env=ctx.get_node_env()
    )


def _set_bang_ast(ctx: AnalyzerContext, form: ISeq) -> SetBang:
    assert form.first == SpecialForm.SET_BANG
    nelems = count(form)

    if nelems != 3:
        raise AnalyzerException(
            "set! forms must contain exactly 3 elements: (set! target value)", form=form
        )

    target = _analyze_form(ctx, runtime.nth(form, 1))
    if not isinstance(target, Assignable):
        raise AnalyzerException(
            f"cannot set! targets of type {type(target)}", form=target
        )

    if not target.is_assignable:
        raise AnalyzerException(
            f"cannot set! target which is not assignable", form=target
        )

    return SetBang(
        form=form,
        target=target,
        val=_analyze_form(ctx, runtime.nth(form, 2)),
        env=ctx.get_node_env(),
    )


def _throw_ast(ctx: AnalyzerContext, form: ISeq) -> Throw:
    assert form.first == SpecialForm.THROW
    return Throw(
        form=form,
        exception=_analyze_form(ctx, runtime.nth(form, 1)),
        env=ctx.get_node_env(),
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
        *catch_statements, catch_ret = map(partial(_analyze_form, ctx), catch_body)
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
                *finally_stmts, finally_ret = map(
                    partial(_analyze_form, ctx), expr.rest
                )
                finally_ = Do(
                    form=expr.rest,
                    statements=vec.vector(finally_stmts),
                    ret=finally_ret,
                    is_body=True,
                    env=ctx.get_node_env(),
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
        [isinstance(node, Catch) for node in catches]
    ), "All catch statements must be catch ops"

    *try_statements, try_ret = try_exprs
    return Try(
        form=form,
        body=Do(
            form=form,
            statements=vec.vector(try_statements),
            ret=try_ret,
            is_body=True,
            env=ctx.get_node_env(),
        ),
        catches=vec.vector(catches),
        finally_=finally_,
        children=vec.v(BODY, CATCHES, FINALLY)
        if finally_ is not None
        else vec.v(BODY, CATCHES),
        env=ctx.get_node_env(),
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

    return VarRef(form=var_sym, var=var, return_var=True, env=ctx.get_node_env())


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
    SpecialForm.LOOP: _loop_ast,
    SpecialForm.QUOTE: _quote_ast,
    SpecialForm.RECUR: _recur_ast,
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
        elif s.name.startswith("."):
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
        form,
        field=form.name,
        target=parent_node,
        is_assignable=True,
        env=ctx.get_node_env(),
    )


def __resolve_namespaced_symbol(  # pylint: disable=too-many-branches
    ctx: AnalyzerContext, form: sym.Symbol
) -> Union[HostField, MaybeClass, MaybeHostForm, VarRef]:
    """Resolve a namespaced symbol into a Python name or Basilisp Var."""
    assert form.ns is not None

    if form.ns == ctx.current_ns.name:
        v = ctx.current_ns.find(sym.symbol(form.name))
        if v is not None:
            return VarRef(form=form, var=v, env=ctx.get_node_env())
    elif form.ns == _BUILTINS_NS:
        class_ = munge(form.name, allow_builtins=True)
        target = getattr(builtins, class_, None)
        if target is None:
            raise AnalyzerException(
                f"cannot resolve builtin function '{class_}'", form=form
            )
        return MaybeClass(
            form=form, class_=class_, target=target, env=ctx.get_node_env()
        )

    if "." in form.name:
        raise AnalyzerException(
            "symbol names may not contain the '.' operator", form=form
        )

    ns_sym = sym.symbol(form.ns)
    if ns_sym in ctx.current_ns.imports or ns_sym in ctx.current_ns.import_aliases:
        # We still import Basilisp code, so we'll want to make sure
        # that the symbol isn't referring to a Basilisp Var first
        v = Var.find(form)
        if v is not None:
            return VarRef(form=form, var=v, env=ctx.get_node_env())

        # Fetch the full namespace name for the aliased namespace/module.
        # We don't need this for actually generating the link later, but
        # we _do_ need it for fetching a reference to the module to check
        # for membership.
        if ns_sym in ctx.current_ns.import_aliases:
            ns = ctx.current_ns.import_aliases[ns_sym]
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
                env=ctx.get_node_env(),
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
            env=ctx.get_node_env(),
        )
    elif ns_sym in ctx.current_ns.aliases:
        aliased_ns: runtime.Namespace = ctx.current_ns.aliases[ns_sym]
        v = Var.find(sym.symbol(form.name, ns=aliased_ns.name))
        if v is None:
            raise AnalyzerException(
                f"unable to resolve symbol '{sym.symbol(form.name, ns_sym.name)}' in this context",
                form=form,
            )
        return VarRef(form=form, var=v, env=ctx.get_node_env())
    elif "." in form.ns:
        return _resolve_nested_symbol(ctx, form)
    else:
        raise AnalyzerException(
            f"unable to resolve symbol '{form}' in this context", form=form
        )


def __resolve_bare_symbol(
    ctx: AnalyzerContext, form: sym.Symbol
) -> Union[MaybeClass, VarRef]:
    """Resolve a non-namespaced symbol into a Python name or a local
    Basilisp Var."""
    assert form.ns is None

    # Look up the symbol in the namespace mapping of the current namespace.
    v = ctx.current_ns.find(form)
    if v is not None:
        return VarRef(form=form, var=v, env=ctx.get_node_env())

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
            env=ctx.get_node_env(),
        )

    assert munged not in vars(ctx.current_ns.module)
    raise AnalyzerException(
        f"unable to resolve symbol '{form}' in this context", form=form
    )


def _resolve_sym(
    ctx: AnalyzerContext, form: sym.Symbol
) -> Union[HostField, MaybeClass, MaybeHostForm, VarRef]:
    """Resolve a Basilisp symbol as a Var or Python name."""
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
            env=ctx.get_node_env(),
        )

    return _resolve_sym(ctx, form)


def _py_dict_node(ctx: AnalyzerContext, form: dict) -> PyDict:
    keys, vals = [], []
    for k, v in form.items():
        keys.append(_analyze_form(ctx, k))
        vals.append(_analyze_form(ctx, v))

    return PyDict(
        form=form, keys=vec.vector(keys), vals=vec.vector(vals), env=ctx.get_node_env()
    )


def _py_list_node(ctx: AnalyzerContext, form: list) -> PyList:
    return PyList(
        form=form,
        items=vec.vector(map(partial(_analyze_form, ctx), form)),
        env=ctx.get_node_env(),
    )


def _py_set_node(ctx: AnalyzerContext, form: set) -> PySet:
    return PySet(
        form=form,
        items=vec.vector(map(partial(_analyze_form, ctx), form)),
        env=ctx.get_node_env(),
    )


def _py_tuple_node(ctx: AnalyzerContext, form: tuple) -> PyTuple:
    return PyTuple(
        form=form,
        items=vec.vector(map(partial(_analyze_form, ctx), form)),
        env=ctx.get_node_env(),
    )


@_with_meta
def _map_node(ctx: AnalyzerContext, form: lmap.Map) -> MapNode:
    keys, vals = [], []
    for k, v in form.items():
        keys.append(_analyze_form(ctx, k))
        vals.append(_analyze_form(ctx, v))

    return MapNode(
        form=form, keys=vec.vector(keys), vals=vec.vector(vals), env=ctx.get_node_env()
    )


@_with_meta
def _set_node(ctx: AnalyzerContext, form: lset.Set) -> SetNode:
    return SetNode(
        form=form,
        items=vec.vector(map(partial(_analyze_form, ctx), form)),
        env=ctx.get_node_env(),
    )


@_with_meta
def _vector_node(ctx: AnalyzerContext, form: vec.Vector) -> VectorNode:
    return VectorNode(
        form=form,
        items=vec.vector(map(partial(_analyze_form, ctx), form)),
        env=ctx.get_node_env(),
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
    )

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
        env=ctx.get_node_env(),
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
        elif ctx.is_quoted:
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
    ctx = AnalyzerContext("<Macroexpand>", should_macroexpand=False)
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
    return analyze_form(AnalyzerContext("<Macroexpand>"), form).form
