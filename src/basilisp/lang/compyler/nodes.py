from abc import ABC, abstractmethod
from enum import Enum
from typing import Collection, Union, Optional, Iterable, Generic, TypeVar, Callable

import attr

import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.seq as lseq
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
from basilisp.lang.runtime import Var
from basilisp.lang.typing import LispForm
from basilisp.lang.util import munge

BODY = kw.keyword("body")
CLASS = kw.keyword("class")
LOCAL = kw.keyword("local")
STATEMENTS = kw.keyword("statements")
RET = kw.keyword("ret")
METHODS = kw.keyword("methods")
PARAMS = kw.keyword("params")
TARGET = kw.keyword("target")
ARGS = kw.keyword("args")
FN = kw.keyword("fn")
BINDINGS = kw.keyword("bindings")
KEYS = kw.keyword("keys")
VALS = kw.keyword("vals")
EXPR = kw.keyword("expr")
EXPRS = kw.keyword("exprs")
ITEMS = kw.keyword("items")
EXCEPTION = kw.keyword("exception")
META = kw.keyword("meta")


class NodeOp(Enum):
    BINDING = kw.keyword("binding")
    CATCH = kw.keyword("catch")
    CONST = kw.keyword("const")
    DEF = kw.keyword("def")
    DO = kw.keyword("do")
    FN = kw.keyword("fn")
    FN_METHOD = kw.keyword("fn-method")
    HOST_CALL = kw.keyword("host-call")
    HOST_FIELD = kw.keyword("host-field")
    HOST_INTEROP = kw.keyword("host-interop")
    IF = kw.keyword("if")
    INVOKE = kw.keyword("invoke")
    LET = kw.keyword("let")
    LETFN = kw.keyword("letfn")
    LOCAL = kw.keyword("local")
    LOOP = kw.keyword("loop")
    MAP = kw.keyword("map")
    MAYBE_CLASS = kw.keyword("maybe-class")
    MAYBE_HOST_FORM = kw.keyword("maybe-host-form")
    QUOTE = kw.keyword("quote")
    RECUR = kw.keyword("recur")
    SET = kw.keyword("set")
    SET_BANG = kw.keyword("set!")
    THROW = kw.keyword("throw")
    TRY = kw.keyword("try")
    VAR = kw.keyword("var")
    VECTOR = kw.keyword("vector")
    WITH_META = kw.keyword("with-meta")


T = TypeVar("T")


class Node(ABC, Generic[T]):
    __slots__ = ()

    @property
    @abstractmethod
    def op(self) -> NodeOp:
        pass

    @property
    @abstractmethod
    def form(self) -> T:
        pass

    @property
    @abstractmethod
    def children(self) -> Collection[kw.Keyword]:
        pass

    @property
    @abstractmethod
    def top_level(self) -> bool:
        pass

    def assoc(self, **kwargs):
        return attr.evolve(self, **kwargs)

    def visit(self, f: Callable[..., None], *args, **kwargs):
        """Visit all descendants of this node, calling f(node, *args, **kwargs)
        on each before visiting its descendants recursively."""
        for child_kw in self.children:
            child_attr = munge(child_kw.name)

            if child_attr.endswith("s"):
                iter_child: Iterable[Node] = getattr(self, child_attr)
                assert iter_child is not None, "Listed child must not be none"
                for item in iter_child:
                    f(item, *args, **kwargs)
                    item.visit(f, *args, **kwargs)
            else:
                child: Node = getattr(self, child_attr)
                assert child is not None, "Listed child must not be none"
                f(child, *args, **kwargs)
                child.visit(f, *args, **kwargs)


class Assignable(ABC):
    __slots__ = ()

    @property
    @abstractmethod
    def is_assignable(self) -> bool:
        pass


class ConstType(Enum):
    NIL = kw.Keyword("nil")
    MAP = kw.keyword("map")
    SET = kw.keyword("set")
    VECTOR = kw.keyword("vector")
    BOOL = kw.keyword("bool")
    KEYWORD = kw.keyword("keyword")
    SYMBOL = kw.keyword("symbol")
    STRING = kw.keyword("string")
    NUMBER = kw.keyword("number")
    DECIMAL = kw.keyword("decimal")
    FRACTION = kw.keyword("fraction")
    RECORD = kw.keyword("record")
    SEQ = kw.keyword("seq")
    CHAR = kw.keyword("char")
    REGEX = kw.keyword("regex")
    CLASS = kw.keyword("class")
    INST = kw.keyword("inst")
    UUID = kw.keyword("uuid")
    UNKNOWN = kw.keyword("unknown")


NodeMeta = Union[None, "Const", "Map"]
ReaderLispForm = Union[LispForm, lseq.Seq]
SpecialForm = Union[llist.List, lseq.Seq]
LoopID = sym.Symbol


class LocalType(Enum):
    ARG = kw.keyword("arg")
    CATCH = kw.keyword("catch")
    FN = kw.keyword("fn")
    LET = kw.keyword("let")
    LETFN = kw.keyword("letfn")
    LOOP = kw.keyword("loop")


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Binding(Node[sym.Symbol]):
    form: sym.Symbol
    name: sym.Symbol
    local: LocalType
    arg_id: Optional[int] = None
    is_variadic: bool = False
    init: Optional[Node] = None
    meta: NodeMeta = None
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.BINDING
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Catch(Node[SpecialForm]):
    form: SpecialForm
    class_: "MaybeClass"
    local: Binding
    body: "Do"
    children: Collection[kw.Keyword] = vec.v(CLASS, LOCAL, BODY)
    op: NodeOp = NodeOp.CATCH
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Const(Node[ReaderLispForm]):
    form: ReaderLispForm
    type: ConstType
    val: ReaderLispForm
    is_literal: bool
    meta: NodeMeta = None
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.CONST
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Def(Node[SpecialForm]):
    form: SpecialForm
    name: sym.Symbol
    init: Optional[Node]
    doc: Optional[str]
    meta: NodeMeta = None
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.DEF
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Do(Node[SpecialForm]):
    form: SpecialForm
    statements: Iterable[Node]
    ret: Node
    is_body: bool = False
    children: Collection[kw.Keyword] = vec.v(STATEMENTS, RET)
    op: NodeOp = NodeOp.DO
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Fn(Node[SpecialForm]):
    form: SpecialForm
    max_fixed_arity: int
    methods: Iterable["FnMethod"]
    local: Optional[Binding] = None
    is_variadic: bool = False
    children: Collection[kw.Keyword] = vec.v(METHODS)
    op: NodeOp = NodeOp.FN
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class FnMethod(Node[SpecialForm]):
    form: SpecialForm
    loop_id: LoopID
    params: Iterable[Binding]
    fixed_arity: int
    body: Do
    is_variadic: bool = False
    children: Collection[kw.Keyword] = vec.v(PARAMS, BODY)
    op: NodeOp = NodeOp.FN_METHOD
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HostCall(Node[SpecialForm]):
    form: SpecialForm
    method: sym.Symbol
    target: Node
    args: Iterable[Node]
    children: Collection[kw.Keyword] = vec.v(TARGET, ARGS)
    op: NodeOp = NodeOp.HOST_CALL
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HostField(Node[SpecialForm], Assignable):
    form: SpecialForm
    field: sym.Symbol
    target: Node
    is_assignable: bool = True
    children: Collection[kw.Keyword] = vec.v(TARGET)
    op: NodeOp = NodeOp.HOST_FIELD
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HostInterop(Node[SpecialForm], Assignable):
    form: SpecialForm
    m_or_f: sym.Symbol
    target: Node
    args: Iterable[Node] = ()
    is_assignable: bool = True
    children: Collection[kw.Keyword] = vec.v(TARGET)
    op: NodeOp = NodeOp.HOST_INTEROP
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class If(Node[SpecialForm]):
    form: SpecialForm
    test: Node
    then: Node
    else_: Node = Const(form=None, type=ConstType.NIL, val=None, is_literal=True)
    children: Collection[kw.Keyword] = vec.v(TARGET)
    op: NodeOp = NodeOp.IF
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Invoke(Node[SpecialForm]):
    form: SpecialForm
    fn: Node
    args: Iterable[Node]
    meta: NodeMeta = None
    children: Collection[kw.Keyword] = vec.v(FN, ARGS)
    op: NodeOp = NodeOp.INVOKE
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Let(Node[SpecialForm]):
    form: SpecialForm
    bindings: Iterable[Binding]
    body: Do
    children: Collection[kw.Keyword] = vec.v(BINDINGS, BODY)
    op: NodeOp = NodeOp.LET
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class LetFn(Node[SpecialForm]):
    form: SpecialForm
    bindings: Iterable[Binding]
    body: Do
    children: Collection[kw.Keyword] = vec.v(BINDINGS, BODY)
    op: NodeOp = NodeOp.LETFN
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Local(Node[sym.Symbol], Assignable):
    form: sym.Symbol
    name: sym.Symbol
    local: LocalType
    is_assignable: bool = False
    arg_id: Optional[int] = None
    is_variadic: bool = False
    children: Collection[kw.Keyword] = vec.v(BINDINGS, BODY)
    op: NodeOp = NodeOp.LETFN
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Loop(Node[SpecialForm]):
    form: SpecialForm
    bindings: Iterable[Binding]
    body: Do
    loop_id: LoopID
    children: Collection[kw.Keyword] = vec.v(BINDINGS, BODY)
    op: NodeOp = NodeOp.LOOP
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Map(Node[lmap.Map]):
    form: lmap.Map
    keys: Iterable[Node]
    vals: Iterable[Node]
    children: Collection[kw.Keyword] = vec.v(KEYS, VALS)
    op: NodeOp = NodeOp.MAP
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class MaybeClass(Node[sym.Symbol]):
    form: sym.Symbol
    class_: sym.Symbol
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.MAYBE_CLASS
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class MaybeHostForm(Node[sym.Symbol]):
    form: sym.Symbol
    class_: sym.Symbol
    field: sym.Symbol
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.MAYBE_HOST_FORM
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Quote(Node[SpecialForm]):
    form: SpecialForm
    expr: Const
    is_literal: bool = True
    children: Collection[kw.Keyword] = vec.v(EXPR)
    op: NodeOp = NodeOp.QUOTE
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Recur(Node[SpecialForm]):
    form: SpecialForm
    exprs: Iterable[Node]
    loop_id: LoopID
    children: Collection[kw.Keyword] = vec.v(EXPRS)
    op: NodeOp = NodeOp.RECUR
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Set(Node[lset.Set]):
    form: lset.Set
    items: Iterable[Node]
    children: Collection[kw.Keyword] = vec.v(ITEMS)
    op: NodeOp = NodeOp.SET
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SetBang(Node[SpecialForm]):
    form: SpecialForm
    target: Node
    val: Union[HostField, HostInterop, Local, "VarRef"]
    children: Collection[kw.Keyword] = vec.v(ITEMS)
    op: NodeOp = NodeOp.SET_BANG
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Throw(Node[SpecialForm]):
    form: SpecialForm
    exception: Node
    children: Collection[kw.Keyword] = vec.v(EXCEPTION)
    op: NodeOp = NodeOp.THROW
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Try(Node[SpecialForm]):
    form: SpecialForm
    body: Do
    catches: Iterable[Catch]
    children: Collection[kw.Keyword]
    finally_: Optional[Do] = None
    op: NodeOp = NodeOp.TRY
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class VarRef(Node[sym.Symbol], Assignable):
    form: sym.Symbol
    var: Var
    is_assignable: bool
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.VAR
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Vector(Node[vec.Vector]):
    form: vec.Vector
    items: Iterable[Node]
    children: Collection[kw.Keyword] = vec.v(ITEMS)
    op: NodeOp = NodeOp.VECTOR
    top_level: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class WithMeta(Node[LispForm]):
    form: LispForm
    meta: Union[Const, Map]
    expr: Union[Fn, Map, Set, Vector]
    children: Collection[kw.Keyword] = vec.v(META, EXPR)
    op: NodeOp = NodeOp.WITH_META
    top_level: bool = False


ParentNode = Union[
    Const,
    Def,
    Do,
    Fn,
    HostCall,
    HostField,
    HostInterop,
    If,
    Invoke,
    Let,
    LetFn,
    Loop,
    Map,
    MaybeClass,
    MaybeHostForm,
    Quote,
    Set,
    SetBang,
    Throw,
    Try,
    VarRef,
    Vector,
    WithMeta,
]
ChildOnlyNode = Union[Binding, Catch, FnMethod, Local, Recur]
AnyNode = Union[ParentNode, ChildOnlyNode]
