from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Collection,
    Union,
    Optional,
    Iterable,
    Generic,
    TypeVar,
    Callable,
    Tuple,
    Dict,
)

import attr

import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.seq as lseq
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
from basilisp.lang.runtime import Namespace, Var
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
VAL = kw.keyword("val")
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
TEST = kw.keyword("test")
THEN = kw.keyword("then")
ELSE = kw.keyword("else")


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
    IMPORT = kw.keyword("import")
    IMPORT_ALIAS = kw.keyword("import-alias")
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
        raise NotImplementedError()

    @property
    @abstractmethod
    def form(self) -> T:
        raise NotImplementedError()

    @property
    @abstractmethod
    def children(self) -> Collection[kw.Keyword]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def raw_forms(self) -> Collection[LispForm]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def top_level(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def env(self) -> "NodeEnv":
        raise NotImplementedError()

    def assoc(self, **kwargs):
        return attr.evolve(self, **kwargs)

    def visit(self, f: Callable[..., None], *args, **kwargs):
        """Visit all immediate children of this node, calling
        f(child, *args, **kwargs) on each child."""
        for child_kw in self.children:
            child_attr = munge(child_kw.name)

            if child_attr.endswith("s"):
                iter_child: Iterable[Node] = getattr(self, child_attr)
                assert iter_child is not None, "Listed child must not be none"
                for item in iter_child:
                    f(item, *args, **kwargs)
            else:
                child: Node = getattr(self, child_attr)
                assert child is not None, "Listed child must not be none"
                f(child, *args, **kwargs)

    def fix_missing_locations(
        self, start_loc: Optional[Tuple[int, int]] = None
    ) -> "Node":
        """Return a transformed copy of this node with location in this node's
        environment updated to match the `start_loc` if given, or using its
        existing location otherwise. All child nodes will be recursively
        transformed and replaced. Child nodes will use their parent node
        location if they do not have one."""
        if self.env.line is None or self.env.col is None:
            loc = start_loc
        else:
            loc = (self.env.line, self.env.col)

        assert loc is not None and all(
            [e is not None for e in loc]
        ), "Must specify location information"

        new_attrs: Dict[str, Union[NodeEnv, Node, Iterable[Node]]] = {
            "env": attr.evolve(self.env, line=loc[0], col=loc[1])
        }
        for child_kw in self.children:
            child_attr = munge(child_kw.name)
            assert child_attr != "env", "Node environment already set"

            if child_attr.endswith("s"):
                iter_child: Iterable[Node] = getattr(self, child_attr)
                assert iter_child is not None, "Listed child must not be none"
                new_children = []
                for item in iter_child:
                    new_children.append(item.fix_missing_locations(start_loc))
                new_attrs[child_attr] = vec.vector(new_children)
            else:
                child: Node = getattr(self, child_attr)
                assert child is not None, "Listed child must not be none"
                new_attrs[child_attr] = child.fix_missing_locations(start_loc)

        return self.assoc(**new_attrs)


class Assignable(ABC):
    __slots__ = ()

    @property
    @abstractmethod
    def is_assignable(self) -> bool:
        raise NotImplementedError()


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
LoopID = str


class LocalType(Enum):
    ARG = kw.keyword("arg")
    CATCH = kw.keyword("catch")
    FN = kw.keyword("fn")
    LET = kw.keyword("let")
    LETFN = kw.keyword("letfn")
    LOOP = kw.keyword("loop")


@attr.s(auto_attribs=True, frozen=True, slots=True)
class NodeEnv:
    ns: Namespace
    file: str
    line: Optional[int] = None
    col: Optional[int] = None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Binding(Node[sym.Symbol]):
    form: sym.Symbol
    name: str
    local: LocalType
    env: NodeEnv
    arg_id: Optional[int] = None
    is_variadic: bool = False
    init: Optional[Node] = None
    meta: NodeMeta = None
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.BINDING
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Catch(Node[SpecialForm]):
    form: SpecialForm
    class_: Union["MaybeClass", "MaybeHostForm"]
    local: Binding
    body: "Do"
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.v(CLASS, LOCAL, BODY)
    op: NodeOp = NodeOp.CATCH
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Const(Node[ReaderLispForm]):
    form: ReaderLispForm
    type: ConstType
    val: ReaderLispForm
    is_literal: bool
    env: NodeEnv
    meta: NodeMeta = None
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.CONST
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Def(Node[SpecialForm]):
    form: SpecialForm
    name: sym.Symbol
    var: Var
    init: Optional[Node]
    doc: Optional[str]
    env: NodeEnv
    meta: NodeMeta = None
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.DEF
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Do(Node[SpecialForm]):
    form: SpecialForm
    statements: Iterable[Node]
    ret: Node
    env: NodeEnv
    is_body: bool = False
    children: Collection[kw.Keyword] = vec.v(STATEMENTS, RET)
    op: NodeOp = NodeOp.DO
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Fn(Node[SpecialForm]):
    form: SpecialForm
    max_fixed_arity: int
    methods: Collection["FnMethod"]
    env: NodeEnv
    local: Optional[Binding] = None
    is_variadic: bool = False
    children: Collection[kw.Keyword] = vec.v(METHODS)
    op: NodeOp = NodeOp.FN
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class FnMethod(Node[SpecialForm]):
    form: SpecialForm
    loop_id: LoopID
    params: Iterable[Binding]
    fixed_arity: int
    body: Do
    env: NodeEnv
    is_variadic: bool = False
    children: Collection[kw.Keyword] = vec.v(PARAMS, BODY)
    op: NodeOp = NodeOp.FN_METHOD
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HostCall(Node[SpecialForm]):
    form: SpecialForm
    method: str
    target: Node
    args: Iterable[Node]
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.v(TARGET, ARGS)
    op: NodeOp = NodeOp.HOST_CALL
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HostField(Node[SpecialForm], Assignable):
    form: SpecialForm
    field: str
    target: Node
    env: NodeEnv
    is_assignable: bool = True
    children: Collection[kw.Keyword] = vec.v(TARGET)
    op: NodeOp = NodeOp.HOST_FIELD
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class If(Node[SpecialForm]):
    form: SpecialForm
    test: Node
    then: Node
    env: NodeEnv
    else_: Node
    children: Collection[kw.Keyword] = vec.v(TEST, THEN, ELSE)
    op: NodeOp = NodeOp.IF
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Import(Node[SpecialForm]):
    form: SpecialForm
    aliases: Iterable["ImportAlias"]
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.IMPORT
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ImportAlias(Node[Union[sym.Symbol, vec.Vector]]):
    form: Union[sym.Symbol, vec.Vector]
    name: str
    alias: str
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.IMPORT_ALIAS
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Invoke(Node[SpecialForm]):
    form: SpecialForm
    fn: Node
    args: Iterable[Node]
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.v(FN, ARGS)
    op: NodeOp = NodeOp.INVOKE
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Let(Node[SpecialForm]):
    form: SpecialForm
    bindings: Iterable[Binding]
    body: Do
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.v(BINDINGS, BODY)
    op: NodeOp = NodeOp.LET
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class LetFn(Node[SpecialForm]):
    form: SpecialForm
    bindings: Iterable[Binding]
    body: Do
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.v(BINDINGS, BODY)
    op: NodeOp = NodeOp.LETFN
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Local(Node[sym.Symbol], Assignable):
    form: sym.Symbol
    name: str
    local: LocalType
    env: NodeEnv
    is_assignable: bool = False
    arg_id: Optional[int] = None
    is_variadic: bool = False
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.LOCAL
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Loop(Node[SpecialForm]):
    form: SpecialForm
    bindings: Iterable[Binding]
    body: Do
    loop_id: LoopID
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.v(BINDINGS, BODY)
    op: NodeOp = NodeOp.LOOP
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Map(Node[lmap.Map]):
    form: lmap.Map
    keys: Iterable[Node]
    vals: Iterable[Node]
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.v(KEYS, VALS)
    op: NodeOp = NodeOp.MAP
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class MaybeClass(Node[sym.Symbol]):
    form: sym.Symbol
    class_: str
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.MAYBE_CLASS
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class MaybeHostForm(Node[sym.Symbol]):
    form: sym.Symbol
    class_: str
    field: str
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.MAYBE_HOST_FORM
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Quote(Node[SpecialForm]):
    form: SpecialForm
    expr: Const
    env: NodeEnv
    is_literal: bool = True
    children: Collection[kw.Keyword] = vec.v(EXPR)
    op: NodeOp = NodeOp.QUOTE
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Recur(Node[SpecialForm]):
    form: SpecialForm
    exprs: Iterable[Node]
    loop_id: LoopID
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.v(EXPRS)
    op: NodeOp = NodeOp.RECUR
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Set(Node[lset.Set]):
    form: lset.Set
    items: Iterable[Node]
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.v(ITEMS)
    op: NodeOp = NodeOp.SET
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SetBang(Node[SpecialForm]):
    form: SpecialForm
    target: Union[Assignable, Node]
    val: Node
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.v(TARGET, VAL)
    op: NodeOp = NodeOp.SET_BANG
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Throw(Node[SpecialForm]):
    form: SpecialForm
    exception: Node
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.v(EXCEPTION)
    op: NodeOp = NodeOp.THROW
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Try(Node[SpecialForm]):
    form: SpecialForm
    body: Do
    catches: Iterable[Catch]
    children: Collection[kw.Keyword]
    env: NodeEnv
    finally_: Optional[Do] = None
    op: NodeOp = NodeOp.TRY
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class VarRef(Node[sym.Symbol], Assignable):
    form: sym.Symbol
    var: Var
    env: NodeEnv
    return_var: bool = False
    children: Collection[kw.Keyword] = vec.Vector.empty()
    op: NodeOp = NodeOp.VAR
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()

    @property
    def is_assignable(self) -> bool:
        return self.var.dynamic


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Vector(Node[vec.Vector]):
    form: vec.Vector
    items: Iterable[Node]
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.v(ITEMS)
    op: NodeOp = NodeOp.VECTOR
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class WithMeta(Node[LispForm]):
    form: LispForm
    meta: Union[Const, Map]
    expr: Union[Fn, Map, Set, Vector]
    env: NodeEnv
    children: Collection[kw.Keyword] = vec.v(META, EXPR)
    op: NodeOp = NodeOp.WITH_META
    top_level: bool = False
    raw_forms: Collection[LispForm] = vec.Vector.empty()


ParentNode = Union[
    Const,
    Def,
    Do,
    Fn,
    HostCall,
    HostField,
    If,
    Import,
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
ChildOnlyNode = Union[Binding, Catch, FnMethod, ImportAlias, Local, Recur]
AnyNode = Union[ParentNode, ChildOnlyNode]
SpecialFormNode = Union[
    Def,
    Do,
    Fn,
    If,
    HostCall,
    HostField,
    Import,
    Invoke,
    Let,
    Loop,
    Quote,
    Recur,
    SetBang,
    Throw,
    Try,
    VarRef,
]
