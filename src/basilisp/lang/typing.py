import uuid
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from re import Pattern
from typing import Any, Protocol, Union

from basilisp.lang import keyword as kw
from basilisp.lang import list as llist
from basilisp.lang import map as lmap
from basilisp.lang import queue as lqueue
from basilisp.lang import set as lset
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec
from basilisp.lang.interfaces import (
    IPersistentMap,
    IPersistentSet,
    IRecord,
    ISeq,
    IType,
)
from basilisp.lang.tagged import TaggedLiteral

CompilerOpts = IPersistentMap[kw.Keyword, bool]

IterableLispForm = Union[
    llist.PersistentList, lmap.PersistentMap, lset.PersistentSet, vec.PersistentVector
]
LispNumber = Union[int, float, Decimal, Fraction]
LispForm = Union[
    bool,
    bytes,
    complex,
    datetime,
    Decimal,
    int,
    float,
    Fraction,
    kw.Keyword,
    llist.PersistentList,
    lmap.PersistentMap,
    None,
    Pattern,
    lqueue.PersistentQueue,
    lset.PersistentSet,
    str,
    sym.Symbol,
    vec.PersistentVector,
    uuid.UUID,
]
PyCollectionForm = Union[dict, list, set, tuple]
ReaderForm = Union[LispForm, IRecord, ISeq, IType, PyCollectionForm, TaggedLiteral]
SpecialForm = Union[llist.PersistentList, ISeq]


class Comparable(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...


class BasilispFunction(Protocol):
    _basilisp_fn: bool
    arities: IPersistentSet[kw.Keyword | int]
    meta: IPersistentMap | None

    def __call__(self, *args, **kwargs): ...

    def apply_to(self, args: list, rest: ISeq | None): ...

    def with_meta(self, meta: IPersistentMap | None) -> "BasilispFunction": ...
