from typing import Optional, TypeVar, cast

from pyrsistent import PList, plist  # noqa # pylint: disable=unused-import
from pyrsistent._plist import _EMPTY_PLIST

from basilisp.lang.interfaces import IPersistentList, IPersistentMap, ISeq, IWithMeta
from basilisp.lang.obj import seq_lrepr as _seq_lrepr
from basilisp.lang.seq import EMPTY

T = TypeVar("T")


class List(IPersistentList[T], ISeq[T], IWithMeta):
    """Basilisp List. Delegates internally to a pyrsistent.PList object.

    Do not instantiate directly. Instead use the l() and list() factory
    methods below."""

    __slots__ = ("_inner", "_meta")

    def __init__(self, wrapped: "PList[T]", meta=None) -> None:
        self._inner = wrapped
        self._meta = meta

    def __bool__(self):
        return True

    def __getitem__(self, item):
        if isinstance(item, slice):
            return List(self._inner[item])
        return self._inner[item]

    def __hash__(self):
        return hash(self._inner)

    def __len__(self):
        return len(self._inner)

    def _lrepr(self, **kwargs) -> str:
        return _seq_lrepr(self._inner, "(", ")", meta=self._meta, **kwargs)

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "List":
        return list(self._inner, meta=meta)

    @property
    def is_empty(self):
        return self._inner is _EMPTY_PLIST

    @property
    def first(self):
        try:
            return self._inner.first
        except AttributeError:
            return None

    @property
    def rest(self) -> ISeq[T]:
        if self._inner.rest is _EMPTY_PLIST:
            return EMPTY
        return List(self._inner.rest)

    def cons(self, *elems: T) -> "List[T]":
        l = self._inner
        for elem in elems:
            l = l.cons(elem)
        return List(l)

    @staticmethod
    def empty(meta=None) -> "List":  # pylint:disable=arguments-differ
        return l(meta=meta)

    def peek(self):
        return self.first

    def pop(self) -> "List[T]":
        if self.is_empty:
            raise IndexError("Cannot pop an empty list")
        return cast(List, self.rest)


def list(members, meta=None) -> List:  # pylint:disable=redefined-builtin
    """Creates a new list."""
    return List(  # pylint: disable=abstract-class-instantiated
        plist(iterable=members), meta=meta
    )


def l(*members, meta=None) -> List:  # noqa
    """Creates a new list from members."""
    return List(  # pylint: disable=abstract-class-instantiated
        plist(iterable=members), meta=meta
    )
