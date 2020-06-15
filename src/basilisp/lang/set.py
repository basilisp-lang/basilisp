from collections.abc import Set as _PySet
from typing import AbstractSet, Iterable, Optional, TypeVar

from immutables import Map as _Map

from basilisp.lang.interfaces import (
    ILispObject,
    IPersistentMap,
    IPersistentSet,
    ISeq,
    IWithMeta,
)
from basilisp.lang.obj import seq_lrepr as _seq_lrepr
from basilisp.lang.seq import sequence

T = TypeVar("T")


class Set(IPersistentSet[T], ILispObject, IWithMeta):
    """Basilisp Set. Delegates internally to a pyrsistent.PSet object.

    Do not instantiate directly. Instead use the s() and set() factory
    methods below."""

    __slots__ = ("_inner", "_meta")

    def __init__(self, members: Optional[Iterable[T]], meta=None) -> None:
        self._inner = _Map((m, m) for m in (members or ()))
        self._meta = meta

    def __bool__(self):
        return True

    def __call__(self, key, default=None):
        if key in self:
            return key
        return None

    def __contains__(self, item):
        return item in self._inner

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, AbstractSet):
            return NotImplemented
        return _PySet.__eq__(self, other)

    def __hash__(self):
        return self._hash()  # type: ignore[attr-defined]

    def __iter__(self):
        yield from self._inner.keys()

    def __len__(self):
        return len(self._inner)

    def _lrepr(self, **kwargs):
        return _seq_lrepr(self._inner, "#{", "}", meta=self._meta, **kwargs)

    issubset = _PySet.__le__
    issuperset = _PySet.__ge__

    def difference(self, *others):
        e = self
        for other in others:
            e = e - other
        return e

    def intersection(self, *others):
        e = self
        for other in others:
            e = e & other
        return e

    def symmetric_difference(self, *others):
        e = self._inner
        for other in others:
            e = e ^ other
        return e

    def union(self, *others):
        e = self._inner
        for other in others:
            e = e | other
        return e

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "Set[T]":
        return set(self._inner, meta=meta)

    def cons(self, *elems: T) -> "Set[T]":
        with self._inner.mutate() as m:
            for elem in elems:
                m.set(elem, elem)
            return Set(m.finish(), meta=self.meta)

    def disj(self, *elems: T) -> "Set[T]":
        with self._inner.mutate() as m:
            for elem in elems:
                try:
                    del m[elem]
                except KeyError:
                    pass
            return Set(m.finish(), meta=self.meta)

    @staticmethod
    def empty() -> "Set":
        return s()

    def seq(self) -> ISeq[T]:
        return sequence(self)


def set(members: Iterable[T], meta=None) -> Set[T]:  # pylint:disable=redefined-builtin
    """Creates a new set."""
    return Set(members, meta=meta)


def s(*members: T, meta=None) -> Set[T]:
    """Creates a new set from members."""
    return Set(members, meta=meta)
