from collections.abc import Set as _PySet
from typing import AbstractSet, Iterable, Optional, TypeVar

from immutables import Map as _Map

from basilisp.lang.interfaces import (
    IEvolveableCollection,
    ILispObject,
    IPersistentMap,
    IPersistentSet,
    ISeq,
    ITransientSet,
    IWithMeta,
)
from basilisp.lang.obj import seq_lrepr as _seq_lrepr
from basilisp.lang.seq import sequence

try:
    from immutables._map import MapMutation  # pylint: disable=unused-import
except ImportError:
    from immutables.map import MapMutation  # type: ignore[misc]


T = TypeVar("T")


class TransientSet(ITransientSet[T]):
    __slots__ = ("_inner",)

    def __init__(self, evolver: "MapMutation[T, T]") -> None:
        self._inner = evolver

    def __bool__(self):
        return True

    def __call__(self, key, default=None):
        if key in self:
            return key
        return default

    def __contains__(self, item):
        return item in self._inner

    def __eq__(self, other):
        return self is other

    def __len__(self):
        return len(self._inner)

    def cons_transient(self, *elems: T) -> "TransientSet":
        for elem in elems:
            self._inner.set(elem, elem)
        return self

    def disj_transient(self, *elems: T) -> "TransientSet":
        for elem in elems:
            try:
                del self._inner[elem]
            except KeyError:
                pass
        return self

    def to_persistent(self) -> "Set[T]":
        return Set(self._inner.finish())


class Set(
    IPersistentSet[T], IEvolveableCollection[TransientSet], ILispObject, IWithMeta,
):
    """Basilisp Set. Delegates internally to a immutables.Map object.

    Do not instantiate directly. Instead use the s() and set() factory
    methods below."""

    __slots__ = ("_inner", "_meta")

    def __init__(self, m: "_Map[T, T]", meta: Optional[IPersistentMap] = None) -> None:
        self._inner = m
        self._meta = meta

    @classmethod
    def from_iterable(
        cls, members: Optional[Iterable[T]], meta: Optional[IPersistentMap] = None
    ) -> "Set":
        return Set(_Map((m, m) for m in (members or ())), meta=meta)

    _from_iterable = from_iterable

    def __bool__(self):
        return True

    def __call__(self, key, default=None):
        if key in self:
            return key
        return default

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

    def to_transient(self) -> TransientSet:
        return TransientSet(self._inner.mutate())


def set(  # pylint:disable=redefined-builtin
    members: Iterable[T], meta: Optional[IPersistentMap] = None
) -> Set[T]:
    """Creates a new set."""
    return Set.from_iterable(members, meta=meta)


def s(*members: T, meta: Optional[IPersistentMap] = None) -> Set[T]:
    """Creates a new set from members."""
    return Set.from_iterable(members, meta=meta)
