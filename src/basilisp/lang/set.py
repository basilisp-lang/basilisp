from collections.abc import Iterable
from collections.abc import Set as _PySet
from typing import AbstractSet, Optional, TypeVar

from immutables import Map as _Map
from immutables import MapMutation
from typing_extensions import Unpack

from basilisp.lang.interfaces import (
    IEvolveableCollection,
    ILispObject,
    IPersistentMap,
    IPersistentSet,
    ISeq,
    ITransientSet,
    IWithMeta,
)
from basilisp.lang.obj import PrintSettings
from basilisp.lang.obj import seq_lrepr as _seq_lrepr
from basilisp.lang.seq import sequence

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

    def to_persistent(self) -> "PersistentSet[T]":
        return PersistentSet(self._inner.finish())


class PersistentSet(
    IPersistentSet[T],
    IEvolveableCollection[TransientSet],
    ILispObject,
    IWithMeta,
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
    ) -> "PersistentSet":
        return PersistentSet(_Map((m, m) for m in (members or ())), meta=meta)

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
        return self._hash()

    def __iter__(self):
        yield from self._inner.keys()

    def __len__(self):
        return len(self._inner)

    def _lrepr(self, **kwargs: Unpack[PrintSettings]):
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

    def with_meta(self, meta: Optional[IPersistentMap]) -> "PersistentSet[T]":
        return set(self._inner, meta=meta)

    def cons(self, *elems: T) -> "PersistentSet[T]":  # type: ignore[return]
        with self._inner.mutate() as m:
            for elem in elems:
                m.set(elem, elem)
            return PersistentSet(m.finish(), meta=self.meta)

    def disj(self, *elems: T) -> "PersistentSet[T]":  # type: ignore[return]
        with self._inner.mutate() as m:
            for elem in elems:
                try:
                    del m[elem]
                except KeyError:
                    pass
            return PersistentSet(m.finish(), meta=self.meta)

    def empty(self) -> "PersistentSet":
        return EMPTY.with_meta(self._meta)

    def seq(self) -> Optional[ISeq[T]]:
        if len(self._inner) == 0:
            return None
        return sequence(self)

    def to_transient(self) -> TransientSet:
        return TransientSet(self._inner.mutate())


EMPTY = PersistentSet.from_iterable(())


def set(  # pylint:disable=redefined-builtin
    members: Iterable[T], meta: Optional[IPersistentMap] = None
) -> PersistentSet[T]:
    """Creates a new set."""
    return PersistentSet.from_iterable(members, meta=meta)


def s(*members: T, meta: Optional[IPersistentMap] = None) -> PersistentSet[T]:
    """Creates a new set from members."""
    return PersistentSet.from_iterable(members, meta=meta)
