from typing import Optional

from pyrsistent import PSet, pset

from basilisp.lang.collection import Collection
from basilisp.lang.map import Map
from basilisp.lang.meta import Meta
from basilisp.lang.obj import LispObject
from basilisp.lang.seq import Seqable, Seq, sequence


class Set(Collection, LispObject, Meta, Seqable):
    """Basilisp Set. Delegates internally to a pyrsistent.PSet object.

    Do not instantiate directly. Instead use the s() and set() factory
    methods below."""

    __slots__ = ("_inner", "_meta")

    def __init__(self, wrapped: PSet, meta=None) -> None:
        self._inner = wrapped
        self._meta = meta

    def __call__(self, key, default=None):
        if key in self:
            return key
        return None

    def __contains__(self, item):
        return item in self._inner

    def __eq__(self, other):
        return self._inner == other

    def __ge__(self, other):
        return self._inner >= other

    def __gt__(self, other):
        return self._inner > other

    def __le__(self, other):
        return self._inner <= other

    def __lt__(self, other):
        return self._inner < other

    def __hash__(self):
        return hash(self._inner)

    def __iter__(self):
        yield from self._inner

    def __len__(self):
        return len(self._inner)

    def _lrepr(self, **kwargs):
        return LispObject.seq_lrepr(self._inner, "#{", "}", meta=self._meta, **kwargs)

    def difference(self, *others):
        e = self._inner
        for other in others:
            e = e.difference(other)
        return Set(e)

    def intersection(self, *others):
        e = self._inner
        for other in others:
            e = e.intersection(other)
        return Set(e)

    def symmetric_difference(self, *others):
        e = self._inner
        for other in others:
            e = e.symmetric_difference(other)
        return Set(e)

    def union(self, *others):
        e = self._inner
        for other in others:
            e = e.union(other)
        return Set(e)

    def isdisjoint(self, s):
        return self._inner.isdisjoint(s)

    def issuperset(self, other):
        return self._inner >= other

    def issubset(self, other):
        return self._inner <= other

    @property
    def meta(self) -> Optional[Map]:
        return self._meta

    def with_meta(self, meta: Map) -> "Set":
        new_meta = meta if self._meta is None else self._meta.update(meta)
        return set(self._inner, meta=new_meta)

    def cons(self, *elems) -> "Set":
        e = self._inner.evolver()
        for elem in elems:
            e.add(elem)
        return Set(e.persistent(), meta=self.meta)

    def disj(self, *elems) -> "Set":
        e = self._inner.evolver()
        for elem in elems:
            try:
                e.remove(elem)
            except KeyError:
                pass
        return Set(e.persistent(), meta=self.meta)

    @staticmethod
    def empty() -> "Set":
        return s()

    def seq(self) -> Seq:
        return sequence(self)


def set(members, meta=None) -> Set:  # pylint:disable=redefined-builtin
    """Creates a new set."""
    return Set(pset(members), meta=meta)


def s(*members, meta=None) -> Set:
    """Creates a new set from members."""
    return Set(pset(members), meta=meta)
