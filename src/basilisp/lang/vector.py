from pyrsistent import PVector, pvector

from basilisp.lang.associative import Associative
from basilisp.lang.collection import Collection
from basilisp.lang.meta import Meta
from basilisp.lang.obj import LispObject
from basilisp.lang.seq import Seqable, Seq, sequence


class Vector(Associative, Collection, LispObject, Meta, Seqable):
    """Basilisp Vector. Delegates internally to a pyrsistent.PVector object.
    Do not instantiate directly. Instead use the v() and vec() factory
    methods below."""

    __slots__ = ("_inner", "_meta")

    def __init__(self, wrapped: PVector, meta=None) -> None:
        self._inner = wrapped
        self._meta = meta

    def __eq__(self, other):
        return self._inner == other

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Vector(self._inner[item])
        return self._inner[item]

    def __hash__(self):
        return hash(self._inner)

    def __iter__(self):
        yield from self._inner

    def __len__(self):
        return len(self._inner)

    def _lrepr(self, **kwargs) -> str:
        return LispObject.seq_lrepr(self._inner, "[", "]", meta=self._meta, **kwargs)

    @property
    def meta(self):
        return self._meta

    def with_meta(self, meta) -> "Vector":
        new_meta = meta if self._meta is None else self._meta.update(meta)
        return vector(self._inner, meta=new_meta)

    def cons(self, *elems) -> "Vector":
        e = self._inner.evolver()
        for elem in elems:
            e.append(elem)
        return Vector(e.persistent(), meta=self.meta)

    def assoc(self, *kvs):
        return Vector(self._inner.mset(*kvs))

    def contains(self, k):
        return 0 <= k < len(self._inner)

    def entry(self, k, default=None):
        try:
            return self._inner[k]
        except IndexError:
            return default

    @staticmethod
    def empty() -> "Vector":
        return v()

    def seq(self) -> Seq:
        return sequence(self)


def vector(members, meta=None) -> Vector:
    """Creates a new vector."""
    return Vector(pvector(members), meta=meta)


def v(*members, meta=None) -> Vector:
    """Creates a new vector from members."""
    return Vector(pvector(members), meta=meta)
