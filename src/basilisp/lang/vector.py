from pyrsistent import PVector, pvector  # noqa # pylint: disable=unused-import

from basilisp.lang.interfaces import (
    ILispObject,
    IMeta,
    IPersistentMap,
    IPersistentVector,
    ISeq,
)
from basilisp.lang.obj import seq_lrepr as _seq_lrepr
from basilisp.lang.seq import sequence
from typing import Iterable, Optional, TypeVar

T = TypeVar("T")


class Vector(ILispObject, IMeta, IPersistentVector[T]):
    """Basilisp Vector. Delegates internally to a pyrsistent.PVector object.
    Do not instantiate directly. Instead use the v() and vec() factory
    methods below."""

    __slots__ = ("_inner", "_meta")

    def __init__(self, wrapped: "PVector[T]", meta=None) -> None:
        self._inner = wrapped
        self._meta = meta

    def __contains__(self, item):
        return item in self._inner

    def __eq__(self, other):
        if hasattr(other, "__len__"):
            return self._inner == other
        return all(e1 == e2 for e1, e2 in zip(self._inner, other))

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
        return _seq_lrepr(self._inner, "[", "]", meta=self._meta, **kwargs)

    @property
    def meta(self):
        return self._meta

    def with_meta(self, meta) -> "Vector[T]":
        new_meta = meta if self._meta is None else self._meta.update(meta)
        return vector(self._inner, meta=new_meta)

    def cons(self, *elems: T) -> "Vector[T]":  # type: ignore
        e = self._inner.evolver()
        for elem in elems:
            e.append(elem)
        return Vector(e.persistent(), meta=self.meta)

    def assoc(self, *kvs: T) -> "Vector[T]":
        return Vector(self._inner.mset(*kvs))  # type: ignore

    def contains(self, k):
        return 0 <= k < len(self._inner)

    def entry(self, k, default=None):
        try:
            return self._inner[k]
        except IndexError:
            return default

    @staticmethod
    def empty() -> "Vector[T]":
        return v()

    def seq(self) -> ISeq[T]:  # type: ignore
        return sequence(self)

    def peek(self) -> Optional[T]:
        if len(self) == 0:
            return None
        return self[-1]

    def pop(self) -> "Vector[T]":
        if len(self) == 0:
            raise IndexError("Cannot pop an empty vector")
        return self[:-1]


def vector(members: Iterable[T], meta: Optional[IPersistentMap] = None) -> Vector[T]:
    """Creates a new vector."""
    return Vector(pvector(members), meta=meta)


def v(*members: T, meta: Optional[IPersistentMap] = None) -> Vector[T]:
    """Creates a new vector from members."""
    return Vector(pvector(members), meta=meta)
