from pyrsistent import PVector, pvector

from basilisp.lang.meta import Meta
from basilisp.lang.util import lrepr


class Vector(Meta):
    __slots__ = ('_inner', '_meta',)

    def __init__(self, wrapped: PVector, meta=None) -> None:
        self._inner = wrapped
        self._meta = meta

    def __repr__(self):
        return "[{vec}]".format(vec=" ".join(map(lrepr, self._inner)))

    def __eq__(self, other):
        return self._inner == other

    def __getitem__(self, item):
        return self._inner[item]

    def __hash__(self):
        return hash(self._inner)

    def __iter__(self):
        yield from self._inner

    def __len__(self):
        return len(self._inner)

    @property
    def meta(self):
        return self._meta

    def with_meta(self, meta) -> "Vector":
        new_meta = meta if self._meta is None else self._meta.update(
            meta)
        return vector(self._inner, meta=new_meta)

    def conj(self, elem) -> "Vector":
        return Vector(self._inner.append(elem), meta=self.meta)

    @staticmethod
    def empty() -> "Vector":
        return v()


def vector(members, meta=None) -> Vector:
    """Creates a new vector."""
    return Vector(pvector(members), meta=meta)


def v(*members, meta=None) -> Vector:
    """Creates a new vector from members."""
    return Vector(pvector(members), meta=meta)
