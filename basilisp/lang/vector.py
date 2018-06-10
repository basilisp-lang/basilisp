from pyrsistent import PVector, pvector
from wrapt import ObjectProxy

from basilisp.lang.meta import Meta
from basilisp.lang.util import lrepr


class Vector(ObjectProxy, Meta):
    __slots__ = ('_self_meta', )

    def __init__(self, wrapped: PVector, meta=None) -> None:
        super(Vector, self).__init__(wrapped)
        self._self_meta = meta

    def __repr__(self):
        return "[{vec}]".format(vec=" ".join(map(lrepr, self)))

    @property
    def meta(self):
        return self._self_meta

    def with_meta(self, meta) -> "Vector":
        new_meta = meta if self._self_meta is None else self._self_meta.update(
            meta)
        return vector(self.__wrapped__, meta=new_meta)

    def conj(self, elem) -> "Vector":
        return Vector(self.append(elem), meta=self.meta)

    @staticmethod
    def empty() -> "Vector":
        return v()


def vector(members, meta=None) -> Vector:
    """Creates a new vector."""
    return Vector(pvector(members), meta=meta)


def v(*members, meta=None) -> Vector:
    """Creates a new vector from members."""
    return Vector(pvector(members), meta=meta)
