import pyrsistent
import wrapt
from apylisp.lang.util import lrepr


class Vector(wrapt.ObjectProxy):
    __slots__ = ('_self_meta', )

    def __init__(self, wrapped, meta=None):
        super(Vector, self).__init__(wrapped)
        self._self_meta = meta

    def __repr__(self):
        return "[{vec}]".format(vec=" ".join(map(lrepr, self)))

    @property
    def meta(self):
        return self._self_meta

    def with_meta(self, meta):
        new_meta = meta if self._self_meta is None else self._self_meta.update(
            meta)
        return vector(self.__wrapped__, meta=new_meta)

    def conj(self, elem):
        return Vector(self.append(elem))

    def empty(self):
        return v()


def vector(members, meta=None):
    """Creates a new vector."""
    return Vector(pyrsistent.pvector(members), meta=meta)


def v(*members, meta=None):
    """Creates a new vector from members."""
    return Vector(pyrsistent.pvector(members), meta=meta)
