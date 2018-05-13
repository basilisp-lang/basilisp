import pyrsistent
import wrapt
from basilisp.lang.util import lrepr


class List(wrapt.ObjectProxy):
    __slots__ = ('_self_meta', )

    def __init__(self, wrapped, meta=None):
        super(List, self).__init__(wrapped)
        self._self_meta = meta

    def __repr__(self):
        return "({list})".format(list=" ".join(map(lrepr, self)))

    @property
    def meta(self):
        return self._self_meta

    def with_meta(self, meta):
        new_meta = meta if self._self_meta is None else self._self_meta.update(
            meta)
        return list(self.__wrapped__, meta=new_meta)

    @property
    def rest(self):
        return List(self.__wrapped__.rest)

    def conj(self, elem):
        return self.cons(elem)

    def empty(self):
        return l()


def list(members, meta=None):
    """Creates a new list."""
    return List(pyrsistent.plist(iterable=members), meta=meta)


def l(*members, meta=None):
    """Creates a new list from members."""
    return List(pyrsistent.plist(iterable=members), meta=meta)
