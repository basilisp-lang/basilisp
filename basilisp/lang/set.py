import pyrsistent
from wrapt import ObjectProxy

from basilisp.lang.meta import Meta
from basilisp.lang.util import lrepr


class Set(ObjectProxy, Meta):
    __slots__ = ('_self_meta', )

    def __init__(self, wrapped, meta=None):
        super(Set, self).__init__(wrapped)
        self._self_meta = meta

    def __repr__(self):
        return "#{{{set}}}".format(set=" ".join(map(lrepr, self)))

    @property
    def meta(self):
        return self._self_meta

    def with_meta(self, meta):
        new_meta = meta if self._self_meta is None else self._self_meta.update(
            meta)
        return set(self.__wrapped__, meta=new_meta)

    def conj(self, elem):
        return self.add(elem)

    def empty(self):
        return s()


def set(members, meta=None):
    """Creates a new set."""
    return Set(pyrsistent.pset(members), meta=meta)


def s(*members, meta=None):
    """Creates a new set from members."""
    return Set(pyrsistent.pset(members), meta=meta)
