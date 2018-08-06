from typing import Optional

from pyrsistent import PSet, pset
from wrapt import ObjectProxy

from basilisp.lang.map import Map
from basilisp.lang.meta import Meta
from basilisp.lang.util import lrepr


class Set(ObjectProxy, Meta):
    __slots__ = ('_self_meta', )

    def __init__(self, wrapped: PSet, meta=None) -> None:
        super(Set, self).__init__(wrapped)
        self._self_meta = meta

    def __repr__(self):
        return "#{{{set}}}".format(set=" ".join(map(lrepr, self)))

    @property
    def meta(self) -> Optional[Map]:
        return self._self_meta

    def with_meta(self, meta: Map) -> "Set":
        new_meta = meta if self._self_meta is None else self._self_meta.update(
            meta)
        return set(self.__wrapped__, meta=new_meta)

    def conj(self, elem) -> "Set":
        return Set(self.add(elem), meta=self.meta)

    @staticmethod
    def empty() -> "Set":
        return s()


def set(members, meta=None) -> Set:
    """Creates a new set."""
    return Set(pset(members), meta=meta)


def s(*members, meta=None) -> Set:
    """Creates a new set from members."""
    return Set(pset(members), meta=meta)
