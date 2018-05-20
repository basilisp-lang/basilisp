from pyrsistent import plist, PList
from wrapt import ObjectProxy

from basilisp.lang.meta import Meta
from basilisp.lang.util import lrepr


class List(ObjectProxy, Meta):
    __slots__ = ('_self_meta', )

    def __init__(self, wrapped: PList, meta=None) -> None:
        super(List, self).__init__(wrapped)
        self._self_meta = meta

    def __repr__(self):
        return "({list})".format(list=" ".join(map(lrepr, self)))

    @property  # type: ignore
    def meta(self):
        return self._self_meta

    def with_meta(self, meta) -> "List":
        new_meta = meta if self._self_meta is None else self._self_meta.update(
            meta)
        return list(self.__wrapped__, meta=new_meta)

    @property
    def rest(self) -> "List":
        return List(self.__wrapped__.rest)

    def conj(self, elem) -> "List":
        return List(self.cons(elem))

    @staticmethod
    def empty() -> "List":
        return l()


def list(members, meta=None) -> List:
    """Creates a new list."""
    return List(plist(iterable=members), meta=meta)


def l(*members, meta=None) -> List:
    """Creates a new list from members."""
    return List(plist(iterable=members), meta=meta)
