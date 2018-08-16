from pyrsistent import plist, PList

from basilisp.lang.meta import Meta
from basilisp.lang.seq import Seq
from basilisp.lang.util import lrepr


class List(Meta, Seq):
    """Basilisp List. Delegates internally to a pyrsistent.PList object.

    Do not instantiate directly. Instead use the l() and list() factory
    methods below."""
    __slots__ = ('_inner', '_meta',)

    def __init__(self, wrapped: PList, meta=None) -> None:
        self._inner = wrapped
        self._meta = meta

    def __repr__(self):
        return "({list})".format(list=" ".join(map(lrepr, self._inner)))

    def __eq__(self, other):
        return self._inner == other

    def __getitem__(self, item):
        if isinstance(item, slice):
            return List(self._inner[item])
        return self._inner[item]

    def __hash__(self):
        return hash(self._inner)

    def __len__(self):
        return len(self._inner)

    @property
    def meta(self):
        return self._meta

    def with_meta(self, meta) -> "List":
        new_meta = meta if self._meta is None else self._meta.update(
            meta)
        return list(self._inner, meta=new_meta)

    @property
    def first(self):
        try:
            return self._inner.first
        except AttributeError:
            return None

    @property
    def rest(self) -> "List":
        return List(self._inner.rest)

    def conj(self, elem) -> "List":
        return List(self._inner.cons(elem))

    def cons(self, elem) -> "List":
        return List(self._inner.cons(elem))

    @staticmethod
    def empty(meta=None) -> "List":
        return l(meta=meta)


def list(members, meta=None) -> List:  # pylint:disable=redefined-builtin
    """Creates a new list."""
    return List(plist(iterable=members), meta=meta)


def l(*members, meta=None) -> List:
    """Creates a new list from members."""
    return List(plist(iterable=members), meta=meta)
