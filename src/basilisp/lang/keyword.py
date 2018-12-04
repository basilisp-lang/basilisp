from typing import Optional

from pyrsistent import pmap, PMap

import basilisp.lang.associative as lassoc
import basilisp.lang.atom as atom
from basilisp.lang.obj import LispObject

__INTERN = atom.Atom(pmap())


class Keyword(LispObject):
    __slots__ = ("_name", "_ns")

    def __init__(self, name: str, ns: Optional[str] = None) -> None:
        self._name = name
        self._ns = ns

    @property
    def name(self) -> str:
        return self._name

    @property
    def ns(self) -> Optional[str]:
        return self._ns

    def _lrepr(self, **kwargs) -> str:
        if self._ns is not None:
            return ":{ns}/{name}".format(ns=self._ns, name=self._name)
        return ":{name}".format(name=self._name)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(str(self))

    def __call__(self, m: lassoc.Associative, default=None):
        try:
            return m.entry(self, default)
        except AttributeError:
            return None


def __get_or_create(kw_cache: PMap, h: int, name: str, ns: Optional[str]) -> PMap:
    """Private swap function used to either get the interned keyword
    instance from the input string."""
    if h in kw_cache:
        return kw_cache
    kw = Keyword(name, ns=ns)
    return kw_cache.set(h, kw)


def keyword(
    name: str, ns: Optional[str] = None, kw_cache: atom.Atom = __INTERN
) -> Keyword:
    """Create a new keyword."""
    h = hash((name, ns))
    return kw_cache.swap(__get_or_create, h, name, ns)[h]
