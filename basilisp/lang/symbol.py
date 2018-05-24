from typing import Optional

from basilisp.lang.map import Map
from basilisp.lang.meta import Meta


class Symbol(Meta):
    __slots__ = ('_name', '_ns', '_meta')

    def __init__(self, name: str, ns: Optional[str] = None, meta=None) -> None:
        self._name = name
        self._ns = ns
        self._meta = meta

    @property
    def name(self) -> str:
        return self._name

    @property
    def ns(self) -> Optional[str]:
        return self._ns

    @property
    def meta(self) -> Optional[Map]:
        return self._meta

    def with_meta(self, meta: Map) -> "Symbol":
        new_meta = meta if self._meta is None else self._meta.update(meta)
        return Symbol(self._name, self._ns, meta=new_meta)

    def __str__(self):
        if self._ns is not None:
            return "{ns}/{name}".format(ns=self._ns, name=self._name)
        return "{name}".format(name=self._name)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return False
        return self._ns == other._ns and self._name == other._name

    def __hash__(self):
        return hash(str(self))


def symbol(name: str, ns: Optional[str] = None, meta=None) -> Symbol:
    """Create a new symbol."""
    return Symbol(name, ns=ns, meta=meta)
