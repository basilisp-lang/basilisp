from typing import Optional

from basilisp.lang.meta import Meta
from basilisp.lang.obj import LispObject, lrepr
from basilisp.lang.util import munge


class Symbol(LispObject, Meta):
    __slots__ = ("_name", "_ns", "_meta")

    def __init__(self, name: str, ns: Optional[str] = None, meta=None) -> None:
        self._name = name
        self._ns = ns
        self._meta = meta

    def _lrepr(self, **kwargs) -> str:
        print_meta = kwargs["print_meta"]

        if self._ns is not None:
            sym_repr = "{ns}/{name}".format(ns=self._ns, name=self._name)
        else:
            sym_repr = self._name

        if print_meta and self._meta:
            return f"^{lrepr(self._meta, **kwargs)} {sym_repr}"
        return sym_repr

    @property
    def name(self) -> str:
        return self._name

    @property
    def ns(self) -> Optional[str]:
        return self._ns

    @property
    def meta(self):
        return self._meta

    def with_meta(self, meta) -> "Symbol":
        new_meta = meta if self._meta is None else self._meta.update(meta)
        return Symbol(self._name, self._ns, meta=new_meta)

    def as_python_sym(self) -> str:
        if self.ns is not None:
            return f"{munge(self.ns)}.{munge(self.name)}"
        return f"{munge(self.name)}"

    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return False
        return self._ns == other._ns and self._name == other._name

    def __hash__(self):
        return hash(str(self))


def symbol(name: str, ns: Optional[str] = None, meta=None) -> Symbol:
    """Create a new symbol."""
    return Symbol(name, ns=ns, meta=meta)
