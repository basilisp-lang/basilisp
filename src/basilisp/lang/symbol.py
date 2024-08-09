from functools import total_ordering
from typing import Optional, Union

from typing_extensions import Unpack

from basilisp.lang.interfaces import (
    IAssociative,
    ILispObject,
    INamed,
    IPersistentMap,
    IPersistentSet,
    IWithMeta,
)
from basilisp.lang.obj import PrintSettings, lrepr
from basilisp.lang.util import munge


@total_ordering
class Symbol(ILispObject, INamed, IWithMeta):
    __slots__ = ("_name", "_ns", "_meta", "_hash")

    def __init__(
        self, name: str, ns: Optional[str] = None, meta: Optional[IPersistentMap] = None
    ) -> None:
        self._name = name
        self._ns = ns
        self._meta = meta
        self._hash = hash((ns, name))

    def _lrepr(self, **kwargs: Unpack[PrintSettings]) -> str:
        print_meta = kwargs["print_meta"]

        if self._ns is not None:
            sym_repr = f"{self._ns}/{self._name}"
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

    @classmethod
    def with_name(cls, name: str, ns: Optional[str] = None) -> "Symbol":
        return Symbol(name, ns=ns)

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "Symbol":
        return Symbol(self._name, self._ns, meta=meta)

    def as_python_sym(self) -> str:
        if self.ns is not None:
            return f"{munge(self.ns)}.{munge(self.name)}"
        return munge(self.name)

    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return False
        return self._ns == other._ns and self._name == other._name

    def __hash__(self):
        return self._hash

    def __lt__(self, other):
        if other is None:  # pragma: no cover
            return False
        if not isinstance(other, Symbol):
            return NotImplemented
        if self._ns is None and other._ns is None:
            return self._name < other._name
        if self._ns is None:
            return True
        if other._ns is None:
            return False
        return self._ns < other._ns or self._name < other._name

    def __call__(self, m: Union[IAssociative, IPersistentSet], default=None):
        if isinstance(m, IPersistentSet):
            return self if self in m else default
        try:
            return m.val_at(self, default)
        except (AttributeError, TypeError):
            return None


def symbol(
    name: str, ns: Optional[str] = None, meta: Optional[IPersistentMap] = None
) -> Symbol:
    """Create a new symbol."""
    return Symbol(name, ns=ns, meta=meta)
