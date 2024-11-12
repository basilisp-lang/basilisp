from typing import Optional, TypeVar, Union

from typing_extensions import Unpack

from basilisp.lang.interfaces import ILispObject, ILookup
from basilisp.lang.keyword import keyword
from basilisp.lang.obj import PrintSettings, lrepr
from basilisp.lang.symbol import Symbol

K = TypeVar("K")
V = TypeVar("V")
T = Union[None, V, Symbol]

_TAG_KW = keyword("tag")
_FORM_KW = keyword("form")


class TaggedLiteral(ILispObject, ILookup[K, T]):
    """Basilisp TaggedLiteral. https://clojure.org/reference/reader#tagged_literals"""

    __slots__ = ("_tag", "_form", "_hash")

    def __init__(self, tag: Symbol, form) -> None:
        self._tag = tag
        self._form = form
        self._hash: Optional[int] = None

    @property
    def tag(self) -> Symbol:
        return self._tag

    @property
    def form(self):
        return self._form

    def __bool__(self):
        return True

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, TaggedLiteral):
            return NotImplemented
        return self._tag == other._tag and self._form == other._form

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self._tag, self._form))
        return self._hash

    def __getitem__(self, item):
        return self.val_at(item)

    def val_at(self, k: K, default: Optional[V] = None) -> T:
        if k == _TAG_KW:
            return self._tag
        elif k == _FORM_KW:
            return self._form
        else:
            return default

    def _lrepr(self, **kwargs: Unpack[PrintSettings]) -> str:
        return f"#{self._tag} {lrepr(self._form, **kwargs)}"


def tagged_literal(tag: Symbol, form):
    """Construct a data representation of a tagged literal from a
    tag symbol and a form."""
    if not isinstance(tag, Symbol):
        raise TypeError(f"tag must be a Symbol, not '{type(tag)}'")
    return TaggedLiteral(tag, form)
