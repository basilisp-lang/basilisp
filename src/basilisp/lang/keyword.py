from typing import Iterable, Optional

from pyrsistent import PMap, pmap

import basilisp.lang.atom as atom
from basilisp.lang.interfaces import IAssociative, ILispObject

__INTERN: atom.Atom["PMap[int, Keyword]"] = atom.Atom(pmap())


class Keyword(ILispObject):
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
        return hash((self._name, self._ns))

    def __call__(self, m: IAssociative, default=None):
        try:
            return m.val_at(self, default)
        except AttributeError:
            return None


def complete(
    text: str, kw_cache: atom.Atom["PMap[int, Keyword]"] = __INTERN
) -> Iterable[str]:
    """Return an iterable of possible completions for the given text."""
    assert text.startswith(":")
    interns = kw_cache.deref()
    text = text[1:]

    if "/" in text:
        prefix, suffix = text.split("/", maxsplit=1)
        results = filter(
            lambda kw: (kw.ns is not None and kw.ns == prefix)
            and kw.name.startswith(suffix),
            interns.itervalues(),
        )
    else:
        results = filter(
            lambda kw: kw.name.startswith(text)
            or (kw.ns is not None and kw.ns.startswith(text)),
            interns.itervalues(),
        )

    return map(str, results)


def __get_or_create(
    kw_cache: "PMap[int, Keyword]", h: int, name: str, ns: Optional[str]
) -> PMap:
    """Private swap function used to either get the interned keyword
    instance from the input string."""
    if h in kw_cache:
        return kw_cache
    kw = Keyword(name, ns=ns)
    return kw_cache.set(h, kw)


def keyword(
    name: str,
    ns: Optional[str] = None,
    kw_cache: atom.Atom["PMap[int, Keyword]"] = __INTERN,
) -> Keyword:
    """Create a new keyword."""
    h = hash((name, ns))
    return kw_cache.swap(__get_or_create, h, name, ns)[h]
