import threading
from typing import Iterable, Optional

import basilisp.lang.map as lmap
from basilisp.lang.interfaces import IAssociative, ILispObject

_LOCK = threading.Lock()
_INTERN: lmap.Map[int, "Keyword"] = lmap.Map.empty()


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
        return self is other or (
            isinstance(other, Keyword)
            and (self._name, self._ns) == (other._name, other._ns)
        )

    def __hash__(self):
        return hash((self._name, self._ns))

    def __call__(self, m: IAssociative, default=None):
        try:
            return m.val_at(self, default)
        except AttributeError:
            return None

    def __reduce__(self):
        return keyword, (self._name, self._ns)


def complete(
    text: str, kw_cache: Optional[lmap.Map[int, Keyword]] = None,
) -> Iterable[str]:
    """Return an iterable of possible completions for the given text."""
    assert text.startswith(":")
    text = text[1:]

    if kw_cache is None:
        kw_cache = _INTERN

    if "/" in text:
        prefix, suffix = text.split("/", maxsplit=1)
        results = filter(
            lambda kw: (kw.ns is not None and kw.ns == prefix)
            and kw.name.startswith(suffix),
            kw_cache.values(),
        )
    else:
        results = filter(
            lambda kw: kw.name.startswith(text)
            or (kw.ns is not None and kw.ns.startswith(text)),
            kw_cache.values(),
        )

    return map(str, results)


def keyword(name: str, ns: Optional[str] = None) -> Keyword:
    """Create a new keyword with name and optional namespace.

    Keywords are stored in a global cache by their hash. If a keyword with the same
    hash already exists in the cache, that keyword will be returned. If no keyword
    exists in the global cache, one will be created, entered into the cache, and then
    returned."""
    global _INTERN

    h = hash((name, ns))
    with _LOCK:
        found = _INTERN.val_at(h)
        if found:
            return found
        kw = Keyword(name, ns=ns)
        _INTERN = _INTERN.assoc(h, kw)
        return kw
