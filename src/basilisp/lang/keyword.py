import threading
from collections.abc import Iterable, Mapping
from functools import total_ordering

from typing_extensions import Unpack

from basilisp.lang.interfaces import (
    IAssociative,
    ILispObject,
    INamed,
    IPersistentSet,
)
from basilisp.lang.obj import PrintSettings

_LOCK = threading.Lock()
_INTERN: dict[tuple[str, str | None], "Keyword"] = {}


@total_ordering
class Keyword(ILispObject, INamed):
    __slots__ = ("_hash", "_name", "_ns")

    def __init__(self, name: str, ns: str | None = None) -> None:
        self._name = name
        self._ns = ns
        self._hash = hash_kw(name, ns)

    @property
    def name(self) -> str:
        return self._name

    @property
    def ns(self) -> str | None:
        return self._ns

    @classmethod
    def with_name(cls, name: str, ns: str | None = None) -> "Keyword":
        return keyword(name, ns=ns)

    def _lrepr(self, **kwargs: Unpack[PrintSettings]) -> str:
        if self._ns is not None:
            return f":{self._ns}/{self._name}"
        return f":{self._name}"

    def __eq__(self, other):
        return self is other or (
            isinstance(other, Keyword)
            and (self._name, self._ns) == (other._name, other._ns)
        )

    def __hash__(self):
        return self._hash

    def __lt__(self, other):
        if other is None:  # pragma: no cover
            return False
        if not isinstance(other, Keyword):
            return NotImplemented
        if self._ns is None and other._ns is None:
            return self._name < other._name
        if self._ns is None:
            return True
        if other._ns is None:
            return False
        return self._ns < other._ns or self._name < other._name

    def __call__(self, m: IAssociative | IPersistentSet, default=None):
        if isinstance(m, IPersistentSet):
            return self if self in m else default
        try:
            return m.val_at(self, default)
        except (AttributeError, TypeError):
            return default

    def __reduce__(self):
        return keyword, (self._name, self._ns)


def complete(
    text: str,
    kw_cache: Mapping[tuple[str, str | None], Keyword] | None = None,
) -> Iterable[str]:
    """Return an iterable of possible completions for the given text."""
    assert text.startswith(":")
    text = text[1:]

    if kw_cache is None:
        # Snapshot the interns under lock so we can use it without holding
        # the lock.
        with _LOCK:
            kw_cache = dict(_INTERN)

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


def hash_kw(name: str, ns: str | None = None) -> int:
    """Return the hash of a potential Keyword instance by its name and namespace."""
    return hash((name, ns))


def find_keyword(name: str, ns: str | None = None) -> Keyword | None:
    """Return the already-interned keyword named by `name` and `ns`, if one exists.
    If the keyword with that name is not interned, return None."""
    with _LOCK:
        return _INTERN.get((name, ns))


def keyword(name: str, ns: str | None = None) -> Keyword:
    """Return a keyword with name `name` and optional namespace `ns`.

    Keyword instances are interned, so an existing object may be returned if one
    with the same name and namespace are already interned."""
    with _LOCK:
        found = _INTERN.get((name, ns))
        if found:
            return found
        kw = Keyword(name, ns=ns)
        _INTERN[name, ns] = kw
        return kw
