import threading
from collections.abc import Iterable
from functools import total_ordering
from typing import Optional, Union

from typing_extensions import Unpack

from basilisp.lang import map as lmap
from basilisp.lang.interfaces import (
    IAssociative,
    ILispObject,
    INamed,
    IPersistentMap,
    IPersistentSet,
)
from basilisp.lang.obj import PrintSettings

_LOCK = threading.Lock()
_INTERN: IPersistentMap[int, "Keyword"] = lmap.EMPTY


@total_ordering
class Keyword(ILispObject, INamed):
    __slots__ = ("_name", "_ns", "_hash")

    def __init__(self, name: str, ns: Optional[str] = None) -> None:
        self._name = name
        self._ns = ns
        self._hash = hash_kw(name, ns)

    @property
    def name(self) -> str:
        return self._name

    @property
    def ns(self) -> Optional[str]:
        return self._ns

    @classmethod
    def with_name(cls, name: str, ns: Optional[str] = None) -> "Keyword":
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

    def __call__(self, m: Union[IAssociative, IPersistentSet], default=None):
        if isinstance(m, IPersistentSet):
            return self if self in m else default
        try:
            return m.val_at(self, default)
        except (AttributeError, TypeError):
            return default

    def __reduce__(self):
        return keyword_from_hash, (self._hash, self._name, self._ns)


def complete(
    text: str,
    kw_cache: Optional[IPersistentMap[int, Keyword]] = None,
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


def hash_kw(name: str, ns: Optional[str] = None) -> int:
    """Return the hash of a potential Keyword instance by its name and namespace."""
    return hash((name, ns))


def keyword_from_hash(kw_hash: int, name: str, ns: Optional[str] = None) -> Keyword:
    """Return the interned keyword with the hash `kw_hash` or create and intern a new
    keyword with name `name` and optional namespace `ns`.

    Keywords are stored in a global cache by their hash. If a keyword with the same
    hash already exists in the cache, that keyword will be returned. If no keyword
    exists in the global cache, one will be created, entered into the cache, and then
    returned.

    This function is an optimization primarily meant for the compiler. Keyword hashes
    are pre-computed during compilation so repeated lookups for the same keyword do not
    require recomputing the hash. In some brief testing, this yielded significant
    performance improvements when creating the same keyword repeatedly."""
    global _INTERN

    with _LOCK:
        found = _INTERN.val_at(kw_hash)
        if found:
            return found
        kw = Keyword(name, ns=ns)
        _INTERN = _INTERN.assoc(kw_hash, kw)
        return kw


def find_keyword(name: str, ns: Optional[str] = None) -> Optional[Keyword]:
    """Return the already-interned keyword named by `name` and `ns`, if one exists.
    If the keyword with that name is not interned, return None."""
    with _LOCK:
        return _INTERN.val_at(hash_kw(name, ns))


def keyword(name: str, ns: Optional[str] = None) -> Keyword:
    """Return a keyword with name `name` and optional namespace `ns`.

    Keyword instances are interned, so an existing object may be returned if one
    with the same name and namespace are already interned."""
    return keyword_from_hash(hash_kw(name, ns), name, ns=ns)
