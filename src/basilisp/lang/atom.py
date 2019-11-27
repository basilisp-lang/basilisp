import threading
from typing import Callable, Generic, Optional, TypeVar

import atomos.atom

from basilisp.lang.interfaces import IDeref, IPersistentMap, IReference

T = TypeVar("T")


class Atom(IDeref[T], IReference, Generic[T]):
    __slots__ = ("_atom", "_meta", "_lock")

    # pylint: disable=assigning-non-slot
    def __init__(self, state: T) -> None:
        self._atom = atomos.atom.Atom(state)
        self._meta: Optional[IPersistentMap] = None
        self._lock = threading.Lock()

    @property
    def meta(self) -> Optional["IPersistentMap"]:
        with self._lock:
            return self._meta

    def alter_meta(self, f: Callable[..., IPersistentMap], *args) -> IPersistentMap:
        with self._lock:
            self._meta = f(self._meta, *args)
            return self._meta

    def reset_meta(self, meta: IPersistentMap) -> IPersistentMap:
        with self._lock:
            self._meta = meta
            return meta

    def compare_and_set(self, old, new):
        return self._atom.compare_and_set(old, new)

    def deref(self) -> T:
        return self._atom.deref()

    def reset(self, v: T) -> T:
        return self._atom.reset(v)

    def swap(self, f: Callable[..., T], *args, **kwargs) -> T:
        return self._atom.swap(f, *args, **kwargs)
