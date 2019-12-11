import threading
from typing import Callable, Generic, Optional, TypeVar

import atomos.atom

from basilisp.lang.interfaces import IDeref, IPersistentMap
from basilisp.lang.reference import ReferenceBase

T = TypeVar("T")


class Atom(IDeref[T], ReferenceBase, Generic[T]):
    __slots__ = ("_atom", "_meta", "_lock")

    # pylint: disable=assigning-non-slot
    def __init__(self, state: T) -> None:
        self._atom = atomos.atom.Atom(state)
        self._meta: Optional[IPersistentMap] = None
        self._lock = threading.Lock()

    def compare_and_set(self, old, new):
        return self._atom.compare_and_set(old, new)

    def deref(self) -> T:
        return self._atom.deref()

    def reset(self, v: T) -> T:
        return self._atom.reset(v)

    def swap(self, f: Callable[..., T], *args, **kwargs) -> T:
        return self._atom.swap(f, *args, **kwargs)
