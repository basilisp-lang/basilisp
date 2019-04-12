from typing import Callable, Generic, TypeVar

import atomos.atom

from basilisp.lang.interfaces import IDeref

T = TypeVar("T")


class Atom(IDeref[T], Generic[T]):
    __slots__ = ("_atom",)

    def __init__(self, state: T) -> None:
        self._atom = atomos.atom.Atom(state)  # pylint: disable=assigning-non-slot

    def compare_and_set(self, old, new):
        return self._atom.compare_and_set(old, new)

    def deref(self) -> T:
        return self._atom.deref()

    def reset(self, v: T) -> T:
        return self._atom.reset(v)

    def swap(self, f: Callable[..., T], *args, **kwargs) -> T:
        return self._atom.swap(f, *args, **kwargs)
