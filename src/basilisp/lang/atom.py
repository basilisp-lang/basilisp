from typing import Generic, TypeVar, Callable

import atomos.atom

from basilisp.lang.deref import Deref

T = TypeVar("T")


class Atom(Deref, Generic[T]):
    __slots__ = ("_atom",)

    def __init__(self, state: T) -> None:
        self._atom = atomos.atom.Atom(state)

    def compare_and_set(self, old, new):
        return self._atom.compare_and_set(old, new)

    def deref(self) -> T:
        return self._atom.deref()

    def reset(self, v: T) -> T:
        return self._atom.reset(v)

    def swap(self, f: Callable[..., T], *args, **kwargs) -> T:
        return self._atom.swap(f, *args, **kwargs)
