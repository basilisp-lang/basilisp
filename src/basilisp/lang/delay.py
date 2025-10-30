from collections.abc import Callable
from typing import Generic, TypeVar

import attr

from basilisp.lang import atom as atom
from basilisp.lang.interfaces import IDeref, IPending

T = TypeVar("T")


@attr.frozen
class _DelayState(Generic[T]):
    f: Callable[[], T]
    value: T | None
    computed: bool = False


class Delay(IDeref[T], IPending):
    __slots__ = ("_state",)

    def __init__(self, f: Callable[[], T]) -> None:
        self._state = atom.Atom(_DelayState(f=f, value=None, computed=False))

    @staticmethod
    def __deref(state: _DelayState) -> _DelayState:
        if state.computed:
            return state
        else:
            return _DelayState(f=state.f, value=state.f(), computed=True)

    def deref(self) -> T | None:
        return self._state.swap(self.__deref).value

    @property
    def is_realized(self) -> bool:
        return self._state.deref().computed
