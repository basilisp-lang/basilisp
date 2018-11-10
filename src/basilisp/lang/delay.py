from typing import Callable, TypeVar

import basilisp.lang.atom as atom
import basilisp.lang.map as lmap
from basilisp.lang.deref import Deref

T = TypeVar("T")


class Delay(Deref[T]):
    __slots__ = ("_state",)

    def __init__(self, f: Callable[[], T]) -> None:
        self._state = atom.Atom(  # pylint:disable=assigning-non-slot
            lmap.m(f=f, value=None, computed=False)
        )

    @staticmethod
    def __deref(m: lmap.Map):
        if m["computed"]:
            return m
        else:
            return m.assoc("value", m["f"](), "computed", True)

    def deref(self) -> T:
        return self._state.swap(Delay.__deref).value

    @property
    def is_realized(self) -> bool:
        return self._state.deref()["computed"]
