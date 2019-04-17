from typing import Callable, Generic, Optional, TypeVar

import attr

import basilisp.lang.atom as atom
from basilisp.lang.interfaces import IDeref

T = TypeVar("T")


# Use attrs `these` for now as there is an open bug around slotted
# generic classes: https://github.com/python-attrs/attrs/issues/313
@attr.s(
    auto_attribs=True,
    frozen=True,
    these={"f": attr.ib(), "value": attr.ib(), "computed": attr.ib(default=False)},
)
class _DelayState(Generic[T]):
    f: Callable[[], T]
    value: Optional[T]
    computed: bool


class Delay(IDeref[T]):
    __slots__ = ("_state",)

    def __init__(self, f: Callable[[], T]) -> None:
        self._state = atom.Atom(  # pylint:disable=assigning-non-slot
            _DelayState(f=f, value=None, computed=False)
        )

    @staticmethod
    def __deref(state: _DelayState) -> _DelayState:
        if state.computed:
            return state
        else:
            return _DelayState(f=state.f, value=state.f(), computed=True)

    def deref(self) -> Optional[T]:
        return self._state.swap(Delay.__deref).value

    @property
    def is_realized(self) -> bool:
        return self._state.deref().computed
