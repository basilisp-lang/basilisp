from typing import Callable, Any

import basilisp.lang.atom as atom
import basilisp.lang.map as lmap


class Delay:
    __slots__ = ('_state',)

    def __init__(self, f: Callable[[], Any]) -> None:
        self._state = atom.Atom(lmap.m(f=f, value=None, computed=False))

    def deref(self) -> Any:
        def __deref(m: lmap.Map):
            if m["computed"]:
                return m
            else:
                return m.assoc("value", m["f"](), "computed", True)

        return self._state.swap(__deref).value
