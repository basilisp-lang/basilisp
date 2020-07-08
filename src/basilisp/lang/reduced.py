from typing import TypeVar

import attr

from basilisp.lang.interfaces import IDeref

T = TypeVar("T")


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Reduced(IDeref[T]):
    value: T

    def deref(self) -> T:
        return self.value
