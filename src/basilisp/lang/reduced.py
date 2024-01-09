from typing import TypeVar

import attr

from basilisp.lang.interfaces import IDeref

T = TypeVar("T")


@attr.frozen
class Reduced(IDeref[T]):
    value: T

    def deref(self) -> T:
        return self.value
