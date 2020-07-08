import sys
from typing import Callable, Generic, TypeVar

import attr

T = TypeVar("T")

_USE_SLOTS = (sys.version_info >= (3, 7))


@attr.s(auto_attribs=True, slots=_USE_SLOTS, these={"value": attr.ib()})
class Volatile(Generic[T]):
    """A volatile reference container. Volatile references do not provide atomic
    semantics, but they may be useful as a mutable reference container in a
    single-threaded context."""

    value: T

    def reset(self, v: T) -> T:
        self.value = v
        return self.value

    def swap(self, f: Callable[..., T], *args, **kwargs) -> T:
        self.value = f(self.value, *args, **kwargs)
        return self.value
