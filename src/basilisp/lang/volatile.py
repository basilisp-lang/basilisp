from typing import Callable, Optional, TypeVar

import attr
from typing_extensions import Concatenate, ParamSpec

from basilisp.lang.interfaces import IDeref

T = TypeVar("T")
P = ParamSpec("P")


@attr.define
class Volatile(IDeref[T]):
    """A volatile reference container. Volatile references do not provide atomic
    semantics, but they may be useful as a mutable reference container in a
    single-threaded context."""

    value: T

    def deref(self) -> Optional[T]:
        return self.value

    def reset(self, v: T) -> T:
        self.value = v
        return self.value

    def swap(
        self, f: Callable[Concatenate[T, P], T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        self.value = f(self.value, *args, **kwargs)
        return self.value
