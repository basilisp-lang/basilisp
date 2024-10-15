import contextlib
import time
from collections.abc import Iterable, Sequence
from typing import Callable, Generic, Optional, TypeVar


@contextlib.contextmanager
def timed(f: Optional[Callable[[int], None]] = None):
    """Time the execution of code in the with-block, calling the function
    f (if it is given) with the resulting time in nanoseconds."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    if f:
        ns = int((end - start) * 1_000_000_000)
        f(ns)


T = TypeVar("T")
U = TypeVar("U")


class Maybe(Generic[T]):
    __slots__ = ("_inner",)

    def __init__(self, inner: Optional[T]) -> None:
        self._inner = inner

    def __eq__(self, other):
        if isinstance(other, Maybe):
            return self._inner == other.value
        return self._inner == other

    def __repr__(self):
        return repr(self._inner)

    def __str__(self):
        return str(self._inner)

    def or_else(self, else_fn: Callable[[], T]) -> T:
        if self._inner is None:
            return else_fn()
        return self._inner

    def or_else_get(self, else_v: T) -> T:
        if self._inner is None:
            return else_v
        return self._inner

    def or_else_raise(self, raise_fn: Callable[[], Exception]) -> T:
        if self._inner is None:
            raise raise_fn()
        return self._inner

    def map(self, f: Callable[[T], U]) -> "Maybe[U]":
        if self._inner is None:
            return Maybe(None)
        return Maybe(f(self._inner))

    @property
    def value(self) -> Optional[T]:
        return self._inner

    @property
    def is_present(self) -> bool:
        return self._inner is not None


def partition(coll: Sequence[T], n: int) -> Iterable[tuple[T, ...]]:
    """Partition `coll` into groups of size `n`."""
    assert n > 0
    start = 0
    stop = n
    while stop <= len(coll):
        yield tuple(e for e in coll[start:stop])
        start += n
        stop += n
    if start < len(coll) < stop:
        stop = len(coll)
        yield tuple(e for e in coll[start:stop])
