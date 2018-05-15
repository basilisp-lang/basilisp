from typing import Generic, TypeVar, Callable, Optional

T = TypeVar('T')


class Maybe(Generic[T]):
    __slots__ = ('_inner', )

    def __init__(self, inner: Optional[T]) -> None:
        self._inner = inner

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
