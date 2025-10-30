from collections.abc import Callable
from concurrent.futures import Future as _Future  # noqa # pylint: disable=unused-import
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from concurrent.futures import TimeoutError as _TimeoutError
from typing import TypeVar

import attr
from typing_extensions import ParamSpec

from basilisp.lang.interfaces import IBlockingDeref, IPending

T = TypeVar("T")
P = ParamSpec("P")


@attr.frozen(eq=True, repr=False)
class Future(IBlockingDeref[T], IPending):
    _future: "_Future[T]"

    def __repr__(self):  # pragma: no cover
        return self._future.__repr__()

    def cancel(self) -> bool:
        return self._future.cancel()

    def cancelled(self) -> bool:
        return self._future.cancelled()

    def deref(
        self, timeout: float | None = None, timeout_val: T | None = None
    ) -> T | None:
        try:
            return self._future.result(timeout=timeout)
        except _TimeoutError:
            return timeout_val

    def done(self) -> bool:
        return self._future.done()

    @property
    def is_realized(self) -> bool:
        return self.done()

    # Pass `Future.result(timeout=...)` through so `Executor.map(...)` can
    # still work with this Future wrapper.
    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout=timeout)


# Basilisp's standard Future executor is the `ThreadPoolExecutor`, but since
# it is set via a dynamic variable, it can be rebound using the binding macro.
# Callers may wish to use a process pool if they have CPU bound work.


class ProcessPoolExecutor(_ProcessPoolExecutor):  # pragma: no cover
    def __init__(self, max_workers: int | None = None):
        super().__init__(max_workers=max_workers)

    # pylint: disable=arguments-differ
    def submit(  # type: ignore[override]
        self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> "Future[T]":
        return Future(super().submit(fn, *args, **kwargs))


class ThreadPoolExecutor(_ThreadPoolExecutor):
    def __init__(
        self,
        max_workers: int | None = None,
        thread_name_prefix: str = "basilisp-futures",
    ):
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)

    # pylint: disable=arguments-differ
    def submit(  # type: ignore[override]
        self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> "Future[T]":
        return Future(super().submit(fn, *args, **kwargs))
