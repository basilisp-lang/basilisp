from concurrent.futures import (  # pylint: disable=unused-import
    Future as _Future,
    ProcessPoolExecutor as _ProcessPoolExecutor,
    ThreadPoolExecutor as _ThreadPoolExecutor,
    TimeoutError as _TimeoutError,
)
from typing import Callable, Optional, TypeVar

import attr

from basilisp.lang.interfaces import IBlockingDeref

T = TypeVar("T")


@attr.s(auto_attribs=True, cmp=False, frozen=True, slots=True)
class Future(IBlockingDeref[T]):
    _future: "_Future[T]"

    def cancel(self) -> bool:
        return self._future.cancel()

    def cancelled(self) -> bool:
        return self._future.cancelled()

    def deref(
        self, timeout: Optional[float] = None, timeout_val: Optional[T] = None
    ) -> Optional[T]:
        try:
            return self._future.result(timeout=timeout)
        except _TimeoutError:
            return timeout_val

    def done(self) -> bool:
        return self._future.done()

    @property
    def is_realized(self) -> bool:
        return self.done()


class ProcessPoolExecutor(_ProcessPoolExecutor):
    def __init__(self, max_workers: Optional[int] = None):
        super().__init__(max_workers=max_workers)

    def submit(  # type: ignore
        self, fn: Callable[..., T], *args, **kwargs
    ) -> "Future[T]":
        fut = super().submit(fn, *args, **kwargs)
        return Future(fut)


class ThreadPoolExecutor(_ThreadPoolExecutor):
    def __init__(
        self,
        max_workers: Optional[int] = None,
        thread_name_prefix: str = "basilisp-futures",
    ):
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)

    def submit(  # type: ignore
        self, fn: Callable[..., T], *args, **kwargs
    ) -> "Future[T]":
        fut = super().submit(fn, *args, **kwargs)
        return Future(fut)
