from concurrent.futures import (
    Future as _Future,
    ProcessPoolExecutor as _ProcessPoolExecutor,
    ThreadPoolExecutor as _ThreadPoolExecutor,
    TimeoutError as _TimeoutError,
)
from typing import Optional, TypeVar

import attr

from basilisp.lang.interfaces import IBlockingDeref

T = TypeVar("T")


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Future(IBlockingDeref):
    _future: _Future

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
    def __init__(self):
        super().__init__()

    def submit(self, fn, *args, **kwargs) -> Future:
        fut = super().submit(fn, *args, **kwargs)
        return Future(fut)


class ThreadPoolExecutor(_ThreadPoolExecutor):
    def __init__(self):
        super().__init__(thread_name_prefix="basilisp-futures")

    def submit(self, fn, *args, **kwargs) -> Future:
        fut = super().submit(fn, *args, **kwargs)
        return Future(fut)
