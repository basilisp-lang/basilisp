import threading
from typing import Optional, TypeVar

from basilisp.lang.interfaces import IBlockingDeref

T = TypeVar("T")


class Promise(IBlockingDeref[T]):
    __slots__ = ("_condition", "_is_delivered", "_value")

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._is_delivered = False
        self._value: Optional[T] = None

    def deliver(self, value: T) -> None:
        with self._condition:
            if not self._is_delivered:
                self._is_delivered = True
                self._value = value
                self._condition.notify_all()

    def deref(
        self, timeout: Optional[float] = None, timeout_val: Optional[T] = None
    ) -> Optional[T]:
        with self._condition:
            if self._condition.wait_for(lambda: self._is_delivered, timeout=timeout):
                return self._value
            else:
                return timeout_val

    @property
    def is_realized(self) -> bool:
        with self._condition:
            return self._is_delivered
