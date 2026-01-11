import threading
from typing import TypeVar

from basilisp.lang.interfaces import IBlockingDeref, IPending

T = TypeVar("T")


class Promise(IBlockingDeref[T], IPending):
    __slots__ = ("_condition", "_is_delivered", "_value")

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._is_delivered = False
        self._value: T | None = None

    def deliver(self, value: T) -> None:
        with self._condition:
            if not self._is_delivered:
                self._is_delivered = True
                self._value = value
                self._condition.notify_all()

    __call__ = deliver

    def deref(
        self, timeout: float | None = None, timeout_val: T | None = None
    ) -> T | None:
        with self._condition:
            if self._condition.wait_for(lambda: self._is_delivered, timeout=timeout):
                return self._value
            else:
                return timeout_val

    @property
    def is_realized(self) -> bool:
        with self._condition:
            return self._is_delivered
