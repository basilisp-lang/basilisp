from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar("T")


class Deref(ABC, Generic[T]):
    __slots__ = ()

    @abstractmethod
    def deref(self) -> T:
        raise NotImplementedError()
