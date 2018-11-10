from abc import ABC, abstractmethod
from typing import TypeVar, Mapping

K = TypeVar("K")
V = TypeVar("V")


class Associative(ABC, Mapping[K, V]):
    __slots__ = ()

    @abstractmethod
    def assoc(self, *kvs) -> "Associative":
        raise NotImplementedError()

    @abstractmethod
    def contains(self, k):
        raise NotImplementedError()

    @abstractmethod
    def entry(self, k, default=None):
        raise NotImplementedError()
