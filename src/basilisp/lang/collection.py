from abc import ABC, abstractmethod


class Collection(ABC):
    __slots__ = ()

    @abstractmethod
    def cons(self, *elems):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def empty() -> "Collection":
        raise NotImplementedError()
