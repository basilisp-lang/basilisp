from abc import ABC, abstractmethod


class Meta(ABC):
    __slots__ = ()

    @property
    @abstractmethod
    def meta(self):
        raise NotImplementedError()

    @abstractmethod
    def with_meta(self, meta) -> "Meta":
        raise NotImplementedError()
