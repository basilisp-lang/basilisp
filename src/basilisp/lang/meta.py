from abc import ABC, abstractmethod


class Meta(ABC):
    slots = ()

    @property
    @abstractmethod
    def meta(self):
        raise NotImplementedError()

    @abstractmethod
    def with_meta(self, meta) -> "Meta":
        raise NotImplementedError()
