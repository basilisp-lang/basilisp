from abc import ABC, abstractmethod


class Meta(ABC):
    slots = ()

    @property
    @abstractmethod
    def meta(self):
        raise NotImplemented()

    @abstractmethod
    def with_meta(self, meta):
        raise NotImplemented()
