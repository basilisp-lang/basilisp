from abc import ABC, abstractmethod
from typing import Optional

from basilisp.lang.map import Map


class Meta(ABC):
    slots = ()

    @property
    @abstractmethod
    def meta(self) -> Optional[Map]:
        raise NotImplemented()

    @abstractmethod
    def with_meta(self, meta: Map) -> "Meta":
        raise NotImplemented()