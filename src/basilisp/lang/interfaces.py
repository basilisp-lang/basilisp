from abc import ABC, abstractmethod
from typing import (AbstractSet, Collection, Generic, Iterable, Mapping, Optional, TypeVar)

K = TypeVar("K")
V = TypeVar("V")


class IAssociative(ABC, Mapping[K, V]):
    __slots__ = ()

    @abstractmethod
    def assoc(self, *kvs) -> "IAssociative[K, V]":
        raise NotImplementedError()

    @abstractmethod
    def contains(self, k: K) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def entry(self, k: K, default: Optional[V] = None) -> Optional[V]:
        raise NotImplementedError()


T = TypeVar("T")


class IDeref(ABC, Generic[T]):
    __slots__ = ()

    @abstractmethod
    def deref(self) -> T:
        raise NotImplementedError()


class IMeta(ABC):
    __slots__ = ()

    @property
    @abstractmethod
    def meta(self) -> Optional["IPersistentMap"]:
        raise NotImplementedError()

    @abstractmethod
    def with_meta(self, meta: "IPersistentMap") -> "IMeta":
        raise NotImplementedError()


class IPersistentCollection(Generic[T]):
    __slots__ = ()

    @abstractmethod
    def cons(self, *elems: T) -> "IPersistentCollection[T]":
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def empty() -> "IPersistentCollection[T]":
        raise NotImplementedError()


class IPersistentStack(IPersistentCollection[T]):
    @abstractmethod
    def peek(self) -> Optional[T]:
        raise NotImplementedError()

    @abstractmethod
    def pop(self) -> "IPersistentStack[T]":
        raise NotImplementedError()


class IPersistentList(IPersistentStack[T]):
    pass


class IPersistentMap(IAssociative[K, V]):
    @abstractmethod
    def dissoc(self, *ks: K) -> "IPersistentMap[K, V]":
        raise NotImplementedError()


class IPersistentSet(IPersistentCollection[T], AbstractSet[T]):
    @abstractmethod
    def disj(self, *elems: T) -> "IPersistentSet[T]":
        raise NotImplementedError()


class IPersistentVector(IAssociative[int, T], Collection[T], IPersistentStack[T]):
    pass


class ISeq(Iterable[T]):
    __slots__ = ()

    @property
    @abstractmethod
    def is_empty(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def first(self) -> Optional[T]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def rest(self) -> "ISeq[T]":
        raise NotImplementedError()

    @abstractmethod
    def cons(self, elem):
        raise NotImplementedError()


class ISeqable(ABC, Iterable[T]):
    __slots__ = ()

    def seq(self) -> ISeq[T]:
        raise NotImplementedError()
