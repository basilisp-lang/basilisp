import itertools
from abc import ABC, abstractmethod
from typing import AbstractSet, Generic, Iterable, Mapping, Optional, Sequence, TypeVar

from basilisp.lang.obj import LispObject

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
    def deref(self) -> Optional[T]:
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
    __slots__ = ()

    @abstractmethod
    def peek(self) -> Optional[T]:
        raise NotImplementedError()

    @abstractmethod
    def pop(self) -> "IPersistentStack[T]":
        raise NotImplementedError()


class IPersistentList(IPersistentStack[T]):
    __slots__ = ()


class IPersistentMap(IAssociative[K, V]):
    __slots__ = ()

    @abstractmethod
    def dissoc(self, *ks: K) -> "IPersistentMap[K, V]":
        raise NotImplementedError()


class IPersistentSet(AbstractSet[T], IPersistentCollection[T]):
    __slots__ = ()

    @abstractmethod
    def disj(self, *elems: T) -> "IPersistentSet[T]":
        raise NotImplementedError()


class IPersistentVector(  # type: ignore
    IAssociative[int, T], IPersistentStack[T], Sequence[T]
):
    __slots__ = ()


class ISeq(LispObject, Iterable[T]):
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
    def cons(self, elem: T) -> "ISeq[T]":
        raise NotImplementedError()

    def _lrepr(self, **kwargs):
        return LispObject.seq_lrepr(iter(self), "(", ")", **kwargs)

    def __eq__(self, other):
        sentinel = object()
        for e1, e2 in itertools.zip_longest(self, other, fillvalue=sentinel):
            if bool(e1 is sentinel) or bool(e2 is sentinel):
                return False
            if e1 != e2:
                return False
        return True

    def __iter__(self):
        o = self
        while o:
            yield o.first
            o = o.rest


class ISeqable(ABC, Iterable[T]):
    __slots__ = ()

    def seq(self) -> ISeq[T]:
        raise NotImplementedError()
