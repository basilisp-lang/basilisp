from abc import ABC, abstractmethod
from typing import Iterator, Optional, TypeVar, Iterable, Any, cast

import basilisp.lang.list as llist
from basilisp.lang.meta import Meta
from basilisp.lang.util import lrepr

T = TypeVar('T')


class Seq(ABC, Iterable[T]):
    __slots__ = ()

    def __repr__(self):
        return "({list})".format(list=" ".join(map(lrepr, self)))

    @property
    @abstractmethod
    def first(self) -> T:
        raise NotImplemented()

    @property
    @abstractmethod
    def rest(self) -> Optional["Seq[T]"]:
        raise NotImplemented()

    @abstractmethod
    def cons(self, elem):
        raise NotImplemented()

    def __eq__(self, other):
        for e1, e2 in zip(self, other):
            if e1 != e2:
                return False
        return True

    def __iter__(self):
        o = self
        while o:
            yield o.first
            o = o.rest


class Cons(Seq, Meta):
    __slots__ = ('_first', '_rest', '_meta')

    def __init__(self, first=None, seq: Optional[Seq[Any]] = None, meta=None) -> None:
        self._first = first
        self._rest = cast(Seq[Any], seq if seq is not None else llist.List.empty())
        self._meta = meta

    @property
    def first(self):
        return self._first

    @property
    def rest(self) -> Seq[Any]:
        return self._rest

    def cons(self, elem) -> "Cons":
        return Cons(elem, self)

    @property
    def meta(self):
        return self._meta

    def with_meta(self, meta) -> "Cons":
        new_meta = meta if self._meta is None else self._meta.update(
            meta)
        return Cons(first=self._first, seq=self._rest, meta=new_meta)


class _Sequence(Seq[T]):
    """Sequences are a thin wrapper over Python Iterable values so they can
    satisfy the Basilisp `Seq` interface.

    Sequences are singly linked lists which lazily traverse the input Iterable.

    Do not directly instantiate a Sequence. Instead use the `sequence` function
    below."""
    __slots__ = ('_first', '_seq')

    def __init__(self, s: Iterator, first: T) -> None:
        self._seq = s
        self._first = first

    @property
    def first(self) -> T:
        return self._first

    @property
    def rest(self) -> Optional["Seq[T]"]:
        try:
            n = next(self._seq)
            return _Sequence(self._seq, n)
        except StopIteration:
            return llist.List.empty()

    def cons(self, elem):
        return Cons(elem, self)


def sequence(s: Iterable):
    """Create a Sequence from Iterable s."""
    try:
        i = iter(s)
        return _Sequence(i, next(i))
    except StopIteration:
        return llist.List.empty()


Seq.register(_Sequence)
Seq.register(llist.List)
