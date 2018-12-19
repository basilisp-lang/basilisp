import itertools
from abc import ABC, abstractmethod
from typing import Iterator, Optional, TypeVar, Iterable, Any, Callable

from basilisp.lang.meta import Meta
from basilisp.lang.obj import LispObject
from basilisp.util import Maybe

T = TypeVar("T")


class Seq(LispObject, Iterable[T]):
    __slots__ = ()

    def _lrepr(self, **kwargs):
        return LispObject.seq_lrepr(iter(self), "(", ")", **kwargs)

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
    def rest(self) -> "Seq[T]":
        raise NotImplementedError()

    @abstractmethod
    def cons(self, elem):
        raise NotImplementedError()

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


class Seqable(ABC, Iterable[T]):
    __slots__ = ()

    def seq(self) -> Seq[T]:
        raise NotImplementedError()


class _EmptySequence(Seq[T]):
    def __repr__(self):
        return "()"

    def __bool__(self):
        return False

    @property
    def is_empty(self) -> bool:
        return True

    @property
    def first(self) -> Optional[T]:
        return None

    @property
    def rest(self) -> Seq[T]:
        return self

    def cons(self, elem):
        return Cons(elem, self)


EMPTY: Seq = _EmptySequence()


class Cons(Seq, Meta):
    __slots__ = ("_first", "_rest", "_meta")

    def __init__(self, first=None, seq: Optional[Seq[Any]] = None, meta=None) -> None:
        self._first = first
        self._rest = Maybe(seq).or_else_get(EMPTY)
        self._meta = meta

    @property
    def is_empty(self) -> bool:
        return False

    @property
    def first(self) -> Optional[Any]:
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
        new_meta = meta if self._meta is None else self._meta.update(meta)
        return Cons(first=self._first, seq=self._rest, meta=new_meta)


class _Sequence(Seq[T]):
    """Sequences are a thin wrapper over Python Iterable values so they can
    satisfy the Basilisp `Seq` interface.

    Sequences are singly linked lists which lazily traverse the input Iterable.

    Do not directly instantiate a Sequence. Instead use the `sequence` function
    below."""

    __slots__ = ("_first", "_seq", "_rest")

    def __init__(self, s: Iterator, first: T) -> None:
        self._seq = s  # pylint:disable=assigning-non-slot
        self._first = first  # pylint:disable=assigning-non-slot
        self._rest: Optional[Seq] = None  # pylint:disable=assigning-non-slot

    @property
    def is_empty(self) -> bool:
        return False

    @property
    def first(self) -> Optional[T]:
        return self._first

    @property
    def rest(self) -> "Seq[T]":
        if self._rest:
            return self._rest

        try:
            n = next(self._seq)
            self._rest = _Sequence(self._seq, n)  # pylint:disable=assigning-non-slot
        except StopIteration:
            self._rest = EMPTY  # pylint:disable=assigning-non-slot

        return self._rest

    def cons(self, elem):
        return Cons(elem, self)


class LazySeq(Seq[T]):
    """LazySeqs are wrappers for delaying sequence computation. Create a LazySeq
    with a function that can either return None or a Seq. If a Seq is returned,
    the LazySeq is a proxy to that Seq."""

    __slots__ = ("_gen", "_realized", "_seq")

    def __init__(self, gen: Callable[[], Optional[Seq]]) -> None:
        self._gen = gen  # pylint:disable=assigning-non-slot
        self._realized = False  # pylint:disable=assigning-non-slot
        self._seq: Optional[Seq] = None  # pylint:disable=assigning-non-slot

    def _realize(self):
        if not self._realized:
            self._seq = self._gen()  # pylint:disable=assigning-non-slot
            self._realized = True  # pylint:disable=assigning-non-slot

    @property
    def is_empty(self) -> bool:
        if not self._realized:
            self._realize()
            return self.is_empty
        if self._seq is None or self._seq.is_empty:
            return True
        return False

    @property
    def first(self) -> Optional[T]:
        if not self._realized:
            self._realize()
        try:
            return self._seq.first  # type: ignore
        except AttributeError:
            return None

    @property
    def rest(self) -> "Seq[T]":
        if not self._realized:
            self._realize()
        try:
            return self._seq.rest  # type: ignore
        except AttributeError:
            return EMPTY

    def cons(self, elem):
        return Cons(elem, self)

    @property
    def is_realized(self):
        return self._realized

    def __iter__(self):
        o = self
        while o:
            first = o.first
            if isinstance(o, LazySeq):
                if o.is_empty:
                    return
            yield first
            o = o.rest


def sequence(s: Iterable) -> Seq[Any]:
    """Create a Sequence from Iterable s."""
    try:
        i = iter(s)
        return _Sequence(i, next(i))
    except StopIteration:
        return EMPTY
