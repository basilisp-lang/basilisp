from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar

from basilisp.lang.interfaces import IMeta, ISeq
from basilisp.util import Maybe

T = TypeVar("T")


class _EmptySequence(ISeq[T]):
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
    def rest(self) -> ISeq[T]:
        return self

    def cons(self, elem):
        return Cons(elem, self)


EMPTY: ISeq = _EmptySequence()


class Cons(ISeq[T], IMeta):
    __slots__ = ("_first", "_rest", "_meta")

    def __init__(self, first=None, seq: Optional[ISeq[T]] = None, meta=None) -> None:
        self._first = first
        self._rest = Maybe(seq).or_else_get(EMPTY)
        self._meta = meta

    @property
    def is_empty(self) -> bool:
        return False

    @property
    def first(self) -> Optional[T]:
        return self._first

    @property
    def rest(self) -> ISeq[T]:
        return self._rest

    def cons(self, elem: T) -> "Cons[T]":
        return Cons(elem, self)

    @property
    def meta(self):
        return self._meta

    def with_meta(self, meta) -> "Cons":
        new_meta = meta if self._meta is None else self._meta.update(meta)
        return Cons(first=self._first, seq=self._rest, meta=new_meta)


class _Sequence(ISeq[T]):
    """Sequences are a thin wrapper over Python Iterable values so they can
    satisfy the Basilisp `ISeq` interface.

    Sequences are singly linked lists which lazily traverse the input Iterable.

    Do not directly instantiate a Sequence. Instead use the `sequence` function
    below."""

    __slots__ = ("_first", "_seq", "_rest")

    def __init__(self, s: Iterator, first: T) -> None:
        self._seq = s  # pylint:disable=assigning-non-slot
        self._first = first  # pylint:disable=assigning-non-slot
        self._rest: Optional[ISeq] = None  # pylint:disable=assigning-non-slot

    @property
    def is_empty(self) -> bool:
        return False

    @property
    def first(self) -> Optional[T]:
        return self._first

    @property
    def rest(self) -> "ISeq[T]":
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


class LazySeq(ISeq[T]):
    """LazySeqs are wrappers for delaying sequence computation. Create a LazySeq
    with a function that can either return None or a Seq. If a Seq is returned,
    the LazySeq is a proxy to that Seq."""

    __slots__ = ("_gen", "_realized", "_seq")

    def __init__(self, gen: Callable[[], Optional[ISeq[T]]]) -> None:
        self._gen = gen  # pylint:disable=assigning-non-slot
        self._realized = False  # pylint:disable=assigning-non-slot
        self._seq: Optional[ISeq[T]] = None  # pylint:disable=assigning-non-slot

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
    def rest(self) -> "ISeq[T]":
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
        if o:
            first = o.first
            if isinstance(o, LazySeq):
                if o.is_empty:
                    return
            yield first
            yield from o.rest


def sequence(s: Iterable) -> ISeq[Any]:
    """Create a Sequence from Iterable s."""
    try:
        i = iter(s)
        return _Sequence(i, next(i))
    except StopIteration:
        return EMPTY
