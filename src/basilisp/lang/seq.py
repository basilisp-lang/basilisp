import functools
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar

from basilisp.lang.interfaces import (
    IPersistentMap,
    ISeq,
    ISeqable,
    ISequential,
    IWithMeta,
)
from basilisp.util import Maybe

T = TypeVar("T")


class _EmptySequence(IWithMeta, ISequential, ISeq[T]):
    def __repr__(self):
        return "()"

    def __bool__(self):
        return True

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return None

    def with_meta(self, meta: Optional[IPersistentMap]) -> "Cons[T]":  # type: ignore
        return Cons(meta=meta)

    @property
    def is_empty(self) -> bool:
        return True

    @property
    def first(self) -> Optional[T]:
        return None

    @property
    def rest(self) -> ISeq[T]:
        return self

    def cons(self, elem: T) -> ISeq[T]:
        return Cons(elem, self)


EMPTY: ISeq = _EmptySequence()


class Cons(ISeq[T], ISequential, IWithMeta):
    __slots__ = ("_first", "_rest", "_meta")

    def __init__(
        self,
        first=None,
        seq: Optional[ISeq[T]] = None,
        meta: Optional[IPersistentMap] = None,
    ) -> None:
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
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "Cons[T]":
        return Cons(first=self._first, seq=self._rest, meta=meta)


class _Sequence(IWithMeta, ISequential, ISeq[T]):
    """Sequences are a thin wrapper over Python Iterable values so they can
    satisfy the Basilisp `ISeq` interface.

    Sequences are singly linked lists which lazily traverse the input Iterable.

    Do not directly instantiate a Sequence. Instead use the `sequence` function
    below."""

    __slots__ = ("_first", "_seq", "_rest", "_meta")

    # pylint:disable=assigning-non-slot
    def __init__(
        self, s: Iterator[T], first: T, *, meta: Optional[IPersistentMap] = None
    ) -> None:
        self._seq = s
        self._first = first
        self._rest: Optional[ISeq] = None
        self._meta = meta

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "_Sequence[T]":
        return _Sequence(self._seq, self._first, meta=meta)

    @property
    def is_empty(self) -> bool:
        return False

    @property
    def first(self) -> Optional[T]:
        return self._first

    # pylint:disable=assigning-non-slot
    @property
    def rest(self) -> "ISeq[T]":
        if self._rest:
            return self._rest

        try:
            n = next(self._seq)
            self._rest = _Sequence(self._seq, n)
        except StopIteration:
            self._rest = EMPTY

        return self._rest

    def cons(self, elem):
        return Cons(elem, self)


LazySeqGenerator = Callable[[], Optional[ISeq[T]]]


class LazySeq(IWithMeta, ISequential, ISeq[T]):
    """LazySeqs are wrappers for delaying sequence computation. Create a LazySeq
    with a function that can either return None or a Seq. If a Seq is returned,
    the LazySeq is a proxy to that Seq."""

    __slots__ = ("_gen", "_seq", "_meta")

    # pylint:disable=assigning-non-slot
    def __init__(
        self,
        gen: Optional[LazySeqGenerator],
        seq: Optional[ISeq[T]] = None,
        *,
        meta: Optional[IPersistentMap] = None,
    ) -> None:
        self._gen: Optional[LazySeqGenerator] = gen
        self._seq: Optional[ISeq[T]] = seq
        self._meta = meta

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "LazySeq[T]":
        return LazySeq(self._gen, seq=self._seq, meta=meta)

    # pylint:disable=assigning-non-slot
    def _realize(self):
        if self._gen is not None:
            self._seq = to_seq(self._gen())
            self._gen = None

    @property
    def is_empty(self) -> bool:
        self._realize()
        return self._seq is None

    @property
    def first(self) -> Optional[T]:
        self._realize()
        try:
            return self._seq.first  # type: ignore
        except AttributeError:
            return None

    @property
    def rest(self) -> "ISeq[T]":
        self._realize()
        try:
            return self._seq.rest  # type: ignore
        except AttributeError:
            return EMPTY

    def cons(self, elem):
        return Cons(elem, self)

    @property
    def is_realized(self):
        return self._gen is None


def sequence(s: Iterable) -> ISeq[Any]:
    """Create a Sequence from Iterable s."""
    try:
        i = iter(s)
        return _Sequence(i, next(i))
    except StopIteration:
        return EMPTY


def _seq_or_nil(s: ISeq) -> Optional[ISeq]:
    """Return None if a ISeq is empty, the ISeq otherwise."""
    if s.is_empty:
        return None
    return s


@functools.singledispatch
def to_seq(o) -> Optional[ISeq]:
    """Coerce the argument o to a ISeq. If o is None, return None."""
    return _seq_or_nil(sequence(o))


@to_seq.register(type(None))
def _to_seq_none(_) -> None:
    return None


@to_seq.register(ISeq)
def _to_seq_iseq(o: ISeq) -> Optional[ISeq]:
    return _seq_or_nil(o)


@to_seq.register(ISeqable)
def _to_seq_iseqable(o: ISeqable) -> Optional[ISeq]:
    return _seq_or_nil(o.seq())
