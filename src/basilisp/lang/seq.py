from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar

from basilisp.lang.interfaces import IPersistentMap, ISeq, IWithMeta
from basilisp.util import Maybe

T = TypeVar("T")


class _EmptySequence(IWithMeta, ISeq[T]):
    def __repr__(self):
        return "()"

    def __bool__(self):
        return False

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


class Cons(ISeq[T], IWithMeta):
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


class _Sequence(IWithMeta, ISeq[T]):
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


class LazySeq(IWithMeta, ISeq[T]):
    """LazySeqs are wrappers for delaying sequence computation. Create a LazySeq
    with a function that can either return None or a Seq. If a Seq is returned,
    the LazySeq is a proxy to that Seq."""

    __slots__ = ("_gen", "_realized", "_seq", "_meta")

    # pylint:disable=assigning-non-slot
    def __init__(
        self,
        gen: Callable[[], Optional[ISeq[T]]],
        *,
        meta: Optional[IPersistentMap] = None,
    ) -> None:
        self._gen = gen
        self._realized = False
        self._seq: Optional[ISeq[T]] = None
        self._meta = meta

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "LazySeq[T]":
        return LazySeq(self._gen, meta=meta)

    # pylint:disable=assigning-non-slot
    def _realize(self):
        if not self._realized:
            self._seq = self._gen()
            self._realized = True

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


def sequence(s: Iterable) -> ISeq[Any]:
    """Create a Sequence from Iterable s."""
    try:
        i = iter(s)
        return _Sequence(i, next(i))
    except StopIteration:
        return EMPTY
