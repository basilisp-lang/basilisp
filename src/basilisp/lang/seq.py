import functools
from collections.abc import Callable, Iterable
from typing import Optional, TypeVar, overload

from basilisp.lang.interfaces import (
    IPersistentMap,
    ISeq,
    ISeqable,
    ISequential,
    IWithMeta,
)

T = TypeVar("T")


class _EmptySequence(IWithMeta, ISequential, ISeq[T]):
    __slots__ = ("_meta",)

    def __init__(self, meta: IPersistentMap | None = None):
        self._meta = meta

    def __repr__(self):
        return "()"

    def __bool__(self):
        return True

    def seq(self) -> ISeq[T] | None:
        return None

    @property
    def meta(self) -> IPersistentMap | None:
        return self._meta

    def with_meta(self, meta: IPersistentMap | None) -> "_EmptySequence[T]":
        return _EmptySequence(meta=meta)

    @property
    def is_empty(self) -> bool:
        return True

    @property
    def first(self) -> T | None:
        return None

    @property
    def rest(self) -> ISeq[T]:
        return self

    def cons(self, *elems: T) -> ISeq[T]:  # type: ignore[override]
        l: ISeq = self
        for elem in elems:
            l = Cons(elem, l)
        return l

    def empty(self):
        return EMPTY


EMPTY: ISeq = _EmptySequence()


class Cons(ISeq[T], ISequential, IWithMeta):
    __slots__ = ("_first", "_rest", "_meta")

    def __init__(
        self,
        first: T,
        seq: ISeq[T] | None = None,
        meta: IPersistentMap | None = None,
    ) -> None:
        self._first = first
        self._rest = EMPTY if seq is None else seq
        self._meta = meta

    @property
    def is_empty(self) -> bool:
        return False

    @property
    def first(self) -> T | None:
        return self._first

    @property
    def rest(self) -> ISeq[T]:
        return self._rest

    def cons(self, *elems: T) -> "Cons[T]":
        l = self
        for elem in elems:
            l = Cons(elem, l)
        return l

    def empty(self):
        return EMPTY

    @property
    def meta(self) -> IPersistentMap | None:
        return self._meta

    def with_meta(self, meta: IPersistentMap | None) -> "Cons[T]":
        return Cons(self._first, seq=self._rest, meta=meta)


LazySeqGenerator = Callable[[], Optional[ISeq[T]]]


try:
    import basilisp._lang

    _LazySeq = basilisp._lang.seq.LazySeq
except ImportError:
    pass
else:
    class LazySeq(_LazySeq, IWithMeta, ISequential, ISeq[T]):
        """LazySeqs are wrappers for delaying sequence computation. Create a LazySeq
        with a function that can either return None or a Seq. If a Seq is returned,
        the LazySeq is a proxy to that Seq.

        Callers should never provide the `seq` argument -- this is provided only to
        support `with_meta` returning a new LazySeq instance."""

        __slots__ = ()

        def __new__(
            cls,
            gen: LazySeqGenerator | None,
            seq: ISeq[T] | None = None,
            *,
            meta: IPersistentMap | None = None,
        ):
            return super().__new__(cls, gen, seq, meta=meta)

        def cons(self, *elems: T) -> ISeq[T]:  # type: ignore[override]
            l: ISeq = self
            for elem in elems:
                l = Cons(elem, l)
            return l


def sequence(s: Iterable[T], support_single_use: bool = False) -> ISeq[T]:
    """Create a Sequence from Iterable `s`.

    By default, raise a ``TypeError`` if `s` is a single-use
    Iterable, unless `fail_single_use` is ``True``.

    """
    i = iter(s)

    if not support_single_use and i is s:
        raise TypeError(
            f"Can't create sequence out of single-use iterable object, please use iterator-seq instead. Iterable Object type: {type(s)}"
        )

    def _next_elem() -> ISeq[T]:
        try:
            e = next(i)
        except StopIteration:
            return EMPTY
        else:
            return Cons(e, LazySeq(_next_elem))

    return LazySeq(_next_elem)


def iterator_sequence(s: Iterable[T]) -> ISeq[T]:
    """Create a Sequence from any iterable `s`."""
    return sequence(s, support_single_use=True)


@overload
def _seq_or_nil(s: None) -> None: ...


@overload
def _seq_or_nil(s: ISeq) -> ISeq | None: ...


def _seq_or_nil(s):
    """Return None if a ISeq is empty, the ISeq otherwise."""
    if s is None or s.is_empty:
        return None
    return s


@functools.singledispatch
def to_seq(o) -> ISeq | None:
    """Coerce the argument o to a ISeq. If o is None, return None."""
    return _seq_or_nil(sequence(o))


@to_seq.register(type(None))
def _to_seq_none(_) -> None:
    return None


@to_seq.register(ISeq)
def _to_seq_iseq(o: ISeq) -> ISeq | None:
    return _seq_or_nil(o)


@to_seq.register(LazySeq)
def _to_seq_lazyseq(o: LazySeq) -> ISeq | None:
    # Force evaluation of the LazySeq by calling o.seq() directly.
    return o.seq()


@to_seq.register(ISeqable)
def _to_seq_iseqable(o: ISeqable) -> ISeq | None:
    return _seq_or_nil(o.seq())
