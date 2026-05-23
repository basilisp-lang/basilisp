import functools
from collections.abc import Iterable
from typing import TypeVar

from basilisp.lang.interfaces import (
    IPersistentMap,
    ISeq,
    ISequential,
    IWithMeta,
)

T = TypeVar("T")

try:
    import basilisp._lang

    _Cons = basilisp._lang.seq.Cons
    _EmptySequenceNative = basilisp._lang.seq.EmptySequence
    _LazySeq: type = basilisp._lang.seq.LazySeq
    to_seq = basilisp._lang.seq.to_seq
except ImportError:
    from basilisp.lang._pyseq import EMPTY, Cons, LazySeq, to_seq

else:

    class _EmptySequence(_EmptySequenceNative, IWithMeta, ISequential, ISeq[T]):
        __slots__ = ()

        def with_meta(self, meta: IPersistentMap | None) -> "_EmptySequence[T]":
            return _EmptySequence(meta=meta)

        def cons(self, *elems: T) -> ISeq[T]:  # type: ignore[override]
            l: ISeq = self
            for elem in elems:
                l = Cons(elem, l)
            return l

    EMPTY = _EmptySequence()

    class Cons(_Cons, ISeq[T], ISequential, IWithMeta):
        __slots__ = ()

        def with_meta(self, meta: IPersistentMap | None) -> "Cons[T]":
            return Cons(self.first, rest=self.rest, meta=meta)

        def cons(self, *elems: T) -> "Cons[T]":
            l = self
            for elem in elems:
                l = Cons(elem, l)
            return l

    class LazySeq(_LazySeq, IWithMeta, ISequential, ISeq[T]):
        """LazySeqs are wrappers for delaying sequence computation. Create a LazySeq
        with a function that can either return None or a Seq. If a Seq is returned,
        the LazySeq is a proxy to that Seq.

        Callers should never provide the `seq` argument -- this is provided only to
        support `with_meta` returning a new LazySeq instance."""

        __slots__ = ()

        def with_meta(self, meta: IPersistentMap | None) -> "LazySeq[T]":
            return LazySeq(None, seq=self.seq(), meta=meta)

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
