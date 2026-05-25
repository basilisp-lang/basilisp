# pylint: disable=abstract-class-instantiated
from typing import Iterable, TypeVar

import basilisp._lang
from basilisp.lang.interfaces import (
    ISeq,
    ISequential,
    IWithMeta,
)

T = TypeVar("T")


_Cons: type = basilisp._lang.seq.Cons  # type: ignore[attr-defined]
_EmptySequenceNative: type = basilisp._lang.seq.EmptySequence  # type: ignore[attr-defined]
_LazySeq: type = basilisp._lang.seq.LazySeq  # type: ignore[attr-defined]
sequence = basilisp._lang.seq.sequence  # type: ignore[attr-defined]
to_seq = basilisp._lang.seq.to_seq  # type: ignore[attr-defined]


class _EmptySequence(_EmptySequenceNative, IWithMeta, ISequential, ISeq[T]):
    __slots__ = ()


EMPTY: ISeq = _EmptySequence()


class Cons(_Cons, ISeq[T], ISequential, IWithMeta):
    __slots__ = ()


class LazySeq(_LazySeq, IWithMeta, ISequential, ISeq[T]):
    """LazySeqs are wrappers for delaying sequence computation. Create a LazySeq
    with a function that can either return None or a Seq. If a Seq is returned,
    the LazySeq is a proxy to that Seq.

    Callers should never provide the `seq` argument -- this is provided only to
    support `with_meta` returning a new LazySeq instance."""

    __slots__ = ()


def iterator_sequence(s: Iterable[T]) -> ISeq[T]:
    """Create a Sequence from any iterable `s`."""
    return sequence(s, support_single_use=True)


__all__ = ("EMPTY", "Cons", "LazySeq", "iterator_sequence", "sequence", "to_seq")
