# pylint: disable=abstract-class-instantiated,import-error,no-name-in-module
from typing import Iterable, TypeVar

from basilisp._lang.seq import Cons as _Cons
from basilisp._lang.seq import EmptySequence as _EmptySequenceNative
from basilisp._lang.seq import LazySeq as _LazySeq
from basilisp._lang.seq import sequence, to_seq
from basilisp.lang.interfaces import (
    ISeq,
    ISequential,
    IWithMeta,
)

T = TypeVar("T")


class _EmptySequence(_EmptySequenceNative[T], IWithMeta, ISequential, ISeq[T]):
    """
    An empty seq.

    Generally referenced using the static value :py:data:`EMPTY` rather than created
    dynamically.
    """

    __slots__ = ()


EMPTY: ISeq = _EmptySequence()


class Cons(_Cons[T], ISeq[T], ISequential, IWithMeta):
    """
    Cons cells are essentially linked-list types for ISeq.

    When ``(cons ...)`` is called on most other ISeq types, the resulting type will be
    Cons cells.
    """

    __slots__ = ()


class LazySeq(_LazySeq[T], IWithMeta, ISequential, ISeq[T]):
    """LazySeqs are wrappers for delaying sequence computation. Create a LazySeq
    with a function that can either return None or a Seq. If a Seq is returned,
    the LazySeq is a proxy to that Seq.

    Callers should never provide the ``seq`` argument -- this is provided only to
    support ``with_meta`` returning a new LazySeq instance."""

    __slots__ = ()


def iterator_sequence(s: Iterable[T]) -> ISeq[T]:
    """Create a Sequence from any iterable `s`."""
    return sequence(s, support_single_use=True)


__all__ = ("EMPTY", "Cons", "LazySeq", "iterator_sequence", "sequence", "to_seq")
