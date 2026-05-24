from typing import Iterable, TypeVar

from basilisp.lang.interfaces import ISeq

try:
    from basilisp.lang.seq._nativeseq import (
        EMPTY,
        Cons,
        LazySeq,
        sequence,
        to_seq,
    )
except ImportError:
    from basilisp.lang.seq._pyseq import (  # type: ignore[assignment]
        EMPTY,
        Cons,
        LazySeq,
        sequence,
        to_seq,
    )

T = TypeVar("T")


def iterator_sequence(s: Iterable[T]) -> ISeq[T]:
    """Create a Sequence from any iterable `s`."""
    return sequence(s, support_single_use=True)


__all__ = ("EMPTY", "Cons", "LazySeq", "iterator_sequence", "sequence", "to_seq")
