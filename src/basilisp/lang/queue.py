from typing import Optional, TypeVar

from pyrsistent import PDeque, pdeque  # noqa # pylint: disable=unused-import

from basilisp.lang.interfaces import (
    ILispObject,
    IPersistentList,
    IPersistentMap,
    ISeq,
    IWithMeta,
    seq_equals,
)
from basilisp.lang.obj import seq_lrepr as _seq_lrepr
from basilisp.lang.seq import sequence

T = TypeVar("T")


class PersistentQueue(IPersistentList[T], IWithMeta, ILispObject):
    """Basilisp Queue. Delegates internally to a pyrsistent.PDeque object.

    Do not instantiate directly. Instead use the q() and queue() factory
    methods below."""

    __slots__ = ("_inner", "_meta")

    def __init__(self, wrapped: "PDeque[T]", meta=None) -> None:
        self._inner = wrapped
        self._meta = meta

    def __bool__(self):
        return True

    def __eq__(self, other):
        if self is other:
            return True
        if hasattr(other, "__len__") and len(self) != len(other):
            return False
        return seq_equals(self, other)

    def __hash__(self):
        return hash(self._inner)

    def __iter__(self):
        yield from self._inner

    def __len__(self):
        return len(self._inner)

    def _lrepr(self, **kwargs) -> str:
        return _seq_lrepr(self._inner, "#queue (", ")", meta=self._meta, **kwargs)

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "PersistentQueue":
        return queue(self._inner, meta=meta)

    def cons(self, *elems: T) -> "PersistentQueue[T]":
        return PersistentQueue(self._inner.extend(elems))

    @staticmethod
    def empty() -> "PersistentQueue":
        return EMPTY

    def peek(self):
        try:
            return self._inner.left
        except IndexError:
            return None

    def pop(self) -> "PersistentQueue[T]":
        if len(self._inner) == 0:
            raise IndexError("Cannot pop an empty queue")
        return PersistentQueue(self._inner.popleft())

    def seq(self) -> ISeq[T]:
        return sequence(self)


EMPTY: PersistentQueue = PersistentQueue(pdeque())


def queue(members, meta=None) -> PersistentQueue:  # pylint:disable=redefined-builtin
    """Creates a new queue."""
    return PersistentQueue(  # pylint: disable=abstract-class-instantiated
        pdeque(iterable=members), meta=meta
    )


def q(*members, meta=None) -> PersistentQueue:  # noqa
    """Creates a new queue from members."""
    return PersistentQueue(  # pylint: disable=abstract-class-instantiated
        pdeque(iterable=members), meta=meta
    )
