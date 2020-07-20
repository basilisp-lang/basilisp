from typing import Optional, TypeVar, cast

from pyrsistent import PDeque, pdeque
from pyrsistent._plist import _EMPTY_PLIST

from basilisp.lang.interfaces import IPersistentList, IPersistentMap, ISeq, IWithMeta
from basilisp.lang.obj import seq_lrepr as _seq_lrepr
from basilisp.lang.seq import EMPTY

T = TypeVar("T")


class PersistentQueue(IPersistentList[T], ISeq[T], IWithMeta):
    """Basilisp Queue. Delegates internally to a pyrsistent.PQueue object.

    Do not instantiate directly. Instead use the q() and queue() factory
    methods below."""

    __slots__ = ("_inner", "_meta")

    def __init__(self, wrapped: "PDeque[T]", meta=None) -> None:
        self._inner = wrapped
        self._meta = meta

    def __bool__(self):
        return True

    def __hash__(self):
        return hash(self._inner)

    def __len__(self):
        return len(self._inner)

    def _lrepr(self, **kwargs) -> str:
        return _seq_lrepr(self._inner, "(", ")", meta=self._meta, **kwargs)

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "List":
        return queue(self._inner, meta=meta)

    @property
    def is_empty(self):
        return len(self._inner) == 0

    @property
    def first(self):
        return self._inner.left

    @property
    def rest(self) -> ISeq[T]:
        return PersistentQueue(self._inner.popleft())

    def cons(self, *elems: T) -> "PersistentQueue[T]":
        return PersistentQueue(self._inner.extend(elems))

    @staticmethod
    def empty(meta=None) -> "PersistentQueue":  # pylint:disable=arguments-differ
        return q(meta=meta)

    def peek(self):
        return self.first

    def pop(self) -> "PersistentQueue[T]":
        if self.is_empty:
            raise IndexError("Cannot pop an empty queue")
        return cast(PersistentQueue, self.rest)


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
