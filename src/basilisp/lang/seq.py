import functools
import threading
from collections.abc import Iterable
from typing import Callable, Optional, TypeVar, overload

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
    __slots__ = ("_meta",)

    def __init__(self, meta: Optional[IPersistentMap] = None):
        self._meta = meta

    def __repr__(self):
        return "()"

    def __bool__(self):
        return True

    def seq(self) -> Optional[ISeq[T]]:
        return None

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "_EmptySequence[T]":
        return _EmptySequence(meta=meta)

    @property
    def is_empty(self) -> bool:
        return True

    @property
    def first(self) -> Optional[T]:
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

    def cons(self, *elems: T) -> "Cons[T]":
        l = self
        for elem in elems:
            l = Cons(elem, l)
        return l

    def empty(self):
        return EMPTY

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "Cons[T]":
        return Cons(self._first, seq=self._rest, meta=meta)


LazySeqGenerator = Callable[[], Optional[ISeq[T]]]


class LazySeq(IWithMeta, ISequential, ISeq[T]):
    """LazySeqs are wrappers for delaying sequence computation. Create a LazySeq
    with a function that can either return None or a Seq. If a Seq is returned,
    the LazySeq is a proxy to that Seq.

    Callers should never provide the `seq` argument -- this is provided only to
    support `with_meta` returning a new LazySeq instance."""

    __slots__ = ("_gen", "_obj", "_seq", "_lock", "_meta")

    def __init__(
        self,
        gen: Optional[LazySeqGenerator],
        seq: Optional[ISeq[T]] = None,
        *,
        meta: Optional[IPersistentMap] = None,
    ) -> None:
        self._gen: Optional[LazySeqGenerator] = gen
        self._obj: Optional[ISeq[T]] = None
        self._seq: Optional[ISeq[T]] = seq
        self._lock = threading.RLock()
        self._meta = meta

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "LazySeq[T]":
        return LazySeq(None, seq=self.seq(), meta=meta)

    # LazySeqs have a fairly complex inner state, in spite of the simple interface.
    # Calls from Basilisp code should be providing the only generator seed function.
    # Calls to `(seq ...)` cause the LazySeq to cache the generator function locally
    # (as explained in _compute_seq), clear the _gen attribute, and cache the results
    # of that generator function call as _obj. _obj may be None, some other ISeq, or
    # perhaps another LazySeq. Finally, the LazySeq attempts to consume all returned
    # LazySeq objects before calling `(seq ...)` on the result, which is cached in the
    # _seq attribute.

    def _compute_seq(self) -> Optional[ISeq[T]]:
        if self._gen is not None:
            # This local caching of the generator function and clearing of self._gen
            # is absolutely critical for supporting co-recursive lazy sequences.
            #
            # The original example that prompted this change is below:
            #
            #   (def primes (remove
            #                (fn [x] (some #(zero? (mod x %)) primes))
            #                (iterate inc 2)))
            #
            #   (take 5 primes)  ;; => stack overflow
            #
            # If we don't clear self._gen, each successive call to (some ... primes)
            # will end up forcing the primes LazySeq object to call self._gen, rather
            # than caching the results, allowing examination of the partial seq
            # computed up to that point.
            gen = self._gen
            self._gen = None
            self._obj = gen()
        return self._obj if self._obj is not None else self._seq

    def seq(self) -> Optional[ISeq[T]]:
        with self._lock:
            self._compute_seq()
            if self._obj is not None:
                o = self._obj
                self._obj = None
                # Consume any additional lazy sequences returned immediately, so we
                # have a "real" concrete sequence to proxy to.
                #
                # The common idiom with LazySeqs is to return
                # (cons value (lazy-seq ...)) from the generator function, so this will
                # only result in evaluating away instances where _another_ LazySeq is
                # returned rather than a cons cell with a concrete first value. This
                # loop will not consume the LazySeq in the rest position of the cons.
                while isinstance(o, LazySeq):
                    o = o._compute_seq()  # type: ignore
                self._seq = to_seq(o)
            return self._seq

    @property
    def is_empty(self) -> bool:
        return self.seq() is None

    @property
    def first(self) -> Optional[T]:
        if self.is_empty:
            return None
        return self.seq().first  # type: ignore[union-attr]

    @property
    def rest(self) -> "ISeq[T]":
        if self.is_empty:
            return EMPTY
        return self.seq().rest  # type: ignore[union-attr]

    def cons(self, *elems: T) -> ISeq[T]:  # type: ignore[override]
        l: ISeq = self
        for elem in elems:
            l = Cons(elem, l)
        return l

    @property
    def is_realized(self):
        with self._lock:
            return self._gen is None

    def empty(self):
        return EMPTY


def sequence(s: Iterable[T]) -> ISeq[T]:
    """Create a Sequence from Iterable s."""
    i = iter(s)

    def _next_elem() -> ISeq[T]:
        try:
            e = next(i)
        except StopIteration:
            return EMPTY
        else:
            return Cons(e, LazySeq(_next_elem))

    return LazySeq(_next_elem)


@overload
def _seq_or_nil(s: None) -> None: ...


@overload
def _seq_or_nil(s: ISeq) -> Optional[ISeq]: ...


def _seq_or_nil(s):
    """Return None if a ISeq is empty, the ISeq otherwise."""
    if s is None or s.is_empty:
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


@to_seq.register(LazySeq)
def _to_seq_lazyseq(o: LazySeq) -> Optional[ISeq]:
    # Force evaluation of the LazySeq by calling o.seq() directly.
    return o.seq()


@to_seq.register(ISeqable)
def _to_seq_iseqable(o: ISeqable) -> Optional[ISeq]:
    return _seq_or_nil(o.seq())
