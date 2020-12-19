import itertools
from abc import ABC, abstractmethod
from typing import (
    AbstractSet,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from basilisp.lang.obj import LispObject as _LispObject
from basilisp.lang.obj import seq_lrepr

T = TypeVar("T")


class IDeref(Generic[T], ABC):
    __slots__ = ()

    @abstractmethod
    def deref(self) -> Optional[T]:
        raise NotImplementedError()


class IBlockingDeref(IDeref[T]):
    __slots__ = ()

    # pylint: disable=arguments-differ
    @abstractmethod
    def deref(
        self, timeout: Optional[float] = None, timeout_val: Optional[T] = None
    ) -> Optional[T]:
        raise NotImplementedError()


class ICounted(ABC):
    """ICounted types can produce their length in constant time.

    All of the builtin collections are ICounted, except Lists whose length is
    determined by counting all of the elements in the list in linear time."""

    __slots__ = ()


class IIndexed(ICounted, ABC):
    """IIndexed types can be accessed by index.

    Of the builtin collections, only Vectors are IIndexed. IIndexed types respond
    True to the `indexed?` predicate."""

    __slots__ = ()


# Making this interface Generic causes the __repr__ to differ between
# Python 3.6 and 3.7, which affects a few simple test assertions.
# Since there is little benefit to this type being Generic, I'm leaving
# it as is for now.
class IExceptionInfo(Exception, ABC):
    __slots__ = ()

    @property
    @abstractmethod
    def data(self) -> "IPersistentMap":
        raise NotImplementedError()


K = TypeVar("K")
V = TypeVar("V")


class IMapEntry(Generic[K, V], ABC):
    __slots__ = ()

    @property
    @abstractmethod
    def key(self) -> K:
        raise NotImplementedError()

    @property
    @abstractmethod
    def value(self) -> V:
        raise NotImplementedError()


class IMeta(ABC):
    __slots__ = ()

    @property
    @abstractmethod
    def meta(self) -> Optional["IPersistentMap"]:
        raise NotImplementedError()


T_with_meta = TypeVar("T_with_meta", bound="IWithMeta")


class IWithMeta(IMeta):
    __slots__ = ()

    @abstractmethod
    def with_meta(self: T_with_meta, meta: "Optional[IPersistentMap]") -> T_with_meta:
        raise NotImplementedError()


ILispObject = _LispObject


class IReference(IMeta):
    __slots__ = ()

    @abstractmethod
    def alter_meta(self, f: Callable[..., "IPersistentMap"], *args) -> "IPersistentMap":
        raise NotImplementedError()

    @abstractmethod
    def reset_meta(self, meta: "IPersistentMap") -> "IPersistentMap":
        raise NotImplementedError()


class IReversible(Generic[T]):
    """IReversible types can produce a sequences of their elements in reverse in
    constant time.

    Of the builtin collections, only Vectors are IReversible. IIndexed types respond
    True to the `reversible` predicate."""

    __slots__ = ()

    @abstractmethod
    def rseq(self) -> "ISeq[T]":
        raise NotImplementedError()


class ISeqable(Iterable[T]):
    """ISeqable types can produce sequences of their elements, but are not ISeqs.

    All of the builtin collections are ISeqable, except Lists which directly
    implement ISeq. Values of type ISeqable respond True to the `seqable?` predicate."""

    __slots__ = ()

    @abstractmethod
    def seq(self) -> "ISeq[T]":
        raise NotImplementedError()


class ISequential(ABC):
    """ISequential is a marker interface for sequential types.

    Lists and Vectors are both considered ISequential and respond True to the
    `sequential?` predicate."""

    __slots__ = ()


class ILookup(Generic[K, V], ABC):
    __slots__ = ()

    @abstractmethod
    def val_at(self, k: K, default: Optional[V] = None) -> Optional[V]:
        raise NotImplementedError()


T_pcoll = TypeVar("T_pcoll", bound="IPersistentCollection", covariant=True)


class IPersistentCollection(ISeqable[T]):
    __slots__ = ()

    @abstractmethod
    def cons(self: T_pcoll, *elems: T) -> "T_pcoll":
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def empty() -> "IPersistentCollection[T]":
        raise NotImplementedError()


T_assoc = TypeVar("T_assoc", bound="IAssociative")


class IAssociative(ILookup[K, V], IPersistentCollection[IMapEntry[K, V]]):
    __slots__ = ()

    @abstractmethod
    def assoc(self: T_assoc, *kvs) -> T_assoc:
        raise NotImplementedError()

    @abstractmethod
    def contains(self, k: K) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def entry(self, k: K) -> Optional[IMapEntry[K, V]]:
        raise NotImplementedError()


T_stack = TypeVar("T_stack", bound="IPersistentStack")


class IPersistentStack(IPersistentCollection[T]):
    __slots__ = ()

    @abstractmethod
    def peek(self) -> Optional[T]:
        raise NotImplementedError()

    @abstractmethod
    def pop(self: T_stack) -> T_stack:
        raise NotImplementedError()


class IPersistentList(ISequential, IPersistentStack[T]):
    __slots__ = ()


T_map = TypeVar("T_map", bound="IPersistentMap")


class IPersistentMap(ICounted, Mapping[K, V], IAssociative[K, V]):
    __slots__ = ()

    @abstractmethod
    def cons(  # type: ignore[override]
        self: T_map, *elems: Union[IMapEntry[K, V], "IPersistentMap[K, V]", None]
    ) -> T_map:
        raise NotImplementedError()

    @abstractmethod
    def dissoc(self: T_map, *ks: K) -> T_map:
        raise NotImplementedError()


T_set = TypeVar("T_set", bound="IPersistentSet")


class IPersistentSet(AbstractSet[T], ICounted, IPersistentCollection[T]):
    __slots__ = ()

    @abstractmethod
    def disj(self: T_set, *elems: T) -> T_set:
        raise NotImplementedError()


T_vec = TypeVar("T_vec", bound="IPersistentVector")


class IPersistentVector(
    Sequence[T],
    IAssociative[int, T],
    IIndexed,
    IReversible[T],
    ISequential,
    IPersistentStack[T],
):
    __slots__ = ()

    @abstractmethod
    def assoc(self: T_vec, *kvs) -> T_vec:  # type: ignore[override]
        raise NotImplementedError()

    @abstractmethod
    def cons(self: T_vec, *elems: T) -> T_vec:  # type: ignore[override]
        raise NotImplementedError()

    @abstractmethod
    def seq(self) -> "ISeq[T]":  # type: ignore[override]
        raise NotImplementedError()


T_tcoll = TypeVar("T_tcoll", bound="ITransientCollection", covariant=True)

# Including ABC as a base seems to cause catastrophic meltdown.
class IEvolveableCollection(Generic[T_tcoll]):
    @abstractmethod
    def to_transient(self) -> T_tcoll:
        raise NotImplementedError()


class ITransientCollection(Generic[T]):
    __slots__ = ()

    @abstractmethod
    def cons_transient(self: T_tcoll, *elems: T) -> "T_tcoll":
        raise NotImplementedError()

    @abstractmethod
    def to_persistent(self: T_tcoll) -> "IPersistentCollection[T]":
        raise NotImplementedError()


T_tassoc = TypeVar("T_tassoc", bound="ITransientAssociative")


class ITransientAssociative(ILookup[K, V], ITransientCollection[IMapEntry[K, V]]):
    __slots__ = ()

    @abstractmethod
    def assoc_transient(self: T_tassoc, *kvs) -> T_tassoc:
        raise NotImplementedError()

    @abstractmethod
    def contains_transient(self, k: K) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def entry_transient(self, k: K) -> Optional[IMapEntry[K, V]]:
        raise NotImplementedError()


T_tmap = TypeVar("T_tmap", bound="ITransientMap")


class ITransientMap(ICounted, ITransientAssociative[K, V]):
    __slots__ = ()

    @abstractmethod
    def cons_transient(  # type: ignore[override]
        self: T_tmap, *elems: Union[IMapEntry[K, V], "IPersistentMap[K, V]", None]
    ) -> T_tmap:
        raise NotImplementedError()

    @abstractmethod
    def dissoc_transient(self: T_tmap, *ks: K) -> T_tmap:
        raise NotImplementedError()


T_tset = TypeVar("T_tset", bound="ITransientSet")


class ITransientSet(ICounted, ITransientCollection[T]):
    __slots__ = ()

    @abstractmethod
    def disj_transient(self: T_tset, *elems: T) -> T_tset:
        raise NotImplementedError()


T_tvec = TypeVar("T_tvec", bound="ITransientVector")


class ITransientVector(
    ITransientAssociative[int, T],
    IIndexed,
):
    __slots__ = ()

    @abstractmethod
    def assoc_transient(self: T_tvec, *kvs) -> T_tvec:  # type: ignore[override]
        raise NotImplementedError()

    @abstractmethod
    def cons_transient(self: T_tvec, *elems: T) -> T_tvec:  # type: ignore[override]
        raise NotImplementedError()

    @abstractmethod
    def pop_transient(self: T_tvec) -> T_tvec:
        raise NotImplementedError()


class IRecord(ILispObject):
    """IRecord is a marker interface for types def'ed by `defrecord` forms.

    All types created by `defrecord` are automatically marked with IRecord."""

    __slots__ = ()

    @classmethod
    @abstractmethod
    def create(cls, m: IPersistentMap) -> "IRecord":
        """Class method constructor from an IPersistentMap instance."""
        raise NotImplementedError()

    def _lrepr(self, **kwargs) -> str:
        return self._record_lrepr(kwargs)

    @abstractmethod
    def _record_lrepr(self, kwargs: Mapping) -> str:
        """Translation method converting Python keyword arguments into a
        Python dict.

        Basilisp methods and functions cannot formally accept Python keyword
        arguments, so this method is called by `_lrepr` with the keyword
        arguments cast to a Python dict."""
        raise NotImplementedError()


def seq_equals(s1, s2) -> bool:
    """Return True if two sequences contain the exactly the same elements in the
    same order. Return False if one sequence is shorter than the other."""
    assert isinstance(s1, (ISeq, ISequential))

    if not isinstance(s2, (ISeq, ISequential)):
        return NotImplemented

    sentinel = object()
    for e1, e2 in itertools.zip_longest(s1, s2, fillvalue=sentinel):  # type: ignore[arg-type]
        if bool(e1 is sentinel) or bool(e2 is sentinel):
            return False
        if e1 != e2:
            return False
    return True


class ISeq(ILispObject, ISeqable[T]):
    __slots__ = ()

    class _SeqIter(Iterator[T]):
        """Stateful iterator for sequence types.

        This is primarily useful for avoiding blowing the stack on a long (or infinite)
        sequence. It is not safe to use `yield` statements to iterate over sequences,
        since they accrete one Python stack frame per sequence element."""

        __slots__ = ("_cur",)

        def __init__(self, seq: "ISeq[T]"):
            self._cur = seq

        def __next__(self):
            if not self._cur:
                raise StopIteration
            v = self._cur.first
            if self._cur.is_empty:
                raise StopIteration
            self._cur = self._cur.rest
            return v

        def __repr__(self):  # pragma: no cover
            return repr(self._cur)

    @property
    @abstractmethod
    def is_empty(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def first(self) -> Optional[T]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def rest(self) -> "ISeq[T]":
        raise NotImplementedError()

    @abstractmethod
    def cons(self, elem: T) -> "ISeq[T]":
        raise NotImplementedError()

    def seq(self) -> "ISeq[T]":
        return self

    def _lrepr(self, **kwargs):
        return seq_lrepr(iter(self), "(", ")", **kwargs)

    def __eq__(self, other):
        if self is other:
            return True
        return seq_equals(self, other)

    def __hash__(self):
        return hash(tuple(self))

    def __iter__(self):
        return self._SeqIter(self)


class IType(ABC):
    """IType is a marker interface for types def'ed by `deftype` forms.

    All types created by `deftype` are automatically marked with IType."""

    __slots__ = ()
