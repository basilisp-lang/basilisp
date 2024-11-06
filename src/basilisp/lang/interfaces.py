import itertools
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence, Sized
from typing import (
    AbstractSet,
    Any,
    Callable,
    Final,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import Self, Unpack

from basilisp.lang.obj import LispObject as _LispObject
from basilisp.lang.obj import PrintSettings, seq_lrepr

T = TypeVar("T")


class IDeref(Generic[T], ABC):
    """``IDeref`` types are reference container types which return their contained
    value via :lpy:fn:`deref` .

    .. seealso::

       :py:class:`IBlockingDeref`"""

    __slots__ = ()

    @abstractmethod
    def deref(self) -> Optional[T]:
        raise NotImplementedError()


class IBlockingDeref(IDeref[T]):
    """``IBlockingDeref`` types are reference container types which may block returning
    their contained value. The contained value can be fetched with a timeout and default
    via :lpy:fn:`deref` .

    .. seealso::

       :py:class:`IDeref`"""

    __slots__ = ()

    @abstractmethod
    def deref(
        self, timeout: Optional[float] = None, timeout_val: Optional[T] = None
    ) -> Optional[T]:
        raise NotImplementedError()


class ICounted(Sized, ABC):
    """``ICounted`` is a marker interface for types can produce their length in
    constant time.

    All the builtin collections are ``ICounted``, except Lists whose length is
    determined by counting all the elements in the list in linear time.

    .. seealso::

       :lpy:fn:`counted?`"""

    __slots__ = ()


class IIndexed(ICounted, ABC):
    """``IIndexed`` is a marker interface for types can be accessed by index.

    Of the builtin collections, only Vectors are ``IIndexed`` . ``IIndexed`` types
    respond ``True`` to the :lpy:fn:`indexed?` predicate.

    .. seealso::

       :lpy:fn:`indexed?`"""

    __slots__ = ()


T_ExceptionInfo = TypeVar("T_ExceptionInfo", bound="IPersistentMap")


class IExceptionInfo(Exception, Generic[T_ExceptionInfo], ABC):
    """``IExceptionInfo`` types are exception types which contain an optional
    :py:class:`IPersistentMap` data element of contextual information about the thrown
    exception.

    .. seealso::

       :lpy:fn:`ex-data`"""

    __slots__ = ()

    @property
    @abstractmethod
    def data(self) -> T_ExceptionInfo:
        raise NotImplementedError()


K = TypeVar("K")
V = TypeVar("V")


class IMapEntry(Generic[K, V], ABC):
    """``IMapEntry`` values are produced :lpy:fn:`seq` ing over any
    :py:class:`IAssociative` (such as a Basilisp map).

    .. seealso::

       :lpy:fn:`key` , :lpy:fn:`val` , :lpy:fn:`map-entry?`"""

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
    """``IMeta`` types can optionally include a map of metadata.

    Persistent data types metadata cannot be mutated, but many of these data types
    also implement :py:class:`IWithMeta` which allows creating a copy of the structure
    with new metadata.

    .. seealso::

       :lpy:fn:`meta`"""

    __slots__ = ()

    @property
    @abstractmethod
    def meta(self) -> Optional["IPersistentMap"]:
        raise NotImplementedError()


class IWithMeta(IMeta):
    """``IWithMeta`` are :py:class:`IMeta` types which can create copies of themselves
    with new metadata.

    .. seealso::

       :lpy:fn:`with-meta`"""

    __slots__ = ()

    @abstractmethod
    def with_meta(self, meta: "Optional[IPersistentMap]") -> Self:
        raise NotImplementedError()


class INamed(ABC):
    """``INamed`` instances are symbolic identifiers with a name and optional
    namespace.

    .. seealso::

       :lpy:fn:`name` , :lpy:fn:`namespace`"""

    __slots__ = ()

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def ns(self) -> Optional[str]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def with_name(cls, name: str, ns: Optional[str] = None) -> Self:
        """Create a new instance of this INamed with `name` and optional `ns`."""
        raise NotImplementedError()


ILispObject = _LispObject


class IReference(IMeta):
    """``IReference`` types are mutable reference containers which allow mutation of
    the associated metadata.

    .. seealso::

       :lpy:fn:`alter-meta!` , :lpy:fn:`reset-meta!`"""

    __slots__ = ()

    @abstractmethod
    def alter_meta(
        self, f: Callable[..., Optional["IPersistentMap"]], *args
    ) -> Optional["IPersistentMap"]:
        raise NotImplementedError()

    @abstractmethod
    def reset_meta(
        self, meta: Optional["IPersistentMap"]
    ) -> Optional["IPersistentMap"]:
        raise NotImplementedError()


RefValidator = Callable[[T], bool]
RefWatchKey = Hashable
RefWatcher = Callable[[RefWatchKey, "IRef", T, T], None]


class IRef(IDeref[T]):
    """``IRef`` types are mutable reference containers which support validation of the
    contained value and watchers which are notified when the contained value changes.

    .. seealso::

       :lpy:fn:`add-watch` , :lpy:fn:`remove-watch` , :lpy:fn:`get-validator` ,
       :lpy:fn:`set-validator!`"""

    __slots__ = ()

    @abstractmethod
    def add_watch(self, k: RefWatchKey, wf: RefWatcher[T]) -> "IReference":
        raise NotImplementedError()

    @abstractmethod
    def remove_watch(self, k: RefWatchKey) -> "IReference":
        raise NotImplementedError()

    @abstractmethod
    def get_validator(self) -> Optional[RefValidator[T]]:
        raise NotImplementedError()

    @abstractmethod
    def set_validator(self, vf: Optional[RefValidator[T]] = None) -> None:
        raise NotImplementedError()


class IReversible(Generic[T]):
    """``IReversible`` types can produce a sequences of their elements in reverse in
    constant time.

    Of the builtin collections, only Vectors are ``IReversible``.

    .. seealso::

       :lpy:fn:`reversible?`"""

    __slots__ = ()

    @abstractmethod
    def rseq(self) -> "ISeq[T]":
        raise NotImplementedError()


class ISeqable(Iterable[T]):
    """``ISeqable`` types can produce sequences of their elements, but are not
    :py:class:`ISeq` .

    All the builtin collections are ``ISeqable``, except Lists which directly implement
    :py:class:`ISeq` .

    .. seealso::

       :ref:`seqs` , :lpy:fn:`seq` , :lpy:fn:`seqable?`"""

    __slots__ = ()

    @abstractmethod
    def seq(self) -> "Optional[ISeq[T]]":
        raise NotImplementedError()


class ISequential(ABC):
    """``ISequential`` is a marker interface for sequential types.

    Lists and Vectors are both considered ``ISequential``.

    .. seealso::

       :lpy:fn:`sequential?`"""

    __slots__ = ()


class ILookup(Generic[K, V], ABC):
    """``ILookup`` types allow accessing contained values by a key or index.

    .. seealso::

       :lpy:fn:`get`"""

    __slots__ = ()

    @abstractmethod
    def val_at(self, k: K, default: Optional[V] = None) -> Optional[V]:
        raise NotImplementedError()


class IPersistentCollection(ISeqable[T]):
    """``IPersistentCollection`` types support both fetching empty variants of an
    existing persistent collection and creating a new collection with additional
    members.

    .. seealso::

       :lpy:fn:`conj` , :lpy:fn:`empty` , :lpy:fn:`coll?`"""

    __slots__ = ()

    @abstractmethod
    def cons(self: Self, *elems: T) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def empty(self) -> "IPersistentCollection[T]":
        raise NotImplementedError()


class IAssociative(ILookup[K, V], IPersistentCollection[IMapEntry[K, V]]):
    """``IAssociative`` types support a persistent data structure variant of
    associative operations.

    .. seealso::

       :lpy:fn:`assoc` , :lpy:fn:`contains?`, :lpy:fn:`find` , :lpy:fn:`associative?`
    """

    __slots__ = ()

    @abstractmethod
    def assoc(self: Self, *kvs) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def contains(self, k: K) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def entry(self, k: K) -> Optional[IMapEntry[K, V]]:
        raise NotImplementedError()


class IPersistentStack(IPersistentCollection[T]):
    """``IPersistentStack`` types support a persistent data structure variant of
    classical stack operations.

    .. seealso::

       :lpy:fn:`pop` , :lpy:fn:`peek`
    """

    __slots__ = ()

    @abstractmethod
    def peek(self) -> Optional[T]:
        raise NotImplementedError()

    @abstractmethod
    def pop(self: Self) -> Self:
        raise NotImplementedError()


T_key = TypeVar("T_key")
V_contra = TypeVar("V_contra", contravariant=True)


class ReduceFunction(Protocol[T, V_contra]):
    @overload
    def __call__(self) -> T: ...

    @overload
    def __call__(self, init: T, val: V_contra) -> T: ...

    def __call__(self, *args, **kwargs): ...


ReduceKVFunction = Callable[[T, T_key, V_contra], T]


class IReduce(ABC):
    """``IReduce`` types define custom implementations of ``reduce``.

    Only vectors are ``IReduce`` by default, providing faster iteration than relying on
    ``seq``.

    .. seealso::

       :lpy:fn:`reduce`
    """

    REDUCE_SENTINEL: Final = object()

    __slots__ = ()

    @overload
    def reduce(self, f: ReduceFunction[T, V_contra]) -> T: ...

    @overload
    def reduce(self, f: ReduceFunction[T, V_contra], init: T) -> T: ...

    @abstractmethod
    def reduce(self, f, init=REDUCE_SENTINEL):
        raise NotImplementedError()


class IReduceKV(ABC):
    """``IReduceKV`` types define custom implementations of ``reduce-kv``.

    Both vectors and maps are ``IReduceKV`` by default, providing faster iteration than
    relying on ``seq``. Maps iterate over the key-value pairs as expected, and vectors
    iterate over the index-item pairs of the vector.

    .. seealso::

       :lpy:fn:`reduce-kv`
    """

    __slots__ = ()

    @abstractmethod
    def reduce_kv(self: Self, f: ReduceKVFunction, init: T):
        raise NotImplementedError()


class IPersistentList(ISequential, IPersistentStack[T]):
    """``IPersistentList`` is a marker interface for a singly-linked list."""

    __slots__ = ()


class IPersistentMap(ICounted, Mapping[K, V], IAssociative[K, V]):
    """``IPersistentMap`` types support creating and modifying persistent maps.

    .. seealso::

       :lpy:fn:`conj` , :lpy:fn:`dissoc` , :lpy:fn:`map?`"""

    __slots__ = ()

    @abstractmethod
    def cons(
        self: Self, *elems: Union[IMapEntry[K, V], "IPersistentMap[K, V]", None]
    ) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def dissoc(self: Self, *ks: K) -> Self:
        raise NotImplementedError()


class IPersistentSet(AbstractSet[T], ICounted, IPersistentCollection[T]):
    """``IPersistentSet`` types support creating and modifying persistent sets.

    .. seealso::

       :lpy:fn:`disj` , :lpy:fn:`set?`"""

    __slots__ = ()

    @abstractmethod
    def disj(self: Self, *elems: T) -> Self:
        raise NotImplementedError()


class IPersistentVector(
    Sequence[T],
    IAssociative[int, T],
    IIndexed,
    IReversible[T],
    ISequential,
    IPersistentStack[T],
):
    """``IPersistentVector`` types support creating and modifying persistent vectors.

    .. seealso::

       :lpy:fn:`vector?`"""

    __slots__ = ()

    @abstractmethod
    def assoc(self: Self, *kvs) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def cons(self: Self, *elems: T) -> Self:  # type: ignore[override]
        raise NotImplementedError()

    @abstractmethod
    def seq(self) -> "Optional[ISeq[T]]":  # type: ignore[override]
        raise NotImplementedError()


T_tcoll_co = TypeVar("T_tcoll_co", bound="ITransientCollection", covariant=True)


# Including ABC as a base seems to cause catastrophic meltdown.
class IEvolveableCollection(Generic[T_tcoll_co]):
    """``IEvolveableCollection`` types support creating transient variants of persistent
    data structures which can be modified efficiently and then returned back into
    persistent data structures once modification is complete.

    .. seealso::

       :lpy:fn:`transient`"""

    @abstractmethod
    def to_transient(self) -> T_tcoll_co:
        raise NotImplementedError()


class ITransientCollection(Generic[T]):
    """``ITransientCollection`` types support efficient modification of otherwise
    persistent data structures.

    .. seealso::

       :lpy:fn:`conj!` , :lpy:fn:`persistent!`"""

    __slots__ = ()

    @abstractmethod
    def cons_transient(self: T_tcoll_co, *elems: T) -> "T_tcoll_co":
        raise NotImplementedError()

    @abstractmethod
    def to_persistent(self: T_tcoll_co) -> "IPersistentCollection[T]":
        raise NotImplementedError()


class ITransientAssociative(ILookup[K, V], ITransientCollection[IMapEntry[K, V]]):
    """``ITransientAssociative`` types are the transient counterpart of
    :py:class:`IAssociative` types.

    .. seealso::

       :lpy:fn:`assoc!` , :lpy:fn:`contains?`, :lpy:fn:`find`"""

    __slots__ = ()

    @abstractmethod
    def assoc_transient(self, *kvs) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def contains_transient(self, k: K) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def entry_transient(self, k: K) -> Optional[IMapEntry[K, V]]:
        raise NotImplementedError()


class ITransientMap(ICounted, ITransientAssociative[K, V]):
    """``ITransientMap`` types are the transient counterpart of
    :py:class:`IPersistentMap` types.

    .. seealso::

       :lpy:fn:`dissoc!`"""

    __slots__ = ()

    @abstractmethod
    def cons_transient(
        self, *elems: Union[IMapEntry[K, V], "IPersistentMap[K, V]", None]
    ) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def dissoc_transient(self, *ks: K) -> Self:
        raise NotImplementedError()


class ITransientSet(ICounted, ITransientCollection[T]):
    """``ITransientSet`` types are the transient counterpart of
    :py:class:`IPersistentSet` types.

    .. seealso::

       :lpy:fn:`disj!`"""

    __slots__ = ()

    @abstractmethod
    def disj_transient(self, *elems: T) -> Self:
        raise NotImplementedError()


T_tvec = TypeVar("T_tvec", bound="ITransientVector")


class ITransientVector(
    ITransientAssociative[int, T],
    IIndexed,
):
    """``ITransientVector`` types are the transient counterpart of
    :py:class:`IPersistentVector` types."""

    __slots__ = ()

    @abstractmethod
    def assoc_transient(self: T_tvec, *kvs) -> T_tvec:
        raise NotImplementedError()

    @abstractmethod
    def cons_transient(self: T_tvec, *elems: T) -> T_tvec:  # type: ignore[override]
        raise NotImplementedError()

    @abstractmethod
    def pop_transient(self: T_tvec) -> T_tvec:
        raise NotImplementedError()


class IRecord(ILispObject):
    """``IRecord`` is a marker interface for types :lpy:form:`def` 'ed by
    :lpy:fn:`defrecord` forms.

    All types created by ``defrecord`` are automatically marked with ``IRecord``.

    .. seealso::

       :ref:`data_types_and_records` , :lpy:fn:`defrecord` , :lpy:fn:`record?`
    """

    __slots__ = ()

    @classmethod
    @abstractmethod
    def create(cls, m: IPersistentMap) -> "IRecord":
        """Class method constructor from an :py:class:`IPersistentMap` instance."""
        raise NotImplementedError()

    def _lrepr(self, **kwargs: Unpack[PrintSettings]) -> str:
        return self._record_lrepr(kwargs)

    @abstractmethod
    def _record_lrepr(self, kwargs: PrintSettings) -> str:
        """Translation method converting Python keyword arguments into a
        Python dict.

        Basilisp methods and functions cannot formally accept Python keyword
        arguments, so this method is called by `_lrepr` with the keyword
        arguments cast to a Python dict."""
        raise NotImplementedError()


def seq_equals(s1: Union["ISeq", ISequential], s2: Any) -> bool:
    """Return True if two sequences contain exactly the same elements in the same
    order. Return False if one sequence is shorter than the other."""
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


class ISeq(ILispObject, IPersistentCollection[T]):
    """``ISeq`` types represent a potentially infinite sequence of elements.

    .. seealso::

       :ref:`seqs` , :lpy:fn:`lazy-seq` , :lpy:fn:`seq` , :lpy:fn:`first` ,
       :lpy:fn:`rest` , :lpy:fn:`next` , :lpy:fn:`second` , :lpy:fn:`seq?` ,
       :lpy:fn:`nfirst` , :lpy:fn:`fnext` , :lpy:fn:`nnext` , :lpy:fn:`empty?` ,
       :lpy:fn:`seq?`
    """

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
    def cons(self, *elem: T) -> "ISeq[T]":
        raise NotImplementedError()

    def seq(self) -> "Optional[ISeq[T]]":
        return self

    def _lrepr(self, **kwargs: Unpack[PrintSettings]):
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
    """``IType`` is a marker interface for types :lpy:form:`def` 'ed by
    :lpy:fn:`deftype` forms.

    All types created by ``deftype`` are automatically marked with ``IType``.

    .. seealso::

       :ref:`data_types_and_records`"""

    __slots__ = ()
