from collections.abc import Iterable, Sequence
from functools import total_ordering
from typing import Optional, TypeVar, Union, cast, overload

from pyrsistent import PVector, pvector  # noqa # pylint: disable=unused-import
from pyrsistent.typing import PVectorEvolver
from typing_extensions import Unpack

from basilisp.lang.interfaces import (
    IEvolveableCollection,
    ILispObject,
    IMapEntry,
    IPersistentMap,
    IPersistentVector,
    IReduce,
    IReduceKV,
    ISeq,
    ITransientVector,
    IWithMeta,
    ReduceFunction,
    ReduceKVFunction,
    seq_equals,
)
from basilisp.lang.obj import PrintSettings
from basilisp.lang.obj import seq_lrepr as _seq_lrepr
from basilisp.lang.reduced import Reduced
from basilisp.lang.seq import sequence
from basilisp.util import partition

T = TypeVar("T")

T_reduce = TypeVar("T_reduce")
V_contra = TypeVar("V_contra", contravariant=True)


class TransientVector(ITransientVector[T]):
    __slots__ = ("_inner",)

    def __init__(self, wrapped: "PVectorEvolver[T]") -> None:
        self._inner = wrapped

    def __bool__(self):
        return True

    def __contains__(self, item):
        return item in self._inner

    def __eq__(self, other):
        return self is other

    def __len__(self):
        return len(self._inner)

    def cons_transient(self, *elems: T) -> "TransientVector[T]":  # type: ignore[override]
        for elem in elems:
            self._inner.append(elem)
        return self

    def assoc_transient(self, *kvs: T) -> "TransientVector[T]":
        for i, v in cast("Sequence[tuple[int, T]]", partition(kvs, 2)):
            self._inner.set(i, v)
        return self

    def contains_transient(self, k: int) -> bool:
        return 0 <= k < len(self._inner)

    def entry_transient(self, k: int) -> Optional[IMapEntry[int, T]]:
        try:
            return MapEntry.of(k, self._inner[k])
        except IndexError:
            return None

    def val_at(self, k: int, default=None):
        try:
            return self._inner[k]
        except IndexError:
            return default

    def pop_transient(self) -> "TransientVector[T]":
        if len(self) == 0:
            raise IndexError("Cannot pop an empty vector")
        del self._inner[-1]
        return self

    def to_persistent(self) -> "PersistentVector[T]":
        return PersistentVector(self._inner.persistent())


@total_ordering
class PersistentVector(
    IReduce,
    IReduceKV,
    IPersistentVector[T],
    IEvolveableCollection[TransientVector],
    ILispObject,
    IWithMeta,
):
    """Basilisp Vector. Delegates internally to a pyrsistent.PVector object.
    Do not instantiate directly. Instead use the v() and vec() factory
    methods below."""

    __slots__ = ("_inner", "_meta")

    def __init__(
        self, wrapped: "PVector[T]", meta: Optional[IPersistentMap] = None
    ) -> None:
        self._inner = wrapped
        self._meta = meta

    def __bool__(self):
        return True

    def __contains__(self, item):
        return item in self._inner

    def __eq__(self, other):
        if self is other:
            return True
        if hasattr(other, "__len__") and len(self) != len(other):
            return False
        return seq_equals(self, other)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return PersistentVector(self._inner[item])
        return self._inner[item]

    def __hash__(self):
        return hash(self._inner)

    def __iter__(self):
        yield from self._inner

    def __len__(self):
        return len(self._inner)

    def __call__(self, k: int, default: Optional[T] = None) -> Optional[T]:
        return self.val_at(k, default=default)

    def __lt__(self, other):
        """Return true if the `self` vector is shorter than the
        `other` vector, or the first unequal element in `self` when
        iterating from left to right is less than the corresponding
        `other` element.

        This is to support the comparing and sorting operations of
        vectors in Clojure."""

        if other is None:  # pragma: no cover
            return False
        if not isinstance(other, PersistentVector):
            return NotImplemented
        if len(self) != len(other):
            return len(self) < len(other)

        for x, y in zip(self, other):
            if x < y:
                return True
            elif y < x:
                return False
        return False

    def _lrepr(self, **kwargs: Unpack[PrintSettings]) -> str:
        return _seq_lrepr(self._inner, "[", "]", meta=self._meta, **kwargs)

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "PersistentVector[T]":
        return vector(self._inner, meta=meta)

    def cons(self, *elems: T) -> "PersistentVector[T]":  # type: ignore[override]
        e = self._inner.evolver()
        for elem in elems:
            e.append(elem)
        return PersistentVector(e.persistent(), meta=self.meta)

    def assoc(self, *kvs: T) -> "PersistentVector[T]":
        return PersistentVector(self._inner.mset(*kvs), meta=self._meta)  # type: ignore[arg-type]

    def contains(self, k: int) -> bool:
        return 0 <= k < len(self._inner)

    def entry(self, k: int) -> Optional[IMapEntry[int, T]]:
        try:
            return MapEntry.of(k, self._inner[k])
        except IndexError:
            return None

    def val_at(self, k: int, default: Optional[T] = None) -> Optional[T]:
        try:
            return self._inner[k]
        except (IndexError, TypeError):
            return default

    def empty(self) -> "PersistentVector[T]":
        return EMPTY.with_meta(self._meta)

    def seq(self) -> Optional[ISeq[T]]:  # type: ignore[override]
        if len(self._inner) == 0:
            return None
        return sequence(self)

    def peek(self) -> Optional[T]:
        if len(self) == 0:
            return None
        return self[-1]

    def pop(self) -> "PersistentVector[T]":
        if len(self) == 0:
            raise IndexError("Cannot pop an empty vector")
        return self[:-1]

    def rseq(self) -> ISeq[T]:
        return sequence(reversed(self))

    def to_transient(self) -> TransientVector:
        return TransientVector(self._inner.evolver())

    @overload
    def reduce(self, f: ReduceFunction[T_reduce, V_contra]) -> T_reduce: ...

    @overload
    def reduce(  # pylint: disable=arguments-differ
        self, f: ReduceFunction[T_reduce, V_contra], init: T_reduce
    ) -> T_reduce: ...

    def reduce(self, f, init=IReduce.REDUCE_SENTINEL):
        if init is IReduce.REDUCE_SENTINEL:
            if len(self) == 0:
                return f()
            else:
                init = self._inner[0]
                for item in self._inner[1:]:
                    init = f(init, item)
                    if isinstance(init, Reduced):
                        return init.deref()
        else:
            for item in self._inner:
                init = f(init, item)
                if isinstance(init, Reduced):
                    return init.deref()
        return init

    def reduce_kv(self, f: ReduceKVFunction, init: T_reduce) -> T_reduce:
        for idx, item in enumerate(self._inner):
            init = f(init, idx, item)
            if isinstance(init, Reduced):
                return init.deref()
        return init


K = TypeVar("K")
V = TypeVar("V")


class MapEntry(IMapEntry[K, V], PersistentVector[Union[K, V]]):
    __slots__ = ()

    def __init__(self, wrapped: "PVector[Union[K, V]]") -> None:
        try:
            if not len(wrapped) == 2:
                raise ValueError("Vector arg to map conj must be a pair")
        except TypeError as e:
            raise TypeError(f"Cannot make map entry from {type(wrapped)}") from e

        super().__init__(wrapped)

    @property
    def key(self) -> K:
        return self[0]

    @property
    def value(self) -> V:
        return self[1]

    @staticmethod
    def of(k: K, v: V) -> "MapEntry[K, V]":
        return MapEntry(pvector([k, v]))

    @staticmethod
    def from_vec(v: Sequence[Union[K, V]]) -> "MapEntry[K, V]":
        return MapEntry(pvector(v))


EMPTY: PersistentVector = PersistentVector(pvector(()))


def vector(
    members: Iterable[T], meta: Optional[IPersistentMap] = None
) -> PersistentVector[T]:
    """Creates a new vector."""
    return PersistentVector(pvector(members), meta=meta)


def v(*members: T, meta: Optional[IPersistentMap] = None) -> PersistentVector[T]:
    """Creates a new vector from members."""
    return PersistentVector(pvector(members), meta=meta)
