from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from itertools import chain, islice
from typing import Any, Generic, Protocol, TypeVar, cast

from immutables import Map as _Map
from immutables import MapMutation
from typing_extensions import Self, Unpack

from basilisp.lang.interfaces import (
    IEvolveableCollection,
    ILispObject,
    IMapEntry,
    INamed,
    IPersistentMap,
    IPersistentVector,
    IReduceKV,
    ISeq,
    ITransientMap,
    IWithMeta,
    ReduceKVFunction,
)
from basilisp.lang.obj import (
    MAP_PRINT_SEPARATOR,
    SURPASSED_PRINT_LENGTH,
    SURPASSED_PRINT_LEVEL,
    PrintSettings,
    lrepr,
    process_lrepr_kwargs,
)
from basilisp.lang.reduced import Reduced
from basilisp.lang.seq import iterator_sequence
from basilisp.lang.vector import MapEntry
from basilisp.util import partition

T = TypeVar("T")
_K_contra = TypeVar("_K_contra", contravariant=True)
K = TypeVar("K")
V = TypeVar("V")
T_reduce = TypeVar("T_reduce")
_T_inner = TypeVar("_T_inner", bound=Mapping)
_T_transient_persistent = TypeVar("_T_transient_persistent", bound=IPersistentMap)


class _TransientInner(Protocol[_K_contra, V]):
    """Protocol type which matches both immutables' MapMutation type and dict, enabling
    us to support both in the `_AbstractTransientMap` supertype."""

    def __getitem__(self, key: _K_contra) -> V: ...
    def __setitem__(self, key: _K_contra, value: V) -> None: ...
    def __delitem__(self, key: _K_contra) -> None: ...
    def __contains__(self, key: object) -> bool: ...
    def __len__(self) -> int: ...


_T_transient_inner = TypeVar("_T_transient_inner", bound=_TransientInner)


_ENTRY_SENTINEL = object()


class _AbstractTransientMap(
    ITransientMap[K, V], Generic[_T_transient_inner, _T_transient_persistent, K, V]
):
    __slots__ = ("_inner",)

    def __init__(self, inner: _T_transient_inner) -> None:
        self._inner = inner

    def __bool__(self):
        return True

    def __call__(self, key, default=None):
        try:
            return self._inner[key]
        except KeyError:
            return default

    def __contains__(self, item):
        return item in self._inner

    def __eq__(self, other):
        return self is other

    def __len__(self):
        return len(self._inner)

    def assoc_transient(self, *kvs) -> Self:
        for t in partition(kvs, 2):
            # Clojure allows assoc! to have odd numbers of arguments, setting nil for
            # the missing value.
            if len(t) == 2:
                k, v = t
                self._inner[k] = v
            else:
                self._inner[t[0]] = None
        return self

    def contains_transient(self, k: K) -> bool:
        return k in self._inner

    def dissoc_transient(self, *ks: K) -> Self:
        for k in ks:
            try:
                del self._inner[k]
            except KeyError:
                pass
        return self

    def entry_transient(self, k: K) -> IMapEntry[K, V] | None:
        try:
            v = self._inner[k]
        except KeyError:
            return None
        else:
            return MapEntry.of(k, v)

    def val_at(self, k, default=None):
        try:
            return self._inner[k]
        except KeyError:
            return default

    def cons_transient(  # type: ignore[override]
        self,
        *elems: (
            IPersistentMap[K, V]
            | IMapEntry[K, V]
            | IPersistentVector[K | V]
            | Mapping[K, V]
        ),
    ) -> Self:
        try:
            for elem in elems:
                if isinstance(elem, (IPersistentMap, Mapping)):
                    for k, v in elem.items():
                        self._inner[k] = v
                elif isinstance(elem, IMapEntry):
                    self._inner[elem.key] = elem.value
                elif elem is None:
                    continue
                else:
                    entry: IMapEntry[K, V] = MapEntry.from_vec(elem)
                    self._inner[entry.key] = entry.value
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Argument to map conj must be another Map or castable to MapEntry"
            ) from e
        else:
            return self

    @abstractmethod
    def to_persistent(self) -> _T_transient_persistent:
        raise NotImplementedError()


class TransientMap(
    _AbstractTransientMap[MapMutation[K, V], "PersistentMap[K, V]", K, V]
):
    def to_persistent(self) -> "PersistentMap[K, V]":
        return PersistentMap(self._inner.finish())


class _FrozenDict(dict[K, V]):
    """A frozen dict subtype which is used to ensure that transient maps can
    no longer be modified after being made persistent."""

    def _blocked(self, *args, **kwargs):
        raise ValueError("cannot mutate a frozen dict")

    __setitem__ = __delitem__ = __ior__ = _blocked
    update = pop = popitem = clear = setdefault = _blocked


class TransientArrayMap(
    _AbstractTransientMap[dict[K, V], "PersistentArrayMap[K, V]", K, V]
):
    """Transient array map type to correspond to the persistent array map type.

    This type differs in behavior from Clojure in one important way: Clojure's
    transient array map type silently converts to a transient hash map via `assoc!`
    just as the persistent variants do with standard `assoc`, but the Basilisp
    implementation does not. In Clojure, the transient array map is backed by an
    actual array type, so as that array grows the linear scan required to find and
    replace key/value pairs costs more and more. In Basilisp, we get O(1) get and
    set plus insertion order preservation for free from Python's `dict` type.
    Skipping this mid-mutation type conversion enables transient array map types
    to utilize the speed of Python `dicts` without ever having to pay the cost of
    copy-on-write.

    As best as I can tell, this is not a guaranteed or documented behavior in
    Clojure, so if this ends up being costly at runtime for users then we can
    change the behavior later without impacting the public API. Users should
    in general be programming to the public interfaces, not the implementations."""

    def to_persistent(self) -> "PersistentArrayMap[K, V]":
        persistent = PersistentArrayMap(self._inner)
        self._inner = _FrozenDict(self._inner)
        return persistent


def map_lrepr(  # pylint: disable=too-many-locals
    entries: Callable[[], Iterable[tuple[Any, Any]]],
    start: str,
    end: str,
    meta: IPersistentMap | None = None,
    **kwargs: Unpack[PrintSettings],
) -> str:
    """Produce a Lisp representation of an associative collection, bookended
    with the start and end string supplied. The entries argument must be a
    callable which will produce tuples of key-value pairs.

    If the keyword argument print_namespace_maps is True and all keys
    share the same namespace, then print the namespace of the keys at
    the beginning of the map instead of beside the keys.

    The keyword arguments will be passed along to lrepr for the sequence
    elements."""
    print_level = kwargs["print_level"]
    if isinstance(print_level, int) and print_level < 1:
        return SURPASSED_PRINT_LEVEL

    kwargs = process_lrepr_kwargs(**kwargs)

    def check_same_ns():
        """Check whether all keys in entries belong to the same
        namespace. If they do, return the namespace name; otherwise,
        return None.
        """
        nses = set()
        for k, _ in entries():
            if isinstance(k, INamed):
                nses.add(k.ns)
            else:
                nses.add(None)
            if len(nses) > 1:
                break
        return next(iter(nses)) if len(nses) == 1 else None

    ns_name_shared = check_same_ns() if kwargs["print_namespace_maps"] else None

    entries_updated = entries
    if ns_name_shared:

        def entries_ns_remove():
            for k, v in entries():
                yield (k.with_name(k.name), v)

        entries_updated = entries_ns_remove

    kw_items = kwargs.copy()
    kw_items["human_readable"] = False

    def entry_reprs():
        for k, v in entries_updated():
            yield f"{lrepr(k, **kw_items)} {lrepr(v, **kw_items)}"

    trailer = []
    print_dup = kwargs["print_dup"]
    print_length = kwargs["print_length"]
    if not print_dup and isinstance(print_length, int):
        items = list(islice(entry_reprs(), print_length + 1))
        if len(items) > print_length:
            items.pop()
            trailer.append(SURPASSED_PRINT_LENGTH)
    else:
        items = list(entry_reprs())

    seq_lrepr = MAP_PRINT_SEPARATOR.join(items + trailer)

    ns_prefix = ("#:" + ns_name_shared) if ns_name_shared else ""
    if kwargs["print_meta"] and meta:
        kwargs_meta = kwargs
        kwargs_meta["print_level"] = None
        return f"^{lrepr(meta,**kwargs_meta)} {ns_prefix}{start}{seq_lrepr}{end}"

    return f"{ns_prefix}{start}{seq_lrepr}{end}"


@lrepr.register(dict)
def _lrepr_py_dict(o: dict, **kwargs: Unpack[PrintSettings]) -> str:
    return f"#py {map_lrepr(o.items, '{', '}', **kwargs)}"


class _AbstractPersistentMap(
    IPersistentMap[K, V],
    IEvolveableCollection[
        ITransientMap
    ],  # K, V argument suppressed due to astroid bug in <= 4.3.4
    IReduceKV,
    ILispObject,
    IWithMeta,
    Generic[_T_inner, K, V],
):

    __slots__ = ("_hash", "_inner", "_meta")

    def __init__(
        self,
        m: _T_inner,
        meta: IPersistentMap | None = None,
    ) -> None:
        self._inner: _T_inner = m
        self._meta = meta
        self._hash: int | None = None

    @classmethod
    @abstractmethod
    def from_coll(
        cls,
        members: Mapping[K, V] | Iterable[tuple[K, V]],
        meta: IPersistentMap | None = None,
    ) -> Self:
        raise NotImplementedError()

    def __bool__(self):
        return True

    def __call__(self, key, default=None):
        return self._inner.get(key, default)

    def __contains__(self, item):
        return item in self._inner

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Mapping):
            return NotImplemented
        if len(self._inner) != len(other):
            return False
        if type(self) is type(other):
            return self._inner == other._inner
        return all(k in other and v == other[k] for k, v in self._inner.items())

    def __getitem__(self, item):
        return self._inner[item]

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(frozenset(self._inner.items()))
        return self._hash

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)

    def _lrepr(self, **kwargs: Unpack[PrintSettings]):
        return map_lrepr(
            self._inner.items,
            start="{",
            end="}",
            meta=self._meta,
            **kwargs,
        )

    @property
    def meta(self) -> IPersistentMap | None:
        return self._meta

    @abstractmethod
    def with_meta(self, meta: IPersistentMap | None) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def assoc(self, *kvs):
        raise NotImplementedError()

    def contains(self, k):
        return k in self._inner

    @abstractmethod
    def dissoc(self, *ks):
        raise NotImplementedError()

    @abstractmethod
    def empty(self) -> Self:
        raise NotImplementedError()

    def entry(self, k: K) -> MapEntry[K, V] | None:
        v = self._inner.get(k, cast("V", _ENTRY_SENTINEL))
        if v is _ENTRY_SENTINEL:
            return None
        return MapEntry.of(k, v)

    def val_at(self, k, default=None):
        return self._inner.get(k, default)

    @abstractmethod
    def cons(  # type: ignore[override]
        self,
        *elems: (
            IPersistentMap[K, V]
            | IMapEntry[K, V]
            | IPersistentVector[K | V]
            | Mapping[K, V]
        ),
    ) -> Self:
        raise NotImplementedError()

    def seq(self) -> ISeq[IMapEntry[K, V]] | None:
        if len(self._inner) == 0:
            return None
        return iterator_sequence((MapEntry.of(k, v) for k, v in self._inner.items()))

    @abstractmethod
    def to_transient(self) -> ITransientMap[K, V]:
        raise NotImplementedError()

    def reduce_kv(self, f: ReduceKVFunction, init: T_reduce) -> T_reduce:
        for k, v in self._inner.items():
            init = f(init, k, v)
            if isinstance(init, Reduced):
                return init.deref()
        return init


class PersistentMap(_AbstractPersistentMap[_Map[K, V], K, V]):
    """Basilisp Map. Delegates internally to a immutables.Map object.
    Do not instantiate directly. Instead use the m() and map() factory
    methods below."""

    @classmethod
    def from_coll(
        cls,
        members: Mapping[K, V] | Iterable[tuple[K, V]],
        meta: IPersistentMap | None = None,
    ) -> "PersistentMap[K, V]":
        return cls(_Map(members), meta=meta)

    def with_meta(self, meta: IPersistentMap | None) -> "PersistentMap[K, V]":
        return PersistentMap(self._inner, meta=meta)

    def assoc(self, *kvs):
        with self._inner.mutate() as m:
            for k, v in partition(kvs, 2):
                m[k] = v
            return PersistentMap(m.finish(), meta=self._meta)

    def dissoc(self, *ks):
        with self._inner.mutate() as m:
            for k in ks:
                try:
                    del m[k]
                except KeyError:
                    pass
            return PersistentMap(m.finish(), meta=self._meta)

    def empty(self) -> "PersistentMap[K, V]":
        return EMPTY.with_meta(self._meta)

    def cons(  # type: ignore[override, return]
        self,
        *elems: (
            IPersistentMap[K, V]
            | IMapEntry[K, V]
            | IPersistentVector[K | V]
            | Mapping[K, V]
        ),
    ) -> "PersistentMap[K, V]":
        with self._inner.mutate() as m:
            try:
                for elem in elems:
                    if isinstance(elem, (IPersistentMap, Mapping)):
                        m.update(elem.items())
                    elif isinstance(elem, IMapEntry):
                        m.set(elem.key, elem.value)
                    elif elem is None:
                        continue
                    else:
                        entry: IMapEntry[K, V] = MapEntry.from_vec(elem)
                        m.set(entry.key, entry.value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "Argument to map conj must be another Map or castable to MapEntry"
                ) from e
            else:
                return PersistentMap(m.finish(), meta=self.meta)

    def to_transient(self) -> TransientMap[K, V]:
        return TransientMap(self._inner.mutate())


EMPTY: PersistentMap = PersistentMap.from_coll(())


class PersistentArrayMap(_AbstractPersistentMap[dict[K, V], K, V]):
    """Basilisp Array Map. Delegates internally to a Python dict.

    Do not instantiate directly. Instead use the array_map() factory
    methods below.

    Array maps preserve insertion order (since that order is guaranteed by Python
    dicts by default). Updates copy the inner dict before applying changes."""

    _THRESHOLD = 8

    @classmethod
    def from_coll(
        cls,
        members: Mapping[K, V] | Iterable[tuple[K, V]],
        meta: IPersistentMap | None = None,
    ) -> "PersistentArrayMap[K, V]":
        return cls(dict(members), meta=meta)

    def with_meta(self, meta: IPersistentMap | None) -> "PersistentArrayMap[K, V]":
        return PersistentArrayMap(self._inner, meta=meta)

    @classmethod
    def _do_assoc(
        cls,
        d: dict[K, V],
        it: Iterator[tuple[K, V]],
        meta: IPersistentMap | None = None,
    ) -> "PersistentMap[K, V] | PersistentArrayMap[K, V]":
        while True:
            try:
                k, v = next(it)
            except StopIteration:
                break
            else:
                if len(d) >= cls._THRESHOLD and k not in d:
                    return PersistentMap.from_coll(
                        chain.from_iterable([d.items(), ((k, v),), it]), meta=meta
                    )
                d[k] = v

        return PersistentArrayMap(d, meta=meta)

    def assoc(self, *kvs):
        d = dict(self._inner)
        return self._do_assoc(d, iter(partition(kvs, 2)), meta=self._meta)

    def dissoc(self, *ks: K) -> "PersistentArrayMap[K, V]":
        return PersistentArrayMap(
            {k: v for k, v in self._inner.items() if k not in ks}, meta=self._meta
        )

    def empty(self) -> "PersistentArrayMap[K, V]":
        return EMPTY_ARRAY_MAP.with_meta(meta=self._meta)

    def cons(  # type: ignore[override]
        self,
        *elems: (
            IPersistentMap[K, V]
            | IMapEntry[K, V]
            | IPersistentVector[K | V]
            | Mapping[K, V]
        ),
    ) -> "PersistentMap[K, V] | PersistentArrayMap[K, V]":

        def _cons_iter() -> Iterable[tuple[K, V]]:
            try:
                for elem in elems:
                    if isinstance(elem, (IPersistentMap, Mapping)):
                        yield from elem.items()
                    elif isinstance(elem, IMapEntry):
                        yield elem.key, elem.value
                    elif elem is None:
                        continue
                    else:
                        entry: IMapEntry[K, V] = MapEntry.from_vec(elem)
                        yield entry.key, entry.value
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "Argument to map conj must be another Map or castable to MapEntry"
                ) from e

        d = dict(self._inner)
        return self._do_assoc(d, iter(_cons_iter()), meta=self._meta)

    def to_transient(self) -> TransientArrayMap[K, V]:
        return TransientArrayMap(dict(self._inner))


EMPTY_ARRAY_MAP: PersistentArrayMap = PersistentArrayMap.from_coll(())


def map(  # pylint:disable=redefined-builtin
    kvs: Mapping[K, V], meta: IPersistentMap | None = None
) -> PersistentMap[K, V]:
    """Creates a new map."""
    # For some reason, creating a new `immutables.Map` instance from an existing
    # `basilisp.lang.map.PersistentMap` instance causes issues because the `__iter__`
    # returns only the keys rather than tuple of key/value pairs, even though it
    # adheres to the `Mapping` protocol. Passing the `.items()` directly bypasses
    # this problem.
    return PersistentMap.from_coll(kvs.items(), meta=meta)


def m(**kvs) -> PersistentMap[str, V]:
    """Creates a new map from keyword arguments."""
    return PersistentMap.from_coll(kvs)


def array_map(*pairs) -> PersistentArrayMap:
    if len(pairs) % 2:
        raise ValueError("array_map requires an even number of arguments")
    return PersistentArrayMap.from_coll(partition(pairs, 2))


def hash_map(*pairs) -> PersistentMap:
    if len(pairs) % 2:
        raise ValueError("hash_map requires an even number of arguments")
    return PersistentMap.from_coll(partition(pairs, 2))
