from builtins import map as pymap
from collections.abc import Iterable, Mapping
from itertools import islice
from typing import Any, Callable, Optional, TypeVar, Union, cast

from immutables import Map as _Map
from immutables import MapMutation
from typing_extensions import Unpack

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
    PRINT_SEPARATOR,
    SURPASSED_PRINT_LENGTH,
    SURPASSED_PRINT_LEVEL,
    PrintSettings,
    lrepr,
    process_lrepr_kwargs,
)
from basilisp.lang.reduced import Reduced
from basilisp.lang.seq import sequence
from basilisp.lang.vector import MapEntry
from basilisp.util import partition

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
T_reduce = TypeVar("T_reduce")

_ENTRY_SENTINEL = object()


class TransientMap(ITransientMap[K, V]):
    __slots__ = ("_inner",)

    def __init__(self, evolver: "MapMutation[K, V]") -> None:
        self._inner = evolver

    def __bool__(self):
        return True

    def __call__(self, key, default=None):
        return self._inner.get(key, default)

    def __contains__(self, item):
        return item in self._inner

    def __eq__(self, other):
        return self is other

    def __len__(self):
        return len(self._inner)

    def assoc_transient(self, *kvs) -> "TransientMap":
        for k, v in partition(kvs, 2):
            self._inner[k] = v
        return self

    def contains_transient(self, k: K) -> bool:
        return k in self._inner

    def dissoc_transient(self, *ks: K) -> "TransientMap[K, V]":
        for k in ks:
            try:
                del self._inner[k]
            except KeyError:
                pass
        return self

    def entry_transient(self, k: K) -> Optional[IMapEntry[K, V]]:
        v = self._inner.get(k, cast("V", _ENTRY_SENTINEL))
        if v is _ENTRY_SENTINEL:
            return None
        return MapEntry.of(k, v)

    def val_at(self, k, default=None):
        return self._inner.get(k, default)

    def cons_transient(  # type: ignore[override]
        self,
        *elems: Union[
            IPersistentMap[K, V],
            IMapEntry[K, V],
            IPersistentVector[Union[K, V]],
            Mapping[K, V],
        ],
    ) -> "TransientMap[K, V]":
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

    def to_persistent(self) -> "PersistentMap[K, V]":
        return PersistentMap(self._inner.finish())


def map_lrepr(  # pylint: disable=too-many-locals
    entries: Callable[[], Iterable[tuple[Any, Any]]],
    start: str,
    end: str,
    meta: Optional[IPersistentMap] = None,
    **kwargs: Unpack[PrintSettings],
) -> str:
    """Produce a Lisp representation of an associative collection, bookended
    with the start and end string supplied. The entries argument must be a
    callable which will produce tuples of key-value pairs.

    If the keyword argument print_namespace_maps is True and all keys
    share the same namespace, then print the namespace of the keys at
    the beginning of the map instead of beside the keys.

    The keyword arguments will be passed along to lrepr for the sequence
    elements.

    """
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

    seq_lrepr = PRINT_SEPARATOR.join(items + trailer)

    ns_prefix = ("#:" + ns_name_shared) if ns_name_shared else ""
    if kwargs["print_meta"] and meta:
        kwargs_meta = kwargs
        kwargs_meta["print_level"] = None
        return f"^{lrepr(meta,**kwargs_meta)} {ns_prefix}{start}{seq_lrepr}{end}"

    return f"{ns_prefix}{start}{seq_lrepr}{end}"


@lrepr.register(dict)
def _lrepr_py_dict(o: dict, **kwargs: Unpack[PrintSettings]) -> str:
    return f"#py {map_lrepr(o.items, '{', '}', **kwargs)}"


class PersistentMap(
    IPersistentMap[K, V],
    IEvolveableCollection[TransientMap],
    IReduceKV,
    ILispObject,
    IWithMeta,
):
    """Basilisp Map. Delegates internally to a immutables.Map object.
    Do not instantiate directly. Instead use the m() and map() factory
    methods below."""

    __slots__ = ("_inner", "_meta")

    def __init__(
        self,
        m: "_Map[K, V]",
        meta: Optional[IPersistentMap] = None,
    ) -> None:
        self._inner = m
        self._meta = meta

    @classmethod
    def from_coll(
        cls,
        members: Union[Mapping[K, V], Iterable[tuple[K, V]]],
        meta: Optional[IPersistentMap] = None,
    ) -> "PersistentMap[K, V]":
        return PersistentMap(_Map(members), meta=meta)

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
        return self._inner == other

    def __getitem__(self, item):
        return self._inner[item]

    def __hash__(self):
        return hash(self._inner)

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
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "PersistentMap":
        return PersistentMap(self._inner, meta=meta)

    def assoc(self, *kvs):
        with self._inner.mutate() as m:
            for k, v in partition(kvs, 2):
                m[k] = v
            return PersistentMap(m.finish(), meta=self._meta)

    def contains(self, k):
        return k in self._inner

    def dissoc(self, *ks):
        with self._inner.mutate() as m:
            for k in ks:
                try:
                    del m[k]
                except KeyError:
                    pass
            return PersistentMap(m.finish(), meta=self._meta)

    def entry(self, k):
        v = self._inner.get(k, cast("V", _ENTRY_SENTINEL))
        if v is _ENTRY_SENTINEL:
            return None
        return MapEntry.of(k, v)

    def val_at(self, k, default=None):
        return self._inner.get(k, default)

    def update(self, *maps: Mapping[K, V]) -> "PersistentMap":
        m: _Map = self._inner.update(*(m.items() for m in maps))
        return PersistentMap(m, meta=self._meta)

    def update_with(  # type: ignore[return]
        self, merge_fn: Callable[[V, V], V], *maps: Mapping[K, V]
    ) -> "PersistentMap[K, V]":
        with self._inner.mutate() as m:
            for map in maps:
                for k, v in map.items():
                    m.set(k, merge_fn(m[k], v) if k in m else v)
            return PersistentMap(m.finish(), meta=self._meta)

    def cons(  # type: ignore[override, return]
        self,
        *elems: Union[
            IPersistentMap[K, V],
            IMapEntry[K, V],
            IPersistentVector[Union[K, V]],
            Mapping[K, V],
        ],
    ) -> "PersistentMap[K, V]":
        with self._inner.mutate() as m:
            try:
                for elem in elems:
                    if isinstance(elem, (IPersistentMap, Mapping)):
                        for k, v in elem.items():
                            m.set(k, v)
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

    def empty(self) -> "PersistentMap":
        return EMPTY.with_meta(self._meta)

    def seq(self) -> Optional[ISeq[IMapEntry[K, V]]]:
        if len(self._inner) == 0:
            return None
        return sequence(MapEntry.of(k, v) for k, v in self._inner.items())

    def to_transient(self) -> TransientMap[K, V]:
        return TransientMap(self._inner.mutate())

    def reduce_kv(self, f: ReduceKVFunction, init: T_reduce) -> T_reduce:
        for k, v in self._inner.items():
            init = f(init, k, v)
            if isinstance(init, Reduced):
                return init.deref()
        return init


EMPTY: PersistentMap = PersistentMap.from_coll(())


def map(  # pylint:disable=redefined-builtin
    kvs: Mapping[K, V], meta: Optional[IPersistentMap] = None
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


def from_entries(entries: Iterable[MapEntry[K, V]]) -> PersistentMap[K, V]:  # type: ignore[return]
    with _Map().mutate() as m:  # type: ignore[var-annotated]
        for entry in entries:
            m.set(entry.key, entry.value)
        return PersistentMap(m.finish())


def hash_map(*pairs) -> PersistentMap:
    entries = pymap(lambda v: MapEntry.of(v[0], v[1]), partition(pairs, 2))
    return from_entries(entries)
