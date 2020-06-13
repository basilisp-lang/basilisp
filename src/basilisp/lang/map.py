from builtins import map as pymap
from typing import Callable, Iterable, Mapping, Optional, Tuple, TypeVar, Union, cast

from immutables import Map as _Map

from basilisp.lang.interfaces import (
    ILispObject,
    IMapEntry,
    IPersistentMap,
    IPersistentVector,
    ISeq,
    IWithMeta,
)
from basilisp.lang.obj import map_lrepr as _map_lrepr
from basilisp.lang.seq import sequence
from basilisp.lang.vector import MapEntry
from basilisp.util import partition

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


_ENTRY_SENTINEL = object()


class Map(IPersistentMap[K, V], ILispObject, IWithMeta):
    """Basilisp Map. Delegates internally to a immutables.Map object.
    Do not instantiate directly. Instead use the m() and map() factory
    methods below."""

    __slots__ = ("_inner", "_meta")

    def __init__(
        self,
        members: Union[Mapping[K, V], Iterable[Tuple[K, V]]],
        meta: Optional[IPersistentMap] = None,
    ) -> None:
        self._inner = _Map(members)
        self._meta = meta

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

    def _lrepr(self, **kwargs):
        return _map_lrepr(
            self._inner.items, start="{", end="}", meta=self._meta, **kwargs
        )

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "Map":
        return Map(self._inner, meta=meta)

    def assoc(self, *kvs):
        with self._inner.mutate() as m:
            for k, v in partition(kvs, 2):
                m[k] = v
            return Map(m.finish())

    def contains(self, k):
        return k in self._inner

    def dissoc(self, *ks):
        with self._inner.mutate() as m:
            for k in ks:
                try:
                    del m[k]
                except KeyError:
                    pass
            return Map(m.finish())

    def entry(self, k):
        v = self._inner.get(k, cast("V", _ENTRY_SENTINEL))
        if v is _ENTRY_SENTINEL:
            return None
        return MapEntry.of(k, v)

    def val_at(self, k, default=None):
        return self._inner.get(k, default)

    def update(self, *maps: Mapping[K, V]) -> "Map":
        m: _Map = self._inner.update(*(m.items() for m in maps))
        return Map(m)

    def update_with(
        self, merge_fn: Callable[[V, V], V], *maps: Mapping[K, V]
    ) -> "Map[K, V]":
        with self._inner.mutate() as m:
            for map in maps:
                for k, v in map.items():
                    m.set(k, merge_fn(m[k], v) if k in m else v)
            return Map(m.finish())

    def cons(  # type: ignore[override]
        self,
        *elems: Union[
            IPersistentMap[K, V],
            IMapEntry[K, V],
            IPersistentVector[Union[K, V]],
            Mapping[K, V],
        ],
    ) -> "Map[K, V]":
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
            except (TypeError, ValueError):
                raise ValueError(
                    "Argument to map conj must be another Map or castable to MapEntry"
                )
            else:
                return Map(m.finish(), meta=self.meta)

    @staticmethod
    def empty() -> "Map":
        return m()

    def seq(self) -> ISeq[IMapEntry[K, V]]:
        return sequence(MapEntry.of(k, v) for k, v in self._inner.items())


def map(kvs: Mapping[K, V], meta=None) -> Map[K, V]:  # pylint:disable=redefined-builtin
    """Creates a new map."""
    # For some reason, creating a new `immutables.Map` instance from an existing
    # `basilisp.lang.map.Map` instance causes issues because the `__iter__` returns
    # only the keys rather than tuple of key/value pairs, even though it adheres to
    # the `Mapping` protocol. Passing the `.items()` directly bypasses this problem.
    return Map(kvs.items(), meta=meta)


def m(**kvs) -> Map[str, V]:
    """Creates a new map from keyword arguments."""
    return Map(kvs)


def from_entries(entries: Iterable[MapEntry[K, V]]) -> Map[K, V]:
    with _Map().mutate() as m:  # type: ignore[var-annotated]
        for entry in entries:
            m.set(entry.key, entry.value)
        return Map(m.finish())


def hash_map(*pairs) -> Map:
    entries = pymap(lambda v: MapEntry.of(v[0], v[1]), partition(pairs, 2))
    return from_entries(entries)
