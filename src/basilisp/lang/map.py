from builtins import map as pymap
from typing import Callable, Dict, Iterable, Mapping, Sequence, TypeVar, Union

from pyrsistent import (  # noqa # pylint: disable=unused-import
    PMap,
    PVector,
    pmap,
    pvector,
)

from basilisp.lang.interfaces import (
    IMapEntry,
    IMeta,
    IPersistentCollection,
    IPersistentMap,
    ISeq,
    ISeqable,
)
from basilisp.lang.obj import LispObject
from basilisp.lang.seq import sequence
from basilisp.lang.vector import Vector
from basilisp.util import partition

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class MapEntry(IMapEntry[K, V], Vector[Union[K, V]]):  # type: ignore
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


class Map(
    IPersistentCollection[MapEntry[K, V]],
    LispObject,
    IMeta,
    ISeqable[MapEntry[K, V]],
    IPersistentMap[K, V],
):
    """Basilisp Map. Delegates internally to a pyrsistent.PMap object.
    Do not instantiate directly. Instead use the m() and map() factory
    methods below."""

    __slots__ = ("_inner", "_meta")

    def __init__(self, wrapped: "PMap[K, V]", meta=None) -> None:
        self._inner = wrapped
        self._meta = meta

    def __call__(self, key, default=None):
        return self._inner.get(key, default)

    def __contains__(self, item):
        return item in self._inner

    def __eq__(self, other):
        return self._inner == other

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def __getitem__(self, item):
        return self._inner[item]

    def __hash__(self):
        return hash(self._inner)

    def __iter__(self):
        for k, v in self._inner.iteritems():
            yield MapEntry.of(k, v)

    def __len__(self):
        return len(self._inner)

    def _lrepr(self, **kwargs):
        return LispObject.map_lrepr(
            self._inner.iteritems, start="{", end="}", meta=self._meta, **kwargs
        )

    def items(self):
        return self._inner.items()

    def keys(self):
        return self._inner.keys()

    def values(self):
        return self._inner.values()

    @property
    def meta(self):
        return self._meta

    def with_meta(self, meta: "IPersistentMap") -> "Map":
        new_meta = meta if self._meta is None else self._meta.update(meta)
        return Map(self._inner, meta=new_meta)

    def assoc(self, *kvs):
        m = self._inner.evolver()
        for k, v in partition(kvs, 2):
            m[k] = v
        return Map(m.persistent())

    def contains(self, k):
        if k in self._inner:
            return True
        return False

    def dissoc(self, *ks):
        m = self._inner.evolver()
        for k in ks:
            try:
                del m[k]
            except KeyError:
                pass
        return Map(m.persistent())

    def entry(self, k, default=None):
        return self._inner.get(k, default)

    def update(self, *maps: Mapping[K, V]) -> "Map":
        m: PMap = self._inner.update(*maps)
        return Map(m)

    def update_with(
        self, merge_fn: Callable[[V, V], V], *maps: Mapping[K, V]
    ) -> "Map[K, V]":
        m: PMap = self._inner.update_with(merge_fn, *maps)
        return Map(m)

    def cons(
        self,
        *elems: Union["Map[K, V]", Dict[K, V], MapEntry[K, V], Vector[Union[K, V]]],
    ) -> "Map[K, V]":
        e = self._inner.evolver()
        try:
            for elem in elems:
                if isinstance(elem, Map):
                    for entry in elem:
                        e.set(entry.key, entry.value)
                elif isinstance(elem, dict):
                    for k, v in elem.items():
                        e.set(k, v)
                elif isinstance(elem, MapEntry):
                    e.set(elem.key, elem.value)
                else:
                    entry = MapEntry.from_vec(elem)
                    e.set(entry.key, entry.value)
            return Map(e.persistent(), meta=self.meta)
        except (TypeError, ValueError):
            raise ValueError(
                "Argument to map conj must be another Map or castable to MapEntry"
            )

    @staticmethod
    def empty() -> "Map":
        return m()

    def seq(self) -> ISeq[MapEntry[K, V]]:
        return sequence(self)


def map(kvs: Mapping[K, V], meta=None) -> Map[K, V]:  # pylint:disable=redefined-builtin
    """Creates a new map."""
    return Map(pmap(initial=kvs), meta=meta)


def m(**kvs) -> Map[str, V]:
    """Creates a new map from keyword arguments."""
    return Map(pmap(initial=kvs))


def from_entries(entries: Iterable[MapEntry[K, V]]) -> Map[K, V]:
    m = pmap().evolver()  # type: ignore
    for entry in entries:
        m.set(entry.key, entry.value)
    return Map(m.persistent())


def hash_map(*pairs) -> Map:
    entries = pymap(lambda v: MapEntry.of(v[0], v[1]), partition(pairs, 2))
    return from_entries(entries)
