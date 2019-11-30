from builtins import map as pymap
from typing import Callable, Iterable, Mapping, Optional, TypeVar, Union, cast

from pyrsistent import PMap, pmap  # noqa # pylint: disable=unused-import

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


class Map(ILispObject, IWithMeta, IPersistentMap[K, V]):
    """Basilisp Map. Delegates internally to a pyrsistent.PMap object.
    Do not instantiate directly. Instead use the m() and map() factory
    methods below."""

    __slots__ = ("_inner", "_meta")

    def __init__(
        self, wrapped: "PMap[K, V]", meta: Optional[IPersistentMap] = None
    ) -> None:
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
        return _map_lrepr(
            self._inner.iteritems, start="{", end="}", meta=self._meta, **kwargs
        )

    def items(self):
        return self._inner.items()

    def keys(self):
        return self._inner.keys()

    def values(self):
        return self._inner.values()

    @property
    def meta(self) -> Optional[IPersistentMap]:
        return self._meta

    def with_meta(self, meta: Optional[IPersistentMap]) -> "Map":
        return Map(self._inner, meta=meta)

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

    def entry(self, k):
        v = self._inner.get(k, cast("V", _ENTRY_SENTINEL))
        if v is _ENTRY_SENTINEL:
            return None
        return MapEntry.of(k, v)

    def val_at(self, k, default=None):
        return self._inner.get(k, default)

    def update(self, *maps: Mapping[K, V]) -> "Map":
        m: PMap = self._inner.update(*maps)
        return Map(m)

    def update_with(
        self, merge_fn: Callable[[V, V], V], *maps: Mapping[K, V]
    ) -> "Map[K, V]":
        m: PMap = self._inner.update_with(merge_fn, *maps)
        return Map(m)

    def cons(  # type: ignore[override]
        self,
        *elems: Union[
            IPersistentMap[K, V],
            IMapEntry[K, V],
            IPersistentVector[Union[K, V]],
            Mapping[K, V],
        ],
    ) -> "Map[K, V]":
        e = self._inner.evolver()
        try:
            for elem in elems:
                if isinstance(elem, (IPersistentMap, Mapping)) and not isinstance(
                    elem, IPersistentVector
                ):
                    # Vectors are handled in the final else block, since we
                    # do not want to treat them as Mapping types for this
                    # particular usage.
                    for k, v in elem.items():
                        e.set(k, v)
                elif isinstance(elem, IMapEntry):
                    e.set(elem.key, elem.value)
                elif elem is None:
                    continue
                else:
                    # This block leniently allows nearly any 2 element sequential
                    # type including Vectors, Python lists, and Python tuples.
                    entry: IMapEntry[K, V] = MapEntry.from_vec(elem)
                    e.set(entry.key, entry.value)
        except (TypeError, ValueError):
            raise ValueError(
                "Argument to map conj must be another Map or castable to MapEntry"
            )
        else:
            return Map(e.persistent(), meta=self.meta)

    @staticmethod
    def empty() -> "Map":
        return m()

    def seq(self) -> ISeq[IMapEntry[K, V]]:
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
