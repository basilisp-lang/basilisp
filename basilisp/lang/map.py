from collections import Sequence
from typing import Any, Optional

from functional import seq
from pyrsistent import pmap, PMap

import basilisp.lang.vector as vec
from basilisp.lang.meta import Meta
from basilisp.lang.seq import Seqable, sequence, Seq
from basilisp.lang.util import lrepr


class MapEntry:
    __slots__ = ('_inner',)

    def __init__(self, wrapped: vec.Vector) -> None:
        try:
            if not len(wrapped) == 2:
                raise ValueError("Vector arg to map conj must be a pair")
        except TypeError as e:
            raise TypeError(f"Cannot make map entry from {type(wrapped)}") from e

        self._inner = wrapped

    def __repr__(self):
        return lrepr(self._inner)

    def __eq__(self, other):
        return self._inner == other

    def __getitem__(self, item):
        return self._inner[item]

    def __hash__(self):
        return hash(self._inner)

    def __iter__(self):
        yield from self._inner

    def __len__(self):
        return len(self._inner)

    @property
    def key(self):
        return self[0]

    @property
    def value(self):
        return self[1]

    @staticmethod
    def of(k, v) -> "MapEntry":
        return MapEntry(vec.v(k, v))

    @staticmethod
    def from_vec(v: Sequence) -> "MapEntry":
        return MapEntry(vec.vector(v))


class Map(Meta, Seqable):
    """Basilisp Map. Delegates internally to a pyrsistent.PMap object.

    Do not instantiate directly. Instead use the m() and map() factory
    methods below."""
    __slots__ = ('_inner', '_meta',)

    def __init__(self, wrapped: PMap, meta=None) -> None:
        self._inner = wrapped
        self._meta = meta

    def __repr__(self):
        kvs = [
            "{k} {v}".format(k=lrepr(k), v=lrepr(v))
            for k, v in self._inner.iteritems()
        ]
        return "{{{kvs}}}".format(kvs=" ".join(kvs))

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

    def items(self):
        return self._inner.items()

    def keys(self):
        return self._inner.keys()

    def values(self):
        return self._inner.values()

    @property
    def meta(self) -> "Optional[Map]":
        return self._meta

    def with_meta(self, meta: "Map") -> "Map":
        new_meta = meta if self._meta is None else self._meta.update(
            meta)
        return map(self._inner, meta=new_meta)

    def discard(self, *ks) -> "Map":
        m: PMap = self._inner
        for k in ks:
            m = m.discard(k)
        return map(m)

    def update(self, *maps) -> "Map":
        m: PMap = self._inner.update(*maps)
        return map(m)

    def _conj(self, entry: MapEntry) -> "Map":
        try:
            return Map(self._inner.set(entry.key, entry.value), meta=self.meta)
        except AttributeError:
            raise ValueError(
                "Argument to map conj must be castable to MapEntry")

    def conj(self, entry: Any) -> "Map":
        try:
            return Map(self._inner.set(entry.key, entry.value), meta=self.meta)
        except AttributeError:
            return self._conj(MapEntry.from_vec(entry))

    @staticmethod
    def empty() -> "Map":
        return m()

    def seq(self) -> Seq:
        return sequence(self)


def map(kvs, meta=None) -> Map:
    """Creates a new map."""
    return Map(pmap(initial=kvs), meta=meta)


def m(**kvs) -> Map:
    """Creates a new map from keyword arguments."""
    return Map(pmap(initial=kvs))


def from_entries(entries):
    m = pmap().evolver()
    for entry in entries:
        m.set(entry.key, entry.value)
    return Map(m.persistent())


def hash_map(*pairs) -> Map:
    entries = seq(pairs).grouped(2).map(lambda v: MapEntry.of(v[0], v[1])).to_list()
    return from_entries(entries)
