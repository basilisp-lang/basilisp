from pyrsistent import pmap, PMap
from wrapt import ObjectProxy

import basilisp.lang.vector as vec
from basilisp.lang.meta import Meta
from basilisp.lang.util import lrepr


class MapEntry(ObjectProxy):
    def __init__(self, wrapped):
        super().__init__(wrapped)
        try:
            if not len(wrapped) == 2:
                raise ValueError("Vector arg to map conj must be a pair")
        except TypeError as e:
            raise TypeError("Cannot make map entry from ") from e

    def __repr__(self):
        return lrepr(self.__wrapped__)

    @property
    def key(self):
        return self[0]

    @property
    def value(self):
        return self[1]

    @staticmethod
    def of(k, v):
        return MapEntry(vec.v(k, v))

    @staticmethod
    def from_vec(v):
        return MapEntry(vec.vector(v))


class Map(ObjectProxy, Meta):
    __slots__ = ('_self_meta', )

    def __init__(self, wrapped: PMap, meta=None) -> None:
        super(Map, self).__init__(wrapped)
        self._self_meta = meta

    def __repr__(self):
        kvs = [
            "{k} {v}".format(k=lrepr(k), v=lrepr(v))
            for k, v in self.iteritems()
        ]
        return "{{{kvs}}}".format(kvs=" ".join(kvs))

    def __call__(self, key, default=None):
        return self.get(key, default)

    @property
    def meta(self):
        return self._self_meta

    def with_meta(self, meta) -> "Map":
        new_meta = meta if self._self_meta is None else self._self_meta.update(
            meta)
        return map(self.__wrapped__, meta=new_meta)

    def update(self, *maps) -> "Map":
        m: PMap = self.__wrapped__.update(*maps)
        return map(m)

    def __iter__(self):
        for k, v in self.iteritems():
            yield MapEntry.of(k, v)

    def _conj(self, entry: MapEntry) -> "Map":
        try:
            return Map(self.set(entry.key, entry.value), meta=self.meta)
        except AttributeError:
            raise ValueError(
                "Argument to map conj must be castable to MapEntry")

    def conj(self, entry: MapEntry) -> "Map":
        try:
            return Map(self.set(entry.key, entry.value), meta=self.meta)
        except AttributeError:
            return self._conj(MapEntry.from_vec(entry))

    def empty(self) -> "Map":
        return m()


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
    return m.persistent()
