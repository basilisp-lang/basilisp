import pyrsistent
import wrapt
import basilisp.lang.vector as vec
from basilisp.lang.util import lrepr


class MapEntry(wrapt.ObjectProxy):
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


def map_entry(k, v):
    return MapEntry(vec.v(k, v))


class Map(wrapt.ObjectProxy):
    __slots__ = ('_self_meta', )

    def __init__(self, wrapped, meta=None):
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

    def with_meta(self, meta):
        new_meta = meta if self._self_meta is None else self._self_meta.update(
            meta)
        return map(self.__wrapped__, meta=new_meta)

    def update(self, *maps):
        m = self.__wrapped__.update(*maps)
        return map(m)

    def __iter__(self):
        for k, v in self.iteritems():
            yield map_entry(k, v)

    def _conj(self, entry):
        try:
            return self.set(entry.key, entry.value)
        except AttributeError:
            raise ValueError(
                "Argument to map conj must be castable to MapEntry")

    def conj(self, entry):
        try:
            return self.set(entry.key, entry.value)
        except AttributeError:
            return self._conj(self, map_entry(entry))

    def empty(self):
        return m()


def map(kvs, meta=None):
    """Creates a new map."""
    return Map(pyrsistent.pmap(initial=kvs), meta=meta)


def m(**kvs):
    """Creates a new map from keyword arguments."""
    return Map(pyrsistent.pmap(initial=kvs))


def from_entries(entries):
    m = pyrsistent.pmap().evolver()
    for entry in entries:
        m.set(entry.key, entry.value)
    return m.persistent()
