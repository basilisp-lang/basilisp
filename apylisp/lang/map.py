import pyrsistent


class Map:
    def __init__(self, kvs=None):
        self._members = pyrsistent.pmap(kvs)

    def __repr__(self):
        kvs = ["{k} {v}".format(k=repr(k), v=repr(v))
              for k, v in self._members.items()]
        return "{{{kvs}}}".format(kvs=" ".join(kvs))


def map(kvs):
    """Creates a new map."""
    return Map(kvs=kvs)


def m(**kvs):
    """Creates a new map from keyword arguments."""
    return Map(kvs=kvs)
