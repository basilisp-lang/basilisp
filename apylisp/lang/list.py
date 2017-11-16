import pyrsistent


class List:
    def __init__(self, members=()):
        self._members = pyrsistent.plist(members)

    def __repr__(self):
        return "({list})".format(list=" ".join(map(repr, self._members)))

    def __eq__(self, other):
        return self._members == self._members


def list(members):
    """Creates a new list."""
    return List(members=members)


def l(*members):
    """Creates a new list from members."""
    return List(members=members)
