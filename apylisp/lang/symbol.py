import ast


class Symbol:
    def __init__(self, name, ns=None, meta=None):
        self._name = name
        self._ns = ns
        self._meta = meta

    @property
    def name(self):
        return self._name

    @property
    def ns(self):
        return self._ns

    @property
    def meta(self):
        return self._meta

    def __str__(self):
        if self._ns is not None:
            return "{ns}/{name}".format(ns=self._ns, name=self._name)
        return "{name}".format(name=self._name)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (self._ns == other._ns and
                self._name == other._name)

    def __hash__(self):
        return hash(str(self))

    def to_ast(self):
        return ast.Name(self._name, ast.Load())


def symbol(name, ns=None):
    """Create a new symbol."""
    return Symbol(name, ns=ns)
