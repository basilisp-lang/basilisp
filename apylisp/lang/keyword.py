import ast


__INTERN = {}


def _as_str(name, ns=None):
    if ns is not None:
        return "{ns}/{name}".format(ns=ns, name=name)
    return "{name}".format(name=name)


class Keyword:
    def __init__(self, name, ns=None):
        self._name = name
        self._ns = ns

    @property
    def name(self):
        return self._name

    @property
    def ns(self):
        return self._ns

    def __str__(self):
        return _as_str(self.name, ns=self.ns)

    def __repr__(self):
        return ":{me}".format(me=str(self))

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(str(self))

    def to_ast(self):
        return ast.Expr(value=ast.Call(func=ast.Name(id='lang.keyword.keyword',
                                                     ctx=ast.Load()),
                                       args=[ast.Str(self._name)],
                                       keywords=[ast.keyword(arg='ns', value=ast.Str(self._ns))]))


def keyword(name, ns=None):
    """Create a new keyword."""
    s = _as_str(name, ns=ns)
    if s in __INTERN:
        return __INTERN[s]
    kw = Keyword(name, ns=ns)
    __INTERN[s] = kw
    return kw
