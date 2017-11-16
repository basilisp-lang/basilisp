import ast
import pyrsistent


class Set:
    def __init__(self, members=()):
        self._members = pyrsistent.pset(members)

    def __repr__(self):
        return "#\{{set}\}".format(set=" ".join(map(repr, self._members)))

    def __eq__(self, other):
        return self._members == other._members

    def to_ast(self):
        elems_ast = map(lambda elem: elem.to_ast(),
                        pyrsistent.thaw(self._members))
        return ast.Expr(value=ast.Call(
            func=ast.Name(id='lang.set.set', ctx=ast.Load()),
            args=[ast.List(elems_ast, ast.Load())]))


def set(members):
    """Creates a new set."""
    return Set(members=members)


def s(*members):
    """Creates a new set from members."""
    return Set(members=members)
