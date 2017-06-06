import ast
import pyrsistent


class Vector:
    def __init__(self, members=None):
        self._members = pyrsistent.pvector(members)

    def __repr__(self):
        return "[{vec}]".format(vec=" ".join(map(repr, self._members)))

    def to_ast(self):
        elems_ast = map(lambda elem: elem.to_ast(), pyrsistent.thaw(self._members))
        return ast.Expr(value=ast.Call(func=ast.Name(id='lang.vector.vector',
                                                     ctx=ast.Load()),
                                       args=[ast.List(elems_ast, ast.Load())]))



def vector(members):
    """Creates a new vector."""
    return Vector(members=members)


def v(*members):
    """Creates a new vector from members."""
    return Vector(members=members)
