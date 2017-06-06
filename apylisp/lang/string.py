import ast


class String(str):
    def __repr__(self):
        return '"{v}"'.format(v=self)

    def to_ast(self):
        return ast.Str(self)


def string(s):
    return String(s)
