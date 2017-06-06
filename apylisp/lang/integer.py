import ast


class Integer(int):
    def to_ast(self):
        return ast.Num(self)


def integer(i):
    return Integer(i)
