import ast


class Float(float):
    def to_ast(self):
        return ast.Num(self)



def lfloat(f):
    return Float(f)
