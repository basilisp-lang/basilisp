import basilisp.lang.symbol as sym


class SpecialForm:
    CATCH = sym.symbol("catch")
    DEF = sym.symbol("def")
    DO = sym.symbol("do")
    FINALLY = sym.symbol("finally")
    FN = sym.symbol("fn*")
    IF = sym.symbol("if")
    IMPORT = sym.symbol("import*")
    INTEROP_CALL = sym.symbol(".")
    INTEROP_PROP = sym.symbol(".-")
    LET = sym.symbol("let*")
    LOOP = sym.symbol("loop*")
    QUOTE = sym.symbol("quote")
    RECUR = sym.symbol("recur")
    SET_BANG = sym.symbol("set!")
    THROW = sym.symbol("throw")
    TRY = sym.symbol("try")
    VAR = sym.symbol("var")


AMPERSAND = sym.symbol("&")
