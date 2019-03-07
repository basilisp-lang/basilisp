import basilisp.lang.keyword as kw
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

DEFAULT_COMPILER_FILE_PATH = "NO_SOURCE_PATH"

SYM_DYNAMIC_META_KEY = kw.keyword("dynamic")
SYM_MACRO_META_KEY = kw.keyword("macro")
SYM_NO_WARN_ON_REDEF_META_KEY = kw.keyword("no-warn-on-redef")
SYM_NO_WARN_WHEN_UNUSED_META_KEY = kw.keyword("no-warn-when-unused")
SYM_REDEF_META_KEY = kw.keyword("redef")

COL_KW = kw.keyword("col")
DOC_KW = kw.keyword("doc")
FILE_KW = kw.keyword("file")
LINE_KW = kw.keyword("line")
NAME_KW = kw.keyword("name")
NS_KW = kw.keyword("ns")
