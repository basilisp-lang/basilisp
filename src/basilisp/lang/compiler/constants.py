from basilisp.lang import keyword as kw
from basilisp.lang import symbol as sym


class SpecialForm:
    AWAIT = sym.symbol("await")
    CATCH = sym.symbol("catch")
    DEF = sym.symbol("def")
    DEFTYPE = sym.symbol("deftype*")
    DO = sym.symbol("do")
    FINALLY = sym.symbol("finally")
    FN = sym.symbol("fn*")
    IF = sym.symbol("if")
    IMPORT = sym.symbol("import*")
    INTEROP_CALL = sym.symbol(".")
    INTEROP_PROP = sym.symbol(".-")
    LET = sym.symbol("let*")
    LETFN = sym.symbol("letfn*")
    LOOP = sym.symbol("loop*")
    QUOTE = sym.symbol("quote")
    RECUR = sym.symbol("recur")
    REIFY = sym.symbol("reify*")
    REQUIRE = sym.symbol("require*")
    SET_BANG = sym.symbol("set!")
    THROW = sym.symbol("throw")
    TRY = sym.symbol("try")
    VAR = sym.symbol("var")


AMPERSAND = sym.symbol("&")

DEFAULT_COMPILER_FILE_PATH = "NO_SOURCE_PATH"

SYM_ABSTRACT_META_KEY = kw.keyword("abstract")
SYM_ASYNC_META_KEY = kw.keyword("async")
SYM_KWARGS_META_KEY = kw.keyword("kwargs")
SYM_PRIVATE_META_KEY = kw.keyword("private")
SYM_CLASSMETHOD_META_KEY = kw.keyword("classmethod")
SYM_DEFAULT_META_KEY = kw.keyword("default")
SYM_DYNAMIC_META_KEY = kw.keyword("dynamic")
SYM_PROPERTY_META_KEY = kw.keyword("property")
SYM_MACRO_META_KEY = kw.keyword("macro")
SYM_MUTABLE_META_KEY = kw.keyword("mutable")
SYM_NO_WARN_ON_REDEF_META_KEY = kw.keyword("no-warn-on-redef")
SYM_NO_WARN_ON_SHADOW_META_KEY = kw.keyword("no-warn-on-shadow")
SYM_NO_WARN_WHEN_UNUSED_META_KEY = kw.keyword("no-warn-when-unused")
SYM_REDEF_META_KEY = kw.keyword("redef")
SYM_STATICMETHOD_META_KEY = kw.keyword("staticmethod")

ARGLISTS_KW = kw.keyword("arglists")
COL_KW = kw.keyword("col")
DOC_KW = kw.keyword("doc")
FILE_KW = kw.keyword("file")
LINE_KW = kw.keyword("line")
NAME_KW = kw.keyword("name")
NS_KW = kw.keyword("ns")

VAR_IS_PROTOCOL_META_KEY = kw.keyword("protocol", "basilisp.core")
