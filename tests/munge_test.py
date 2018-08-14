import builtins
import keyword

from basilisp.lang.util import munge


def test_munge_disallows_syms():
    assert "__PLUS__" == munge("+")
    assert "_" == munge("-")
    assert "__STAR__ns__STAR__" == munge("*ns*")
    assert "__DIV__" == munge("/")
    assert "__LT__" == munge("<")
    assert "__GT__" == munge(">")
    assert "send__BANG__" == munge("send!")
    assert "__EQ__" == munge("=")
    assert "string__Q__" == munge("string?")
    assert "__IDIV__" == munge("\\")
    assert "__AMP__form" == munge("&form")


def test_munge_disallows_python_builtins():
    for name in builtins.__dict__.keys():
        assert f"{name}_" == munge(name, allow_builtins=False)


def test_munge_disallows_python_kws():
    for kw in keyword.kwlist:
        assert f"{kw}_" == munge(kw)
