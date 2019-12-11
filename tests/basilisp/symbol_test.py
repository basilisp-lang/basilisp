import basilisp.lang.map as lmap
from basilisp.lang.keyword import keyword

from basilisp.lang.symbol import symbol


def test_symbol_name_and_ns():
    sym = symbol("sym", ns="ns")
    assert sym.name == "sym"
    assert sym.ns == "ns"

    sym = symbol("sym")
    assert sym.name == "sym"
    assert sym.ns is None


def test_symbol_meta():
    assert symbol("sym").meta is None
    meta = lmap.m(type=symbol("str"))
    assert symbol("sym", meta=meta).meta == meta


def test_symbol_with_meta():
    sym = symbol("sym")
    assert sym.meta is None

    meta1 = lmap.m(type=symbol("str"))
    sym1 = symbol("sym", meta=meta1)
    assert sym1.meta == meta1

    meta2 = lmap.m(tag=keyword("async"))
    sym2 = sym1.with_meta(meta2)
    assert sym1 is not sym2
    assert sym1 == sym2
    assert sym2.meta == lmap.m(tag=keyword("async"))

    meta3 = lmap.m(tag=keyword("macro"))
    sym3 = sym2.with_meta(meta3)
    assert sym2 is not sym3
    assert sym2 == sym3
    assert sym3.meta == lmap.m(tag=keyword("macro"))


def test_symbol_str_and_repr():
    sym = symbol("sym", ns="ns")
    assert str(sym) == "ns/sym"
    assert repr(sym) == "ns/sym"

    sym = symbol("sym", ns="some.ns")
    assert str(sym) == "some.ns/sym"
    assert repr(sym) == "some.ns/sym"

    sym = symbol("sym")
    assert str(sym) == "sym"
    assert repr(sym) == "sym"
