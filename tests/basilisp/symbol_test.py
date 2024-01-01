import pickle

import pytest

from basilisp.lang import map as lmap
from basilisp.lang import set as lset
from basilisp.lang import vector as lvector
from basilisp.lang.keyword import keyword
from basilisp.lang.symbol import Symbol, symbol


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


def test_symbol_as_function():
    sym = symbol("kw")
    assert None is sym(None)

    assert 1 == sym(lmap.map({sym: 1}))
    assert "hi" == sym(lmap.map({sym: "hi"}))
    assert None is sym(lmap.map({"hi": sym}))

    assert sym == sym(lset.s(sym))
    assert None is sym(lset.s(1))
    assert "hi" is sym(lset.s(1), default="hi")

    assert None is sym(lvector.v(1))


@pytest.mark.parametrize(
    "o",
    [
        symbol("kw1"),
        symbol("very-long-name"),
        symbol("kw1", ns="namespaced.keyword"),
        symbol("long-named-kw", ns="also.namespaced.keyword"),
    ],
)
def test_symbol_pickleability(pickle_protocol: int, o: Symbol):
    assert o == pickle.loads(pickle.dumps(o, protocol=pickle_protocol))


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
