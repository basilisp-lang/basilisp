import pytest

from basilisp.lang import keyword as kw
from basilisp.lang import map as lmap
from basilisp.lang import multifn as multifn
from basilisp.lang import symbol as sym


def test_multi_function():
    def dispatch(v) -> kw.Keyword:
        if v == "i":
            return kw.keyword("a")
        elif v == "ii":
            return kw.keyword("b")
        return kw.keyword("default")

    def fn_a(v) -> str:
        return "1"

    def fn_b(v) -> str:
        return "2"

    def fn_default(v) -> str:
        return "BLAH"

    f = multifn.MultiFunction(sym.symbol("test-fn"), dispatch, kw.keyword("default"))
    f.add_method(kw.keyword("a"), fn_a)
    f.add_method(kw.keyword("b"), fn_b)
    f.add_method(kw.keyword("default"), fn_default)

    assert (
        lmap.map(
            {
                kw.keyword("a"): fn_a,
                kw.keyword("b"): fn_b,
                kw.keyword("default"): fn_default,
            }
        )
        == f.methods
    )

    assert kw.keyword("default") == f.default

    assert fn_a is f.get_method(kw.keyword("a"))
    assert fn_b is f.get_method(kw.keyword("b"))
    assert fn_default is f.get_method(kw.keyword("default"))
    assert fn_default is f.get_method(kw.keyword("other"))

    assert "1" == f("i")
    assert "2" == f("ii")
    assert "BLAH" == f("iii")
    assert "BLAH" == f("whatever")

    f.remove_method(kw.keyword("b"))

    assert "1" == f("i")
    assert "BLAH" == f("ii")
    assert "BLAH" == f("iii")
    assert "BLAH" == f("whatever")

    f.remove_all_methods()

    assert lmap.EMPTY == f.methods

    with pytest.raises(NotImplementedError):
        f("blah")
