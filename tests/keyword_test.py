from basilisp.lang.keyword import keyword


def test_keyword_identity_equals():
    assert keyword("kw") is keyword("kw")
    assert keyword("kw") == keyword("kw")

    assert keyword("kw", ns="some.ns") is not keyword("kw", ns="other.ns")
    assert keyword("kw", ns="some.ns") is not keyword("kw")


def test_keyword_name_and_ns():
    kw = keyword("kw", ns="ns")
    assert kw.name == "kw"
    assert kw.ns == "ns"

    kw = keyword("kw")
    assert kw.name == "kw"
    assert kw.ns is None


def test_keyword_str_and_repr():
    kw = keyword("kw", ns="ns")
    assert str(kw) == "ns/kw"
    assert repr(kw) == ":ns/kw"

    kw = keyword("kw", ns="some.ns")
    assert str(kw) == "some.ns/kw"
    assert repr(kw) == ":some.ns/kw"

    kw = keyword("kw")
    assert str(kw) == "kw"
    assert repr(kw) == ":kw"
