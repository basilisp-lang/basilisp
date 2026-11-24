from basilisp.lang.symbol import Symbol, symbol
from basilisp.lang.tagged import tagged_literal


def test_tagged_literal():
    tag = symbol("tag")
    form = 1
    tagged = tagged_literal(tag, form)
    assert tagged.tag == tag
    assert tagged.form == form


def test_tagged_literal_str_and_repr():
    tag = symbol("tag")
    form = 1
    tagged = tagged_literal(tag, form)
    assert str(tagged) == "#tag 1"
    assert repr(tagged) == "#tag 1"


def _produce_tagged() -> dict[Symbol, str]:
    return {tagged_literal(Symbol("inst"), "2026-01-01"): "value"}


def _consume_tagged(d: dict[Symbol, str]):
    return d.get(tagged_literal(Symbol("inst"), "2026-01-01"))


def test_tagged_literal_pickleability_cross_process(pickle_cross_process):
    assert pickle_cross_process(_produce_tagged, _consume_tagged) == "value"
