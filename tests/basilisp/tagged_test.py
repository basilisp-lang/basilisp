from basilisp.lang.symbol import symbol
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
