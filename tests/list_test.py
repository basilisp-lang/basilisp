import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.meta as meta
import basilisp.lang.seq as lseq
from basilisp.lang.keyword import keyword
from basilisp.lang.symbol import symbol


def test_list_meta_interface():
    assert isinstance(llist.l(), meta.Meta)
    assert issubclass(llist.List, meta.Meta)


def test_list_seq_interface():
    assert isinstance(llist.l(), lseq.Seq)
    assert issubclass(llist.List, lseq.Seq)


def test_list_conj():
    meta = lmap.m(tag="async")
    l1 = llist.l(keyword("kw1"), meta=meta)
    l2 = l1.conj(keyword("kw2"))
    assert l1 is not l2
    assert l1 != l2
    assert len(l2) == 2
    assert meta == l1.meta
    assert l2.meta is None


def test_list_meta():
    assert llist.l("vec").meta is None
    meta = lmap.m(type=symbol("str"))
    assert llist.l("vec", meta=meta).meta == meta


def test_list_with_meta():
    l1 = llist.l("vec")
    assert l1.meta is None

    meta1 = lmap.m(type=symbol("str"))
    l2 = llist.l("vec", meta=meta1)
    assert l2.meta == meta1

    meta2 = lmap.m(tag=keyword("async"))
    l3 = l2.with_meta(meta2)
    assert l2 is not l3
    assert l2 == l3
    assert l3.meta == lmap.m(type=symbol("str"), tag=keyword("async"))

    meta3 = lmap.m(tag=keyword("macro"))
    l4 = l3.with_meta(meta3)
    assert l3 is not l4
    assert l3 == l4
    assert l4.meta == lmap.m(type=symbol("str"), tag=keyword("macro"))


def test_list_first():
    assert None is llist.List.empty().first
    assert None is llist.l().first
    assert 1 == llist.l(1).first
    assert 1 == llist.l(1, 2).first


def test_list_rest():
    assert llist.l().rest == llist.l()
    assert llist.l(keyword("kw1")).rest == llist.l()
    assert llist.l(keyword("kw1"), keyword("kw2")).rest == llist.l(keyword("kw2"))


def test_list_repr():
    l = llist.l()
    assert repr(l) == "()"

    l = llist.l(keyword("kw1"))
    assert repr(l) == "(:kw1)"

    l = llist.l(keyword("kw1"), keyword("kw2"))
    assert repr(l) == "(:kw1 :kw2)"
