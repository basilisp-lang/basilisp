import basilisp.lang.map as lmap
import basilisp.lang.set as lset
from basilisp.lang.interfaces import (
    IMeta,
    IPersistentCollection,
    IPersistentSet,
    ISeqable,
)
from basilisp.lang.keyword import keyword
from basilisp.lang.obj import LispObject
from basilisp.lang.symbol import symbol


def test_list_collection_interface():
    assert isinstance(lset.s(), IPersistentCollection)
    assert issubclass(lset.Set, IPersistentCollection)


def test_list_meta_interface():
    assert isinstance(lset.s(), IMeta)
    assert issubclass(lset.Set, IMeta)


def test_set_object_interface():
    assert isinstance(lset.s(), LispObject)
    assert issubclass(lset.Set, LispObject)


def test_set_seqable_interface():
    assert isinstance(lset.s(), ISeqable)
    assert issubclass(lset.Set, ISeqable)


def test_set_set_interface():
    assert isinstance(lset.s(), IPersistentSet)
    assert issubclass(lset.Set, IPersistentSet)


def test_set_conj():
    meta = lmap.m(tag="async")
    s1 = lset.s(keyword("kw1"), meta=meta)
    s2 = s1.cons(keyword("kw2"))
    assert s1 is not s2
    assert s1 != s2
    assert len(s2) == 2
    assert meta == s1.meta
    assert meta == s2.meta


def test_set_meta():
    assert lset.s("vec").meta is None
    meta = lmap.m(type=symbol("str"))
    assert lset.s("vec", meta=meta).meta == meta


def test_set_with_meta():
    s1 = lset.s("vec")
    assert s1.meta is None

    meta1 = lmap.m(type=symbol("str"))
    s2 = lset.s("vec", meta=meta1)
    assert s2.meta == meta1

    meta2 = lmap.m(tag=keyword("async"))
    s3 = s2.with_meta(meta2)
    assert s2 is not s3
    assert s2 == s3
    assert s3.meta == lmap.m(type=symbol("str"), tag=keyword("async"))

    meta3 = lmap.m(tag=keyword("macro"))
    s4 = s3.with_meta(meta3)
    assert s3 is not s4
    assert s3 == s4
    assert s4.meta == lmap.m(type=symbol("str"), tag=keyword("macro"))


def test_set_repr():
    s = lset.s()
    assert repr(s) == "#{}"

    s = lset.s(keyword("kw1"))
    assert repr(s) == "#{:kw1}"

    s = lset.s(keyword("kw1"), keyword("kw2"))
    assert repr(s) in ["#{:kw1 :kw2}", "#{:kw2 :kw1}"]
