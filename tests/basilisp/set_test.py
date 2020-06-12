import pickle
import typing

import pytest

import basilisp.lang.map as lmap
import basilisp.lang.set as lset
from basilisp.lang.interfaces import (
    ICounted,
    ILispObject,
    IPersistentCollection,
    IPersistentSet,
    ISeqable,
    IWithMeta,
)
from basilisp.lang.keyword import keyword
from basilisp.lang.symbol import symbol


@pytest.mark.parametrize(
    "interface",
    [
        typing.AbstractSet,
        ICounted,
        ILispObject,
        IPersistentCollection,
        IPersistentSet,
        ISeqable,
        IWithMeta,
    ],
)
def test_set_interface_membership(interface):
    assert isinstance(lset.s(), interface)
    assert issubclass(lset.Set, interface)


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
    assert s3.meta == lmap.m(tag=keyword("async"))

    meta3 = lmap.m(tag=keyword("macro"))
    s4 = s3.with_meta(meta3)
    assert s3 is not s4
    assert s3 == s4
    assert s4.meta == lmap.m(tag=keyword("macro"))


@pytest.mark.parametrize(
    "o",
    [
        lset.s(),
        lset.s(keyword("kw1")),
        lset.s(keyword("kw1"), 2),
        lset.s(keyword("kw1"), 2, None, "nothingness"),
        lset.s(keyword("kw1"), lset.s("string", 4)),
    ],
)
def test_set_pickleability(pickle_protocol: int, o: lset.Set):
    assert o == pickle.loads(pickle.dumps(o, protocol=pickle_protocol))


@pytest.mark.parametrize(
    "l,str_repr",
    [
        (lset.s(), {"#{}"}),
        (lset.s(keyword("kw1")), {"#{:kw1}"}),
        (lset.s(keyword("kw1"), keyword("kw2")), {"#{:kw1 :kw2}", "#{:kw2 :kw1}"}),
    ],
)
def test_set_repr(l: lset.Set, str_repr: typing.Set[str]):
    assert repr(l) in str_repr
