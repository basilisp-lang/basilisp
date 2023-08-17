import pickle

import pytest

from basilisp.lang import map as lmap
from basilisp.lang import vector as vec
from basilisp.lang.interfaces import (
    IAssociative,
    ICounted,
    ILispObject,
    ILookup,
    IPersistentCollection,
    IPersistentStack,
    IPersistentVector,
    IReversible,
    ISeqable,
    ISequential,
    IWithMeta,
)
from basilisp.lang.keyword import keyword
from basilisp.lang.symbol import symbol


@pytest.mark.parametrize(
    "interface",
    [
        IAssociative,
        ICounted,
        ILispObject,
        ILookup,
        IPersistentCollection,
        IPersistentStack,
        IPersistentVector,
        IReversible,
        ISeqable,
        ISequential,
        IWithMeta,
    ],
)
def test_vector_interface_membership(interface):
    assert isinstance(vec.v(), interface)
    assert issubclass(vec.PersistentVector, interface)


def test_vector_slice():
    assert isinstance(vec.v(1, 2, 3)[1:], vec.PersistentVector)


def test_assoc():
    v = vec.PersistentVector.empty()
    assert vec.v("a") == v.assoc(0, "a")
    assert vec.PersistentVector.empty() == v
    assert vec.vector(["a", "b"]) == v.assoc(0, "a", 1, "b")

    v1 = vec.v("a")
    assert vec.v("c", "b") == v1.assoc(0, "c", 1, "b")
    assert vec.v("a", "b") == v1.assoc(1, "b")


def test_vector_bool():
    assert True is bool(vec.PersistentVector.empty())


def test_contains():
    assert True is vec.v("a").contains(0)
    assert True is vec.v("a", "b").contains(1)
    assert False is vec.v("a", "b").contains(2)
    assert False is vec.v("a", "b").contains(-1)
    assert False is vec.PersistentVector.empty().contains(0)
    assert False is vec.PersistentVector.empty().contains(1)
    assert False is vec.PersistentVector.empty().contains(-1)


def test_py_contains():
    assert "a" in vec.v("a")
    assert "a" in vec.v("a", "b")
    assert "b" in vec.v("a", "b")
    assert "c" not in vec.PersistentVector.empty()
    assert "c" not in vec.v("a")
    assert "c" not in vec.v("a", "b")


def test_vector_cons():
    meta = lmap.m(tag="async")
    v1 = vec.v(keyword("kw1"), meta=meta)
    v2 = v1.cons(keyword("kw2"))
    assert v1 is not v2
    assert v1 != v2
    assert len(v2) == 2
    assert meta == v1.meta
    assert meta == v2.meta


def test_entry():
    assert vec.MapEntry.of(0, "a") == vec.v("a").entry(0)
    assert vec.MapEntry.of(1, "b") == vec.v("a", "b").entry(1)
    assert None is vec.v("a", "b").entry(2)
    assert vec.MapEntry.of(-1, "b") == vec.v("a", "b").entry(-1)
    assert None is vec.PersistentVector.empty().entry(0)
    assert None is vec.PersistentVector.empty().entry(1)
    assert None is vec.PersistentVector.empty().entry(-1)


def test_val_at():
    assert "a" == vec.v("a").val_at(0)
    assert "b" == vec.v("a", "b").val_at(1)
    assert None is vec.v("a", "b").val_at(2)
    assert "b" == vec.v("a", "b").val_at(-1)
    assert None is vec.PersistentVector.empty().val_at(0)
    assert None is vec.PersistentVector.empty().val_at(1)
    assert None is vec.PersistentVector.empty().val_at(-1)


def test_peek():
    assert None is vec.v().peek()

    assert 1 == vec.v(1).peek()
    assert 2 == vec.v(1, 2).peek()
    assert 3 == vec.v(1, 2, 3).peek()


def test_pop():
    with pytest.raises(IndexError):
        vec.v().pop()

    assert vec.PersistentVector.empty() == vec.v(1).pop()
    assert vec.v(1) == vec.v(1, 2).pop()
    assert vec.v(1, 2) == vec.v(1, 2, 3).pop()


def test_vector_seq():
    assert None is vec.PersistentVector.empty().seq()
    assert vec.v(1) == vec.v(1).seq()
    assert vec.v(1, 2) == vec.v(1, 2).seq()
    assert vec.v(1, 2, 3) == vec.v(1, 2, 3).seq()


def test_vector_meta():
    assert vec.v("vec").meta is None
    meta = lmap.m(type=symbol("str"))
    assert vec.v("vec", meta=meta).meta == meta


def test_vector_with_meta():
    vec0 = vec.v("vec")
    assert vec0.meta is None

    meta1 = lmap.m(type=symbol("str"))
    vec1 = vec.v("vec", meta=meta1)
    assert vec1.meta == meta1

    meta2 = lmap.m(tag=keyword("async"))
    vec2 = vec1.with_meta(meta2)
    assert vec1 is not vec2
    assert vec1 == vec2
    assert vec2.meta == lmap.m(tag=keyword("async"))

    meta3 = lmap.m(tag=keyword("macro"))
    vec3 = vec2.with_meta(meta3)
    assert vec2 is not vec3
    assert vec2 == vec3
    assert vec3.meta == lmap.m(tag=keyword("macro"))

def test_vector_less_than():
    assert False == (vec.v() < vec.v())
    assert False == (vec.v() > vec.v())
    assert False == (vec.v(1) < vec.v(1))
    assert False == (vec.v(1) > vec.v(1))
    assert False == (vec.v(1, 2) < vec.v(1, 2))
    assert False == (vec.v(1, 2) > vec.v(1, 2))
    assert True  == (vec.v(1, 2) < vec.v(1, 3))
    assert False == (vec.v(1, 3) < vec.v(1, 2))
    assert True  == (vec.v(1, 2) < vec.v(1, 2, 3))
    assert False == (vec.v(1, 2, 3) < vec.v(1, 2))

@pytest.mark.parametrize(
    "o",
    [
        vec.v(),
        vec.v(keyword("kw1")),
        vec.v(keyword("kw1"), 2),
        vec.v(keyword("kw1"), 2, None, "nothingness"),
        vec.v(keyword("kw1"), vec.v("string", 4)),
    ],
)
def test_vector_pickleability(pickle_protocol: int, o: vec.PersistentVector):
    assert o == pickle.loads(pickle.dumps(o, protocol=pickle_protocol))


@pytest.mark.parametrize(
    "l,str_repr",
    [
        (vec.v(), "[]"),
        (vec.v(keyword("kw1")), "[:kw1]"),
        (vec.v(keyword("kw1"), keyword("kw2")), "[:kw1 :kw2]"),
    ],
)
def test_vector_repr(l: vec.PersistentVector, str_repr: str):
    assert repr(l) == str_repr
