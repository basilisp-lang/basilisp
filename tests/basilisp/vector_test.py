import pickle

import pytest

import basilisp.lang.map as lmap
import basilisp.lang.vector as vector
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
    assert isinstance(vector.v(), interface)
    assert issubclass(vector.Vector, interface)


def test_vector_slice():
    assert isinstance(vector.v(1, 2, 3)[1:], vector.Vector)


def test_assoc():
    v = vector.Vector.empty()
    assert vector.v("a") == v.assoc(0, "a")
    assert vector.Vector.empty() == v
    assert vector.vector(["a", "b"]) == v.assoc(0, "a", 1, "b")

    v1 = vector.v("a")
    assert vector.v("c", "b") == v1.assoc(0, "c", 1, "b")
    assert vector.v("a", "b") == v1.assoc(1, "b")


def test_contains():
    assert True is vector.v("a").contains(0)
    assert True is vector.v("a", "b").contains(1)
    assert False is vector.v("a", "b").contains(2)
    assert False is vector.v("a", "b").contains(-1)
    assert False is vector.Vector.empty().contains(0)
    assert False is vector.Vector.empty().contains(1)
    assert False is vector.Vector.empty().contains(-1)


def test_py_contains():
    assert "a" in vector.v("a")
    assert "a" in vector.v("a", "b")
    assert "b" in vector.v("a", "b")
    assert "c" not in vector.Vector.empty()
    assert "c" not in vector.v("a")
    assert "c" not in vector.v("a", "b")


def test_vector_cons():
    meta = lmap.m(tag="async")
    v1 = vector.v(keyword("kw1"), meta=meta)
    v2 = v1.cons(keyword("kw2"))
    assert v1 is not v2
    assert v1 != v2
    assert len(v2) == 2
    assert meta == v1.meta
    assert meta == v2.meta


def test_entry():
    assert "a" == vector.v("a").val_at(0)
    assert "b" == vector.v("a", "b").val_at(1)
    assert None is vector.v("a", "b").val_at(2)
    assert "b" == vector.v("a", "b").val_at(-1)
    assert None is vector.Vector.empty().val_at(0)
    assert None is vector.Vector.empty().val_at(1)
    assert None is vector.Vector.empty().val_at(-1)


def test_peek():
    assert None is vector.v().peek()

    assert 1 == vector.v(1).peek()
    assert 2 == vector.v(1, 2).peek()
    assert 3 == vector.v(1, 2, 3).peek()


def test_pop():
    with pytest.raises(IndexError):
        vector.v().pop()

    assert vector.Vector.empty() == vector.v(1).pop()
    assert vector.v(1) == vector.v(1, 2).pop()
    assert vector.v(1, 2) == vector.v(1, 2, 3).pop()


def test_vector_meta():
    assert vector.v("vec").meta is None
    meta = lmap.m(type=symbol("str"))
    assert vector.v("vec", meta=meta).meta == meta


def test_vector_with_meta():
    vec = vector.v("vec")
    assert vec.meta is None

    meta1 = lmap.m(type=symbol("str"))
    vec1 = vector.v("vec", meta=meta1)
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


@pytest.mark.parametrize(
    "o",
    [
        vector.v(),
        vector.v(keyword("kw1")),
        vector.v(keyword("kw1"), 2),
        vector.v(keyword("kw1"), 2, None, "nothingness"),
        vector.v(keyword("kw1"), vector.v("string", 4)),
    ],
)
def test_vector_pickleability(pickle_protocol: int, o: vector.Vector):
    assert o == pickle.loads(pickle.dumps(o, protocol=pickle_protocol))


@pytest.mark.parametrize(
    "l,str_repr",
    [
        (vector.v(), "[]"),
        (vector.v(keyword("kw1")), "[:kw1]"),
        (vector.v(keyword("kw1"), keyword("kw2")), "[:kw1 :kw2]"),
    ],
)
def test_vector_repr(l: vector.Vector, str_repr: str):
    assert repr(l) == str_repr
