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
    IReduce,
    IReduceKV,
    IReversible,
    ISeqable,
    ISequential,
    IWithMeta,
)
from basilisp.lang.keyword import keyword
from basilisp.lang.reduced import Reduced
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
        IReduce,
        IReduceKV,
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
    v = vec.EMPTY
    assert vec.v("a") == v.assoc(0, "a")
    assert vec.EMPTY == v
    assert vec.vector(["a", "b"]) == v.assoc(0, "a", 1, "b")

    meta = lmap.m(meta=True)
    v1 = vec.v("a", meta=meta)
    assert vec.v("c", "b") == v1.assoc(0, "c", 1, "b")
    assert vec.v("a", "b") == v1.assoc(1, "b")
    assert v1.assoc(1, "b").meta == meta
    assert v1.assoc(1, "b", 2, "c").meta == meta


def test_vector_empty():
    meta = lmap.map({"meta": 1})
    v1 = vec.v(1, 2, 3, meta=meta)
    assert v1.empty() == vec.EMPTY
    assert v1.empty().meta == meta
    assert vec.EMPTY.meta is None


def test_vector_bool():
    assert True is bool(vec.EMPTY)


def test_contains():
    assert True is vec.v("a").contains(0)
    assert True is vec.v("a", "b").contains(1)
    assert False is vec.v("a", "b").contains(2)
    assert False is vec.v("a", "b").contains(-1)
    assert False is vec.EMPTY.contains(0)
    assert False is vec.EMPTY.contains(1)
    assert False is vec.EMPTY.contains(-1)


def test_py_contains():
    assert "a" in vec.v("a")
    assert "a" in vec.v("a", "b")
    assert "b" in vec.v("a", "b")
    assert "c" not in vec.EMPTY
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
    assert None is vec.EMPTY.entry(0)
    assert None is vec.EMPTY.entry(1)
    assert None is vec.EMPTY.entry(-1)


def test_vector_callable():
    assert "a" == vec.v("a")(0)
    assert "b" == vec.v("a", "b")(1)
    assert "b" == vec.v("a", "b")(-1)


@pytest.mark.parametrize("v,idx", [(vec.v("a", "b"), 2), (vec.EMPTY, 0), (vec.EMPTY, 1), (vec.EMPTY, -1)])
def test_vector_callable_index_must_be_valid(v, idx):
    with pytest.raises(IndexError):
        v(idx)


def test_val_at():
    assert "a" == vec.v("a").val_at(0)
    assert "b" == vec.v("a", "b").val_at(1)
    assert None is vec.v("a", "b").val_at(2)
    assert "b" == vec.v("a", "b").val_at(-1)
    assert None is vec.EMPTY.val_at(0)
    assert None is vec.EMPTY.val_at(1)
    assert None is vec.EMPTY.val_at(-1)
    assert None is vec.EMPTY.val_at(keyword("blah"))
    assert "default" == vec.EMPTY.val_at(keyword("blah"), "default")
    assert None is vec.EMPTY.val_at("key")
    assert "default" == vec.EMPTY.val_at("key", "default")


def test_peek():
    assert None is vec.v().peek()

    assert 1 == vec.v(1).peek()
    assert 2 == vec.v(1, 2).peek()
    assert 3 == vec.v(1, 2, 3).peek()


def test_pop():
    with pytest.raises(IndexError):
        vec.v().pop()

    assert vec.EMPTY == vec.v(1).pop()
    assert vec.v(1) == vec.v(1, 2).pop()
    assert vec.v(1, 2) == vec.v(1, 2, 3).pop()


def test_vector_seq():
    assert None is vec.EMPTY.seq()
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


@pytest.mark.parametrize(
    "result,v1,v2",
    [
        (False, vec.v(), vec.v()),
        (False, vec.v(1), vec.v(1)),
        (False, vec.v(1, 2), vec.v(1, 2)),
        (True, vec.v(1, 2), vec.v(1, 3)),
        (False, vec.v(1, 3), vec.v(1, 2)),
        (True, vec.v(1, 2), vec.v(1, 2, 3)),
        (False, vec.v(1, 2, 3), vec.v(1, 2)),
    ],
)
def test_vector_less_than(result, v1, v2):
    assert result == (v1 < v2)


@pytest.mark.parametrize(
    "result,v1,v2",
    [
        (False, vec.v(), vec.v()),
        (False, vec.v(1), vec.v(1)),
        (False, vec.v(1, 2), vec.v(1, 2)),
        (False, vec.v(1, 2), vec.v(1, 3)),
        (True, vec.v(1, 3), vec.v(1, 2)),
        (False, vec.v(1, 2), vec.v(1, 2, 3)),
        (True, vec.v(1, 2, 3), vec.v(1, 2)),
    ],
)
def test_vector_greater_than(result, v1, v2):
    assert result == (v1 > v2)


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


@pytest.fixture
def add():
    def _add(*args):
        if len(args) == 0:
            return 0
        elif len(args) == 1:
            return args[0]
        else:
            if args[0] > 20:
                return Reduced(args[0] + args[1])
            return args[0] + args[1]

    return _add


@pytest.mark.parametrize(
    "coll,res",
    [
        (vec.vector([]), 0),
        (vec.vector([1]), 1),
        (vec.vector([1, 2, 3]), 6),
        (vec.vector([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 28),
    ],
)
def test_vector_reduce(add, coll: vec.PersistentVector, res):
    assert coll.reduce(add) == res


@pytest.mark.parametrize(
    "coll,res, init",
    [
        (vec.vector([]), 45, 45),
        (vec.vector([1]), 46, 45),
        (vec.vector([1, 2, 3]), 10, 4),
        (vec.vector([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 46, 45),
    ],
)
def test_vector_reduce_init(add, coll: vec.PersistentVector, res, init):
    assert coll.reduce(add, init) == res


def test_vector_reduce_kv():
    init = lmap.map({keyword("ks"): vec.v(), keyword("vs"): vec.v()})

    def reduce_vec(acc: lmap.PersistentMap, i, v):
        ks, vs = acc.get(keyword("ks")), acc.get(keyword("vs"))
        res = acc.assoc(keyword("ks"), ks.cons(i), keyword("vs"), vs.cons(v))
        if i < 2:
            return res
        return Reduced(res)

    assert init == vec.v().reduce_kv(reduce_vec, init)
    assert (
        lmap.map(
            {
                keyword("ks"): vec.v(0, 1, 2),
                keyword("vs"): vec.vector([keyword(s) for s in ("a", "b", "c")]),
            }
        )
        == vec.vector([keyword(s) for s in ("a", "b", "c")]).reduce_kv(reduce_vec, init)
        == vec.vector([keyword(s) for s in ("a", "b", "c", "d", "e", "f")]).reduce_kv(
            reduce_vec, init
        )
    )


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
