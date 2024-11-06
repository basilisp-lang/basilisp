import pickle
from collections.abc import Mapping

import pytest

from basilisp.lang import map as lmap
from basilisp.lang import set as lset
from basilisp.lang.interfaces import (
    IAssociative,
    ICounted,
    ILispObject,
    IMapEntry,
    IPersistentCollection,
    IPersistentMap,
    IReduceKV,
    ISeqable,
    IWithMeta,
)
from basilisp.lang.keyword import keyword
from basilisp.lang.symbol import symbol
from basilisp.lang.vector import MapEntry, v


def test_map_entry_interface_membership():
    assert isinstance(lmap.MapEntry.of("a", "b"), IMapEntry)
    assert issubclass(lmap.MapEntry, IMapEntry)


@pytest.mark.parametrize(
    "interface",
    [
        IAssociative,
        ICounted,
        ILispObject,
        IPersistentCollection,
        IPersistentMap,
        IReduceKV,
        ISeqable,
        IWithMeta,
        Mapping,
    ],
)
def test_map_interface_membership(interface):
    assert isinstance(lmap.m(), interface)
    assert issubclass(lmap.PersistentMap, interface)


def test_assoc():
    m = lmap.m()
    assert lmap.map({"k": 1}) == m.assoc("k", 1)
    assert lmap.EMPTY == m
    assert lmap.map({"a": 1, "b": 2}) == m.assoc("a", 1, "b", 2)

    m1 = lmap.map({"a": 3})
    assert lmap.map({"a": 1, "b": 2}) == m1.assoc("a", 1, "b", 2)
    assert lmap.map({"a": 3, "b": 2}) == m1.assoc("b", 2)

    meta = lmap.map({"meta": True})
    m2 = lmap.map({"b": 3}, meta=meta)
    assert m2.assoc("b", 4).meta == meta
    assert m2.assoc("c", 8, "d", 12).meta == meta


def test_map_bool():
    assert True is bool(lmap.EMPTY)


def test_map_as_function():
    assert None is lmap.m()(1)
    assert None is lmap.m()("abc")
    assert None is lmap.map({1: True, "2": 2, 3: "string"})("abc")
    assert "string" == lmap.map({1: True, "2": 2, 3: "string"})(3)
    assert 2 == lmap.map({1: True, "2": 2, 3: "string"})("2")


def test_contains():
    assert True is lmap.map({"a": 1}).contains("a")
    assert False is lmap.map({"a": 1}).contains("b")
    assert False is lmap.EMPTY.contains("a")


def test_dissoc():
    assert lmap.EMPTY == lmap.EMPTY.dissoc("a")
    assert lmap.EMPTY == lmap.EMPTY.dissoc("a", "b", "c")

    m1 = lmap.map({"a": 3})
    assert m1 == m1.dissoc("b")
    assert lmap.EMPTY == m1.dissoc("a")

    meta = lmap.map({"meta": True})
    m2 = lmap.map({"a": 3, "b": 2}, meta=meta)
    assert lmap.map({"a": 3}) == m2.dissoc("b")
    assert m2.dissoc("b").meta == meta
    assert lmap.map({"b": 2}) == m2.dissoc("a")
    assert m2.dissoc("a").meta == meta
    assert lmap.EMPTY == m2.dissoc("a", "b")
    assert m2.dissoc("a", "b").meta == meta


def test_map_empty():
    meta = lmap.map({"meta": 1})
    m1 = lmap.map({"a": 1}, meta=meta)
    assert m1.empty() == lmap.EMPTY
    assert m1.empty().meta == meta
    assert lmap.EMPTY.meta is None


def test_entry():
    assert MapEntry.of("a", 1) == lmap.map({"a": 1}).entry("a")
    assert None is lmap.map({"a": 1}).entry("b")
    assert None is lmap.EMPTY.entry("a")


def test_val_at():
    assert 1 == lmap.map({"a": 1}).val_at("a")
    assert None is lmap.map({"a": 1}).val_at("b")
    assert None is lmap.EMPTY.val_at("a")


def test_map_cons():
    meta = lmap.m(tag="async")
    m1 = lmap.map({"first": "Chris"}, meta=meta)
    m2 = m1.cons({"last": "Cronk"})
    assert m1 is not m2
    assert m1 != m2
    assert len(m2) == 2
    assert meta == m1.meta
    assert meta == m2.meta
    assert "Chris" == m1.get("first")
    assert not m1.contains("last")
    assert "Cronk" == m2.get("last")
    assert "Chris" == m2.get("first")

    meta = lmap.m(tag="async")
    m1 = lmap.map({"first": "Chris"}, meta=meta)
    m2 = m1.cons(MapEntry.of("last", "Cronk"))
    assert m1 is not m2
    assert m1 != m2
    assert len(m2) == 2
    assert meta == m1.meta
    assert meta == m2.meta
    assert "Chris" == m1.get("first")
    assert not m1.contains("last")
    assert "Cronk" == m2.get("last")
    assert "Chris" == m2.get("first")

    meta = lmap.m(tag="async")
    m1 = lmap.map({"first": "Chris"}, meta=meta)
    m2 = m1.cons(["last", "Cronk"])
    assert m1 is not m2
    assert m1 != m2
    assert len(m2) == 2
    assert meta == m1.meta
    assert meta == m2.meta
    assert "Chris" == m1.get("first")
    assert not m1.contains("last")
    assert "Cronk" == m2.get("last")
    assert "Chris" == m2.get("first")

    meta = lmap.m(tag="async")
    m1 = lmap.map({"first": "Chris"}, meta=meta)
    m2 = m1.cons(["last", "Cronk"], v("middle", "L"))
    assert m1 is not m2
    assert m1 != m2
    assert len(m2) == 3
    assert meta == m1.meta
    assert meta == m2.meta
    assert "Chris" == m1.get("first")
    assert not m1.contains("middle")
    assert not m1.contains("last")
    assert "Cronk" == m2.get("last")
    assert "L" == m2.get("middle")
    assert "Chris" == m2.get("first")

    with pytest.raises(ValueError):
        m1 = lmap.map({"first": "Chris"})
        m1.cons(["last"])


def test_map_meta():
    assert lmap.m(tag="str").meta is None
    meta = lmap.m(type=symbol("str"))
    assert lmap.map({"type": "vec"}, meta=meta).meta == meta


def test_map_with_meta():
    m1 = lmap.m(key1="vec")
    assert m1.meta is None

    meta1 = lmap.m(type=symbol("str"))
    m2 = lmap.map({"key1": "vec"}, meta=meta1)
    assert m2.meta == meta1

    meta2 = lmap.m(tag=keyword("async"))
    m3 = m2.with_meta(meta2)
    assert m2 is not m3
    assert m2 == m3
    assert m3.meta == lmap.m(tag=keyword("async"))

    meta3 = lmap.m(tag=keyword("macro"))
    m4 = m3.with_meta(meta3)
    assert m3 is not m4
    assert m3 == m4
    assert m4.meta == lmap.m(tag=keyword("macro"))


def test_map_seq():
    assert None is lmap.EMPTY.seq()
    assert {v("a", 1)} == set(lmap.map({"a": 1}).seq())
    assert {v("a", 1), v("b", 2)} == set(lmap.map({"a": 1, "b": 2}).seq())
    assert {v("a", 1), v("b", 2), v("c", 3)} == set(
        lmap.map({"a": 1, "b": 2, "c": 3}).seq()
    )


def test_map_reduce_kv():

    def reduce_map(acc: lset.PersistentSet, key, val):
        return acc.cons(v(key, val))

    assert lset.s() == lmap.m().reduce_kv(reduce_map, lset.s())
    assert lset.s(v("a", 1), v("b", 2), v("c", 3)) == lmap.map(
        {"a": 1, "b": 2, "c": 3}
    ).reduce_kv(reduce_map, lset.s())


def test_map_repr():
    m = lmap.m()
    assert repr(m) == "{}"

    m = lmap.map({keyword("key"): "val"})
    assert repr(m) == '{:key "val"}'

    m = lmap.map({keyword("key1"): "val1", keyword("key2"): 3})
    assert repr(m) in ['{:key2 3 :key1 "val1"}', '{:key1 "val1" :key2 3}']


def test_hash_map_creator():
    assert lmap.m() == lmap.hash_map()
    assert lmap.map({1: 2}) == lmap.hash_map(1, 2)
    assert lmap.map({1: 2, 3: 4}) == lmap.hash_map(1, 2, 3, 4)

    with pytest.raises(IndexError):
        lmap.hash_map(1, 2, 3)


@pytest.mark.parametrize(
    "o",
    [
        lmap.m(),
        lmap.map({"a": 2}),
        lmap.map({"a": 2, None: "NOTHINGNESS"}),
        lmap.map({"a": 2, keyword("b"): lmap.map({keyword("c"): "string"})}),
    ],
)
def test_map_pickleability(pickle_protocol: int, o: lmap.PersistentMap):
    assert o == pickle.loads(pickle.dumps(o, protocol=pickle_protocol))
