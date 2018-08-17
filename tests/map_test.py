import pytest

import basilisp.lang.associative as lassoc
import basilisp.lang.map as lmap
import basilisp.lang.meta as meta
import basilisp.lang.seq as lseq
from basilisp.lang.keyword import keyword
from basilisp.lang.map import MapEntry
from basilisp.lang.symbol import symbol


def test_map_associative_interface():
    assert isinstance(lmap.m(), lassoc.Associative)
    assert issubclass(lmap.Map, lassoc.Associative)


def test_map_meta_interface():
    assert isinstance(lmap.m(), meta.Meta)
    assert issubclass(lmap.Map, meta.Meta)


def test_map_seqable_interface():
    assert isinstance(lmap.m(), lseq.Seqable)
    assert issubclass(lmap.Map, lseq.Seqable)


def test_assoc():
    m = lmap.m()
    assert lmap.map({"k": 1}) == m.assoc("k", 1)
    assert lmap.Map.empty() == m
    assert lmap.map({"a": 1, "b": 2}) == m.assoc("a", 1, "b", 2)

    m1 = lmap.map({"a": 3})
    assert lmap.map({"a": 1, "b": 2}) == m1.assoc("a", 1, "b", 2)
    assert lmap.map({"a": 3, "b": 2}) == m1.assoc("b", 2)


def test_contains():
    assert True is lmap.map({"a": 1}).contains("a")
    assert False is lmap.map({"a": 1}).contains("b")
    assert False is lmap.Map.empty().contains("a")


def test_dissoc():
    assert lmap.Map.empty() == lmap.Map.empty().dissoc("a")
    assert lmap.Map.empty() == lmap.Map.empty().dissoc("a", "b", "c")

    m1 = lmap.map({"a": 3})
    assert m1 == m1.dissoc("b")
    assert lmap.Map.empty() == m1.dissoc("a")

    m2 = lmap.map({"a": 3, "b": 2})
    assert lmap.map({"a": 3}) == m2.dissoc("b")
    assert lmap.map({"b": 2}) == m2.dissoc("a")
    assert lmap.Map.empty() == m2.dissoc("a", "b")


def test_entry():
    assert 1 == lmap.map({"a": 1}).entry("a")
    assert None is lmap.map({"a": 1}).entry("b")
    assert None is lmap.Map.empty().entry("a")


def test_map_conj():
    meta = lmap.m(tag="async")
    m1 = lmap.map({"first": "Chris"}, meta=meta)
    m2 = m1.conj(MapEntry.of("last", "Cronk"))
    assert m1 is not m2
    assert m1 != m2
    assert len(m2) == 2
    assert meta == m1.meta
    assert meta == m2.meta

    meta = lmap.m(tag="async")
    m1 = lmap.map({"first": "Chris"}, meta=meta)
    m2 = m1.conj(["last", "Cronk"])
    assert m1 is not m2
    assert m1 != m2
    assert len(m2) == 2
    assert meta == m1.meta
    assert meta == m2.meta

    with pytest.raises(ValueError):
        m1 = lmap.map({"first": "Chris"})
        m1.conj(["last"])


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
    assert m3.meta == lmap.m(type=symbol("str"), tag=keyword("async"))

    meta3 = lmap.m(tag=keyword("macro"))
    m4 = m3.with_meta(meta3)
    assert m3 is not m4
    assert m3 == m4
    assert m4.meta == lmap.m(type=symbol("str"), tag=keyword("macro"))


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
