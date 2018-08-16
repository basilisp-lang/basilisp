import basilisp.lang.map as lmap
import basilisp.lang.meta as meta
import basilisp.lang.seq as lseq
import basilisp.lang.vector as vector
from basilisp.lang.keyword import keyword
from basilisp.lang.symbol import symbol


def test_vector_meta_interface():
    assert isinstance(vector.v(), meta.Meta)
    assert issubclass(vector.Vector, meta.Meta)


def test_vector_seqable_interface():
    assert isinstance(vector.v(), lseq.Seqable)
    assert issubclass(vector.Vector, lseq.Seqable)


def test_vector_slice():
    assert isinstance(vector.v(1, 2, 3)[1:], vector.Vector)


def test_vector_conj():
    meta = lmap.m(tag="async")
    v1 = vector.v(keyword("kw1"), meta=meta)
    v2 = v1.conj(keyword("kw2"))
    assert v1 is not v2
    assert v1 != v2
    assert len(v2) == 2
    assert meta == v1.meta
    assert meta == v2.meta


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
    assert vec2.meta == lmap.m(type=symbol("str"), tag=keyword("async"))

    meta3 = lmap.m(tag=keyword("macro"))
    vec3 = vec2.with_meta(meta3)
    assert vec2 is not vec3
    assert vec2 == vec3
    assert vec3.meta == lmap.m(type=symbol("str"), tag=keyword("macro"))


def test_vector_repr():
    v = vector.v()
    assert repr(v) == "[]"

    v = vector.v(keyword("kw1"))
    assert repr(v) == "[:kw1]"

    v = vector.v(keyword("kw1"), keyword("kw2"))
    assert repr(v) == "[:kw1 :kw2]"
