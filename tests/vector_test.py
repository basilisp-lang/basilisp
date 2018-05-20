import basilisp.lang.map as lmap
import basilisp.lang.vector as vector
from basilisp.lang.keyword import keyword


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
    meta = lmap.m(type=vector.v("str"))
    assert vector.v("vec", meta=meta).meta == meta


def test_vector_with_meta():
    vec = vector.v("vec")
    assert vec.meta is None

    meta1 = lmap.m(type=vector.v("str"))
    vec1 = vector.v("vec", meta=meta1)
    assert vec1.meta == meta1

    meta2 = lmap.m(tag=keyword("async"))
    vec2 = vec1.with_meta(meta2)
    assert vec1 is not vec2
    assert vec1 == vec2
    assert vec2.meta == lmap.m(type=vector.v("str"), tag=keyword("async"))

    meta3 = lmap.m(tag=keyword("macro"))
    vec3 = vec2.with_meta(meta3)
    assert vec2 is not vec3
    assert vec2 == vec3
    assert vec3.meta == lmap.m(type=vector.v("str"), tag=keyword("macro"))


def test_vector_repr():
    v = vector.v()
    assert repr(v) == "[]"

    v = vector.v(keyword("kw1"))
    assert repr(v) == "[:kw1]"

    v = vector.v(keyword("kw1"), keyword("kw2"))
    assert repr(v) == "[:kw1 :kw2]"
