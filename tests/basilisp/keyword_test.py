import pickle

import pytest
from pyrsistent import PMap, pmap

import basilisp.lang.map as lmap
from basilisp.lang.atom import Atom
from basilisp.lang.keyword import Keyword, complete, keyword


def test_keyword_identity_equals():
    assert keyword("kw") is keyword("kw")
    assert keyword("kw") == keyword("kw")

    assert keyword("kw", ns="some.ns") is not keyword("kw", ns="other.ns")
    assert keyword("kw", ns="some.ns") is not keyword("kw")


def test_keyword_name_and_ns():
    kw = keyword("kw", ns="ns")
    assert kw.name == "kw"
    assert kw.ns == "ns"

    kw = keyword("kw")
    assert kw.name == "kw"
    assert kw.ns is None


def test_keyword_str_and_repr():
    kw = keyword("kw", ns="ns")
    assert str(kw) == ":ns/kw"
    assert repr(kw) == ":ns/kw"

    kw = keyword("kw", ns="some.ns")
    assert str(kw) == ":some.ns/kw"
    assert repr(kw) == ":some.ns/kw"

    kw = keyword("kw")
    assert str(kw) == ":kw"
    assert repr(kw) == ":kw"


def test_keyword_as_function():
    kw = keyword("kw")
    assert None is kw(None)

    assert 1 == kw(lmap.map({kw: 1}))
    assert "hi" == kw(lmap.map({kw: "hi"}))
    assert None is kw(lmap.map({"hi": kw}))


@pytest.mark.parametrize(
    "o",
    [
        keyword("kw1"),
        keyword("very-long-name"),
        keyword("kw1", ns="namespaced.keyword"),
        keyword("long-named-kw", ns="also.namespaced.keyword"),
    ],
)
def test_keyword_pickleability(pickle_protocol: int, o: Keyword):
    assert o == pickle.loads(pickle.dumps(o, protocol=pickle_protocol))


class TestKeywordCompletion:
    @pytest.fixture
    def empty_cache(self) -> Atom["PMap[int, Keyword]"]:
        return Atom(pmap())

    def test_empty_cache_no_completion(self, empty_cache: Atom["PMap[int, Keyword]"]):
        assert [] == list(complete(":", kw_cache=empty_cache))

    @pytest.fixture
    def cache(self) -> Atom["PMap[int, Keyword]"]:
        values = [Keyword("kw"), Keyword("ns"), Keyword("kw", ns="ns")]
        return Atom(pmap({hash(v): v for v in values}))

    def test_no_ns_completion(self, cache: Atom["PMap[int, Keyword]"]):
        assert [] == list(complete(":v", kw_cache=cache))
        assert {":kw", ":ns/kw"} == set(complete(":k", kw_cache=cache))
        assert {":kw", ":ns/kw"} == set(complete(":kw", kw_cache=cache))
        assert {":ns", ":ns/kw"} == set(complete(":n", kw_cache=cache))
        assert {":ns", ":ns/kw"} == set(complete(":ns", kw_cache=cache))

    def test_ns_completion(self, cache: Atom["PMap[int, Keyword]"]):
        assert [] == list(complete(":v/", kw_cache=cache))
        assert [] == list(complete(":k/", kw_cache=cache))
        assert [] == list(complete(":kw/", kw_cache=cache))
        assert [] == list(complete(":n/", kw_cache=cache))
        assert [":ns/kw"] == list(complete(":ns/", kw_cache=cache))
        assert [":ns/kw"] == list(complete(":ns/k", kw_cache=cache))
