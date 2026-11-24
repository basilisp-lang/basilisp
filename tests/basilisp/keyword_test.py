import pickle
import secrets

import pytest

from basilisp.lang import map as lmap
from basilisp.lang import set as lset
from basilisp.lang import vector as lvector
from basilisp.lang.keyword import Keyword, complete, find_keyword, keyword


def test_keyword_identity_equals():
    assert keyword("kw") is keyword("kw")
    assert keyword("kw") == keyword("kw")

    assert keyword("kw", ns="some.ns") is not keyword("kw", ns="other.ns")
    assert keyword("kw", ns="some.ns") is not keyword("kw")


def test_find_keyword():
    existing_kw = keyword("existing")
    existing_ns_kw = keyword("existing", ns="with-ns")

    assert existing_kw is find_keyword("existing")
    assert existing_ns_kw is find_keyword("existing", ns="with-ns")

    assert None is find_keyword(f"k{secrets.token_hex(4)}")
    assert None is find_keyword(f"k{secrets.token_hex(4)}", ns="any-ns")


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

    assert kw == kw(lset.s(kw))
    assert None is kw(lset.s(1))
    assert "hi" == kw(lset.s(1), default="hi")

    assert 1 == kw(None, 1)
    assert None is kw(None, None)

    assert None is kw(lvector.v(1))


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


def _produce_keyword() -> tuple[Keyword, Keyword]:
    return keyword("key"), keyword("key", ns="ns")


def _consume_keyword(t: tuple[Keyword, Keyword]):
    key, namespaced = t
    return (
        keyword("key") is key
        and find_keyword("key")
        and keyword("key", ns="ns") is namespaced
        and find_keyword("key", ns="ns")
    )


def test_keyword_pickleability_cross_process(pickle_cross_process):
    assert pickle_cross_process(_produce_keyword, _consume_keyword)


class TestKeywordCompletion:
    @pytest.fixture
    def empty_cache(self) -> dict[tuple[str, str | None], Keyword]:
        return {}

    def test_empty_cache_no_completion(
        self, empty_cache: dict[tuple[str, str | None], Keyword]
    ):
        assert [] == list(complete(":", kw_cache=empty_cache))

    @pytest.fixture
    def cache(self) -> dict[tuple[str, str | None], Keyword]:
        values = [Keyword("kw"), Keyword("ns"), Keyword("kw", ns="ns")]
        return {(v.name, v.ns): v for v in values}

    def test_no_ns_completion(self, cache: dict[tuple[str, str | None], Keyword]):
        assert [] == list(complete(":v", kw_cache=cache))
        assert {":kw", ":ns/kw"} == set(complete(":k", kw_cache=cache))
        assert {":kw", ":ns/kw"} == set(complete(":kw", kw_cache=cache))
        assert {":ns", ":ns/kw"} == set(complete(":n", kw_cache=cache))
        assert {":ns", ":ns/kw"} == set(complete(":ns", kw_cache=cache))

    def test_ns_completion(self, cache: dict[tuple[str, str | None], Keyword]):
        assert [] == list(complete(":v/", kw_cache=cache))
        assert [] == list(complete(":k/", kw_cache=cache))
        assert [] == list(complete(":kw/", kw_cache=cache))
        assert [] == list(complete(":n/", kw_cache=cache))
        assert [":ns/kw"] == list(complete(":ns/", kw_cache=cache))
        assert [":ns/kw"] == list(complete(":ns/k", kw_cache=cache))
