import platform
import sys
from decimal import Decimal
from fractions import Fraction

import pytest

from basilisp.lang import atom as atom
from basilisp.lang import keyword as kw
from basilisp.lang import list as llist
from basilisp.lang import map as lmap
from basilisp.lang import queue as lqueue
from basilisp.lang import runtime as runtime
from basilisp.lang import seq as lseq
from basilisp.lang import set as lset
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec
from basilisp.lang.compiler.constants import SpecialForm
from basilisp.lang.interfaces import ISeq
from basilisp.lang.reduced import Reduced
from tests.basilisp.helpers import get_or_create_ns


def test_is_supported_python_version():
    v = sys.version_info
    assert (v.major, v.minor) in runtime.SUPPORTED_PYTHON_VERSIONS


@pytest.mark.parametrize(
    "feature",
    {
        (3, 8): frozenset(
            map(
                kw.keyword,
                [
                    "lpy38",
                    "default",
                    "lpy",
                    "lpy38-",
                    "lpy38+",
                    "lpy39-",
                    "lpy310-",
                    "lpy311-",
                    "lpy312-",
                    "lpy313-",
                    platform.system().lower(),
                ],
            )
        ),
        (3, 9): frozenset(
            map(
                kw.keyword,
                [
                    "lpy39",
                    "default",
                    "lpy",
                    "lpy39-",
                    "lpy39+",
                    "lpy38+",
                    "lpy310-",
                    "lpy311-",
                    "lpy312-",
                    "lpy313-",
                    platform.system().lower(),
                ],
            )
        ),
        (3, 10): frozenset(
            map(
                kw.keyword,
                [
                    "lpy310",
                    "default",
                    "lpy",
                    "lpy313-",
                    "lpy312-",
                    "lpy311-",
                    "lpy310+",
                    "lpy310-",
                    "lpy39+",
                    "lpy38+",
                    platform.system().lower(),
                ],
            )
        ),
        (3, 11): frozenset(
            map(
                kw.keyword,
                [
                    "lpy311",
                    "default",
                    "lpy",
                    "lpy311+",
                    "lpy313-",
                    "lpy312-",
                    "lpy311-",
                    "lpy310+",
                    "lpy39+",
                    "lpy38+",
                    platform.system().lower(),
                ],
            )
        ),
        (3, 12): frozenset(
            map(
                kw.keyword,
                [
                    "lpy312",
                    "default",
                    "lpy",
                    "lpy311+",
                    "lpy312+",
                    "lpy313-",
                    "lpy312-",
                    "lpy39+",
                    "lpy38+",
                    platform.system().lower(),
                ],
            )
        ),
        (3, 13): frozenset(
            map(
                kw.keyword,
                [
                    "lpy313",
                    "default",
                    "lpy",
                    "lpy311+",
                    "lpy312+",
                    "lpy313+",
                    "lpy313-",
                    "lpy310+",
                    "lpy39+",
                    "lpy38+",
                    platform.system().lower(),
                ],
            )
        ),
    }[(sys.version_info.major, sys.version_info.minor)],
)
def test_reader_default_featureset(feature):
    assert feature in runtime.READER_COND_DEFAULT_FEATURE_SET


def test_first():
    assert None is runtime.first(None)
    assert None is runtime.first(llist.l())
    assert 1 == runtime.first(llist.l(1))
    assert 1 == runtime.first(llist.l(1, 2, 3))
    assert 1 == runtime.first(vec.v(1).seq())
    assert 1 == runtime.first(vec.v(1))
    assert 1 == runtime.first(vec.v(1, 2, 3))


def test_rest():
    assert lseq.EMPTY is runtime.rest(None)
    assert lseq.EMPTY is runtime.rest(llist.l())
    assert lseq.EMPTY is runtime.rest(llist.l(1))
    assert llist.l(2, 3) == runtime.rest(llist.l(1, 2, 3))
    v1 = runtime.rest(vec.v(1).seq())
    assert lseq.EMPTY == v1
    assert v1.is_empty
    v2 = runtime.rest(vec.v(1))
    assert lseq.EMPTY == v2
    assert v2.is_empty
    assert llist.l(2, 3) == runtime.rest(vec.v(1, 2, 3))


def test_nthrest():
    assert None is runtime.nthrest(None, 1)

    assert llist.PersistentList.empty() == runtime.nthrest(
        llist.PersistentList.empty(), 0
    )
    assert lseq.sequence([2, 3, 4, 5, 6]) == runtime.nthrest(
        llist.l(1, 2, 3, 4, 5, 6), 1
    )
    assert lseq.sequence([3, 4, 5, 6]) == runtime.nthrest(llist.l(1, 2, 3, 4, 5, 6), 2)
    assert lseq.sequence([4, 5, 6]) == runtime.nthrest(llist.l(1, 2, 3, 4, 5, 6), 3)
    assert lseq.sequence([5, 6]) == runtime.nthrest(llist.l(1, 2, 3, 4, 5, 6), 4)
    assert lseq.sequence([6]) == runtime.nthrest(llist.l(1, 2, 3, 4, 5, 6), 5)

    assert vec.PersistentVector.empty() == runtime.nthrest(
        vec.PersistentVector.empty(), 0
    )
    assert lseq.sequence([2, 3, 4, 5, 6]) == runtime.nthrest(vec.v(1, 2, 3, 4, 5, 6), 1)
    assert lseq.sequence([3, 4, 5, 6]) == runtime.nthrest(vec.v(1, 2, 3, 4, 5, 6), 2)
    assert lseq.sequence([4, 5, 6]) == runtime.nthrest(vec.v(1, 2, 3, 4, 5, 6), 3)
    assert lseq.sequence([5, 6]) == runtime.nthrest(vec.v(1, 2, 3, 4, 5, 6), 4)
    assert lseq.sequence([6]) == runtime.nthrest(vec.v(1, 2, 3, 4, 5, 6), 5)


def test_next():
    assert None is runtime.next_(None)
    assert None is runtime.next_(llist.l())
    assert None is runtime.next_(llist.l(1))
    assert llist.l(2, 3) == runtime.next_(llist.l(1, 2, 3))
    assert None is runtime.next_(vec.v(1).seq())
    assert None is runtime.next_(vec.v(1))
    assert llist.l(2, 3) == runtime.next_(vec.v(1, 2, 3))


def test_nthnext():
    assert None is runtime.nthnext(None, 1)

    assert None is runtime.nthnext(llist.PersistentList.empty(), 0)
    assert lseq.sequence([2, 3, 4, 5, 6]) == runtime.nthnext(
        llist.l(1, 2, 3, 4, 5, 6), 1
    )
    assert lseq.sequence([3, 4, 5, 6]) == runtime.nthnext(llist.l(1, 2, 3, 4, 5, 6), 2)
    assert lseq.sequence([4, 5, 6]) == runtime.nthnext(llist.l(1, 2, 3, 4, 5, 6), 3)
    assert lseq.sequence([5, 6]) == runtime.nthnext(llist.l(1, 2, 3, 4, 5, 6), 4)
    assert lseq.sequence([6]) == runtime.nthnext(llist.l(1, 2, 3, 4, 5, 6), 5)

    assert None is runtime.nthnext(vec.PersistentVector.empty(), 0)
    assert lseq.sequence([2, 3, 4, 5, 6]) == runtime.nthnext(vec.v(1, 2, 3, 4, 5, 6), 1)
    assert lseq.sequence([3, 4, 5, 6]) == runtime.nthnext(vec.v(1, 2, 3, 4, 5, 6), 2)
    assert lseq.sequence([4, 5, 6]) == runtime.nthnext(vec.v(1, 2, 3, 4, 5, 6), 3)
    assert lseq.sequence([5, 6]) == runtime.nthnext(vec.v(1, 2, 3, 4, 5, 6), 4)
    assert lseq.sequence([6]) == runtime.nthnext(vec.v(1, 2, 3, 4, 5, 6), 5)


def test_cons():
    assert llist.l(None) == runtime.cons(None, None)
    assert llist.l(1) == runtime.cons(1, None)
    assert llist.l(1) == runtime.cons(1, llist.l())
    assert llist.l(1, 2, 3) == runtime.cons(1, llist.l(2, 3))
    assert llist.l(1, 2, 3) == runtime.cons(1, vec.v(2, 3))
    assert llist.l(1, 2, 3) == runtime.cons(1, vec.v(2, 3).seq())


def test_to_seq():
    assert None is runtime.to_seq(None)
    assert None is runtime.to_seq(llist.PersistentList.empty())
    assert None is runtime.to_seq(vec.PersistentVector.empty())
    assert None is runtime.to_seq(lmap.PersistentMap.empty())
    assert None is runtime.to_seq(lset.PersistentSet.empty())
    assert None is runtime.to_seq("")

    assert None is not runtime.to_seq(llist.l(None))
    assert None is not runtime.to_seq(llist.l(None, None, None))
    assert None is not runtime.to_seq(llist.l(1))
    assert None is not runtime.to_seq(vec.v(1))
    assert None is not runtime.to_seq(lmap.map({"a": 1}))
    assert None is not runtime.to_seq(lset.s(1))
    assert None is not runtime.to_seq("string")

    one_elem = llist.l(kw.keyword("kw"))
    assert one_elem == runtime.to_seq(one_elem)

    seqable = vec.v(kw.keyword("kw"))
    assert seqable == runtime.to_seq(seqable)

    v1 = vec.v(kw.keyword("kw"), 1, llist.l("something"), 3)
    s1 = runtime.to_seq(v1)
    assert isinstance(s1, ISeq)
    for v, s in zip(v1, s1):
        assert v == s

    py_list = [1, 2, 3]
    assert llist.l(1, 2, 3) == runtime.to_seq(py_list)


def test_concat():
    s1 = runtime.concat()
    assert lseq.EMPTY == s1
    assert s1.is_empty

    s1 = runtime.concat(llist.PersistentList.empty(), llist.PersistentList.empty())
    assert lseq.EMPTY == s1

    s1 = runtime.concat(llist.PersistentList.empty(), llist.l(1, 2, 3))
    assert s1 == llist.l(1, 2, 3)

    s1 = runtime.concat(llist.l(1, 2, 3), vec.v(4, 5, 6))
    assert s1 == llist.l(1, 2, 3, 4, 5, 6)

    s1 = runtime.concat(lmap.map({"a": 1}), lmap.map({"b": 2}))
    assert s1 == llist.l(vec.v("a", 1), vec.v("b", 2))

    s1 = runtime.concat(vec.v(1, 2), None, "ab")
    assert s1 == llist.l(1, 2, "a", "b")


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


def test_internal_reduce_type_errors(add):
    with pytest.raises(TypeError):
        runtime.internal_reduce(3, add)

    with pytest.raises(TypeError):
        runtime.internal_reduce(False, add, 8)


@pytest.mark.parametrize(
    "coll,res",
    [
        (None, 0),
        (lseq.EMPTY, 0),
        (lseq.sequence([1]), 1),
        (lseq.sequence([1, 2, 3]), 6),
        (vec.vector([1, 2, 3]), 6),
        (lseq.sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 28),
    ],
)
def test_internal_reduce(add, coll, res):
    assert runtime.internal_reduce(coll, add) == res


@pytest.mark.parametrize(
    "coll,res, init",
    [
        (None, 45, 45),
        (lseq.sequence([]), 45, 45),
        (lseq.sequence([1]), 46, 45),
        (lseq.sequence([1, 2, 3]), 10, 4),
        (vec.vector([1, 2, 3]), 10, 4),
        (lseq.sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 46, 45),
    ],
)
def test_internal_reduce_init(add, coll, res, init):
    assert runtime.internal_reduce(coll, add, init) == res


def test_apply():
    assert vec.v() == runtime.apply(vec.v, [[]])
    assert vec.v(1, 2, 3) == runtime.apply(vec.v, [[1, 2, 3]])
    assert vec.v(None, None, None) == runtime.apply(vec.v, [[None, None, None]])
    assert vec.v(vec.v(1, 2, 3), 4, 5, 6) == runtime.apply(
        vec.v, [vec.v(1, 2, 3), [4, 5, 6]]
    )
    assert vec.v(vec.v(1, 2, 3), None, None, None) == runtime.apply(
        vec.v, [vec.v(1, 2, 3), [None, None, None]]
    )


def test_count():
    assert 0 == runtime.count(None)
    assert 0 == runtime.count(vec.v())
    assert 0 == runtime.count("")
    assert 3 == runtime.count(vec.v(1, 2, 3))
    assert 3 == runtime.count("123")


def test_nth():
    assert None is runtime.nth(None, 1)
    assert "not found" == runtime.nth(None, 4, "not found")
    assert "l" == runtime.nth("hello world", 2)
    assert "l" == runtime.nth(["h", "e", "l", "l", "o"], 2)
    assert "l" == runtime.nth(llist.l("h", "e", "l", "l", "o"), 2)
    assert "l" == runtime.nth(vec.v("h", "e", "l", "l", "o"), 2)
    assert "l" == runtime.nth(lseq.sequence(["h", "e", "l", "l", "o"]), 2)

    assert "Z" == runtime.nth(llist.l("h", "e", "l", "l", "o"), 7, "Z")
    assert "Z" == runtime.nth(lseq.sequence(["h", "e", "l", "l", "o"]), 7, "Z")
    assert "Z" == runtime.nth(vec.v("h", "e", "l", "l", "o"), 7, "Z")

    with pytest.raises(IndexError):
        runtime.nth(llist.l("h", "e", "l", "l", "o"), 7)

    with pytest.raises(IndexError):
        runtime.nth(lseq.sequence(["h", "e", "l", "l", "o"]), 7)

    with pytest.raises(IndexError):
        runtime.nth(vec.v("h", "e", "l", "l", "o"), 7)

    with pytest.raises(TypeError):
        runtime.nth(lmap.PersistentMap.empty(), 2)

    with pytest.raises(TypeError):
        runtime.nth(lmap.map({"a": 1, "b": 2, "c": 3}), 2)

    with pytest.raises(TypeError):
        runtime.nth(lset.PersistentSet.empty(), 2)

    with pytest.raises(TypeError):
        runtime.nth(lset.s(1, 2, 3), 2)

    with pytest.raises(TypeError):
        runtime.nth(3, 1)

    with pytest.raises(TypeError):
        runtime.nth(3, 1, "Z")


def test_get():
    assert None is runtime.get(None, "a")
    assert kw.keyword("nada") is runtime.get(None, "a", kw.keyword("nada"))
    assert None is runtime.get(3, "a")
    assert kw.keyword("nada") is runtime.get(3, "a", kw.keyword("nada"))
    assert 1 == runtime.get(lmap.map({"a": 1}), "a")
    assert None is runtime.get(lmap.map({"a": 1}), "b")
    assert 2 == runtime.get(lmap.map({"a": 1}), "b", 2)
    assert 1 == runtime.get(lmap.map({"a": 1}).to_transient(), "a")
    assert None is runtime.get(lmap.map({"a": 1}).to_transient(), "b")
    assert 2 == runtime.get(lmap.map({"a": 1}).to_transient(), "b", 2)
    assert 1 == runtime.get(vec.v(1, 2, 3), 0)
    assert None is runtime.get(vec.v(1, 2, 3), 3)
    assert "nada" == runtime.get(vec.v(1, 2, 3), 3, "nada")
    assert 1 == runtime.get(vec.v(1, 2, 3).to_transient(), 0)
    assert None is runtime.get(vec.v(1, 2, 3).to_transient(), 3)
    assert "nada" == runtime.get(vec.v(1, 2, 3).to_transient(), 3, "nada")
    assert "l" == runtime.get("hello world", 2)
    assert None is runtime.get("hello world", 50)
    assert "nada" == runtime.get("hello world", 50, "nada")
    assert "l" == runtime.get(["h", "e", "l", "l", "o"], 2)
    assert None is runtime.get(["h", "e", "l", "l", "o"], 50)
    assert "nada" == runtime.get(["h", "e", "l", "l", "o"], 50, "nada")
    assert 1 == runtime.get({"a": 1}, "a")
    assert None is runtime.get({"a": 1}, "b")
    assert 2 == runtime.get({"a": 1}, "b", 2)
    assert "a" == runtime.get({"a", "b", "c"}, "a")
    assert None is runtime.get({"a", "b", "c"}, "d")
    assert 2 == runtime.get({"a", "b", "c"}, "d", 2)
    assert "a" == runtime.get(frozenset({"a", "b", "c"}), "a")
    assert None is runtime.get(frozenset({"a", "b", "c"}), "d")
    assert 2 == runtime.get(frozenset({"a", "b", "c"}), "d", 2)
    assert "a" == runtime.get(lset.set({"a", "b", "c"}), "a")
    assert None is runtime.get(lset.set({"a", "b", "c"}), "d")
    assert 2 == runtime.get(lset.set({"a", "b", "c"}), "d", 2)
    assert "a" == runtime.get(lset.set({"a", "b", "c"}).to_transient(), "a")
    assert None is runtime.get(lset.set({"a", "b", "c"}).to_transient(), "d")
    assert 2 == runtime.get(lset.set({"a", "b", "c"}).to_transient(), "d", 2)

    # lists are "supported" by virtue of the fact that `get`-ing them does not fail
    assert None is runtime.get(llist.l(1, 2, 3), 0)
    assert None is runtime.get(llist.l(1, 2, 3), 3)
    assert "nada" == runtime.get(llist.l(1, 2, 3), 0, "nada")


def test_assoc():
    assert lmap.PersistentMap.empty() == runtime.assoc(None)
    assert lmap.map({"a": 1}) == runtime.assoc(None, "a", 1)
    assert lmap.map({"a": 8}) == runtime.assoc(lmap.map({"a": 1}), "a", 8)
    assert lmap.map({"a": 1, "b": "string"}) == runtime.assoc(
        lmap.map({"a": 1}), "b", "string"
    )

    assert vec.v("a") == runtime.assoc(vec.PersistentVector.empty(), 0, "a")
    assert vec.v("c", "b") == runtime.assoc(vec.v("a", "b"), 0, "c")
    assert vec.v("a", "c") == runtime.assoc(vec.v("a", "b"), 1, "c")

    with pytest.raises(IndexError):
        runtime.assoc(vec.PersistentVector.empty(), 1, "a")

    with pytest.raises(TypeError):
        runtime.assoc(llist.PersistentList.empty(), 1, "a")


def test_update():
    assert lmap.map({"a": 1}) == runtime.update(None, "a", lambda _: 1)
    assert lmap.map({"a": 50}) == runtime.update(None, "a", lambda _, x: x, 50)

    assert lmap.map({"a": 2}) == runtime.update(
        lmap.map({"a": 1}), "a", lambda x: x + 1
    )
    assert lmap.map({"a": 4}) == runtime.update(
        lmap.map({"a": 1}), "a", lambda x, y: x * y + 1, 3
    )

    assert lmap.map({"a": 1, "b": "string"}) == runtime.update(
        lmap.map({"a": 1}), "b", lambda _: "string"
    )
    assert lmap.map({"a": 1, "b": "string"}) == runtime.update(
        lmap.map({"a": 1, "b": 583}), "b", lambda _: "string"
    )

    assert vec.v("a") == runtime.update(vec.PersistentVector.empty(), 0, lambda _: "a")
    assert vec.v("yay", "b") == runtime.update(vec.v("a", "b"), 0, lambda x: f"y{x}y")
    assert vec.v("a", "boy") == runtime.update(
        vec.v("a", "b"), 1, lambda x, y: f"{x}{y}", "oy"
    )

    with pytest.raises(TypeError):
        runtime.update(llist.PersistentList.empty(), 1, lambda _: "y")


def test_conj():
    assert llist.l(1) == runtime.conj(None, 1)
    assert llist.l(3, 2, 1) == runtime.conj(None, 1, 2, 3)
    assert llist.l(llist.l(1, 2, 3)) == runtime.conj(None, llist.l(1, 2, 3))

    assert llist.l(1) == runtime.conj(llist.PersistentList.empty(), 1)
    assert llist.l(3, 2, 1) == runtime.conj(llist.PersistentList.empty(), 1, 2, 3)
    assert llist.l(3, 2, 1, 1) == runtime.conj(llist.l(1), 1, 2, 3)
    assert llist.l(llist.l(1, 2, 3), 1) == runtime.conj(llist.l(1), llist.l(1, 2, 3))

    assert lset.s(1) == runtime.conj(lset.PersistentSet.empty(), 1)
    assert lset.s(1, 2, 3) == runtime.conj(lset.PersistentSet.empty(), 1, 2, 3)
    assert lset.s(1, 2, 3) == runtime.conj(lset.s(1), 1, 2, 3)
    assert lset.s(1, lset.s(1, 2, 3)) == runtime.conj(lset.s(1), lset.s(1, 2, 3))

    assert vec.v(1) == runtime.conj(vec.PersistentVector.empty(), 1)
    assert vec.v(1, 2, 3) == runtime.conj(vec.PersistentVector.empty(), 1, 2, 3)
    assert vec.v(1, 1, 2, 3) == runtime.conj(vec.v(1), 1, 2, 3)
    assert vec.v(1, vec.v(1, 2, 3)) == runtime.conj(vec.v(1), vec.v(1, 2, 3))

    assert lmap.map({"a": 1}) == runtime.conj(lmap.PersistentMap.empty(), ["a", 1])
    assert lmap.map({"a": 1, "b": 93}) == runtime.conj(
        lmap.PersistentMap.empty(), ["a", 1], ["b", 93]
    )
    assert lmap.map({"a": 1, "b": 93}) == runtime.conj(
        lmap.map({"a": 8}), ["a", 1], ["b", 93]
    )

    with pytest.raises(ValueError):
        runtime.conj(lmap.map({"a": 8}), "a", 1, "b", 93)

    with pytest.raises(TypeError):
        runtime.conj(3, 1, "a")

    with pytest.raises(TypeError):
        runtime.conj("b", 1, "a")


def test_deref():
    assert 1 == runtime.deref(atom.Atom(1))
    assert vec.PersistentVector.empty() == runtime.deref(
        atom.Atom(vec.PersistentVector.empty())
    )

    with pytest.raises(TypeError):
        runtime.deref(1)

    with pytest.raises(TypeError):
        runtime.deref(vec.PersistentVector.empty())


@pytest.mark.parametrize(
    "v1,v2",
    [
        (0, 0),
        (1, 1),
        (-1, -1),
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (0.0, 0),
        (1.0, 1),
        (-1.0, -1),
        (True, True),
        (False, False),
        (None, None),
        ("", ""),
        ("not empty", "not empty"),
        (Fraction("1/2"), Fraction("1/2")),
        (Decimal("3.14159"), Decimal("3.14159")),
        (llist.PersistentList.empty(), llist.PersistentList.empty()),
        (llist.l(1, 2, 3), llist.l(1, 2, 3)),
        (lmap.PersistentMap.empty(), lmap.PersistentMap.empty()),
        (lmap.map({"a": 1, "b": 2}), lmap.map({"a": 1, "b": 2})),
        (lset.PersistentSet.empty(), lset.PersistentSet.empty()),
        (lqueue.PersistentQueue.empty(), lqueue.PersistentQueue.empty()),
        (lqueue.q(1, 2, 3), lqueue.q(1, 2, 3)),
        (lset.s(1, 2, 3), lset.s(1, 2, 3)),
        (vec.PersistentVector.empty(), vec.PersistentVector.empty()),
        (vec.v(1, 2, 3), vec.v(1, 2, 3)),
        (lseq.EMPTY, lseq.EMPTY),
        (lseq.EMPTY.cons(3).cons(2).cons(1), lseq.EMPTY.cons(3).cons(2).cons(1)),
        (lqueue.q(1, 2, 3), lseq.EMPTY.cons(3).cons(2).cons(1)),
        (vec.v(1, 2, 3), lseq.EMPTY.cons(3).cons(2).cons(1)),
        (llist.PersistentList.empty(), vec.PersistentVector.empty()),
        (llist.l(1, 2, 3), vec.v(1, 2, 3)),
        (lseq.EMPTY, vec.PersistentVector.empty()),
        (lqueue.q(1, 2, 3), vec.v(1, 2, 3)),
        (lqueue.q(1, 2, 3), llist.l(1, 2, 3)),
        (lseq.EMPTY, lqueue.PersistentQueue.empty()),
        (lseq.EMPTY.cons(3).cons(2).cons(1), vec.v(1, 2, 3)),
        (llist.PersistentList.empty(), lseq.EMPTY),
        (lseq.EMPTY.cons(3).cons(2).cons(1), llist.l(1, 2, 3)),
    ],
)
def test_equals(v1, v2):
    assert runtime.equals(v1, v2)
    assert runtime.equals(v2, v1)


@pytest.mark.parametrize(
    "v1,v2",
    [
        (1, True),
        (0, False),
        ("", "not empty"),
        (1, -1),
        (0, 0.00000032),
        (llist.l(1, 2, 3), llist.l(2, 3, 4)),
        (llist.l(1, 2, 3), lqueue.q(2, 3, 4)),
        (llist.l(1, 2, 3), vec.v(2, 3, 4)),
        (llist.l(1, 2, 3), lseq.EMPTY.cons(4).cons(3).cons(2)),
        (lqueue.q(1, 2, 3), vec.v(2, 3, 4)),
        (lqueue.q(1, 2, 3), lseq.EMPTY.cons(4).cons(3).cons(2)),
        (vec.v(1, 2, 3), lseq.EMPTY.cons(4).cons(3).cons(2)),
        (lmap.PersistentMap.empty(), llist.PersistentList.empty()),
        (lmap.PersistentMap.empty(), vec.PersistentVector.empty()),
        (lmap.PersistentMap.empty(), lseq.EMPTY),
        (lmap.PersistentMap.empty(), lqueue.PersistentQueue.empty()),
        (lmap.map({1: "1", 2: "2", 3: "3"}), llist.l(1, 2, 3)),
        (lmap.map({1: "1", 2: "2", 3: "3"}), lqueue.q(1, 2, 3)),
        (lmap.map({1: "1", 2: "2", 3: "3"}), vec.v(1, 2, 3)),
        (lmap.map({1: "1", 2: "2", 3: "3"}), lseq.EMPTY.cons(3).cons(2).cons(1)),
        (lset.PersistentSet.empty(), llist.PersistentList.empty()),
        (lset.PersistentSet.empty(), lmap.PersistentMap.empty()),
        (lset.PersistentSet.empty(), lqueue.PersistentQueue.empty()),
        (lset.PersistentSet.empty(), vec.PersistentVector.empty()),
        (lset.PersistentSet.empty(), lseq.EMPTY),
        (lset.s(1, 2, 3), llist.l(1, 2, 3)),
        (lset.s(1, 2, 3), lmap.PersistentMap.empty()),
        (lset.s(1, 2, 3), lqueue.q(1, 2, 3)),
        (lset.s(1, 2, 3), vec.v(1, 2, 3)),
        (lset.s(1, 2, 3), lseq.EMPTY.cons(3).cons(2).cons(1)),
    ],
)
def test_not_equals(v1, v2):
    assert not runtime.equals(v1, v2)
    assert not runtime.equals(v2, v1)


def test_pop_thread_bindings():
    with pytest.raises(runtime.RuntimeException):
        runtime.pop_thread_bindings()


class TestToPython:
    def test_literal_to_py(self):
        assert None is runtime.to_py(None)
        assert 1 == runtime.to_py(1)
        assert 1.6 == runtime.to_py(1.6)
        assert "string" == runtime.to_py("string")
        assert sym.symbol("sym") == runtime.to_py(sym.symbol("sym"))
        assert sym.symbol("sym", ns="ns") == runtime.to_py(sym.symbol("sym", ns="ns"))
        assert "kw" == runtime.to_py(kw.keyword("kw"))
        assert "kw" == runtime.to_py(kw.keyword("kw", ns="kw"))

    def test_to_dict(self):
        assert {} == runtime.to_py(lmap.PersistentMap.empty())
        assert {"a": 2} == runtime.to_py(lmap.map({"a": 2}))
        assert {"a": 2, "b": "string"} == runtime.to_py(
            lmap.map({"a": 2, kw.keyword("b"): "string"})
        )

    def test_to_list(self):
        assert [] == runtime.to_py(llist.PersistentList.empty())
        assert ["a", 2] == runtime.to_py(llist.l("a", 2))
        assert ["a", 2, None] == runtime.to_py(llist.l("a", 2, None))

        assert [] == runtime.to_py(vec.PersistentVector.empty())
        assert ["a", 2] == runtime.to_py(vec.v("a", 2))
        assert ["a", 2, None] == runtime.to_py(vec.v("a", 2, None))

        assert None is runtime.to_py(runtime.to_seq(vec.PersistentVector.empty()))
        assert ["a", 2] == runtime.to_py(runtime.to_seq(vec.v("a", 2)))
        assert ["a", 2, None] == runtime.to_py(runtime.to_seq(vec.v("a", 2, None)))

    def test_to_set(self):
        assert set() == runtime.to_py(lset.PersistentSet.empty())
        assert {"a", 2} == runtime.to_py(lset.set({"a", 2}))
        assert {"a", 2, "b"} == runtime.to_py(lset.set({"a", 2, kw.keyword("b")}))


class TestToLisp:
    def test_literal_to_lisp(self):
        assert None is runtime.to_lisp(None)
        assert 1 == runtime.to_lisp(1)
        assert 1.6 == runtime.to_lisp(1.6)
        assert "string" == runtime.to_lisp("string")
        assert sym.symbol("sym") == runtime.to_lisp(sym.symbol("sym"))
        assert sym.symbol("sym", ns="ns") == runtime.to_lisp(sym.symbol("sym", ns="ns"))
        assert kw.keyword("kw") == runtime.to_lisp(kw.keyword("kw"))
        assert kw.keyword("kw", ns="ns") == runtime.to_lisp(kw.keyword("kw", ns="ns"))

    def test_to_map(self):
        assert lmap.PersistentMap.empty() == runtime.to_lisp({})
        assert lmap.map({kw.keyword("a"): 2}) == runtime.to_lisp({"a": 2})
        assert lmap.map(
            {kw.keyword("a"): 2, kw.keyword("b"): "string"}
        ) == runtime.to_lisp({"a": 2, "b": "string"})
        assert lmap.map(
            {
                kw.keyword("a"): 2,
                kw.keyword("b"): lmap.map(
                    {
                        kw.keyword("c"): "string",
                        kw.keyword("d"): vec.v("list"),
                        kw.keyword("e"): lset.s("a", "set"),
                        kw.keyword("f"): vec.v("tuple", "not", "list"),
                    }
                ),
            }
        ) == runtime.to_lisp(
            {
                "a": 2,
                "b": {
                    "c": "string",
                    "d": ["list"],
                    "e": {"a", "set"},
                    kw.keyword("f"): ("tuple", "not", "list"),
                },
            }
        )

    def test_to_map_no_keywordize(self):
        assert lmap.PersistentMap.empty() == runtime.to_lisp({})
        assert lmap.map({"a": 2}) == runtime.to_lisp({"a": 2}, keywordize_keys=False)
        assert lmap.map({"a": 2, "b": "string"}) == runtime.to_lisp(
            {"a": 2, "b": "string"}, keywordize_keys=False
        )
        assert lmap.map(
            {
                "a": 2,
                "b": lmap.map(
                    {
                        "c": "string",
                        "d": vec.v("list"),
                        "e": lset.s("a", "set"),
                        kw.keyword("f"): vec.v("tuple", "not", "list"),
                    }
                ),
            }
        ) == runtime.to_lisp(
            {
                "a": 2,
                "b": {
                    "c": "string",
                    "d": ["list"],
                    "e": {"a", "set"},
                    kw.keyword("f"): ("tuple", "not", "list"),
                },
            },
            keywordize_keys=False,
        )

    def test_to_set(self):
        assert lset.PersistentSet.empty() == runtime.to_lisp(set())
        assert lset.set({"a", 2}) == runtime.to_lisp({"a", 2})
        assert lset.set({"a", 2, kw.keyword("b")}) == runtime.to_lisp(
            {"a", 2, kw.keyword("b")}
        )

    def test_to_vec(self):
        assert vec.PersistentVector.empty() == runtime.to_lisp([])
        assert vec.v("a", 2) == runtime.to_lisp(["a", 2])
        assert vec.v("a", 2, None) == runtime.to_lisp(["a", 2, None])

        assert vec.PersistentVector.empty() == runtime.to_lisp(
            vec.PersistentVector.empty()
        )
        assert vec.v("a", 2) == runtime.to_lisp(("a", 2))
        assert vec.v("a", 2, None) == runtime.to_lisp(("a", 2, None))


def test_trampoline_args():
    args = runtime._TrampolineArgs(True)
    assert () == args.args

    args = runtime._TrampolineArgs(False, llist.l(2, 3, 4))
    assert (llist.l(2, 3, 4),) == args.args

    args = runtime._TrampolineArgs(True, llist.l(2, 3, 4))
    assert (2, 3, 4) == args.args

    args = runtime._TrampolineArgs(False, 1, 2, 3, llist.l(4, 5, 6))
    assert (1, 2, 3, llist.l(4, 5, 6)) == args.args

    args = runtime._TrampolineArgs(True, 1, 2, 3, llist.l(4, 5, 6))
    assert (1, 2, 3, 4, 5, 6) == args.args

    args = runtime._TrampolineArgs(True, 1, llist.l(2, 3, 4), 5, 6)
    assert (1, llist.l(2, 3, 4), 5, 6) == args.args


@pytest.mark.parametrize("form", runtime._SPECIAL_FORMS)
def test_is_special_form(form: sym.Symbol):
    assert runtime.is_special_form(form)


class TestResolveAlias:
    @pytest.fixture
    def compiler_special_forms(self) -> lset.PersistentSet:
        return lset.set(
            [v for k, v in SpecialForm.__dict__.items() if isinstance(v, sym.Symbol)]
        )

    def test_runtime_and_compiler_special_forms_in_sync(
        self, compiler_special_forms: lset.PersistentSet
    ):
        assert compiler_special_forms == runtime._SPECIAL_FORMS

    def test_resolve_alias_does_not_resolve_special_forms(self):
        for form in runtime._SPECIAL_FORMS:
            assert form == runtime.resolve_alias(form)

    def test_resolve_alias(self, core_ns):
        ns_name = "resolve-test"
        ns_sym = sym.symbol(ns_name)

        with runtime.ns_bindings(ns_name) as ns:
            runtime.Var.intern(ns_sym, sym.symbol("existing-var"), None)
            assert sym.symbol("existing-var", ns=ns_name) == runtime.resolve_alias(
                sym.symbol("existing-var"), ns=ns
            )

            assert sym.symbol("non-existent-var", ns=ns_name) == runtime.resolve_alias(
                sym.symbol("non-existent-var"), ns=ns
            )

            foo_ns_sym = sym.symbol("zux.bar.foo")
            foo_ns = get_or_create_ns(foo_ns_sym)
            ns.add_alias(foo_ns, sym.symbol("foo"))
            assert sym.symbol(
                "aliased-var", ns=foo_ns_sym.name
            ) == runtime.resolve_alias(sym.symbol("aliased-var", ns="foo"), ns=ns)

            assert sym.symbol(
                "non-existent-alias-var", ns="wee.woo"
            ) == runtime.resolve_alias(
                sym.symbol("non-existent-alias-var", ns="wee.woo"), ns=ns
            )
