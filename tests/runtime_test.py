import pytest

import basilisp.lang.atom as atom
import basilisp.lang.keyword as keyword
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.runtime as runtime
import basilisp.lang.seq as lseq
import basilisp.lang.set as lset
import basilisp.lang.vector as vec


def test_first():
    assert None is runtime.first(None)
    assert None is runtime.first(llist.l())
    assert 1 == runtime.first(llist.l(1))
    assert 1 == runtime.first(llist.l(1, 2, 3))
    assert 1 == runtime.first(vec.v(1).seq())
    assert 1 == runtime.first(vec.v(1))
    assert 1 == runtime.first(vec.v(1, 2, 3))


def test_rest():
    assert None is runtime.rest(None)
    assert lseq.EMPTY is runtime.rest(llist.l())
    assert lseq.EMPTY is runtime.rest(llist.l(1))
    assert llist.l(2, 3) == runtime.rest(llist.l(1, 2, 3))
    assert lseq.EMPTY is runtime.rest(vec.v(1).seq())
    assert lseq.EMPTY is runtime.rest(vec.v(1))
    assert llist.l(2, 3) == runtime.rest(vec.v(1, 2, 3))


def test_nthrest():
    assert None is runtime.nthrest(None, 1)

    assert llist.List.empty() == runtime.nthrest(llist.List.empty(), 0)
    assert lseq.sequence([2, 3, 4, 5, 6]) == runtime.nthrest(llist.l(1, 2, 3, 4, 5, 6), 1)
    assert lseq.sequence([3, 4, 5, 6]) == runtime.nthrest(llist.l(1, 2, 3, 4, 5, 6), 2)
    assert lseq.sequence([4, 5, 6]) == runtime.nthrest(llist.l(1, 2, 3, 4, 5, 6), 3)
    assert lseq.sequence([5, 6]) == runtime.nthrest(llist.l(1, 2, 3, 4, 5, 6), 4)
    assert lseq.sequence([6]) == runtime.nthrest(llist.l(1, 2, 3, 4, 5, 6), 5)

    assert vec.Vector.empty() == runtime.nthrest(vec.Vector.empty(), 0)
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

    assert None is runtime.nthnext(llist.List.empty(), 0)
    assert lseq.sequence([2, 3, 4, 5, 6]) == runtime.nthnext(llist.l(1, 2, 3, 4, 5, 6), 1)
    assert lseq.sequence([3, 4, 5, 6]) == runtime.nthnext(llist.l(1, 2, 3, 4, 5, 6), 2)
    assert lseq.sequence([4, 5, 6]) == runtime.nthnext(llist.l(1, 2, 3, 4, 5, 6), 3)
    assert lseq.sequence([5, 6]) == runtime.nthnext(llist.l(1, 2, 3, 4, 5, 6), 4)
    assert lseq.sequence([6]) == runtime.nthnext(llist.l(1, 2, 3, 4, 5, 6), 5)

    assert None is runtime.nthnext(vec.Vector.empty(), 0)
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
    assert None is runtime.to_seq(llist.List.empty())
    assert None is runtime.to_seq(vec.Vector.empty())
    assert None is runtime.to_seq(lmap.Map.empty())
    assert None is runtime.to_seq(lset.Set.empty())
    assert None is runtime.to_seq("")

    assert None is not runtime.to_seq(llist.l(None))
    assert None is not runtime.to_seq(llist.l(None, None, None))
    assert None is not runtime.to_seq(llist.l(1))
    assert None is not runtime.to_seq(vec.v(1))
    assert None is not runtime.to_seq(lmap.map({"a": 1}))
    assert None is not runtime.to_seq(lset.s(1))
    assert None is not runtime.to_seq("string")

    one_elem = llist.l(keyword.keyword('kw'))
    assert one_elem == runtime.to_seq(one_elem)

    seqable = vec.v(keyword.keyword('kw'))
    assert seqable == runtime.to_seq(seqable)

    v1 = vec.v(keyword.keyword('kw'), 1, llist.l("something"), 3)
    s1 = runtime.to_seq(v1)
    assert isinstance(s1, lseq.Seq)
    for v, s in zip(v1, s1):
        assert v == s

    py_list = [1, 2, 3]
    assert llist.l(1, 2, 3) == runtime.to_seq(py_list)


def test_concat():
    s1 = runtime.concat()
    assert lseq.EMPTY is s1

    s1 = runtime.concat(llist.List.empty(), llist.List.empty())
    assert lseq.EMPTY == s1

    s1 = runtime.concat(llist.List.empty(), llist.l(1, 2, 3))
    assert s1 == llist.l(1, 2, 3)

    s1 = runtime.concat(llist.l(1, 2, 3), vec.v(4, 5, 6))
    assert s1 == llist.l(1, 2, 3, 4, 5, 6)


def test_apply():
    assert vec.v() == runtime.apply(vec.v, [[]])
    assert vec.v(1, 2, 3) == runtime.apply(vec.v, [[1, 2, 3]])
    assert vec.v(None, None, None) == runtime.apply(vec.v, [[None, None, None]])
    assert vec.v(vec.v(1, 2, 3), 4, 5, 6) == runtime.apply(vec.v, [vec.v(1, 2, 3), [4, 5, 6]])
    assert vec.v(vec.v(1, 2, 3), None, None, None) == runtime.apply(vec.v, [vec.v(1, 2, 3), [None, None, None]])


def test_nth():
    assert None is runtime.nth(None, 1)
    assert "l" == runtime.nth("hello world", 2)
    assert "l" == runtime.nth(["h", "e", "l", "l", "o"], 2)
    assert "l" == runtime.nth(llist.l("h", "e", "l", "l", "o"), 2)
    assert "l" == runtime.nth(vec.v("h", "e", "l", "l", "o"), 2)
    assert "l" == runtime.nth(lseq.sequence(["h", "e", "l", "l", "o"]), 2)

    with pytest.raises(IndexError):
        runtime.nth(llist.l("h", "e", "l", "l", "o"), 7)

    with pytest.raises(IndexError):
        runtime.nth(lseq.sequence(["h", "e", "l", "l", "o"]), 7)

    with pytest.raises(TypeError):
        runtime.nth(3, 1)


def test_assoc():
    assert lmap.Map.empty() == runtime.assoc(None)
    assert lmap.map({"a": 1}) == runtime.assoc(None, "a", 1)
    assert lmap.map({"a": 8}) == runtime.assoc(lmap.map({"a": 1}), "a", 8)
    assert lmap.map({"a": 1, "b": "string"}) == runtime.assoc(lmap.map({"a": 1}), "b", "string")

    assert vec.v("a") == runtime.assoc(vec.Vector.empty(), 0, "a")
    assert vec.v("c", "b") == runtime.assoc(vec.v("a", "b"), 0, "c")
    assert vec.v("a", "c") == runtime.assoc(vec.v("a", "b"), 1, "c")

    with pytest.raises(IndexError):
        runtime.assoc(vec.Vector.empty(), 1, "a")

    with pytest.raises(TypeError):
        runtime.assoc(llist.List.empty(), 1, "a")


def test_conj():
    assert llist.l(1) == runtime.conj(None, 1)
    assert llist.l(3, 2, 1) == runtime.conj(None, 1, 2, 3)
    assert llist.l(llist.l(1, 2, 3)) == runtime.conj(None, llist.l(1, 2, 3))

    assert llist.l(1) == runtime.conj(llist.List.empty(), 1)
    assert llist.l(3, 2, 1) == runtime.conj(llist.List.empty(), 1, 2, 3)
    assert llist.l(3, 2, 1, 1) == runtime.conj(llist.l(1), 1, 2, 3)
    assert llist.l(llist.l(1, 2, 3), 1) == runtime.conj(llist.l(1), llist.l(1, 2, 3))

    assert lset.s(1) == runtime.conj(lset.Set.empty(), 1)
    assert lset.s(1, 2, 3) == runtime.conj(lset.Set.empty(), 1, 2, 3)
    assert lset.s(1, 2, 3) == runtime.conj(lset.s(1), 1, 2, 3)
    assert lset.s(1, lset.s(1, 2, 3)) == runtime.conj(lset.s(1), lset.s(1, 2, 3))

    assert vec.v(1) == runtime.conj(vec.Vector.empty(), 1)
    assert vec.v(1, 2, 3) == runtime.conj(vec.Vector.empty(), 1, 2, 3)
    assert vec.v(1, 1, 2, 3) == runtime.conj(vec.v(1), 1, 2, 3)
    assert vec.v(1, vec.v(1, 2, 3)) == runtime.conj(vec.v(1), vec.v(1, 2, 3))

    assert lmap.map({"a": 1}) == runtime.conj(lmap.Map.empty(), ["a", 1])
    assert lmap.map({"a": 1, "b": 93}) == runtime.conj(lmap.Map.empty(), ["a", 1], ["b", 93])
    assert lmap.map({"a": 1, "b": 93}) == runtime.conj(lmap.map({"a": 8}), ["a", 1], ["b", 93])

    with pytest.raises(ValueError):
        runtime.conj(lmap.map({"a": 8}), "a", 1, "b", 93)

    with pytest.raises(TypeError):
        runtime.conj(3, 1, "a")

    with pytest.raises(TypeError):
        runtime.conj("b", 1, "a")


def test_deref():
    assert 1 == runtime.deref(atom.Atom(1))
    assert vec.Vector.empty() == runtime.deref(atom.Atom(vec.Vector.empty()))

    with pytest.raises(TypeError):
        runtime.deref(1)

    with pytest.raises(TypeError):
        runtime.deref(vec.Vector.empty())


def test_swap():
    assert 3 == runtime.swap(atom.Atom(1), lambda x, y: x + y, 2)
    assert vec.v(1) == runtime.swap(atom.Atom(vec.Vector.empty()), lambda v, e: v.cons(e), 1)

    with pytest.raises(AttributeError):
        runtime.swap(1, lambda x, y: x + y, 2)


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
