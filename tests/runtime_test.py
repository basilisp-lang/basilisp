import pytest

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
    assert llist.l() == runtime.rest(llist.l())
    assert llist.l() == runtime.rest(llist.l(1))
    assert llist.l(2, 3) == runtime.rest(llist.l(1, 2, 3))
    assert llist.l() == runtime.rest(vec.v(1).seq())
    assert llist.l() == runtime.rest(vec.v(1))
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
    assert None is runtime.next(None)
    assert None is runtime.next(llist.l())
    assert None is runtime.next(llist.l(1))
    assert llist.l(2, 3) == runtime.next(llist.l(1, 2, 3))
    assert None is runtime.next(vec.v(1).seq())
    assert None is runtime.next(vec.v(1))
    assert llist.l(2, 3) == runtime.next(vec.v(1, 2, 3))


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
    assert llist.l() == s1

    s1 = runtime.concat(llist.List.empty(), llist.List.empty())
    assert llist.l() == s1

    s1 = runtime.concat(llist.List.empty(), llist.l(1, 2, 3))
    assert s1 == llist.l(1, 2, 3)

    s1 = runtime.concat(llist.l(1, 2, 3), vec.v(4, 5, 6))
    assert s1 == llist.l(1, 2, 3, 4, 5, 6)


def test_apply():
    assert vec.v() == runtime.apply(vec.v, [[]])
    assert vec.v(1, 2, 3) == runtime.apply(vec.v, [[1, 2, 3]])
    assert vec.v(vec.v(1, 2, 3), 4, 5, 6) == runtime.apply(vec.v, [vec.v(1, 2, 3), [4, 5, 6]])


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
