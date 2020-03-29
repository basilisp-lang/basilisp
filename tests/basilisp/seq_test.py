import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.runtime as runtime
import basilisp.lang.seq as lseq
import basilisp.lang.vector as vec


def test_to_sequence():
    assert lseq.EMPTY is lseq.sequence([])
    assert lseq.sequence([]).is_empty
    assert llist.l(None) == lseq.sequence([None])
    assert not lseq.sequence([None]).is_empty
    assert llist.l(1) == lseq.sequence([1])
    assert not lseq.sequence([1]).is_empty
    assert llist.l(1, 2, 3) == lseq.sequence([1, 2, 3])
    assert not lseq.sequence([1, 2, 3]).is_empty


def test_lazy_sequence():
    s = lseq.LazySeq(lambda: None)
    assert s.is_empty
    assert None is s.first
    assert lseq.EMPTY is s.rest
    assert s.is_realized
    assert s.is_empty, "LazySeq has been realized and is empty"

    s = lseq.LazySeq(lambda: lseq.EMPTY)
    assert s.is_empty
    assert None is s.first
    assert lseq.EMPTY is s.rest
    assert s.is_realized
    assert s.is_empty, "LazySeq has been realized and is empty"

    s = lseq.LazySeq(lambda: lseq.sequence([1]))
    assert not s.is_empty
    assert 1 == s.first
    assert lseq.EMPTY is s.rest
    assert s.is_realized
    assert not s.is_empty, "LazySeq has been realized and is not empty"

    def lazy_seq():
        def inner_seq():
            def inner_inner_seq():
                return lseq.sequence([3])

            return lseq.LazySeq(inner_inner_seq).cons(2)

        return lseq.LazySeq(inner_seq).cons(1)

    s = lseq.LazySeq(lazy_seq)
    assert not s.is_empty
    assert 1 == s.first
    assert isinstance(s.rest, lseq.LazySeq)
    assert s.is_realized
    assert not s.is_empty, "LazySeq has been realized and is not empty"

    r = s.rest
    assert not r.is_empty
    assert 2 == r.first
    assert isinstance(r.rest, lseq.LazySeq)
    assert r.is_realized
    assert not r.is_empty, "LazySeq has been realized and is not empty"

    t = r.rest
    assert not t.is_empty
    assert 3 == t.first
    assert lseq.EMPTY is t.rest
    assert t.is_realized
    assert not t.is_empty, "LazySeq has been realized and is not empty"

    assert [1, 2, 3] == [e for e in s]


def test_empty_sequence():
    empty = lseq.sequence([])
    assert empty.is_empty
    assert None is empty.first
    assert empty.rest == empty
    assert llist.l(1) == empty.cons(1)
    assert lseq.EMPTY is empty


def test_sequence():
    s = lseq.sequence([1])
    assert not s.is_empty
    assert 1 == s.first
    assert lseq.EMPTY is s.rest
    assert llist.l(2, 1) == s.cons(2)
    assert [1, 2, 3] == [e for e in lseq.sequence([1, 2, 3])]
    assert llist.l(1, 2, 3) == lseq.sequence([1, 2, 3])
    assert llist.l(1, 2, 3) == lseq.sequence(llist.l(1, 2, 3))
    assert llist.l(1, 2, 3) == llist.list(lseq.sequence([1, 2, 3]))

    s = lseq.sequence([1, 2, 3])
    assert not s.is_empty
    assert 2 == s.rest.first
    assert 3 == s.rest.rest.first
    assert None is s.rest.rest.rest.first


def test_seq_iterator():
    s = lseq.sequence([])
    assert vec.Vector.empty() == vec.vector(s)

    s = lseq.sequence(range(10000))
    assert 10000 == len(vec.vector(s))


def test_seq_equals():
    # to_seq should be first to ensure that `ISeq.__eq__` is used
    assert runtime.to_seq(vec.v(1, 2, 3)) == llist.l(1, 2, 3)
    assert False is (runtime.to_seq(vec.v(1, 2, 3)) == kw.keyword("abc"))

    assert lseq.sequence(vec.v(1, 2, 3)) == llist.l(1, 2, 3)
    assert False is (lseq.sequence(vec.v(1, 2, 3)) == kw.keyword("abc"))
