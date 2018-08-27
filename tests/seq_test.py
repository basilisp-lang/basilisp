import basilisp.lang.list as llist
import basilisp.lang.seq as lseq


def test_to_sequence():
    assert lseq.EMPTY is lseq.sequence([])
    assert llist.l(None) == lseq.sequence([None])
    assert llist.l(1) == lseq.sequence([1])
    assert llist.l(1, 2, 3) == lseq.sequence([1, 2, 3])


def test_lazy_sequence():
    s = lseq.LazySeq(lambda: None)
    assert not s.is_empty, "LazySeq has not been realized yet"
    assert None is s.first
    assert lseq.EMPTY is s.rest
    assert s.is_realized
    assert s.is_empty, "LazySeq has been realized and is empty"

    s = lseq.LazySeq(lambda: lseq.EMPTY)
    assert not s.is_empty, "LazySeq has not been realized yet"
    assert None is s.first
    assert lseq.EMPTY is s.rest
    assert s.is_realized
    assert s.is_empty, "LazySeq has been realized and is empty"

    s = lseq.LazySeq(lambda: lseq.sequence([1]))
    assert not s.is_empty, "LazySeq has not been realized yet"
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
    assert not s.is_empty, "LazySeq has not been realized yet"
    assert 1 == s.first
    assert isinstance(s.rest, lseq.LazySeq)
    assert s.is_realized
    assert not s.is_empty, "LazySeq has been realized and is not empty"

    r = s.rest
    assert not r.is_empty, "LazySeq has not been realized yet"
    assert 2 == r.first
    assert isinstance(r.rest, lseq.LazySeq)
    assert r.is_realized
    assert not r.is_empty, "LazySeq has been realized and is not empty"

    t = r.rest
    assert not t.is_empty, "LazySeq has not been realized yet"
    assert 3 == t.first
    assert lseq.EMPTY is t.rest
    assert t.is_realized
    assert not t.is_empty, "LazySeq has been realized and is not empty"

    assert [1, 2, 3] == [e for e in s]


def test_empty_sequence():
    empty = lseq.sequence([])
    assert None is empty.first
    assert empty.rest == empty
    assert llist.l(1) == empty.cons(1)
    assert lseq.EMPTY is empty


def test_sequence():
    s = lseq.sequence([1])
    assert 1 == s.first
    assert lseq.EMPTY is s.rest
    assert llist.l(2, 1) == s.cons(2)
    assert [1, 2, 3] == [e for e in lseq.sequence([1, 2, 3])]
    assert llist.l(1, 2, 3) == lseq.sequence([1, 2, 3])
    assert llist.l(1, 2, 3) == lseq.sequence(llist.l(1, 2, 3))
    assert llist.l(1, 2, 3) == llist.list(lseq.sequence([1, 2, 3]))

    s = lseq.sequence([1, 2, 3])
    assert 2 == s.rest.first
    assert 3 == s.rest.rest.first
    assert None is s.rest.rest.rest.first
