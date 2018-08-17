import basilisp.lang.list as llist
import basilisp.lang.seq as lseq


def test_to_sequence():
    assert lseq.empty() == lseq.sequence([])
    assert llist.l(1) == lseq.sequence([1])
    assert llist.l(1, 2, 3) == lseq.sequence([1, 2, 3])


def test_empty_sequence():
    empty = lseq.sequence([])
    assert None is empty.first
    assert empty.rest == empty
    assert llist.l(1) == empty.cons(1)
    assert lseq.empty() == empty


def test_sequence():
    s = lseq.sequence([1])
    assert 1 == s.first
    assert lseq.empty() == s.rest
    assert llist.l(2, 1) == s.cons(2)
    assert [1, 2, 3] == [e for e in lseq.sequence([1, 2, 3])]
    assert llist.l(1, 2, 3) == lseq.sequence([1, 2, 3])
    assert llist.l(1, 2, 3) == lseq.sequence(llist.l(1, 2, 3))
    assert llist.l(1, 2, 3) == llist.list(lseq.sequence([1, 2, 3]))

    s = lseq.sequence([1, 2, 3])
    assert 2 == s.rest.first
    assert 3 == s.rest.rest.first
    assert None is s.rest.rest.rest.first
