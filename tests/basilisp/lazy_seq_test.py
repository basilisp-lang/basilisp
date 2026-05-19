from basilisp.lang.seq import EMPTY, LazySeq
from basilisp.lang.vector import vector


def test_lazy_seq():
    s = LazySeq(lambda: vector([1]), None)
    assert s.first == 1
    assert s.rest == EMPTY
