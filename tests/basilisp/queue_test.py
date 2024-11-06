import pickle

import pytest

from basilisp.lang import map as lmap
from basilisp.lang import queue as lqueue
from basilisp.lang.interfaces import (
    ILispObject,
    IPersistentCollection,
    IPersistentList,
    IPersistentStack,
    ISeqable,
    ISequential,
    IWithMeta,
)
from basilisp.lang.keyword import keyword
from basilisp.lang.symbol import symbol


@pytest.mark.parametrize(
    "interface",
    [
        ILispObject,
        IPersistentCollection,
        IPersistentList,
        IPersistentStack,
        ISeqable,
        ISequential,
        IWithMeta,
    ],
)
def test_queue_interface_membership(interface):
    assert isinstance(lqueue.q(), interface)
    assert issubclass(lqueue.PersistentQueue, interface)


def test_queue_bool():
    assert True is bool(lqueue.EMPTY)


def test_queue_cons():
    meta = lmap.m(tag="async")
    q1 = lqueue.q(keyword("kw1"), meta=meta)
    q2 = q1.cons(keyword("kw2"))
    assert q1 is not q2
    assert q1 != q2
    assert q2 == lqueue.q(keyword("kw1"), keyword("kw2"))
    assert len(q2) == 2
    assert meta == q1.meta
    assert q2.meta is None
    q3 = q2.cons(3, "four")
    assert q3 == lqueue.q(keyword("kw1"), keyword("kw2"), 3, "four")


def test_queue_equals():
    q = lqueue.EMPTY
    assert q == q
    assert lqueue.q(1, 2, 3) != (1, 2, 3, 4)


def test_queue_peek():
    assert None is lqueue.EMPTY.peek()

    assert 1 == lqueue.q(1).peek()
    assert 1 == lqueue.q(1, 2).peek()
    assert 1 == lqueue.q(1, 2, 3).peek()


def test_queue_pop():
    with pytest.raises(IndexError):
        lqueue.q().pop()

    assert lqueue.EMPTY == lqueue.q(1).pop()
    assert lqueue.q(2) == lqueue.q(1, 2).pop()
    assert lqueue.q(2, 3) == lqueue.q(1, 2, 3).pop()


def test_queue_meta():
    assert lqueue.q("vec").meta is None
    meta = lmap.m(type=symbol("str"))
    assert lqueue.q("vec", meta=meta).meta == meta


def test_queue_with_meta():
    q1 = lqueue.q("vec")
    assert q1.meta is None

    meta1 = lmap.m(type=symbol("str"))
    q2 = lqueue.q("vec", meta=meta1)
    assert q2.meta == meta1

    meta2 = lmap.m(tag=keyword("async"))
    q3 = q2.with_meta(meta2)
    assert q2 is not q3
    assert q2 == q3
    assert q3.meta == lmap.m(tag=keyword("async"))

    meta3 = lmap.m(tag=keyword("macro"))
    q4 = q3.with_meta(meta3)
    assert q3 is not q4
    assert q3 == q4
    assert q4.meta == lmap.m(tag=keyword("macro"))


def test_queue_seq():
    assert None is lqueue.EMPTY.seq()
    assert lqueue.q(1) == lqueue.q(1).seq()
    assert lqueue.q(1, 2) == lqueue.q(1, 2).seq()
    assert lqueue.q(1, 2, 3) == lqueue.q(1, 2, 3).seq()


@pytest.mark.parametrize(
    "o",
    [
        lqueue.q(),
        lqueue.q(keyword("kw1")),
        lqueue.q(keyword("kw1"), 2),
        lqueue.q(keyword("kw1"), 2, None, "nothingness"),
        lqueue.q(keyword("kw1"), lqueue.q("string", 4)),
    ],
)
def test_queue_pickleability(pickle_protocol: int, o: lqueue.PersistentQueue):
    assert o == pickle.loads(pickle.dumps(o, protocol=pickle_protocol))


@pytest.mark.parametrize(
    "l,str_repr",
    [
        (lqueue.EMPTY, "#queue ()"),
        (lqueue.q(keyword("kw1")), "#queue (:kw1)"),
        (lqueue.q(keyword("kw1"), keyword("kw2")), "#queue (:kw1 :kw2)"),
    ],
)
def test_queue_repr(l: lqueue.PersistentQueue, str_repr: str):
    assert repr(l) == str_repr
