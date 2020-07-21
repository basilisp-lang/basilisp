import pickle

import pytest

from basilisp.lang import map as lmap
from basilisp.lang import queue
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
    assert isinstance(queue.q(), interface)
    assert issubclass(queue.PersistentQueue, interface)


def test_queue_bool():
    assert True is bool(queue.PersistentQueue.empty())


def test_queue_cons():
    meta = lmap.m(tag="async")
    q1 = queue.q(keyword("kw1"), meta=meta)
    q2 = q1.cons(keyword("kw2"))
    assert q1 is not q2
    assert q1 != q2
    assert q2 == queue.q(keyword("kw1"), keyword("kw2"))
    assert len(q2) == 2
    assert meta == q1.meta
    assert q2.meta is None
    q3 = q2.cons(3, "four")
    assert q3 == queue.q(keyword("kw1"), keyword("kw2"), 3, "four")


def test_queue_equals():
    q = queue.PersistentQueue.empty()
    assert q == q
    assert queue.q(1, 2, 3) != (1, 2, 3, 4)


def test_queue_peek():
    assert None is queue.PersistentQueue.empty().peek()

    assert 1 == queue.q(1).peek()
    assert 1 == queue.q(1, 2).peek()
    assert 1 == queue.q(1, 2, 3).peek()


def test_queue_pop():
    with pytest.raises(IndexError):
        queue.q().pop()

    assert queue.PersistentQueue.empty() == queue.q(1).pop()
    assert queue.q(2) == queue.q(1, 2).pop()
    assert queue.q(2, 3) == queue.q(1, 2, 3).pop()


def test_queue_meta():
    assert queue.q("vec").meta is None
    meta = lmap.m(type=symbol("str"))
    assert queue.q("vec", meta=meta).meta == meta


def test_queue_with_meta():
    q1 = queue.q("vec")
    assert q1.meta is None

    meta1 = lmap.m(type=symbol("str"))
    q2 = queue.q("vec", meta=meta1)
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


@pytest.mark.parametrize(
    "o",
    [
        queue.q(),
        queue.q(keyword("kw1")),
        queue.q(keyword("kw1"), 2),
        queue.q(keyword("kw1"), 2, None, "nothingness"),
        queue.q(keyword("kw1"), queue.q("string", 4)),
    ],
)
def test_queue_pickleability(pickle_protocol: int, o: queue.PersistentQueue):
    assert o == pickle.loads(pickle.dumps(o, protocol=pickle_protocol))


@pytest.mark.parametrize(
    "l,str_repr",
    [
        (queue.PersistentQueue.empty(), "#queue ()"),
        (queue.q(keyword("kw1")), "#queue (:kw1)"),
        (queue.q(keyword("kw1"), keyword("kw2")), "#queue (:kw1 :kw2)"),
    ],
)
def test_queue_repr(l: queue.PersistentQueue, str_repr: str):
    assert repr(l) == str_repr
