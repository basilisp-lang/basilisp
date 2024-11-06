import pytest

import basilisp.lang.interfaces
from basilisp.lang import atom as atom
from basilisp.lang import keyword as kw
from basilisp.lang import map as lmap
from basilisp.lang import runtime as runtime
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec
from basilisp.lang.exception import ExceptionInfo


@pytest.mark.parametrize(
    "interface",
    [
        basilisp.lang.interfaces.IDeref,
        basilisp.lang.interfaces.IRef,
        basilisp.lang.interfaces.IReference,
    ],
)
def test_atom_deref_interface(interface):
    assert isinstance(atom.Atom(1), interface)
    assert issubclass(atom.Atom, interface)


def test_atom():
    a = atom.Atom(vec.EMPTY)
    assert vec.EMPTY == a.deref()

    assert vec.v(1) == a.swap(lambda v, e: v.cons(e), 1)
    assert vec.v(1) == a.deref()

    assert vec.v(1, 2) == a.swap(lambda v, e: v.cons(e), 2)
    assert vec.v(1, 2) == a.deref()

    assert lmap.EMPTY == a.reset(lmap.EMPTY)
    assert lmap.EMPTY == a.deref()

    assert False is a.compare_and_set("not a map", vec.EMPTY)
    assert True is a.compare_and_set(lmap.EMPTY, "new string")
    assert "new string" == a.deref()


def test_alter_atom_meta():
    a = atom.Atom(None)
    assert a.meta is None

    a.alter_meta(runtime.assoc, "type", sym.symbol("str"))
    assert a.meta == lmap.m(type=sym.symbol("str"))

    a.alter_meta(runtime.assoc, "tag", kw.keyword("async"))
    assert a.meta == lmap.m(type=sym.symbol("str"), tag=kw.keyword("async"))


def test_reset_atom_meta():
    a = atom.Atom(None)
    assert a.meta is None

    a.reset_meta(lmap.map({"type": sym.symbol("str")}))
    assert a.meta == lmap.m(type=sym.symbol("str"))

    a.reset_meta(lmap.m(tag=kw.keyword("async")))
    assert a.meta == lmap.m(tag=kw.keyword("async"))


def test_atom_validators():
    with pytest.raises(ExceptionInfo):
        atom.Atom(None, validator=lambda v: v is not None)

    even_validator = lambda i: isinstance(i, int) and i % 2 == 0
    a = atom.Atom(0, validator=even_validator)
    assert even_validator == a.get_validator()
    assert 2 == a.swap(lambda i: i + 2)

    with pytest.raises(ExceptionInfo):
        a.swap(lambda i: i + 1)

    a.set_validator()
    assert None is a.get_validator()

    assert 1 == a.reset(1)

    odd_validator = lambda i: isinstance(i, int) and i % 2 == 1
    a.set_validator(odd_validator)

    with pytest.raises(ExceptionInfo):
        a.compare_and_set(1, 4)


def test_atom_watchers():
    a = atom.Atom(0)
    assert a is a.remove_watch("nonexistent-watch")

    watcher1_key = kw.keyword("watcher-the-first")
    watcher1_vals = []

    def watcher1(k, ref, old, new):
        assert watcher1_key is k
        assert a is ref
        watcher1_vals.append((old, new))

    a.add_watch(watcher1_key, watcher1)
    a.swap(lambda v: v * 2)  # == 0
    a.reset(4)  # == 4

    watcher2_key = kw.keyword("watcher-the-second")
    watcher2_vals = []

    def watcher2(k, ref, old, new):
        assert watcher2_key is k
        assert a is ref
        watcher2_vals.append((old, new))

    a.add_watch(watcher2_key, watcher2)
    a.swap(lambda v: v * 2)  # == 8

    a.remove_watch(watcher1_key)
    a.reset(10)  # == 10
    a.swap(lambda v: "a" * v)  # == "aaaaaaaaaa"

    # A failing CAS won't notify watches, so we'll test that:
    assert False is a.compare_and_set(10, "ten")

    assert [(0, 0), (0, 4), (4, 8)] == watcher1_vals
    assert [(4, 8), (8, 10), (10, "aaaaaaaaaa")] == watcher2_vals
