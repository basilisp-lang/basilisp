import basilisp.lang.interfaces
from basilisp.lang import atom as atom
from basilisp.lang import keyword as kw
from basilisp.lang import map as lmap
from basilisp.lang import runtime as runtime
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec


def test_atom_deref_interface():
    assert isinstance(atom.Atom(1), basilisp.lang.interfaces.IDeref)
    assert issubclass(atom.Atom, basilisp.lang.interfaces.IDeref)


def test_atom():
    a = atom.Atom(vec.PersistentVector.empty())
    assert vec.PersistentVector.empty() == a.deref()

    assert vec.v(1) == a.swap(lambda v, e: v.cons(e), 1)
    assert vec.v(1) == a.deref()

    assert vec.v(1, 2) == a.swap(lambda v, e: v.cons(e), 2)
    assert vec.v(1, 2) == a.deref()

    assert lmap.PersistentMap.empty() == a.reset(lmap.PersistentMap.empty())
    assert lmap.PersistentMap.empty() == a.deref()


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
