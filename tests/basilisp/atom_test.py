import basilisp.lang.atom as atom
import basilisp.lang.interfaces
import basilisp.lang.keyword as kw
import basilisp.lang.map as lmap
import basilisp.lang.runtime as runtime
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec


def test_atom_deref_interface():
    assert isinstance(atom.Atom(1), basilisp.lang.interfaces.IDeref)
    assert issubclass(atom.Atom, basilisp.lang.interfaces.IDeref)


def test_atom():
    a = atom.Atom(vec.Vector.empty())
    assert vec.Vector.empty() == a.deref()

    assert vec.v(1) == a.swap(lambda v, e: v.cons(e), 1)
    assert vec.v(1) == a.deref()

    assert vec.v(1, 2) == a.swap(lambda v, e: v.cons(e), 2)
    assert vec.v(1, 2) == a.deref()

    assert vec.v(1, 2) == a.reset(lmap.Map.empty())
    assert lmap.Map.empty() == a.deref()


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
