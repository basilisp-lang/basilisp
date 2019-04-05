import basilisp.lang.atom as atom
import basilisp.lang.interfaces
import basilisp.lang.map as lmap
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
