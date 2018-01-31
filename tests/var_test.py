import pyrsistent
import pytest
import apylisp.lang.atom as atom
import apylisp.lang.keyword as kw
import apylisp.lang.namespace as namespace
import apylisp.lang.symbol as sym
import apylisp.lang.var as var
import apylisp.lang.vector as vec


@pytest.fixture
def ns_sym() -> sym.Symbol:
    return sym.symbol('some.ns')


@pytest.fixture
def var_name() -> sym.Symbol:
    return sym.symbol('var-val')


@pytest.fixture
def intern_val():
    return vec.v(kw.keyword("value"))


@pytest.fixture
def ns_cache(ns_sym: sym.Symbol) -> atom.Atom:
    return atom.Atom(pyrsistent.pmap({ns_sym: namespace.Namespace(ns_sym)}))


def test_intern(ns_sym: sym.Symbol, var_name: sym.Symbol, intern_val,
                ns_cache: atom.Atom):
    v = var.intern(ns_sym, var_name, intern_val, ns_cache=ns_cache)
    assert isinstance(v, var.Var)
    assert v.ns.name == ns_sym.name
    assert v.name == var_name
    assert not v.dynamic
    assert v.root == intern_val
    assert v.value == intern_val

    ns = namespace.get_or_create(ns_sym, ns_cache=ns_cache)
    assert ns is not None
    assert v == ns.find(var_name)


def test_find_in_ns(ns_sym: sym.Symbol, var_name: sym.Symbol, intern_val,
                    ns_cache: atom.Atom):
    v = var.intern(ns_sym, var_name, intern_val, ns_cache=ns_cache)
    v_in_ns = var.find_in_ns(ns_sym, var_name, ns_cache=ns_cache)
    assert v == v_in_ns


def test_find(ns_sym: sym.Symbol, var_name: sym.Symbol, intern_val,
              ns_cache: atom.Atom):
    v = var.intern(ns_sym, var_name, intern_val, ns_cache=ns_cache)
    ns_qualified_sym = sym.symbol(var_name.name, ns=ns_sym.name)
    v_in_ns = var.find(ns_qualified_sym, ns_cache=ns_cache)
    assert v == v_in_ns
