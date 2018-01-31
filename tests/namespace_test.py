import pyrsistent
import pytest
import apylisp.lang.atom as atom
import apylisp.lang.namespace as namespace
import apylisp.lang.symbol as sym


@pytest.fixture
def core_ns_sym():
    return sym.symbol(namespace._CORE_NS)


@pytest.fixture
def core_ns(core_ns_sym):
    ns = namespace.Namespace(core_ns_sym)
    return ns


@pytest.fixture
def ns_cache(core_ns_sym: sym.Symbol,
             core_ns: namespace.Namespace) -> atom.Atom:
    return atom.Atom(pyrsistent.pmap({core_ns_sym: core_ns}))


@pytest.fixture
def ns_sym() -> sym.Symbol:
    return sym.symbol("some.ns")


def test_create_ns(ns_sym: sym.Symbol, ns_cache: atom.Atom):
    assert len(ns_cache.deref().keys()) == 1
    ns = namespace.get_or_create(ns_sym, ns_cache=ns_cache)
    assert isinstance(ns, namespace.Namespace)
    assert ns.name == ns_sym.name
    assert len(ns_cache.deref().keys()) == 2


@pytest.fixture
def ns_cache_with_existing_ns(ns_sym: sym.Symbol, core_ns_sym: sym.Symbol,
                              core_ns: namespace.Namespace) -> atom.Atom:
    return atom.Atom(
        pyrsistent.pmap({
            core_ns_sym: core_ns,
            ns_sym: namespace.Namespace(ns_sym)
        }))


def test_get_existing_ns(ns_sym: sym.Symbol,
                         ns_cache_with_existing_ns: atom.Atom):
    assert len(ns_cache_with_existing_ns.deref().keys()) == 2
    ns = namespace.get_or_create(ns_sym, ns_cache=ns_cache_with_existing_ns)
    assert isinstance(ns, namespace.Namespace)
    assert ns.name == ns_sym.name
    assert len(ns_cache_with_existing_ns.deref().keys()) == 2


def test_remove_ns(ns_sym: sym.Symbol, ns_cache_with_existing_ns: atom.Atom):
    assert len(ns_cache_with_existing_ns.deref().keys()) == 2
    ns = namespace.remove(ns_sym, ns_cache=ns_cache_with_existing_ns)
    assert isinstance(ns, namespace.Namespace)
    assert ns.name == ns_sym.name
    assert len(ns_cache_with_existing_ns.deref().keys()) == 1


@pytest.fixture
def other_ns_sym() -> sym.Symbol:
    return sym.symbol("some.other.ns")


def test_remove_non_existent_ns(other_ns_sym: sym.Symbol,
                                ns_cache_with_existing_ns: atom.Atom):
    assert len(ns_cache_with_existing_ns.deref().keys()) == 2
    ns = namespace.remove(other_ns_sym, ns_cache=ns_cache_with_existing_ns)
    assert ns is None
    assert len(ns_cache_with_existing_ns.deref().keys()) == 2
