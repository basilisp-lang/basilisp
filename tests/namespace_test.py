from unittest.mock import patch

import pyrsistent
import pytest

import basilisp.lang.atom as atom
import basilisp.lang.runtime as runtime
import basilisp.lang.symbol as sym
from basilisp.lang.runtime import Namespace


@pytest.fixture
def core_ns_sym():
    return sym.symbol(runtime._CORE_NS)


@pytest.fixture
def core_ns(core_ns_sym):
    ns = Namespace(core_ns_sym)
    return ns


@pytest.fixture
def ns_cache(core_ns_sym: sym.Symbol,
             core_ns: Namespace) -> patch:
    """Patch the Namespace cache with a test fixture."""
    return patch('basilisp.lang.runtime.Namespace._NAMESPACES',
                 new=atom.Atom(pyrsistent.pmap({core_ns_sym: core_ns})))


@pytest.fixture
def ns_sym() -> sym.Symbol:
    return sym.symbol("some.ns")


def test_create_ns(ns_sym: sym.Symbol, ns_cache: patch):
    with ns_cache as cache:
        assert len(cache.deref().keys()) == 1
        ns = Namespace.get_or_create(ns_sym)
        assert isinstance(ns, Namespace)
        assert ns.name == ns_sym.name
        assert len(cache.deref().keys()) == 2


@pytest.fixture
def ns_cache_with_existing_ns(ns_sym: sym.Symbol, core_ns_sym: sym.Symbol,
                              core_ns: Namespace) -> patch:
    """Patch the Namespace cache with a test fixture with an existing namespace."""
    return patch('basilisp.lang.runtime.Namespace._NAMESPACES',
                 atom.Atom(
        pyrsistent.pmap({
            core_ns_sym: core_ns,
            ns_sym: Namespace(ns_sym)
        })))


def test_get_existing_ns(ns_sym: sym.Symbol,
                         ns_cache_with_existing_ns: patch):
    with ns_cache_with_existing_ns as cache:
        assert len(cache.deref().keys()) == 2
        ns = Namespace.get_or_create(ns_sym)
        assert isinstance(ns, Namespace)
        assert ns.name == ns_sym.name
        assert len(cache.deref().keys()) == 2


def test_remove_ns(ns_sym: sym.Symbol, ns_cache_with_existing_ns: patch):
    with ns_cache_with_existing_ns as cache:
        assert len(cache.deref().keys()) == 2
        ns = Namespace.remove(ns_sym)
        assert isinstance(ns, Namespace)
        assert ns.name == ns_sym.name
        assert len(cache.deref().keys()) == 1


@pytest.fixture
def other_ns_sym() -> sym.Symbol:
    return sym.symbol("some.other.ns")


def test_remove_non_existent_ns(other_ns_sym: sym.Symbol,
                                ns_cache_with_existing_ns: patch):
    with ns_cache_with_existing_ns as cache:
        assert len(cache.deref().keys()) == 2
        ns = Namespace.remove(other_ns_sym)
        assert ns is None
        assert len(cache.deref().keys()) == 2
