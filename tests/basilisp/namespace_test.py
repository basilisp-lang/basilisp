from typing import Callable
from unittest.mock import patch

import pytest

import basilisp.lang.atom as atom
import basilisp.lang.keyword as kw
import basilisp.lang.map as lmap
import basilisp.lang.runtime as runtime
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
from basilisp.lang.runtime import Namespace, Var


@pytest.fixture
def core_ns_sym() -> sym.Symbol:
    return sym.symbol(runtime._CORE_NS)


@pytest.fixture
def core_ns(core_ns_sym: sym.Symbol) -> Namespace:
    ns = Namespace(core_ns_sym)
    return ns


@pytest.fixture
def ns_cache(core_ns_sym: sym.Symbol,
             core_ns: Namespace) -> patch:
    """Patch the Namespace cache with a test fixture."""
    return patch('basilisp.lang.runtime.Namespace._NAMESPACES',
                 new=atom.Atom(lmap.map({core_ns_sym: core_ns})))


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
                     lmap.map({
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


def test_gated_import():
    with patch('basilisp.lang.runtime.Namespace.DEFAULT_IMPORTS',
               new=atom.Atom(lset.set([sym.symbol('default')]))), \
         patch('basilisp.lang.runtime.Namespace.GATED_IMPORTS',
               new=lset.set(['gated-default'])):
        Namespace.add_default_import('non-gated-default')
        assert sym.symbol('non-gated-default') not in Namespace.DEFAULT_IMPORTS.deref()

        Namespace.add_default_import('gated-default')
        assert sym.symbol('gated-default') in Namespace.DEFAULT_IMPORTS.deref()


def test_intern_does_not_overwrite(ns_cache: patch):
    with ns_cache:
        ns = Namespace.get_or_create(sym.symbol('ns1'))
        var_sym = sym.symbol('useful-value')

        var_val1 = "cool string"
        var1 = Var(ns, var_sym)
        var1.value = var_val1
        ns.intern(var_sym, var1)

        var_val2 = "lame string"
        var2 = Var(ns, var_sym)
        var2.value = var_val2
        ns.intern(var_sym, var2)

        assert var1 is ns.find(var_sym)
        assert var_val1 == ns.find(var_sym).value

        ns.intern(var_sym, var2, force=True)

        assert var2 is ns.find(var_sym)
        assert var_val2 == ns.find(var_sym).value


@pytest.fixture
def core_map() -> sym.Symbol:
    return sym.symbol('map')


@pytest.fixture
def core_map_fn() -> Callable:
    return map


@pytest.fixture
def core_private() -> sym.Symbol:
    return sym.symbol('private-var')


@pytest.fixture
def core_private_val() -> str:
    return "private-string"


@pytest.fixture
def test_ns() -> str:
    return "test"


@pytest.fixture
def other_ns(core_map: sym.Symbol,
             core_map_fn: Callable,
             core_private: sym.Symbol,
             core_private_val: str,
             test_ns: str) -> Namespace:
    runtime.init_ns_var(which_ns=runtime._CORE_NS)
    core_ns = Namespace.get_or_create(sym.symbol(runtime._CORE_NS))

    # Add a public Var
    map_var = Var(core_ns, core_map)
    map_var.value = core_map_fn
    core_ns.intern(core_map, map_var)

    # Add a private Var
    private_var = Var(core_ns, core_private, meta=lmap.map({kw.keyword('private'): True}))
    private_var.value = core_private_val
    core_ns.intern(core_private, private_var)

    with runtime.ns_bindings(test_ns) as ns:
        yield ns


def test_refer_core(core_ns_sym: sym.Symbol,
                    other_ns: Namespace,
                    core_map: sym.Symbol,
                    core_private: sym.Symbol):
    core_ns = Namespace.get_or_create(core_ns_sym)

    assert core_map in core_ns.interns
    assert core_private in core_ns.interns
    assert core_map in other_ns.refers
    assert core_private not in other_ns.refers


def test_refer(ns_cache: patch):
    with ns_cache:
        ns1 = Namespace.get_or_create(sym.symbol('ns1'))
        var_sym, var_val = sym.symbol('useful-value'), "cool string"
        var = Var(ns1, var_sym)
        var.value = var_val
        ns1.intern(var_sym, var)

        ns2 = Namespace.get_or_create(sym.symbol('ns2'))
        ns2.add_refer(var_sym, var)

        assert var is ns2.get_refer(var_sym)
        assert var_val == ns2.find(var_sym).value


def test_cannot_refer_private(ns_cache: patch):
    with ns_cache:
        ns1 = Namespace.get_or_create(sym.symbol('ns1'))
        var_sym, var_val = sym.symbol('useful-value'), "cool string"
        var = Var(ns1, var_sym, meta=lmap.map({kw.keyword('private'): True}))
        var.value = var_val
        ns1.intern(var_sym, var)

        ns2 = Namespace.get_or_create(sym.symbol('ns2'))
        ns2.add_refer(var_sym, var)

        assert None is ns2.get_refer(var_sym)
        assert None is ns2.find(var_sym)


def test_refer_all(ns_cache: patch):
    with ns_cache:
        ns1 = Namespace.get_or_create(sym.symbol('ns1'))

        var_sym1, var_val1 = sym.symbol('useful-value'), "cool string"
        var1 = Var(ns1, var_sym1)
        var1.value = var_val1
        ns1.intern(var_sym1, var1)

        var_sym2, var_val2 = sym.symbol('private-value'), "private string"
        var2 = Var(ns1, var_sym2, meta=lmap.map({kw.keyword('private'): True}))
        var2.value = var_val2
        ns1.intern(var_sym2, var2)

        var_sym3, var_val3 = sym.symbol('existing-value'), "interned string"
        var3 = Var(ns1, var_sym3)
        var3.value = var_val3
        ns1.intern(var_sym3, var3)

        ns2 = Namespace.get_or_create(sym.symbol('ns2'))
        var_val4 = "some other value"
        var4 = Var(ns2, var_sym3)
        var4.value = var_val4
        ns2.intern(var_sym3, var4)
        ns2.refer_all(ns1)

        assert var1 is ns2.get_refer(var_sym1)
        assert var1 is ns2.find(var_sym1)
        assert var_val1 == ns2.find(var_sym1).value

        assert None is ns2.get_refer(var_sym2)
        assert None is ns2.find(var_sym2)

        assert var3 is ns2.get_refer(var_sym3)
        assert var4 is ns2.find(var_sym3)
        assert var_val4 == ns2.find(var_sym3).value


def test_refer_does_not_shadow_intern(ns_cache: patch):
    with ns_cache:
        ns1 = Namespace.get_or_create(sym.symbol('ns1'))
        var_sym = sym.symbol('useful-value')

        var_val1 = "cool string"
        var1 = Var(ns1, var_sym)
        var1.value = var_val1
        ns1.intern(var_sym, var1)

        ns2 = Namespace.get_or_create(sym.symbol('ns2'))
        var_val2 = "lame string"
        var2 = Var(ns1, var_sym)
        var2.value = var_val2
        ns2.intern(var_sym, var2)

        ns2.add_refer(var_sym, var1)

        assert var1 is ns2.get_refer(var_sym)
        assert var_val2 == ns2.find(var_sym).value


def test_alias(ns_cache: patch):
    with ns_cache as cache:
        ns1 = Namespace.get_or_create(sym.symbol('ns1'))
        ns2 = Namespace.get_or_create(sym.symbol('ns2'))

        ns1.add_alias(sym.symbol('n2'), ns2)

        assert None is ns1.get_alias(sym.symbol('ns2'))
        assert ns2 is ns1.get_alias(sym.symbol('n2'))
