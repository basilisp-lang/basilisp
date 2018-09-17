from unittest.mock import patch

import pytest

import basilisp.lang.atom as atom
import basilisp.lang.keyword as kw
import basilisp.lang.map as lmap
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
from basilisp.lang.runtime import Namespace, Var


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
def ns_cache(ns_sym: sym.Symbol) -> patch:
    return patch('basilisp.lang.runtime.Namespace._NAMESPACES',
                 atom.Atom(lmap.map({ns_sym: Namespace(ns_sym)})))


def test_intern(ns_sym: sym.Symbol, var_name: sym.Symbol, intern_val,
                ns_cache: patch):
    with ns_cache:
        v = Var.intern(ns_sym, var_name, intern_val)
        assert isinstance(v, Var)
        assert v.ns.name == ns_sym.name
        assert v.name == var_name
        assert not v.dynamic
        assert v.root == intern_val
        assert v.value == intern_val

        ns = Namespace.get_or_create(ns_sym)
        assert ns is not None
        assert v == ns.find(var_name)


def test_find_in_ns(ns_sym: sym.Symbol, var_name: sym.Symbol, intern_val,
                    ns_cache: patch):
    with ns_cache:
        v = Var.intern(ns_sym, var_name, intern_val)
        v_in_ns = Var.find_in_ns(ns_sym, var_name)
        assert v == v_in_ns


def test_find(ns_sym: sym.Symbol, var_name: sym.Symbol, intern_val,
              ns_cache: patch):
    with ns_cache:
        v = Var.intern(ns_sym, var_name, intern_val)
        ns_qualified_sym = sym.symbol(var_name.name, ns=ns_sym.name)
        v_in_ns = Var.find(ns_qualified_sym)
        assert v == v_in_ns
