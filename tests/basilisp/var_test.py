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
    return sym.symbol("some.ns")


@pytest.fixture
def var_name() -> sym.Symbol:
    return sym.symbol("var-val")


@pytest.fixture
def intern_val():
    return vec.v(kw.keyword("value"))


@pytest.fixture
def ns_cache(ns_sym: sym.Symbol) -> patch:
    return patch(
        "basilisp.lang.runtime.Namespace._NAMESPACES",
        atom.Atom(lmap.map({ns_sym: Namespace(ns_sym)})),
    )


def test_public_var(
    ns_cache: patch, ns_sym: sym.Symbol, var_name: sym.Symbol, intern_val
):
    with ns_cache:
        v = Var.intern(ns_sym, var_name, intern_val)
        assert not v.is_private


def test_private_var(
    ns_cache: patch, ns_sym: sym.Symbol, var_name: sym.Symbol, intern_val
):
    with ns_cache:
        v = Var.intern(
            ns_sym, var_name, intern_val, meta=lmap.map({kw.keyword("private"): True})
        )
        assert v.is_private


def test_dynamic_var(
    ns_sym: sym.Symbol, var_name: sym.Symbol, intern_val, ns_cache: patch
):
    with ns_cache:
        v = Var.intern(ns_sym, var_name, intern_val, dynamic=True)
        assert v.dynamic
        assert intern_val == v.root
        assert intern_val == v.value

        new_val = kw.keyword("new-val")
        new_val2 = kw.keyword("other-new-val")
        try:
            v.push_bindings(new_val)
            assert v.dynamic
            assert intern_val == v.root
            assert new_val == v.value

            v.value = new_val2
            assert v.dynamic
            assert intern_val == v.root
            assert new_val2 == v.value
        finally:
            v.pop_bindings()

        assert v.dynamic
        assert intern_val == v.root
        assert intern_val == v.value


def test_var_bindings_are_noop_for_non_dynamic_var(
    ns_sym: sym.Symbol, var_name: sym.Symbol, intern_val, ns_cache: patch
):
    with ns_cache:
        v = Var.intern(ns_sym, var_name, intern_val)
        assert not v.dynamic
        assert intern_val == v.root
        assert intern_val == v.value

        new_val = kw.keyword("new-val")
        new_val2 = kw.keyword("other-new-val")
        try:
            v.push_bindings(new_val)
            assert not v.dynamic
            assert intern_val == v.root
            assert intern_val == v.value

            v.value = new_val2
            assert not v.dynamic
            assert new_val2 == v.root
            assert new_val2 == v.value
        finally:
            v.pop_bindings()

        assert not v.dynamic
        assert new_val2 == v.root
        assert new_val2 == v.value


def test_intern(ns_sym: sym.Symbol, var_name: sym.Symbol, intern_val, ns_cache: patch):
    with ns_cache:
        v = Var.intern(ns_sym, var_name, intern_val)
        assert isinstance(v, Var)
        assert ns_sym.name == v.ns.name
        assert var_name == v.name
        assert not v.dynamic
        assert intern_val == v.root
        assert intern_val == v.value

        ns = Namespace.get_or_create(ns_sym)
        assert None is not ns
        assert ns.find(var_name) == v


def test_intern_unbound(ns_sym: sym.Symbol, var_name: sym.Symbol, ns_cache: patch):
    with ns_cache:
        v = Var.intern_unbound(ns_sym, var_name)
        assert isinstance(v, Var)
        assert ns_sym.name == v.ns.name
        assert var_name == v.name
        assert not v.dynamic
        assert None is v.root
        assert None is v.value

        ns = Namespace.get_or_create(ns_sym)
        assert None is not ns
        assert ns.find(var_name) == v


def test_find_in_ns(
    ns_sym: sym.Symbol, var_name: sym.Symbol, intern_val, ns_cache: patch
):
    with ns_cache:
        v = Var.intern(ns_sym, var_name, intern_val)
        v_in_ns = Var.find_in_ns(ns_sym, var_name)
        assert v == v_in_ns


def test_find(ns_sym: sym.Symbol, var_name: sym.Symbol, intern_val, ns_cache: patch):
    with ns_cache:
        v = Var.intern(ns_sym, var_name, intern_val)
        ns_qualified_sym = sym.symbol(var_name.name, ns=ns_sym.name)
        v_in_ns = Var.find(ns_qualified_sym)
        assert v == v_in_ns
