from unittest.mock import patch

import pytest

import basilisp.lang.atom as atom
import basilisp.lang.keyword as kw
import basilisp.lang.map as lmap
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
from basilisp.lang.runtime import Namespace, NamespaceMap, RuntimeException, Var, assoc


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
def ns_cache(ns_sym: sym.Symbol) -> atom.Atom[NamespaceMap]:
    with patch(
        "basilisp.lang.runtime.Namespace._NAMESPACES",
        atom.Atom(lmap.map({ns_sym: Namespace(ns_sym)})),
    ) as ns_cache:
        yield ns_cache


def test_public_var(
    ns_cache: atom.Atom[NamespaceMap],
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
):
    v = Var.intern(ns_sym, var_name, intern_val)
    assert not v.is_private


def test_private_var(
    ns_cache: atom.Atom[NamespaceMap],
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
):
    v = Var.intern(
        ns_sym, var_name, intern_val, meta=lmap.map({kw.keyword("private"): True})
    )
    assert v.is_private


def test_alter_var_meta(
    ns_cache: atom.Atom[NamespaceMap],
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
):
    v = Var.intern(ns_sym, var_name, intern_val)
    assert v.meta is None

    v.alter_meta(assoc, "type", sym.symbol("str"))
    assert v.meta == lmap.m(type=sym.symbol("str"))

    v.alter_meta(assoc, "tag", kw.keyword("async"))
    assert v.meta == lmap.m(type=sym.symbol("str"), tag=kw.keyword("async"))


def test_reset_var_meta(
    ns_cache: atom.Atom[NamespaceMap],
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
):
    v = Var.intern(ns_sym, var_name, intern_val)
    assert v.meta is None

    v.reset_meta(lmap.map({"type": sym.symbol("str")}))
    assert v.meta == lmap.m(type=sym.symbol("str"))

    v.reset_meta(lmap.m(tag=kw.keyword("async")))
    assert v.meta == lmap.m(tag=kw.keyword("async"))


def test_dynamic_var(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
    ns_cache: atom.Atom[NamespaceMap],
):
    v = Var.intern(ns_sym, var_name, intern_val, dynamic=True)
    assert v.is_bound
    assert v.dynamic
    assert not v.is_thread_bound
    assert intern_val == v.root
    assert intern_val == v.value
    assert intern_val == v.deref()

    new_val = kw.keyword("new-val")
    new_val2 = kw.keyword("other-new-val")
    try:
        v.push_bindings(new_val)
        assert v.is_bound
        assert v.dynamic
        assert v.is_thread_bound
        assert intern_val == v.root
        assert new_val == v.value
        assert new_val == v.deref()

        v.value = new_val2
        assert v.is_bound
        assert v.dynamic
        assert v.is_thread_bound
        assert intern_val == v.root
        assert new_val2 == v.value
        assert new_val2 == v.deref()
    finally:
        v.pop_bindings()

    assert v.is_bound
    assert v.dynamic
    assert not v.is_thread_bound
    assert intern_val == v.root
    assert intern_val == v.value
    assert intern_val == v.deref()


def test_var_bindings_are_noop_for_non_dynamic_var(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
    ns_cache: atom.Atom[NamespaceMap],
):
    v = Var.intern(ns_sym, var_name, intern_val)
    assert v.is_bound
    assert not v.dynamic
    assert not v.is_thread_bound
    assert intern_val == v.root
    assert intern_val == v.value
    assert intern_val == v.deref()

    new_val = kw.keyword("new-val")
    new_val2 = kw.keyword("other-new-val")
    try:
        with pytest.raises(RuntimeException):
            v.push_bindings(new_val)

        assert v.is_bound
        assert not v.dynamic
        assert not v.is_thread_bound
        assert intern_val == v.root
        assert intern_val == v.value
        assert intern_val == v.deref()

        v.value = new_val2
        assert v.is_bound
        assert not v.dynamic
        assert not v.is_thread_bound
        assert new_val2 == v.root
        assert new_val2 == v.value
        assert new_val2 == v.deref()
    finally:
        with pytest.raises(RuntimeException):
            v.pop_bindings()

    assert v.is_bound
    assert not v.dynamic
    assert not v.is_thread_bound
    assert new_val2 == v.root
    assert new_val2 == v.value
    assert new_val2 == v.deref()


def test_intern(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
    ns_cache: atom.Atom[NamespaceMap],
):
    v = Var.intern(ns_sym, var_name, intern_val)
    assert isinstance(v, Var)
    assert ns_sym.name == v.ns.name
    assert var_name == v.name
    assert v.is_bound
    assert not v.dynamic
    assert not v.is_thread_bound
    assert intern_val == v.root
    assert intern_val == v.value
    assert intern_val == v.deref()

    ns = Namespace.get_or_create(ns_sym)
    assert None is not ns
    assert ns.find(var_name) == v


def test_intern_unbound(
    ns_sym: sym.Symbol, var_name: sym.Symbol, ns_cache: atom.Atom[NamespaceMap]
):
    v = Var.intern_unbound(ns_sym, var_name)
    assert isinstance(v, Var)
    assert ns_sym.name == v.ns.name
    assert var_name == v.name
    assert not v.is_bound
    assert not v.dynamic
    assert not v.is_thread_bound
    assert None is v.root
    assert None is v.value
    assert None is v.deref()

    ns = Namespace.get_or_create(ns_sym)
    assert None is not ns
    assert ns.find(var_name) == v


def test_dynamic_unbound(
    ns_sym: sym.Symbol, var_name: sym.Symbol, ns_cache: atom.Atom[NamespaceMap]
):
    v = Var.intern_unbound(ns_sym, var_name, dynamic=True)
    assert isinstance(v, Var)
    assert ns_sym.name == v.ns.name
    assert var_name == v.name
    assert not v.is_bound
    assert v.dynamic
    assert not v.is_thread_bound
    assert None is v.root
    assert None is v.value
    assert None is v.deref()

    new_val = kw.keyword("new-val")
    try:
        v.push_bindings(new_val)
        assert v.is_bound
        assert v.dynamic
        assert v.is_thread_bound
        assert None is v.root
        assert new_val == v.value
        assert new_val == v.deref()
    finally:
        v.pop_bindings()

    assert not v.is_bound

    ns = Namespace.get_or_create(ns_sym)
    assert None is not ns
    assert ns.find(var_name) == v


def test_alter_var_root(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
    ns_cache: atom.Atom[NamespaceMap],
):
    v = Var.intern(ns_sym, var_name, intern_val)

    new_root = kw.keyword("new-root")
    alter_args = (1, 2, 3)

    def alter_root(root, *args):
        assert intern_val == root
        assert alter_args == args
        return new_root

    v.alter_root(alter_root, *alter_args)

    assert new_root == v.root
    assert new_root == v.value
    assert new_root == v.deref()


def test_alter_dynamic_var_root(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
    ns_cache: atom.Atom[NamespaceMap],
):
    v = Var.intern(ns_sym, var_name, intern_val, dynamic=True)
    assert v.is_bound
    assert v.dynamic
    assert not v.is_thread_bound
    assert intern_val == v.root
    assert intern_val == v.value
    assert intern_val == v.deref()

    new_val = kw.keyword("new-val")
    new_root = kw.keyword("new-root")
    alter_args = (1, 2, 3)

    def alter_root(root, *args):
        assert intern_val == root
        assert alter_args == args
        return new_root

    try:
        v.push_bindings(new_val)
        v.alter_root(alter_root, *alter_args)
        assert v.is_bound
        assert v.dynamic
        assert v.is_thread_bound
        assert new_root == v.root
        assert new_val == v.value
        assert new_val == v.deref()
    finally:
        v.pop_bindings()

    assert v.is_bound
    assert v.dynamic
    assert not v.is_thread_bound
    assert new_root == v.root
    assert new_root == v.value
    assert new_root == v.deref()


def test_find_in_ns(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
    ns_cache: atom.Atom[NamespaceMap],
):
    v = Var.intern(ns_sym, var_name, intern_val)
    v_in_ns = Var.find_in_ns(ns_sym, var_name)
    assert v == v_in_ns


def test_find(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
    ns_cache: atom.Atom[NamespaceMap],
):
    v = Var.intern(ns_sym, var_name, intern_val)
    ns_qualified_sym = sym.symbol(var_name.name, ns=ns_sym.name)
    v_in_ns = Var.find(ns_qualified_sym)
    assert v == v_in_ns


def test_find_safe(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
    ns_cache: atom.Atom[NamespaceMap],
):
    v = Var.intern(ns_sym, var_name, intern_val)
    ns_qualified_sym = sym.symbol(var_name.name, ns=ns_sym.name)
    v_in_ns = Var.find_safe(ns_qualified_sym)
    assert v == v_in_ns

    with pytest.raises(RuntimeException):
        Var.find_safe(sym.symbol("some-other-var", ns="doesnt.matter"))
