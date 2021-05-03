from unittest.mock import patch

import pytest

from basilisp.lang import atom as atom
from basilisp.lang import keyword as kw
from basilisp.lang import map as lmap
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec
from basilisp.lang.exception import ExceptionInfo
from basilisp.lang.runtime import (
    Namespace,
    NamespaceMap,
    RuntimeException,
    Unbound,
    Var,
    assoc,
)
from tests.basilisp.helpers import get_or_create_ns


@pytest.fixture
def ns_sym() -> sym.Symbol:
    return sym.symbol("some.ns")


@pytest.fixture
def var_name() -> sym.Symbol:
    return sym.symbol("var-val")


@pytest.fixture
def intern_val():
    return vec.v(kw.keyword("value"))


@pytest.fixture(autouse=True)
def ns_cache(ns_sym: sym.Symbol) -> atom.Atom[NamespaceMap]:
    with patch(
        "basilisp.lang.runtime.Namespace._NAMESPACES",
        atom.Atom(lmap.map({ns_sym: Namespace(ns_sym)})),
    ) as ns_cache:
        yield ns_cache


def test_public_var(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
):
    v = Var.intern(ns_sym, var_name, intern_val)
    assert not v.is_private


def test_private_var(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
):
    v = Var.intern(
        ns_sym, var_name, intern_val, meta=lmap.map({kw.keyword("private"): True})
    )
    assert v.is_private


def test_alter_var_meta(
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


def test_var_validators(ns_sym: sym.Symbol, var_name: sym.Symbol):
    v = Var.intern(ns_sym, var_name, 0)

    even_validator = lambda i: isinstance(i, int) and i % 2 == 0
    v.set_validator(even_validator)
    assert even_validator == v.get_validator()

    with pytest.raises(ExceptionInfo):
        v.root = 1

    with pytest.raises(ExceptionInfo):
        v.alter_root(lambda i: i + 1)

    assert 0 == v.root
    v.root = 2

    v.set_validator()
    assert None is v.get_validator()

    v.alter_root(lambda i: i + 1)

    assert 3 == v.root

    with pytest.raises(ExceptionInfo):
        v.set_validator(even_validator)

    odd_validator = lambda i: isinstance(i, int) and i % 2 == 1
    v.set_validator(odd_validator)

    with pytest.raises(ExceptionInfo):
        v.root = 2


def test_var_watchers(ns_sym: sym.Symbol, var_name: sym.Symbol):
    v = Var.intern(ns_sym, var_name, 0)
    assert v is v.remove_watch("nonexistent-watch")

    watcher1_key = kw.keyword("watcher-the-first")
    watcher1_vals = []

    def watcher1(k, ref, old, new):
        assert watcher1_key is k
        assert v is ref
        watcher1_vals.append((old, new))

    v.add_watch(watcher1_key, watcher1)
    v.alter_root(lambda v: v * 2)  # == 0
    v.root = 4  # == 4

    watcher2_key = kw.keyword("watcher-the-second")
    watcher2_vals = []

    def watcher2(k, ref, old, new):
        assert watcher2_key is k
        assert v is ref
        watcher2_vals.append((old, new))

    v.add_watch(watcher2_key, watcher2)
    v.alter_root(lambda v: v * 2)  # == 8

    v.remove_watch(watcher1_key)
    v.root = 10  # == 10
    v.alter_root(lambda v: "a" * v)  # == "aaaaaaaaaa"

    assert [(0, 0), (0, 4), (4, 8)] == watcher1_vals
    assert [(4, 8), (8, 10), (10, "aaaaaaaaaa")] == watcher2_vals


def test_dynamic_var(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
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

    ns = get_or_create_ns(ns_sym)
    assert None is not ns
    assert ns.find(var_name) == v


def test_intern_unbound(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
):
    v = Var.intern_unbound(ns_sym, var_name)
    assert isinstance(v, Var)
    assert ns_sym.name == v.ns.name
    assert var_name == v.name
    assert not v.is_bound
    assert not v.dynamic
    assert not v.is_thread_bound
    assert Unbound(v) == v.root
    assert Unbound(v) == v.value
    assert Unbound(v) == v.deref()

    ns = get_or_create_ns(ns_sym)
    assert None is not ns
    assert ns.find(var_name) == v


def test_dynamic_unbound(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
):
    v = Var.intern_unbound(ns_sym, var_name, dynamic=True)
    assert isinstance(v, Var)
    assert ns_sym.name == v.ns.name
    assert var_name == v.name
    assert not v.is_bound
    assert v.dynamic
    assert not v.is_thread_bound
    assert Unbound(v) == v.root
    assert Unbound(v) == v.value
    assert Unbound(v) == v.deref()

    new_val = kw.keyword("new-val")
    try:
        v.push_bindings(new_val)
        assert v.is_bound
        assert v.dynamic
        assert v.is_thread_bound
        assert Unbound(v) == v.root
        assert new_val == v.value
        assert new_val == v.deref()
    finally:
        v.pop_bindings()

    assert not v.is_bound

    ns = get_or_create_ns(ns_sym)
    assert None is not ns
    assert ns.find(var_name) == v


def test_alter_var_root(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
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
):
    v = Var.intern(ns_sym, var_name, intern_val)
    v_in_ns = Var.find_in_ns(ns_sym, var_name)
    assert v == v_in_ns


def test_find(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
):
    v = Var.intern(ns_sym, var_name, intern_val)
    ns_qualified_sym = sym.symbol(var_name.name, ns=ns_sym.name)
    v_in_ns = Var.find(ns_qualified_sym)
    assert v == v_in_ns


def test_find_safe(
    ns_sym: sym.Symbol,
    var_name: sym.Symbol,
    intern_val,
):
    v = Var.intern(ns_sym, var_name, intern_val)
    ns_qualified_sym = sym.symbol(var_name.name, ns=ns_sym.name)
    v_in_ns = Var.find_safe(ns_qualified_sym)
    assert v == v_in_ns

    with pytest.raises(RuntimeException):
        Var.find_safe(sym.symbol("some-other-var", ns="doesnt.matter"))
