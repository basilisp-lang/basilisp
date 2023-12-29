from unittest.mock import patch

import pytest

from basilisp.lang import atom as atom
from basilisp.lang import keyword as kw
from basilisp.lang import map as lmap
from basilisp.lang import runtime as runtime
from basilisp.lang import symbol as sym
from basilisp.lang.runtime import Namespace, NamespaceMap, Var
from tests.basilisp.helpers import get_or_create_ns


@pytest.fixture
def ns_cache(core_ns_sym: sym.Symbol, core_ns: Namespace) -> atom.Atom[NamespaceMap]:
    """Patch the Namespace cache with a test fixture."""
    with patch(
        "basilisp.lang.runtime.Namespace._NAMESPACES",
        new=atom.Atom(lmap.map({core_ns_sym: core_ns})),
    ) as cache:
        yield cache


@pytest.fixture
def ns_sym() -> sym.Symbol:
    return sym.symbol("some.ns")


def test_create_ns(ns_sym: sym.Symbol, ns_cache: atom.Atom[NamespaceMap]):
    assert len(list(ns_cache.deref().keys())) == 1
    ns = get_or_create_ns(ns_sym)
    assert isinstance(ns, Namespace)
    assert ns.name == ns_sym.name
    assert len(list(ns_cache.deref().keys())) == 2
    assert ns.get_refer(sym.symbol("ns"))


def test_in_ns(ns_sym: sym.Symbol):
    in_ns = Var.find_in_ns(runtime.CORE_NS_SYM, sym.symbol("in-ns"))
    assert in_ns
    ns = in_ns.value(ns_sym)
    assert isinstance(ns, Namespace)
    assert ns.name == ns_sym.name


@pytest.fixture
def ns_cache_with_existing_ns(
    ns_sym: sym.Symbol, core_ns_sym: sym.Symbol, core_ns: Namespace
) -> atom.Atom[NamespaceMap]:
    """Patch the Namespace cache with a test fixture with an existing namespace."""
    with patch(
        "basilisp.lang.runtime.Namespace._NAMESPACES",
        atom.Atom(lmap.map({core_ns_sym: core_ns, ns_sym: Namespace(ns_sym)})),
    ) as cache:
        yield cache


def test_get_existing_ns(
    ns_sym: sym.Symbol, ns_cache_with_existing_ns: atom.Atom[NamespaceMap]
):
    assert len(list(ns_cache_with_existing_ns.deref().keys())) == 2
    ns = get_or_create_ns(ns_sym)
    assert isinstance(ns, Namespace)
    assert ns.name == ns_sym.name
    assert len(list(ns_cache_with_existing_ns.deref().keys())) == 2


def test_remove_ns(
    ns_sym: sym.Symbol, ns_cache_with_existing_ns: atom.Atom[NamespaceMap]
):
    assert len(list(ns_cache_with_existing_ns.deref().keys())) == 2
    ns = Namespace.remove(ns_sym)
    assert isinstance(ns, Namespace)
    assert ns.name == ns_sym.name
    assert len(list(ns_cache_with_existing_ns.deref().keys())) == 1


@pytest.fixture
def other_ns_sym() -> sym.Symbol:
    return sym.symbol("some.other.ns")


def test_remove_non_existent_ns(
    other_ns_sym: sym.Symbol, ns_cache_with_existing_ns: patch
):
    assert len(list(ns_cache_with_existing_ns.deref().keys())) == 2
    ns = Namespace.remove(other_ns_sym)
    assert ns is None
    assert len(list(ns_cache_with_existing_ns.deref().keys())) == 2


def test_alter_ns_meta(
    ns_cache: atom.Atom[NamespaceMap],
    ns_sym: sym.Symbol,
):
    ns = get_or_create_ns(ns_sym)
    assert ns.meta is None

    ns.alter_meta(runtime.assoc, "type", sym.symbol("str"))
    assert ns.meta == lmap.m(type=sym.symbol("str"))

    ns.alter_meta(runtime.assoc, "tag", kw.keyword("async"))
    assert ns.meta == lmap.m(type=sym.symbol("str"), tag=kw.keyword("async"))


def test_reset_ns_meta(
    ns_cache: atom.Atom[NamespaceMap],
    ns_sym: sym.Symbol,
):
    ns = get_or_create_ns(ns_sym)
    assert ns.meta is None

    ns.reset_meta(lmap.map({"type": sym.symbol("str")}))
    assert ns.meta == lmap.m(type=sym.symbol("str"))

    ns.reset_meta(lmap.m(tag=kw.keyword("async")))
    assert ns.meta == lmap.m(tag=kw.keyword("async"))


def test_cannot_remove_core(ns_cache: atom.Atom[NamespaceMap]):
    with pytest.raises(ValueError):
        Namespace.remove(sym.symbol("basilisp.core"))


def test_imports(ns_cache: atom.Atom[NamespaceMap]):
    ns = get_or_create_ns(sym.symbol("ns1"))
    time = __import__("time")
    ns.add_import(sym.symbol("time"), time, sym.symbol("py-time"), sym.symbol("py-tm"))
    assert time == ns.get_import(sym.symbol("time"))
    assert time == ns.get_import(sym.symbol("py-time"))
    assert time == ns.get_import(sym.symbol("py-tm"))
    assert None is ns.get_import(sym.symbol("python-time"))


def test_intern_does_not_overwrite(ns_cache: atom.Atom[NamespaceMap]):
    ns = get_or_create_ns(sym.symbol("ns1"))
    var_sym = sym.symbol("useful-value")

    var_val1 = "cool string"
    var1 = Var(ns, var_sym)
    var1.set_value(var_val1)
    ns.intern(var_sym, var1)

    var_val2 = "lame string"
    var2 = Var(ns, var_sym)
    var2.set_value(var_val2)
    ns.intern(var_sym, var2)

    assert var1 is ns.find(var_sym)
    assert var_val1 == ns.find(var_sym).value

    ns.intern(var_sym, var2, force=True)

    assert var2 is ns.find(var_sym)
    assert var_val2 == ns.find(var_sym).value


def test_unmap(ns_cache: atom.Atom[NamespaceMap]):
    ns = get_or_create_ns(sym.symbol("ns1"))
    var_sym = sym.symbol("useful-value")

    var_val = "cool string"
    var = Var(ns, var_sym)
    var.set_value(var_val)
    ns.intern(var_sym, var)

    assert var is ns.find(var_sym)
    assert var_val == ns.find(var_sym).value

    ns.unmap(var_sym)

    assert None is ns.find(var_sym)


def test_refer(ns_cache: atom.Atom[NamespaceMap]):
    ns1 = get_or_create_ns(sym.symbol("ns1"))
    var_sym, var_val = sym.symbol("useful-value"), "cool string"
    var = Var(ns1, var_sym)
    var.set_value(var_val)
    ns1.intern(var_sym, var)

    ns2 = get_or_create_ns(sym.symbol("ns2"))
    ns2.add_refer(var_sym, var)

    assert var is ns2.get_refer(var_sym)
    assert var_val == ns2.find(var_sym).value


def test_cannot_refer_private(ns_cache: atom.Atom[NamespaceMap]):
    ns1 = get_or_create_ns(sym.symbol("ns1"))
    var_sym, var_val = sym.symbol("useful-value"), "cool string"
    var = Var(ns1, var_sym, meta=lmap.map({kw.keyword("private"): True}))
    var.set_value(var_val)
    ns1.intern(var_sym, var)

    ns2 = get_or_create_ns(sym.symbol("ns2"))
    ns2.add_refer(var_sym, var)

    assert None is ns2.get_refer(var_sym)
    assert None is ns2.find(var_sym)


def test_refer_all(ns_cache: atom.Atom[NamespaceMap]):
    ns1 = get_or_create_ns(sym.symbol("ns1"))

    var_sym1, var_val1 = sym.symbol("useful-value"), "cool string"
    var1 = Var(ns1, var_sym1)
    var1.set_value(var_val1)
    ns1.intern(var_sym1, var1)

    var_sym2, var_val2 = sym.symbol("private-value"), "private string"
    var2 = Var(ns1, var_sym2, meta=lmap.map({kw.keyword("private"): True}))
    var2.set_value(var_val2)
    ns1.intern(var_sym2, var2)

    var_sym3, var_val3 = sym.symbol("existing-value"), "interned string"
    var3 = Var(ns1, var_sym3)
    var3.set_value(var_val3)
    ns1.intern(var_sym3, var3)

    ns2 = get_or_create_ns(sym.symbol("ns2"))
    var_val4 = "some other value"
    var4 = Var(ns2, var_sym3)
    var4.set_value(var_val4)
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


def test_refer_does_not_shadow_intern(ns_cache: atom.Atom[NamespaceMap]):
    ns1 = get_or_create_ns(sym.symbol("ns1"))
    var_sym = sym.symbol("useful-value")

    var_val1 = "cool string"
    var1 = Var(ns1, var_sym)
    var1.set_value(var_val1)
    ns1.intern(var_sym, var1)

    ns2 = get_or_create_ns(sym.symbol("ns2"))
    var_val2 = "lame string"
    var2 = Var(ns1, var_sym)
    var2.set_value(var_val2)
    ns2.intern(var_sym, var2)

    ns2.add_refer(var_sym, var1)

    assert var1 is ns2.get_refer(var_sym)
    assert var_val2 == ns2.find(var_sym).value


def test_alias(ns_cache: atom.Atom[NamespaceMap]):
    ns1 = get_or_create_ns(sym.symbol("ns1"))
    ns2 = get_or_create_ns(sym.symbol("ns2"))

    ns1.add_alias(ns2, sym.symbol("n2"))

    assert None is ns1.get_alias(sym.symbol("ns2"))
    assert ns2 is ns1.get_alias(sym.symbol("n2"))

    ns1.remove_alias(sym.symbol("n2"))

    assert None is ns1.get_alias(sym.symbol("n2"))


class TestCompletion:
    @pytest.fixture
    def ns(self) -> Namespace:
        ns_sym = sym.symbol("test")
        ns = Namespace(ns_sym)

        str_ns_alias = sym.symbol("basilisp.string")
        join_sym = sym.symbol("join")
        chars_sym = sym.symbol("chars")
        str_ns = Namespace(str_ns_alias)
        str_ns.intern(join_sym, Var(ns, join_sym))
        str_ns.intern(
            chars_sym, Var(ns, chars_sym, meta=lmap.map({kw.keyword("private"): True}))
        )
        ns.add_alias(str_ns, str_ns_alias)

        str_alias = sym.symbol("str")
        ns.add_alias(Namespace(str_alias), str_alias)

        str_sym = sym.symbol("str")
        ns.intern(str_sym, Var(ns, str_sym))

        is_string_sym = sym.symbol("string?")
        ns.intern(is_string_sym, Var(ns, is_string_sym))

        time_sym = sym.symbol("time")
        time_alias = sym.symbol("py-time")
        ns.add_import(time_sym, __import__("time"), time_alias)

        core_ns = Namespace(sym.symbol("basilisp.core"))
        map_alias = sym.symbol("map")
        ns.add_refer(map_alias, Var(core_ns, map_alias))

        return ns

    def test_ns_completion(self, ns: Namespace):
        assert {"basilisp.string/"} == set(ns.complete("basilisp.st"))
        assert {"basilisp.string/join"} == set(ns.complete("basilisp.string/j"))
        assert {"str/", "string?", "str"} == set(ns.complete("st"))
        assert {"map"} == set(ns.complete("m"))
        assert {"map"} == set(ns.complete("ma"))

    def test_import_and_alias(self, ns: Namespace):
        assert {"time/"} == set(ns.complete("ti"))
        assert {"time/asctime"} == set(ns.complete("time/as"))
        assert {"py-time/"} == set(ns.complete("py-t"))
        assert {"py-time/asctime"} == set(ns.complete("py-time/as"))
