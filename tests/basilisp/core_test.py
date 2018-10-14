import itertools
import re
from fractions import Fraction
from unittest.mock import Mock

import pytest

import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.runtime as runtime
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
from basilisp.lang.exception import ExceptionInfo
from basilisp.main import init

init()

# Cache the initial state of the `print_generated_python` flag.
__PRINT_GENERATED_PYTHON_FN = runtime.print_generated_python


def setup_module(module):
    """Disable the `print_generated_python` flag so we can safely capture
    stderr and stdout for tests which require those facilities."""
    runtime.print_generated_python = Mock(return_value=False)


def teardown_module(module):
    """Restore the `print_generated_python` flag after we finish running tests."""
    runtime.print_generated_python = __PRINT_GENERATED_PYTHON_FN


import basilisp.core as core

TRUTHY_VALUES = [True,
                 -1, 0, 1,
                 -1.0, 0.0, 1.0,
                 sym.symbol("name"), sym.symbol("name", ns="ns"),
                 kw.keyword("name"), kw.keyword("name", ns="ns"),
                 "", "not empty",
                 llist.List.empty(), llist.l(0), llist.l(False), llist.l(True),
                 lmap.Map.empty(), lmap.map({0: 0}), lmap.map({False: False}), lmap.map({True: True}),
                 lset.Set.empty(), lset.s(0), lset.s(False), lset.s(True),
                 vec.Vector.empty(), vec.v(0), vec.v(False), vec.v(True)]
FALSEY_VALUES = [False, None]

NON_NIL_VALUES = [False, True,
                  -1, 0, 1,
                  -1.0, 0.0, 1.0,
                  sym.symbol("name"), sym.symbol("name", ns="ns"),
                  kw.keyword("name"), kw.keyword("name", ns="ns"),
                  "", "not empty",
                  llist.List.empty(), llist.l(0), llist.l(False), llist.l(True),
                  lmap.Map.empty(), lmap.map({0: 0}), lmap.map({False: False}), lmap.map({True: True}),
                  lset.Set.empty(), lset.s(0), lset.s(False), lset.s(True),
                  vec.Vector.empty(), vec.v(0), vec.v(False), vec.v(True)]
NIL_VALUES = [None]


def test_ex_info():
    with pytest.raises(ExceptionInfo):
        raise core.ex_info("This is just an exception", lmap.m())


def test_last():
    assert None is core.last(llist.List.empty())
    assert 1 == core.last(llist.l(1))
    assert 2 == core.last(llist.l(1, 2))
    assert 3 == core.last(llist.l(1, 2, 3))

    assert None is core.last(vec.Vector.empty())
    assert 1 == core.last(vec.v(1))
    assert 2 == core.last(vec.v(1, 2))
    assert 3 == core.last(vec.v(1, 2, 3))


def test_not():
    for v in FALSEY_VALUES:
        assert True is core.not_(v)

    for v in TRUTHY_VALUES:
        assert False is core.not_(v)


def test_false__Q__():
    assert True is core.false__Q__(False)
    assert False is core.false__Q__(None)

    for v in TRUTHY_VALUES:
        assert False is core.false__Q__(v)


def test_true__Q__():
    assert True is core.true__Q__(True)

    for v in TRUTHY_VALUES:
        if v is not True:
            assert False is core.true__Q__(v)

    for v in FALSEY_VALUES:
        assert False is core.true__Q__(v)


def test_nil__Q__():
    for v in NIL_VALUES:
        assert True is core.nil__Q__(v)

    for v in NON_NIL_VALUES:
        assert False is core.nil__Q__(v)


def test_some__Q__():
    for v in NIL_VALUES:
        assert False is core.some__Q__(v)

    for v in NON_NIL_VALUES:
        assert True is core.some__Q__(v)


def test_any__Q__():
    for v in NIL_VALUES:
        assert True is core.any__Q__(v)

    for v in NON_NIL_VALUES:
        assert True is core.some__Q__(v)


def test___EQ__():
    for v in itertools.chain(NIL_VALUES, NON_NIL_VALUES):
        assert True is core.__EQ__(v)
        assert True is core.__EQ__(v, v)

    assert False is core.__EQ__(1, 1, True)
    assert True is core.__EQ__(1, 1, 1)
    assert True is core.__EQ__(True, True, True)

    assert False is core.__EQ__(0, 0, False)
    assert True is core.__EQ__(0, 0, 0)
    assert True is core.__EQ__(False, False, False)

    assert True is core.__EQ__(None, None, None)

    assert True is core.__EQ__(1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert False is core.__EQ__(1, 2, 1, 1, 1, 1, 1, 1, 1)
    assert False is core.__EQ__(1, 1, 1, 1, 1, 1, 1, 1, 2)


def test_not__EQ__():
    for v in itertools.chain(NIL_VALUES, NON_NIL_VALUES):
        assert False is core.not__EQ__(v), v
        assert False is core.not__EQ__(v, v)

    assert True is core.not__EQ__(1, 1, True)
    assert False is core.not__EQ__(1, 1, 1)
    assert False is core.not__EQ__(True, True, True)

    assert True is core.not__EQ__(0, 0, False)
    assert False is core.not__EQ__(0, 0, 0)
    assert False is core.not__EQ__(False, False, False)

    assert False is core.not__EQ__(None, None, None)

    assert False is core.not__EQ__(1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert True is core.not__EQ__(1, 2, 1, 1, 1, 1, 1, 1, 1)
    assert True is core.not__EQ__(1, 1, 1, 1, 1, 1, 1, 1, 2)


def test___GT__():
    for v in itertools.chain(NIL_VALUES, NON_NIL_VALUES):
        assert True is core.__GT__(v), v

    assert True is core.__GT__(2, 1)
    assert True is core.__GT__(3, 2, 1)
    assert False is core.__GT__(3, 2, 2)
    assert False is core.__GT__(2, 2, 2)
    assert False is core.__GT__(3, 4, 5)


def test___GT____EQ__():
    for v in itertools.chain(NIL_VALUES, NON_NIL_VALUES):
        assert True is core.__GT____EQ__(v), v

    assert True is core.__GT____EQ__(2, 1)
    assert True is core.__GT____EQ__(3, 2, 1)
    assert True is core.__GT____EQ__(3, 2, 2)
    assert True is core.__GT____EQ__(2, 2, 2)
    assert False is core.__GT____EQ__(3, 4, 5)


def test___LT__():
    for v in itertools.chain(NIL_VALUES, NON_NIL_VALUES):
        assert True is core.__LT__(v), v

    assert True is core.__LT__(1, 2)
    assert True is core.__LT__(1, 2, 3)
    assert False is core.__LT__(2, 2, 3)
    assert False is core.__LT__(2, 2, 2)
    assert False is core.__LT__(5, 4, 3)


def test___LT____EQ__():
    for v in itertools.chain(NIL_VALUES, NON_NIL_VALUES):
        assert True is core.__LT____EQ__(v), v

    assert True is core.__LT____EQ__(1, 2)
    assert True is core.__LT____EQ__(1, 2, 3)
    assert True is core.__LT____EQ__(2, 2, 3)
    assert True is core.__LT____EQ__(2, 2, 2)
    assert False is core.__LT____EQ__(5, 4, 3)


def test_str():
    assert "" == core.str_()
    assert "hi" == core.str_("hi")
    assert "1" == core.str_(1)
    assert "hi there i'm chris" == core.str_("hi ", "there ", "i'm ", "chris")
    assert "today is my 1st birthday" == core.str_("today is my ", 1, "st birthday")


def test_name():
    assert "hi" == core.name("hi")
    assert "sym" == core.name(sym.symbol("sym"))
    assert "sym" == core.name(sym.symbol("sym", ns="ns"))
    assert "kw" == core.name(kw.keyword("kw"))
    assert "kw" == core.name(kw.keyword("kw", ns="ns"))


def test_namespace():
    assert None is core.namespace(sym.symbol("sym"))
    assert "ns" == core.namespace(sym.symbol("sym", ns="ns"))
    assert None is core.namespace(kw.keyword("kw"))
    assert "ns" == core.namespace(kw.keyword("kw", ns="ns"))


def test_pos__Q__():
    assert True is core.pos__Q__(1)
    assert True is core.pos__Q__(100)
    assert True is core.pos__Q__(1.0)
    assert True is core.pos__Q__(9999839.874394)
    assert False is core.pos__Q__(0)
    assert False is core.pos__Q__(-1)
    assert False is core.pos__Q__(-100)
    assert False is core.pos__Q__(-1.0)
    assert False is core.pos__Q__(-9999839.874394)


def test_non_neg__Q__():
    assert True is core.non_neg__Q__(1)
    assert True is core.non_neg__Q__(100)
    assert True is core.non_neg__Q__(1.0)
    assert True is core.non_neg__Q__(9999839.874394)
    assert True is core.non_neg__Q__(0)
    assert False is core.non_neg__Q__(-1)
    assert False is core.non_neg__Q__(-100)
    assert False is core.non_neg__Q__(-1.0)
    assert False is core.non_neg__Q__(-9999839.874394)


def test_zero__Q__():
    assert False is core.zero__Q__(1)
    assert False is core.zero__Q__(100)
    assert False is core.zero__Q__(1.0)
    assert False is core.zero__Q__(9999839.874394)
    assert True is core.zero__Q__(0)
    assert False is core.zero__Q__(-1)
    assert False is core.zero__Q__(-100)
    assert False is core.zero__Q__(-1.0)
    assert False is core.zero__Q__(-9999839.874394)


def test_neg__Q__():
    assert False is core.neg__Q__(1)
    assert False is core.neg__Q__(100)
    assert False is core.neg__Q__(1.0)
    assert False is core.neg__Q__(9999839.874394)
    assert False is core.neg__Q__(0)
    assert True is core.neg__Q__(-1)
    assert True is core.neg__Q__(-100)
    assert True is core.neg__Q__(-1.0)
    assert True is core.neg__Q__(-9999839.874394)


def test___PLUS__():
    assert 0 == core.__PLUS__()
    assert -1 == core.__PLUS__(-1)
    assert 0 == core.__PLUS__(0)
    assert 1 == core.__PLUS__(1)
    assert 15 == core.__PLUS__(1, 2, 3, 4, 5)
    assert 5 == core.__PLUS__(1, 2, 3, 4, -5)
    assert -15 == core.__PLUS__(-1, -2, -3, -4, -5)


def test_minus():
    with pytest.raises(runtime.RuntimeException):
        core._()

    assert 1 == core._(-1)
    assert 0 == core._(0)
    assert -1 == core._(1)
    assert -13 == core._(1, 2, 3, 4, 5)
    assert -3 == core._(1, 2, 3, 4, -5)
    assert 13 == core._(-1, -2, -3, -4, -5)


def test___STAR__():
    assert 1 == core.__STAR__()
    assert -1 == core.__STAR__(-1)
    assert 0 == core.__STAR__(0)
    assert 1 == core.__STAR__(1)
    assert 120 == core.__STAR__(1, 2, 3, 4, 5)
    assert -120 == core.__STAR__(1, 2, 3, 4, -5)
    assert -120 == core.__STAR__(-1, -2, -3, -4, -5)


def test___DIV__():
    with pytest.raises(runtime.RuntimeException):
        core.__DIV__()

    with pytest.raises(ZeroDivisionError):
        core.__DIV__(0)

    with pytest.raises(ZeroDivisionError):
        core.__DIV__(3, 0)

    assert -1 == core.__DIV__(-1)
    assert 1 == core.__DIV__(1)
    assert Fraction(1, 2) == core.__DIV__(2)
    assert 0.5 == core.__DIV__(2.0)
    assert Fraction(1, 2) == core.__DIV__(1, 2)
    assert 0.5 == core.__DIV__(1.0, 2)
    assert 0.125 == core.__DIV__(1, 2, 4.0)
    assert Fraction(-1, 120) == core.__DIV__(1, 2, 3, 4, -5)
    assert 0.008333333333333333 == core.__DIV__(1, 2, 3, 4, 5.0)
    assert Fraction(-1, 120) == core.__DIV__(-1, -2, -3, -4, -5)
    assert -0.008333333333333333 == core.__DIV__(-1, -2, -3, -4, -5.0)


def test_mod():
    assert 0 == core.mod(10, 5)
    assert 4 == core.mod(10, 6)
    assert 0 == core.mod(10, 10)
    assert 0 == core.mod(10, -1)
    assert 3 == core.mod(-21, 4)
    assert 3 == core.mod(-2, 5)
    assert 2 == core.mod(-10, 3)
    assert 0.5 == core.mod(1.5, 1)
    assert 6.095000000000027 == core.mod(475.095, 7)
    assert 0.840200000000074 == core.mod(1024.8402, 5.12)
    assert 4.279799999999926 == core.mod(-1024.8402, 5.12)


def test_quot():
    assert 0 == core.quot(1, 2)
    assert 1 == core.quot(2, 2)
    assert 1 == core.quot(3, 2)
    assert 2 == core.quot(4, 2)
    assert 3 == core.quot(10, 3)
    assert 3 == core.quot(11, 3)
    assert 4 == core.quot(12, 3)
    assert 1.0 == core.quot(5.9, 3)
    assert -1.0 == core.quot(-5.9, 3)
    assert -1.0 == core.quot(5.9, -3)
    assert -3 == core.quot(-10, 3)
    assert -3 == core.quot(10, -3)
    assert 3 == core.quot(10, 3)


def test_rem():
    assert 1 == core.rem(10, 9)
    assert 0 == core.rem(2, 2)
    assert -1 == core.rem(-10, 3)
    assert -1 == core.rem(-21, 4)


def test_inc():
    assert 11 == core.inc(10)
    assert 1 == core.inc(0)
    assert 0 == core.inc(-1)
    assert 6.9 == core.inc(5.9)
    assert -4.9 == core.inc(-5.9)


def test_dec():
    assert 9 == core.dec(10)
    assert -1 == core.dec(0)
    assert -2 == core.dec(-1)
    assert 0 == core.dec(1)
    assert 4.9 == core.dec(5.9)
    assert -6.9 == core.dec(-5.9)


def test_even__Q__():
    assert True is core.even__Q__(-10)
    assert True is core.even__Q__(-2)
    assert True is core.even__Q__(0)
    assert True is core.even__Q__(2)
    assert True is core.even__Q__(10)
    assert True is core.even__Q__(-10.0)
    assert True is core.even__Q__(-2.0)
    assert True is core.even__Q__(0.0)
    assert True is core.even__Q__(2.0)
    assert True is core.even__Q__(10.0)

    assert False is core.even__Q__(-11.0)
    assert False is core.even__Q__(-3.0)
    assert False is core.even__Q__(-11)
    assert False is core.even__Q__(-3)
    assert False is core.even__Q__(3)
    assert False is core.even__Q__(11)
    assert False is core.even__Q__(3.0)
    assert False is core.even__Q__(11.0)


def test_odd__Q__():
    assert False is core.odd__Q__(-10)
    assert False is core.odd__Q__(-2)
    assert False is core.odd__Q__(0)
    assert False is core.odd__Q__(2)
    assert False is core.odd__Q__(10)
    assert False is core.odd__Q__(-10.0)
    assert False is core.odd__Q__(-2.0)
    assert False is core.odd__Q__(0.0)
    assert False is core.odd__Q__(2.0)
    assert False is core.odd__Q__(10.0)

    assert True is core.odd__Q__(-11.0)
    assert True is core.odd__Q__(-3.0)
    assert True is core.odd__Q__(-11)
    assert True is core.odd__Q__(-3)
    assert True is core.odd__Q__(3)
    assert True is core.odd__Q__(11)
    assert True is core.odd__Q__(3.0)
    assert True is core.odd__Q__(11.0)


def test_min():
    assert 5 == core.min_(5)
    assert 5 == core.min_(5, 5)
    assert 5 == core.min_(5, 9)
    assert 1 == core.min_(1, 2, 3, 4, 5)
    assert -399 == core.min_(5, 10, -1, 532, -399, 42.3, 99.1937, -33.8)


def test_max():
    assert 5 == core.max_(5)
    assert 5 == core.max_(5, 5)
    assert 9 == core.max_(5, 9)
    assert 5 == core.max_(1, 2, 3, 4, 5)
    assert 532 == core.max_(5, 10, -1, 532, -399, 42.3, 99.1937, -33.8)


def test_sort():
    assert llist.l(1) == core.sort(vec.v(1))
    assert llist.l(1, 2, 3) == core.sort(vec.v(1, 2, 3))
    assert llist.l(1, 2, 3, 4, 5) == core.sort(vec.v(5, 3, 1, 2, 4))


def test_contains__Q__():
    assert True is core.contains__Q__(lmap.map({"a": 1}), "a")
    assert False is core.contains__Q__(lmap.map({"a": 1}), "b")
    assert True is core.contains__Q__(vec.v(1, 2, 3), 0)
    assert True is core.contains__Q__(vec.v(1, 2, 3), 1)
    assert True is core.contains__Q__(vec.v(1, 2, 3), 2)
    assert False is core.contains__Q__(vec.v(1, 2, 3), 3)
    assert False is core.contains__Q__(vec.v(1, 2, 3), -1)


def test_disj():
    assert lset.Set.empty() == core.disj(lset.Set.empty(), "a")
    assert lset.Set.empty() == core.disj(lset.s("a"), "a")
    assert lset.s("b", "d") == core.disj(lset.s("a", "b", "c", "d"), "a", "c", "e")


def test_dissoc():
    assert lmap.Map.empty() == core.dissoc(lmap.map({"a": 1}), "a", "c")
    assert lmap.map({"a": 1}) == core.dissoc(lmap.map({"a": 1}), "b", "c")


def test_get():
    assert 1 == core.get(lmap.map({"a": 1}), "a")
    assert None is core.get(lmap.map({"a": 1}), "b")
    assert 2 == core.get(lmap.map({"a": 1}), "b", 2)
    assert 1 == core.get(vec.v(1, 2, 3), 0)
    assert 2 == core.get(vec.v(1, 2, 3), 1)
    assert 3 == core.get(vec.v(1, 2, 3), 2)
    assert None is core.get(vec.v(1, 2, 3), 3)
    assert 4 == core.get(vec.v(1, 2, 3), 3, 4)
    assert 3 == core.get(vec.v(1, 2, 3), -1)
    assert 2 == core.get(vec.v(1, 2, 3), -2)
    assert 1 == core.get(vec.v(1, 2, 3), -3)
    assert None is core.get(vec.v(1, 2, 3), -4)


def test_range():
    assert llist.l(1) == core.range_(1, 1)
    assert llist.l(1, 2, 3, 4, 5) == core.range_(1, 5)
    assert llist.l(1, 3, 5, 7, 9) == core.range_(1, 10, 2)
    # assert llist.l(1, -1, -3, -5, -7, -9) == core.range_(1, -10, -2)


def test_constantly():
    f = core.constantly("hi")
    assert "hi" == f()
    assert "hi" == f(1)
    assert "hi" == f("what will", "you", "return?")


def test_complement():
    is_even = core.complement(core.odd__Q__)
    assert True is is_even(-10)
    assert True is is_even(-2)
    assert True is is_even(0)
    assert True is is_even(2)
    assert True is is_even(10)
    assert True is is_even(-10.0)
    assert True is is_even(-2.0)
    assert True is is_even(0.0)
    assert True is is_even(2.0)
    assert True is is_even(10.0)

    assert False is is_even(-11.0)
    assert False is is_even(-3.0)
    assert False is is_even(-11)
    assert False is is_even(-3)
    assert False is is_even(3)
    assert False is is_even(11)
    assert False is is_even(3.0)
    assert False is is_even(11.0)


def test_reduce():
    assert 0 == core.reduce(core.__PLUS__, [])
    assert 1 == core.reduce(core.__PLUS__, [1])
    assert 6 == core.reduce(core.__PLUS__, [1, 2, 3])
    assert 45 == core.reduce(core.__PLUS__, 45, [])
    assert 46 == core.reduce(core.__PLUS__, 45, [1])


def test_reduce_with_lazy_seq():
    assert 25 == core.reduce(core.__PLUS__, core.filter_(core.odd__Q__, vec.v(1, 2, 3, 4, 5, 6, 7, 8, 9)))
    assert 25 == core.reduce(core.__PLUS__, 0, core.filter_(core.odd__Q__, vec.v(1, 2, 3, 4, 5, 6, 7, 8, 9)))


def test_interpose():
    assert llist.List.empty() == core.interpose(",", vec.Vector.empty())
    assert llist.l("hi") == core.interpose(",", vec.v("hi"))
    assert llist.l("hi", ",", "there") == core.interpose(",", vec.v("hi", "there"))


def test_comp():
    assert 1 == core.comp()(1)
    assert "hi" == core.comp()("hi")
    assert True is core.comp(core.odd__Q__)(3)
    assert False is core.comp(core.odd__Q__)(2)
    assert 7 == core.comp(core.inc, core.inc, core.dec, lambda v: core.__STAR__(2, v))(3)


def test_juxt():
    assert vec.v(True) == core.juxt(core.odd__Q__)(3)
    assert vec.v(True, False, 4, 2) == core.juxt(core.odd__Q__, core.even__Q__, core.inc, core.dec)(3)


def test_partial():
    assert 3 == core.partial(core.__PLUS__)(3)
    assert 6 == core.partial(core.__PLUS__, 3)(3)
    assert 10 == core.partial(core.__PLUS__, 3, 4)(3)


def test_every__Q__():
    assert True is core.every__Q__(core.odd__Q__, vec.Vector.empty())
    assert True is core.every__Q__(core.odd__Q__, vec.v(3))
    assert True is core.every__Q__(core.odd__Q__, vec.v(3, 5, 7, 9, 11))
    assert False is core.every__Q__(core.odd__Q__, vec.v(2))
    assert False is core.every__Q__(core.odd__Q__, vec.v(3, 5, 7, 9, 2))
    assert False is core.every__Q__(core.odd__Q__, vec.v(2, 3, 5, 7, 9))
    assert False is core.every__Q__(core.odd__Q__, vec.v(3, 5, 2, 7, 9))


def test_not_every__Q__():
    assert False is core.not_every__Q__(core.odd__Q__, vec.Vector.empty())
    assert False is core.not_every__Q__(core.odd__Q__, vec.v(3))
    assert False is core.not_every__Q__(core.odd__Q__, vec.v(3, 5, 7, 9, 11))
    assert True is core.not_every__Q__(core.odd__Q__, vec.v(2))
    assert True is core.not_every__Q__(core.odd__Q__, vec.v(3, 5, 7, 9, 2))
    assert True is core.not_every__Q__(core.odd__Q__, vec.v(2, 3, 5, 7, 9))
    assert True is core.not_every__Q__(core.odd__Q__, vec.v(3, 5, 2, 7, 9))


def test_some():
    assert None is core.some(core.odd__Q__, vec.Vector.empty())
    assert True is core.some(core.odd__Q__, vec.v(3))
    assert True is core.some(core.odd__Q__, vec.v(3, 5, 7, 9, 11))
    assert None is core.some(core.odd__Q__, vec.v(2))
    assert True is core.some(core.odd__Q__, vec.v(3, 5, 7, 9, 2))
    assert True is core.some(core.odd__Q__, vec.v(2, 3, 5, 7, 9))
    assert True is core.some(core.odd__Q__, vec.v(3, 5, 2, 7, 9))
    assert None is core.some(core.odd__Q__, vec.v(2, 4, 6, 8, 10))


def test_not_any__Q__():
    assert True is core.not_any__Q__(core.odd__Q__, vec.Vector.empty())
    assert False is core.not_any__Q__(core.odd__Q__, vec.v(3))
    assert False is core.not_any__Q__(core.odd__Q__, vec.v(3, 5, 7, 9, 11))
    assert True is core.not_any__Q__(core.odd__Q__, vec.v(2))
    assert False is core.not_any__Q__(core.odd__Q__, vec.v(3, 5, 7, 9, 2))
    assert False is core.not_any__Q__(core.odd__Q__, vec.v(2, 3, 5, 7, 9))
    assert False is core.not_any__Q__(core.odd__Q__, vec.v(3, 5, 2, 7, 9))
    assert True is core.not_any__Q__(core.odd__Q__, vec.v(2, 4, 6, 8, 10))


def test_merge():
    assert None is core.merge()
    assert lmap.Map.empty() == core.merge(lmap.Map.empty())
    assert lmap.map({kw.keyword("a"): 1}) == core.merge(lmap.map({kw.keyword("a"): 1}))
    assert lmap.map({kw.keyword("a"): 53, kw.keyword("b"): "hi"}) == core.merge(
        lmap.map({kw.keyword("a"): 1, kw.keyword("b"): "hi"}),
        lmap.map({kw.keyword("a"): 53}))


def test_pr_str():
    assert '' == core.pr_str()
    assert '""' == core.pr_str("")
    assert ':kw' == core.pr_str(kw.keyword('kw'))
    assert ':hi "there" 3' == core.pr_str(kw.keyword('hi'), "there", 3)


def test_prn_str():
    assert '\n' == core.prn_str()
    assert '""\n' == core.prn_str("")
    assert ':kw\n' == core.prn_str(kw.keyword('kw'))
    assert ':hi "there" 3\n' == core.prn_str(kw.keyword('hi'), "there", 3)


def test_print_str():
    assert '' == core.print_str()
    assert '' == core.print_str("")
    assert 'kw' == core.print_str(kw.keyword('kw'))
    assert 'hi there 3' == core.print_str(kw.keyword('hi'), "there", 3)


def test_println_str():
    assert '\n' == core.println_str()
    assert '\n' == core.println_str("")
    assert 'kw\n' == core.println_str(kw.keyword('kw'))
    assert 'hi there 3\n' == core.println_str(kw.keyword('hi'), "there", 3)


def test_re_find():
    assert None is core.re_find(re.compile(r"\d+"), "abcdef")
    assert "12345" == core.re_find(re.compile(r"\d+"), "abc12345def")
    assert vec.v("word then number ", "word then number ", None) == core.re_find(
        re.compile(r"(\D+)|(\d+)"), "word then number 57")
    assert vec.v("57", None, "57") == core.re_find(
        re.compile(r"(\D+)|(\d+)"), "57 number then word")
    assert vec.v("lots", "", "l") == core.re_find(re.compile(r"(\d*)(\S)\S+"), "lots o' digits 123456789")


def test_re_matches():
    assert None is core.re_matches(re.compile(r"hello"), "hello, world")
    assert "hello, world" == core.re_matches(re.compile(r"hello.*"), "hello, world")
    assert vec.v("hello, world", "world") == core.re_matches(re.compile(r"hello, (.*)"), "hello, world")


def test_re_seq():
    assert None is core.seq(core.re_seq(re.compile(r"[a-zA-Z]+"), "134325235234"))
    assert llist.l("1", "1", "0") == core.re_seq(re.compile(r"\d+"), "Basilisp 1.1.0")
    assert llist.l("the", "man", "who", "sold", "the", "world") == core.re_seq(
        re.compile(r"\w+"), "the man who sold the world")
