import itertools
import re
from decimal import Decimal
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


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """Disable the `print_generated_python` flag so we can safely capture
    stderr and stdout for tests which require those facilities."""
    init()
    orig = runtime.print_generated_python
    runtime.print_generated_python = Mock(return_value=False)
    yield
    runtime.print_generated_python = orig


import basilisp.core as core

TRUTHY_VALUES = [
    True,
    -1,
    0,
    1,
    -1.0,
    0.0,
    1.0,
    sym.symbol("name"),
    sym.symbol("name", ns="ns"),
    kw.keyword("name"),
    kw.keyword("name", ns="ns"),
    "",
    "not empty",
    llist.List.empty(),
    llist.l(0),
    llist.l(False),
    llist.l(True),
    lmap.Map.empty(),
    lmap.map({0: 0}),
    lmap.map({False: False}),
    lmap.map({True: True}),
    lset.Set.empty(),
    lset.s(0),
    lset.s(False),
    lset.s(True),
    vec.Vector.empty(),
    vec.v(0),
    vec.v(False),
    vec.v(True),
]

FALSEY_VALUES = [False, None]

NON_NIL_VALUES = [
    False,
    True,
    -1,
    0,
    1,
    -1.0,
    0.0,
    1.0,
    sym.symbol("name"),
    sym.symbol("name", ns="ns"),
    kw.keyword("name"),
    kw.keyword("name", ns="ns"),
    "",
    "not empty",
    llist.List.empty(),
    llist.l(0),
    llist.l(False),
    llist.l(True),
    lmap.Map.empty(),
    lmap.map({0: 0}),
    lmap.map({False: False}),
    lmap.map({True: True}),
    lset.Set.empty(),
    lset.s(0),
    lset.s(False),
    lset.s(True),
    vec.Vector.empty(),
    vec.v(0),
    vec.v(False),
    vec.v(True),
]
NIL_VALUES = [None]

EVEN_INTS = [-10, -2, 0, 2, 10]
EVEN_FLOATS = [-10.0, -2.0, 0.0, 2.0, 10.0]

ODD_INTS = [-11, -3, 3, 11]
ODD_FLOATS = [-11.0, -3.0, 3.0, 11.0]


@pytest.fixture(params=TRUTHY_VALUES)
def truthy_value(request):
    return request.param


@pytest.fixture(params=FALSEY_VALUES)
def falsey_value(request):
    return request.param


@pytest.fixture(params=NON_NIL_VALUES)
def non_nil_value(request):
    return request.param


@pytest.fixture(params=NIL_VALUES)
def nil_value(request):
    return request.param


@pytest.fixture(params=itertools.chain(NON_NIL_VALUES, NIL_VALUES))
def lisp_value(request):
    return request.param


@pytest.fixture(params=EVEN_INTS)
def even_int(request):
    return request.param


@pytest.fixture(params=EVEN_FLOATS)
def even_float(request):
    return request.param


@pytest.fixture(params=itertools.chain(EVEN_INTS, EVEN_FLOATS))
def even_number(request):
    return request.param


@pytest.fixture(params=ODD_INTS)
def odd_int(request):
    return request.param


@pytest.fixture(params=ODD_FLOATS)
def odd_float(request):
    return request.param


@pytest.fixture(params=itertools.chain(ODD_INTS, ODD_FLOATS))
def odd_number(request):
    return request.param


@pytest.fixture(params=itertools.chain(ODD_INTS, EVEN_INTS))
def int_number(request):
    return request.param


@pytest.fixture(params=itertools.chain(ODD_FLOATS, EVEN_FLOATS))
def float_number(request):
    return request.param


@pytest.fixture(params=itertools.chain(ODD_INTS, EVEN_INTS, ODD_FLOATS, EVEN_FLOATS))
def real_number(request):
    return request.param


@pytest.fixture
def complex_number(real_number):
    return complex(0, real_number)


@pytest.fixture
def decimal(even_int, odd_int):
    return Decimal(even_int) / Decimal(odd_int)


@pytest.fixture
def fraction(even_int, odd_int):
    return Fraction(even_int, odd_int)


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


class TestNot:
    def test_falsey(self, falsey_value):
        assert True is core.not_(falsey_value)

    def test_truth(self, truthy_value):
        assert False is core.not_(truthy_value)


class TestEquals:
    def test_equals_to_itself(self, lisp_value):
        assert True is core.__EQ__(lisp_value)
        assert True is core.__EQ__(lisp_value, lisp_value)

    def test_consecutive_equals(self):
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


class TestNotEquals:
    def test_equals_to_itself(self, lisp_value):
        assert False is core.not__EQ__(lisp_value), lisp_value
        assert False is core.not__EQ__(lisp_value, lisp_value)

    def test_consecutive_not_equals(self):
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


class TestComparison:
    def test_one_arg_gt(self, lisp_value):
        assert True is core.__GT__(lisp_value), lisp_value

    def test_gt(self):
        assert True is core.__GT__(2, 1)
        assert True is core.__GT__(3, 2, 1)
        assert False is core.__GT__(3, 2, 2)
        assert False is core.__GT__(2, 2, 2)
        assert False is core.__GT__(3, 4, 5)

    def test_one_arg_ge(self, lisp_value):
        assert True is core.__GT____EQ__(lisp_value), lisp_value

    def test_ge(self):
        assert True is core.__GT____EQ__(2, 1)
        assert True is core.__GT____EQ__(3, 2, 1)
        assert True is core.__GT____EQ__(3, 2, 2)
        assert True is core.__GT____EQ__(2, 2, 2)
        assert False is core.__GT____EQ__(3, 4, 5)

    def test_one_arg_lt(self, lisp_value):
        assert True is core.__LT__(lisp_value), lisp_value

    def test_lt(self):
        assert True is core.__LT__(1, 2)
        assert True is core.__LT__(1, 2, 3)
        assert False is core.__LT__(2, 2, 3)
        assert False is core.__LT__(2, 2, 2)
        assert False is core.__LT__(5, 4, 3)

    def test_one_arg_le(self, lisp_value):
        assert True is core.__LT____EQ__(lisp_value), lisp_value

    def test_le(self):
        assert True is core.__LT____EQ__(1, 2)
        assert True is core.__LT____EQ__(1, 2, 3)
        assert True is core.__LT____EQ__(2, 2, 3)
        assert True is core.__LT____EQ__(2, 2, 2)
        assert False is core.__LT____EQ__(5, 4, 3)

    def test_is_identical(self, lisp_value):
        assert core.identical__Q__(lisp_value, lisp_value)

    def test_is_not_identical(self):
        assert False is core.identical__Q__(object(), object())

    def test_hash(self, lisp_value):
        assert hash(lisp_value) == core.hash_(lisp_value)


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


class TestNumericPredicates:
    @pytest.mark.parametrize("v", [1, 100, 1.0, 9_999_839.874_394])
    def test_is_positive(self, v):
        assert True is core.pos__Q__(v)

    @pytest.mark.parametrize("v", [0, -1, -100, -1.0, -9_999_839.874_394])
    def test_is_not_positive(self, v):
        assert False is core.pos__Q__(v)

    @pytest.mark.parametrize("v", [0, 1, 100, 1.0, 9_999_839.874_394])
    def test_is_non_neg(self, v):
        assert True is core.non_neg__Q__(v)

    @pytest.mark.parametrize("v", [-1, -100, -1.0, -9_999_839.874_394])
    def test_is_not_non_neg(self, v):
        assert False is core.non_neg__Q__(v)

    def test_is_zero(self):
        assert True is core.zero__Q__(0)

    @pytest.mark.parametrize(
        "v", [1, 100, 1.0, 9_999_839.874_394, -1, -100, -1.0, -9_999_839.874_394]
    )
    def test_is_not_zero(self, v):
        assert False is core.zero__Q__(v)

    @pytest.mark.parametrize("v", [-1, -100, -1.0, -9_999_839.874_394])
    def test_is_neg(self, v):
        assert True is core.neg__Q__(v)

    @pytest.mark.parametrize("v", [0, 1, 100, 1.0, 9_999_839.874_394])
    def test_is_not_neg(self, v):
        assert False is core.neg__Q__(1)


class TestArithmetic:
    def test_addition(self):
        assert 0 == core.__PLUS__()
        assert -1 == core.__PLUS__(-1)
        assert 0 == core.__PLUS__(0)
        assert 1 == core.__PLUS__(1)
        assert 15 == core.__PLUS__(1, 2, 3, 4, 5)
        assert 5 == core.__PLUS__(1, 2, 3, 4, -5)
        assert -15 == core.__PLUS__(-1, -2, -3, -4, -5)

    def test_subtraction(self):
        with pytest.raises(runtime.RuntimeException):
            core._()

        assert 1 == core._(-1)
        assert 0 == core._(0)
        assert -1 == core._(1)
        assert -13 == core._(1, 2, 3, 4, 5)
        assert -3 == core._(1, 2, 3, 4, -5)
        assert 13 == core._(-1, -2, -3, -4, -5)

    def test_multiplication(self):
        assert 1 == core.__STAR__()
        assert -1 == core.__STAR__(-1)
        assert 0 == core.__STAR__(0)
        assert 1 == core.__STAR__(1)
        assert 120 == core.__STAR__(1, 2, 3, 4, 5)
        assert -120 == core.__STAR__(1, 2, 3, 4, -5)
        assert -120 == core.__STAR__(-1, -2, -3, -4, -5)

    def test_division(self):
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
        assert 0.008_333_333_333_333_333 == core.__DIV__(1, 2, 3, 4, 5.0)
        assert Fraction(-1, 120) == core.__DIV__(-1, -2, -3, -4, -5)
        assert -0.008_333_333_333_333_333 == core.__DIV__(-1, -2, -3, -4, -5.0)

    @pytest.mark.parametrize(
        "result,x,y",
        [
            (0, 10, 5),
            (4, 10, 6),
            (0, 10, 10),
            (0, 10, -1),
            (3, -21, 4),
            (3, -2, 5),
            (2, -10, 3),
            (0.5, 1.5, 1),
            (6.095_000_000_000_027, 475.095, 7),
            (0.840_200_000_000_074, 1024.8402, 5.12),
            (4.279_799_999_999_926, -1024.8402, 5.12),
        ],
    )
    def test_mod(self, result, x, y):
        assert result == core.mod(x, y)

    @pytest.mark.parametrize(
        "result,x,y",
        [
            (0, 1, 2),
            (1, 2, 2),
            (1, 3, 2),
            (2, 4, 2),
            (3, 10, 3),
            (3, 11, 3),
            (4, 12, 3),
            (1.0, 5.9, 3),
            (-1.0, -5.9, 3),
            (-3, -10, 3),
            (-3, 10, -3),
            (3, 10, 3),
        ],
    )
    def test_quot(self, result, x, y):
        assert result == core.quot(x, y)

    @pytest.mark.parametrize(
        "result,x,y", [(1, 10, 9), (0, 2, 2), (-1, -10, 3), (-1, -21, 4)]
    )
    def test_rem(self, result, x, y):
        assert result == core.rem(x, y)

    @pytest.mark.parametrize(
        "result,x", [(11, 10), (1, 0), (0, -1), (6.9, 5.9), (-4.9, -5.9)]
    )
    def test_inc(self, result, x):
        assert result == core.inc(x)

    @pytest.mark.parametrize(
        "result,x", [(9, 10), (-1, 0), (-2, -1), (0, 1), (4.9, 5.9), (-6.9, -5.9)]
    )
    def test_dec(self, result, x):
        assert result == core.dec(x)


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


def test_numerator(fraction):
    assert fraction.numerator == core.numerator(fraction)


def test_denominator(fraction):
    assert fraction.denominator == core.denominator(fraction)


def test_sort():
    assert llist.l(1) == core.sort(vec.v(1))
    assert llist.l(1, 2, 3) == core.sort(vec.v(1, 2, 3))
    assert llist.l(1, 2, 3, 4, 5) == core.sort(vec.v(5, 3, 1, 2, 4))


class TestIsAny:
    def test_any_always_true(self, lisp_value):
        assert True is core.any__Q__(lisp_value)


class TestIsAssociative:
    @pytest.mark.parametrize("v", [lmap.Map.empty(), vec.Vector.empty()])
    def test_is_associative(self, v):
        assert True is core.associative__Q__(v)

    @pytest.mark.parametrize("v", [llist.List.empty(), lset.Set.empty()])
    def test_is_not_associative(self, v):
        assert False is core.associative__Q__(v)


class TestIsClass:
    @pytest.mark.parametrize(
        "tp", [kw.Keyword, llist.List, lmap.Map, sym.Symbol, vec.Vector]
    )
    def test_is_class(self, tp):
        assert True is core.class__Q__(tp)

    @pytest.mark.parametrize("tp", [kw.keyword("a"), 1, "string", sym.symbol("sym")])
    def test_is_not_class(self, tp):
        assert False is core.class__Q__(tp)


class TestIsColl:
    @pytest.mark.parametrize(
        "v",
        [llist.List.empty(), lmap.Map.empty(), lset.Set.empty(), vec.Vector.empty()],
    )
    def test_is_coll(self, v):
        assert True is core.coll__Q__(v)

    @pytest.mark.parametrize("v", [kw.keyword("a"), 1, "string", sym.symbol("sym")])
    def test_is_not_coll(self, v):
        assert False is core.coll__Q__(v)


class TestIsComplex:
    def test_is_complex(self, complex_number):
        assert True is core.complex__Q__(complex_number)

    def test_real_is_not_complex(self, real_number):
        assert False is core.complex__Q__(real_number)

    def test_fraction_is_not_complex(self, fraction):
        assert False is core.complex__Q__(fraction)

    def test_decimal_is_not_complex(self, decimal):
        assert False is core.complex__Q__(decimal)


class TestIsDecimal:
    def test_is_decimal(self, decimal):
        assert True is core.decimal__Q__(decimal)

    def test_real_is_not_decimal(self, real_number):
        assert False is core.decimal__Q__(real_number)

    def test_fraction_is_not_decimal(self, fraction):
        assert False is core.decimal__Q__(fraction)

    def test_complex_is_not_decimal(self, complex_number):
        assert False is core.decimal__Q__(complex_number)


class TestIsDouble:
    def test_is_double(self, float_number):
        assert True is core.double__Q__(float_number)
        assert True is core.float__Q__(float_number)

    def test_decimal_is_not_double(self, decimal):
        assert False is core.double__Q__(decimal)
        assert False is core.float__Q__(decimal)

    def test_fraction_is_not_double(self, fraction):
        assert False is core.double__Q__(fraction)
        assert False is core.float__Q__(fraction)

    def test_integer_is_not_double(self, int_number):
        assert False is core.double__Q__(int_number)
        assert False is core.float__Q__(int_number)

    def test_complex_is_not_double(self, complex_number):
        assert False is core.double__Q__(complex_number)
        assert False is core.float__Q__(complex_number)


class TestIsEven:
    def test_even_nums_are_even(self, even_number):
        assert True is core.even__Q__(even_number)

    def test_odd_nums_are_not_even(self, odd_number):
        assert False is core.even__Q__(odd_number)


class TestIsFalse:
    def test_false_is_false(self):
        assert True is core.false__Q__(False)

    def test_none_is_not_false(self):
        assert False is core.false__Q__(None)

    def test_truth_values_are_not_false(self, truthy_value):
        assert False is core.false__Q__(truthy_value)


class TestIsFn:
    @pytest.fixture(scope="class")
    def basilisp_fn(self):
        @runtime._basilisp_fn
        def repeat(v):
            while True:
                yield v

        return repeat

    @pytest.fixture(scope="class")
    def py_fn(self):
        return lambda v: v

    def test_is_fn(self, basilisp_fn, py_fn):
        assert True is core.fn__Q__(basilisp_fn)
        assert False is core.fn__Q__(py_fn)

    @pytest.mark.parametrize(
        "v",
        [
            "a",
            1,
            1.6,
            kw.keyword("a"),
            lmap.Map.empty(),
            lset.Set.empty(),
            sym.symbol("a"),
            vec.Vector.empty(),
        ],
    )
    def test_is_not_fn(self, v):
        assert False is core.fn__Q__(v)

    def test_function_is_ifn(self, basilisp_fn, py_fn):
        assert True is core.ifn__Q__(basilisp_fn)
        assert True is core.ifn__Q__(py_fn)

    @pytest.mark.parametrize("v", [kw.keyword("a"), lmap.Map.empty(), lset.Set.empty()])
    def test_other_is_ifn(self, v):
        assert True is core.ifn__Q__(v)

    @pytest.mark.parametrize("v", ["a", 1, 1.6])
    def test_is_not_ifn(self, v):
        assert False is core.ifn__Q__(v)


class TestIsInt:
    def test_is_int(self, int_number):
        assert True is core.integer__Q__(int_number)
        assert True is core.int__Q__(int_number)

    def test_decimal_is_not_int(self, decimal):
        assert False is core.integer__Q__(decimal)
        assert False is core.int__Q__(decimal)

    def test_double_is_not_int(self, float_number):
        assert False is core.integer__Q__(float_number)
        assert False is core.int__Q__(float_number)

    def test_fraction_is_not_int(self, fraction):
        assert False is core.integer__Q__(fraction)
        assert False is core.int__Q__(fraction)

    def test_complex_is_not_int(self, complex_number):
        assert False is core.integer__Q__(complex_number)
        assert False is core.integer__Q__(complex_number)


class TestIsNil:
    def test_nil_values_are_nil(self, nil_value):
        assert True is core.nil__Q__(nil_value)

    def test_non_nil_values_are_not_nil(self, non_nil_value):
        assert False is core.nil__Q__(non_nil_value)


class TestIsOdd:
    def test_odd_nums_are_odd(self, odd_number):
        assert True is core.odd__Q__(odd_number)

    def test_even_nums_are_not_odd(self, even_number):
        assert False is core.odd__Q__(even_number)


class TestIsSome:
    def test_nil_values_are_not_some(self, nil_value):
        assert False is core.some__Q__(nil_value)

    def test_non_nil_values_are_some(self, non_nil_value):
        assert True is core.some__Q__(non_nil_value)


class TestIsTrue:
    def test_true_is_true(self):
        assert True is core.true__Q__(True)

    def test_other_values_are_not_true(self, lisp_value):
        if lisp_value is not True:
            assert False is core.true__Q__(lisp_value)


class TestAssociativeFunctions:
    def test_contains(self):
        assert True is core.contains__Q__(lmap.map({"a": 1}), "a")
        assert False is core.contains__Q__(lmap.map({"a": 1}), "b")
        assert True is core.contains__Q__(vec.v(1, 2, 3), 0)
        assert True is core.contains__Q__(vec.v(1, 2, 3), 1)
        assert True is core.contains__Q__(vec.v(1, 2, 3), 2)
        assert False is core.contains__Q__(vec.v(1, 2, 3), 3)
        assert False is core.contains__Q__(vec.v(1, 2, 3), -1)

    def test_disj(self):
        assert lset.Set.empty() == core.disj(lset.Set.empty(), "a")
        assert lset.Set.empty() == core.disj(lset.s("a"), "a")
        assert lset.s("b", "d") == core.disj(lset.s("a", "b", "c", "d"), "a", "c", "e")

    def test_dissoc(self):
        assert lmap.Map.empty() == core.dissoc(lmap.map({"a": 1}), "a", "c")
        assert lmap.map({"a": 1}) == core.dissoc(lmap.map({"a": 1}), "b", "c")

    def test_get(self):
        assert None is core.get(None, "a")
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

    def test_keys(self):
        assert None is core.keys(lmap.map({}))
        assert llist.l("a") == core.keys(lmap.map({"a": 1}))
        assert lset.s("a", "b") == lset.set(core.keys(lmap.map({"a": 1, "b": 2})))

    def test_vals(self):
        assert None is core.vals(lmap.map({}))
        assert llist.l(1) == core.vals(lmap.map({"a": 1}))
        assert lset.s(1, 2) == lset.set(core.vals(lmap.map({"a": 1, "b": 2})))

    def test_select_keys(self):
        assert lmap.Map.empty() == core.select_keys(
            lmap.Map.empty(), vec.Vector.empty()
        )
        assert lmap.Map.empty() == core.select_keys(
            lmap.Map.empty(), vec.v(kw.keyword("a"), kw.keyword("b"))
        )
        assert lmap.map(
            {kw.keyword("a"): "a", kw.keyword("b"): "b"}
        ) == core.select_keys(
            lmap.map(
                {kw.keyword("a"): "a", kw.keyword("b"): "b", kw.keyword("c"): "c"}
            ),
            vec.v(kw.keyword("a"), kw.keyword("b")),
        )


def test_range():
    assert llist.l(1) == core.range_(1, 1)
    assert llist.l(1, 2, 3, 4, 5) == core.range_(1, 6)
    assert llist.l(1, 3, 5, 7, 9) == core.range_(1, 11, 2)
    # assert llist.l(1, -1, -3, -5, -7, -9) == core.range_(1, -10, -2)


def test_constantly():
    f = core.constantly("hi")
    assert "hi" == f()
    assert "hi" == f(1)
    assert "hi" == f("what will", "you", "return?")


class TestComplement:
    @pytest.fixture(scope="class")
    def is_even(self):
        return core.complement(core.odd__Q__)

    def test_evens_are_even(self, is_even, even_number):
        assert True is is_even(even_number)

    def test_odds_are_not_even(self, is_even, odd_number):
        assert False is is_even(odd_number)


def test_reduce():
    assert 0 == core.reduce(core.__PLUS__, [])
    assert 1 == core.reduce(core.__PLUS__, [1])
    assert 6 == core.reduce(core.__PLUS__, [1, 2, 3])
    assert 45 == core.reduce(core.__PLUS__, 45, [])
    assert 46 == core.reduce(core.__PLUS__, 45, [1])


def test_reduce_with_lazy_seq():
    assert 25 == core.reduce(
        core.__PLUS__, core.filter_(core.odd__Q__, vec.v(1, 2, 3, 4, 5, 6, 7, 8, 9))
    )
    assert 25 == core.reduce(
        core.__PLUS__, 0, core.filter_(core.odd__Q__, vec.v(1, 2, 3, 4, 5, 6, 7, 8, 9))
    )


def test_comp():
    assert 1 == core.comp()(1)
    assert "hi" == core.comp()("hi")
    assert True is core.comp(core.odd__Q__)(3)
    assert False is core.comp(core.odd__Q__)(2)
    assert 7 == core.comp(core.inc, core.inc, core.dec, lambda v: core.__STAR__(2, v))(
        3
    )


def test_juxt():
    assert vec.v(True) == core.juxt(core.odd__Q__)(3)
    assert vec.v(True, False, 4, 2) == core.juxt(
        core.odd__Q__, core.even__Q__, core.inc, core.dec
    )(3)


def test_partial():
    assert 3 == core.partial(core.__PLUS__)(3)
    assert 6 == core.partial(core.__PLUS__, 3)(3)
    assert 10 == core.partial(core.__PLUS__, 3, 4)(3)


class TestIsEvery:
    @pytest.mark.parametrize(
        "coll", [vec.Vector.empty(), vec.v(3), vec.v(3, 5, 7, 9, 11)]
    )
    def test_is_every(self, coll):
        assert True is core.every__Q__(core.odd__Q__, coll)

    @pytest.mark.parametrize(
        "coll",
        [vec.v(2), vec.v(3, 5, 7, 9, 2), vec.v(2, 3, 5, 7, 9), vec.v(3, 5, 2, 7, 9)],
    )
    def test_is_not_every(self, coll):
        assert False is core.every__Q__(core.odd__Q__, coll)


class TestIsNotEvery:
    @pytest.mark.parametrize(
        "coll", [vec.Vector.empty(), vec.v(3), vec.v(3, 5, 7, 9, 11)]
    )
    def test_is_not_every(self, coll):
        assert False is core.not_every__Q__(core.odd__Q__, coll)

    @pytest.mark.parametrize(
        "coll",
        [vec.v(2), vec.v(3, 5, 7, 9, 2), vec.v(2, 3, 5, 7, 9), vec.v(3, 5, 2, 7, 9)],
    )
    def test_not_is_not_every(self, coll):
        assert True is core.not_every__Q__(core.odd__Q__, coll)


class TestSome:
    @pytest.mark.parametrize(
        "coll",
        [
            vec.v(3),
            vec.v(3, 5, 7, 9, 11),
            vec.v(3, 5, 7, 9, 2),
            vec.v(2, 3, 5, 7, 9),
            vec.v(3, 5, 2, 7, 9),
        ],
    )
    def test_is_some(self, coll):
        assert True is core.some(core.odd__Q__, coll)

    @pytest.mark.parametrize(
        "coll", [vec.Vector.empty(), vec.v(2), vec.v(2, 4, 6, 8, 10)]
    )
    def test_is_not_some(self, coll):
        assert None is core.some(core.odd__Q__, coll)


class TestNotAny:
    @pytest.mark.parametrize(
        "coll",
        [
            vec.v(3),
            vec.v(3, 5, 7, 9, 11),
            vec.v(3, 5, 7, 9, 2),
            vec.v(2, 3, 5, 7, 9),
            vec.v(3, 5, 2, 7, 9),
        ],
    )
    def test_is_not_any(self, coll):
        assert False is core.not_any__Q__(core.odd__Q__, coll)

    @pytest.mark.parametrize(
        "coll", [vec.Vector.empty(), vec.v(2), vec.v(2, 4, 6, 8, 10)]
    )
    def test_not_is_not_any(self, coll):
        assert True is core.not_any__Q__(core.odd__Q__, coll)


def test_merge():
    assert None is core.merge()
    assert lmap.Map.empty() == core.merge(lmap.Map.empty())
    assert lmap.map({kw.keyword("a"): 1}) == core.merge(lmap.map({kw.keyword("a"): 1}))
    assert lmap.map({kw.keyword("a"): 53, kw.keyword("b"): "hi"}) == core.merge(
        lmap.map({kw.keyword("a"): 1, kw.keyword("b"): "hi"}),
        lmap.map({kw.keyword("a"): 53}),
    )


def test_map():
    assert llist.List.empty() == core.map_(core.identity, vec.Vector.empty())
    assert llist.l(1, 2, 3) == core.map_(core.identity, vec.v(1, 2, 3))
    assert llist.l(2, 3, 4) == core.map_(core.inc, vec.v(1, 2, 3))

    assert llist.l(5, 7, 9) == core.map_(core.__PLUS__, vec.v(1, 2, 3), vec.v(4, 5, 6))
    assert llist.l(5, 7, 9) == core.map_(
        core.__PLUS__, vec.v(1, 2, 3), core.range_(4, 7)
    )


def test_map_indexed():
    assert llist.l(vec.v(0, 1), vec.v(1, 2), vec.v(2, 3)) == core.map_indexed(
        core.vector, vec.v(1, 2, 3)
    )


def test_mapcat():
    assert llist.List.empty() == core.mapcat(
        lambda x: vec.v(x, x + 1), vec.Vector.empty()
    )
    assert llist.l(1, 2, 2, 3, 3, 4) == core.mapcat(
        lambda x: vec.v(x, x + 1), vec.v(1, 2, 3)
    )
    assert llist.l(1, 4, 2, 5, 3, 6) == core.mapcat(
        core.vector, vec.v(1, 2, 3), vec.v(4, 5, 6)
    )


def test_filter():
    assert llist.List.empty() == core.filter_(core.identity, vec.Vector.empty())
    assert llist.l(1, 3) == core.filter_(core.odd__Q__, vec.v(1, 2, 3, 4))
    assert llist.l(1, 2, 3, 4) == core.filter_(core.identity, vec.v(1, 2, 3, 4))


def test_remove():
    assert llist.List.empty() == core.remove(core.identity, vec.Vector.empty())
    assert llist.l(2, 4) == core.remove(core.odd__Q__, vec.v(1, 2, 3, 4))
    assert llist.List.empty() == core.remove(core.identity, vec.v(1, 2, 3, 4))


def test_take():
    assert llist.List.empty() == core.take(3, vec.Vector.empty())
    assert llist.l(1, 2, 3) == core.take(3, vec.v(1, 2, 3))
    assert llist.l(1, 2) == core.take(2, vec.v(1, 2, 3))
    assert llist.l(1) == core.take(1, vec.v(1, 2, 3))
    assert llist.List.empty() == core.take(0, vec.v(1, 2, 3))


def test_take_while():
    assert llist.List.empty() == core.take_while(core.odd__Q__, vec.Vector.empty())
    assert llist.List.empty() == core.take_while(core.even__Q__, vec.v(1, 3, 5, 7))
    assert llist.List.empty() == core.take_while(core.odd__Q__, vec.v(2, 3, 5, 7))
    assert llist.l(1, 3, 5) == core.take_while(core.odd__Q__, vec.v(1, 3, 5, 2))
    assert llist.l(1, 3, 5, 7) == core.take_while(core.odd__Q__, vec.v(1, 3, 5, 7))
    assert llist.l(1) == core.take_while(core.odd__Q__, vec.v(1, 2, 3, 4))


def test_take_nth():
    assert llist.List.empty() == core.take_nth(0, vec.Vector.empty())
    assert llist.l(1, 1, 1) == core.take(3, core.take_nth(0, vec.v(1)))
    assert llist.l(1, 1, 1) == core.take(3, core.take_nth(0, vec.v(1, 1, 1)))
    assert llist.l(1, 2, 3, 4, 5) == core.take_nth(1, vec.v(1, 2, 3, 4, 5))
    assert llist.l(1, 4) == core.take_nth(3, vec.v(1, 2, 3, 4, 5))


def test_drop():
    assert llist.List.empty() == core.drop(3, vec.Vector.empty())
    assert llist.List.empty() == core.drop(3, vec.v(1, 2, 3))
    assert llist.l(1, 2, 3) == core.drop(0, vec.v(1, 2, 3))
    assert llist.l(2, 3) == core.drop(1, vec.v(1, 2, 3))
    assert llist.l(3) == core.drop(2, vec.v(1, 2, 3))
    assert llist.l(4) == core.drop(3, vec.v(1, 2, 3, 4))


def test_drop_while():
    assert llist.List.empty() == core.drop_while(core.odd__Q__, vec.Vector.empty())
    assert llist.List.empty() == core.drop_while(core.odd__Q__, vec.v(1, 3, 5, 7))
    assert llist.l(2) == core.drop_while(core.odd__Q__, vec.v(1, 3, 5, 2))
    assert llist.l(2, 3, 4) == core.drop_while(core.odd__Q__, vec.v(1, 2, 3, 4))
    assert llist.l(2, 4, 6, 8) == core.drop_while(core.odd__Q__, vec.v(2, 4, 6, 8))


def test_drop_last():
    assert llist.l(1, 2, 3, 4) == core.drop_last(vec.v(1, 2, 3, 4, 5))
    assert llist.l(1, 2, 3) == core.drop_last(2, vec.v(1, 2, 3, 4, 5))
    assert llist.l(1, 2) == core.drop_last(3, vec.v(1, 2, 3, 4, 5))
    assert llist.l(1) == core.drop_last(4, vec.v(1, 2, 3, 4, 5))
    assert llist.List.empty() == core.drop_last(5, vec.v(1, 2, 3, 4, 5))
    assert llist.List.empty() == core.drop_last(6, vec.v(1, 2, 3, 4, 5))
    assert llist.l(1, 2, 3, 4, 5) == core.drop_last(0, vec.v(1, 2, 3, 4, 5))
    assert llist.l(1, 2, 3, 4, 5) == core.drop_last(-1, vec.v(1, 2, 3, 4, 5))


def test_split_at():
    assert vec.v(llist.List.empty(), llist.List.empty()) == core.split_at(
        3, vec.Vector.empty()
    )
    assert vec.v(llist.List.empty(), llist.l(1, 2, 3)) == core.split_at(
        0, vec.v(1, 2, 3)
    )
    assert vec.v(llist.l(1), llist.l(2, 3)) == core.split_at(1, vec.v(1, 2, 3))
    assert vec.v(llist.l(1, 2), llist.l(3)) == core.split_at(2, vec.v(1, 2, 3))
    assert vec.v(llist.l(1, 2, 3), llist.List.empty()) == core.split_at(
        3, vec.v(1, 2, 3)
    )
    assert vec.v(llist.l(1, 2, 3), llist.List.empty()) == core.split_at(
        4, vec.v(1, 2, 3)
    )
    assert vec.v(llist.l(1, 2, 3), llist.l(4)) == core.split_at(3, vec.v(1, 2, 3, 4))


def test_split_with():
    assert vec.v(llist.List.empty(), llist.List.empty()) == core.split_with(
        core.odd__Q__, vec.Vector.empty()
    )
    assert vec.v(llist.l(1), llist.l(2, 3)) == core.split_with(
        core.odd__Q__, vec.v(1, 2, 3)
    )
    assert vec.v(llist.l(1, 3, 5, 7), llist.List.empty()) == core.split_with(
        core.odd__Q__, vec.v(1, 3, 5, 7)
    )
    assert vec.v(llist.List.empty(), llist.l(2, 4, 6, 8)) == core.split_with(
        core.odd__Q__, vec.v(2, 4, 6, 8)
    )


def test_group_by():
    assert lmap.Map.empty() == core.group_by(core.inc, vec.Vector.empty())
    assert lmap.map({True: vec.v(1, 3), False: vec.v(2, 4)}) == core.group_by(
        core.odd__Q__, vec.v(1, 2, 3, 4)
    )


def test_interpose():
    assert llist.List.empty() == core.interpose(",", vec.Vector.empty())
    assert llist.l("hi") == core.interpose(",", vec.v("hi"))
    assert llist.l("hi", ",", "there") == core.interpose(",", vec.v("hi", "there"))


def test_cycle():
    assert llist.l(1, 1, 1) == core.take(3, core.cycle(vec.v(1)))
    assert llist.l(1, 2, 1) == core.take(3, core.cycle(vec.v(1, 2)))
    assert llist.l(1, 2, 3) == core.take(3, core.cycle(vec.v(1, 2, 3)))
    assert llist.l(1, 2, 3, 1, 2, 3) == core.take(6, core.cycle(vec.v(1, 2, 3)))


def test_repeat():
    assert llist.l(1, 1, 1) == core.take(3, core.repeat(1))
    assert llist.l(1, 1, 1, 1, 1, 1) == core.take(6, core.repeat(1))
    assert llist.l(1, 1, 1) == core.repeat(3, 1)


def test_repeatedly():
    assert llist.l("yes", "yes", "yes") == core.take(3, core.repeatedly(lambda: "yes"))
    assert llist.l("yes", "yes", "yes") == core.repeatedly(3, lambda: "yes")


def test_partition():
    assert llist.l(llist.l(1, 2), llist.l(3, 4), llist.l(5, 6)) == core.partition(
        2, core.range_(1, 7)
    )
    assert llist.l(llist.l(1, 2, 3), llist.l(4, 5, 6)) == core.partition(
        3, core.range_(1, 7)
    )

    assert llist.l(
        llist.l(1, 2, 3, 4, 5), llist.l(11, 12, 13, 14, 15), llist.l(21, 22, 23)
    ) == core.partition(5, 10, core.range_(1, 24))
    assert llist.l(
        llist.l(1, 2, 3, 4, 5), llist.l(11, 12, 13, 14, 15), llist.l(21, 22, 23, 24, 25)
    ) == core.partition(5, 10, core.range_(1, 26))

    assert llist.l(
        llist.l(1, 2, 3, 4, 5),
        llist.l(11, 12, 13, 14, 15),
        llist.l(21, 22, 23, kw.keyword("a"), kw.keyword("a")),
    ) == core.partition(5, 10, core.repeat(kw.keyword("a")), core.range_(1, 24))
    assert llist.l(
        llist.l(1, 2, 3, 4, 5), llist.l(11, 12, 13, 14, 15), llist.l(21, 22, 23, 24, 25)
    ) == core.partition(5, 10, core.repeat(kw.keyword("a")), core.range_(1, 26))


class TestPrintFunctions:
    def test_pr_str(self):
        assert "" == core.pr_str()
        assert '""' == core.pr_str("")
        assert ":kw" == core.pr_str(kw.keyword("kw"))
        assert ':hi "there" 3' == core.pr_str(kw.keyword("hi"), "there", 3)

    def test_prn_str(self):
        assert "\n" == core.prn_str()
        assert '""\n' == core.prn_str("")
        assert ":kw\n" == core.prn_str(kw.keyword("kw"))
        assert ':hi "there" 3\n' == core.prn_str(kw.keyword("hi"), "there", 3)

    def test_print_str(self):
        assert "" == core.print_str()
        assert "" == core.print_str("")
        assert ":kw" == core.print_str(kw.keyword("kw"))
        assert ":hi there 3" == core.print_str(kw.keyword("hi"), "there", 3)

    def test_println_str(self):
        assert "\n" == core.println_str()
        assert "\n" == core.println_str("")
        assert ":kw\n" == core.println_str(kw.keyword("kw"))
        assert ":hi there 3\n" == core.println_str(kw.keyword("hi"), "there", 3)


class TestRegexFunctions:
    def test_re_find(self):
        assert None is core.re_find(re.compile(r"\d+"), "abcdef")
        assert "12345" == core.re_find(re.compile(r"\d+"), "abc12345def")
        assert vec.v("word then number ", "word then number ", None) == core.re_find(
            re.compile(r"(\D+)|(\d+)"), "word then number 57"
        )
        assert vec.v("57", None, "57") == core.re_find(
            re.compile(r"(\D+)|(\d+)"), "57 number then word"
        )
        assert vec.v("lots", "", "l") == core.re_find(
            re.compile(r"(\d*)(\S)\S+"), "lots o' digits 123456789"
        )

    def test_re_matches(self):
        assert None is core.re_matches(re.compile(r"hello"), "hello, world")
        assert "hello, world" == core.re_matches(re.compile(r"hello.*"), "hello, world")
        assert vec.v("hello, world", "world") == core.re_matches(
            re.compile(r"hello, (.*)"), "hello, world"
        )

    def test_re_seq(self):
        assert None is core.seq(core.re_seq(re.compile(r"[a-zA-Z]+"), "134325235234"))
        assert llist.l("1", "1", "0") == core.re_seq(
            re.compile(r"\d+"), "Basilisp 1.1.0"
        )
        assert llist.l("the", "man", "who", "sold", "the", "world") == core.re_seq(
            re.compile(r"\w+"), "the man who sold the world"
        )
