import itertools
import os
import re
from decimal import Decimal
from fractions import Fraction
from unittest.mock import Mock
from uuid import UUID

import pytest

from basilisp.lang import keyword as kw
from basilisp.lang import list as llist
from basilisp.lang import map as lmap
from basilisp.lang import runtime as runtime
from basilisp.lang import set as lset
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec
from basilisp.lang.exception import ExceptionInfo
from basilisp.lang.interfaces import IExceptionInfo


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """Disable the `print_generated_python` flag so we can safely capture
    stderr and stdout for tests which require those facilities."""
    orig = runtime.print_generated_python
    runtime.print_generated_python = Mock(return_value=False)
    yield
    runtime.print_generated_python = orig


import basilisp.core as core  # isort:skip

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
    llist.EMPTY,
    llist.l(0),
    llist.l(False),
    llist.l(True),
    lmap.EMPTY,
    lmap.map({0: 0}),
    lmap.map({False: False}),
    lmap.map({True: True}),
    lset.EMPTY,
    lset.s(0),
    lset.s(False),
    lset.s(True),
    vec.EMPTY,
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
    llist.EMPTY,
    llist.l(0),
    llist.l(False),
    llist.l(True),
    lmap.EMPTY,
    lmap.map({0: 0}),
    lmap.map({False: False}),
    lmap.map({True: True}),
    lset.EMPTY,
    lset.s(0),
    lset.s(False),
    lset.s(True),
    vec.EMPTY,
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


class TestIsAny:
    def test_any_always_true(self, lisp_value):
        assert True is core.any__Q__(lisp_value)


class TestIsAssociative:
    @pytest.mark.parametrize("v", [lmap.EMPTY, vec.EMPTY])
    def test_is_associative(self, v):
        assert True is core.associative__Q__(v)

    @pytest.mark.parametrize("v", [llist.EMPTY, lset.EMPTY])
    def test_is_not_associative(self, v):
        assert False is core.associative__Q__(v)


class TestIsClass:
    @pytest.mark.parametrize(
        "tp",
        [
            kw.Keyword,
            llist.PersistentList,
            lmap.PersistentMap,
            sym.Symbol,
            vec.PersistentVector,
        ],
    )
    def test_is_class(self, tp):
        assert True is core.class__Q__(tp)

    @pytest.mark.parametrize("tp", [kw.keyword("a"), 1, "string", sym.symbol("sym")])
    def test_is_not_class(self, tp):
        assert False is core.class__Q__(tp)


class TestIsColl:
    @pytest.mark.parametrize(
        "v",
        [
            llist.EMPTY,
            lmap.EMPTY,
            lset.EMPTY,
            vec.EMPTY,
        ],
    )
    def test_is_coll(self, v):
        assert True is core.coll__Q__(v)

    @pytest.mark.parametrize("v", [kw.keyword("a"), 1, "string", sym.symbol("sym")])
    def test_is_not_coll(self, v):
        assert False is core.coll__Q__(v)


class TestIsFn:
    @pytest.fixture(scope="class")
    def basilisp_fn(self):
        @runtime._basilisp_fn(arities=(1,))
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
            lmap.EMPTY,
            lset.EMPTY,
            sym.symbol("a"),
            vec.EMPTY,
        ],
    )
    def test_is_not_fn(self, v):
        assert False is core.fn__Q__(v)

    def test_function_is_ifn(self, basilisp_fn, py_fn):
        assert True is core.ifn__Q__(basilisp_fn)
        assert True is core.ifn__Q__(py_fn)

    @pytest.mark.parametrize("v", [kw.keyword("a"), lmap.EMPTY, lset.EMPTY])
    def test_other_is_ifn(self, v):
        assert True is core.ifn__Q__(v)

    @pytest.mark.parametrize("v", ["a", 1, 1.6])
    def test_is_not_ifn(self, v):
        assert False is core.ifn__Q__(v)


class TestIsIdent:
    @pytest.mark.parametrize(
        "v",
        [
            kw.keyword("kw"),
            sym.symbol("sym"),
            kw.keyword("kw", ns="ns"),
            sym.symbol("sym", ns="ns"),
            kw.keyword("kw", ns="qualified.ns"),
            sym.symbol("sym", ns="qualified.ns"),
        ],
    )
    def test_is_ident(self, v):
        assert True is core.ident__Q__(v)

    @pytest.mark.parametrize(
        "v",
        [
            kw.keyword("kw", ns="ns"),
            kw.keyword("kw", ns="qualified.ns"),
            sym.symbol("sym", ns="ns"),
            sym.symbol("sym", ns="qualified.ns"),
        ],
    )
    def test_is_qualified_ident(self, v):
        assert True is core.qualified_ident__Q__(v)

    @pytest.mark.parametrize("v", [kw.keyword("kw"), sym.symbol("sym")])
    def test_is_not_qualified_ident(self, v):
        assert False is core.qualified_ident__Q__(v)

    @pytest.mark.parametrize(
        "v",
        [
            kw.keyword("kw", ns="ns"),
            kw.keyword("kw", ns="qualified.ns"),
            sym.symbol("sym", ns="ns"),
            sym.symbol("sym", ns="qualified.ns"),
        ],
    )
    def test_is_simple_ident(self, v):
        assert False is core.simple_ident__Q__(v)

    @pytest.mark.parametrize("v", [kw.keyword("kw"), sym.symbol("sym")])
    def test_is_not_simple_ident(self, v):
        assert True is core.simple_ident__Q__(v)

    @pytest.mark.parametrize("v", [kw.keyword("kw"), kw.keyword("kw", ns="ns")])
    def test_is_keyword(self, v):
        assert True is core.keyword__Q__(v)

    @pytest.mark.parametrize(
        "v", [kw.keyword("kw", ns="ns"), kw.keyword("kw", ns="qualified.ns")]
    )
    def test_is_qualified_keyword(self, v):
        assert True is core.qualified_keyword__Q__(v)

    @pytest.mark.parametrize("v", [sym.symbol("sym"), sym.symbol("sym", ns="ns")])
    def test_is_symbol(self, v):
        assert True is core.symbol__Q__(v)

    @pytest.mark.parametrize(
        "v", [sym.symbol("sym", ns="ns"), sym.symbol("sym", ns="qualified.ns")]
    )
    def test_is_qualified_symbol(self, v):
        assert True is core.qualified_symbol__Q__(v)


class TestNumericPredicates:
    def test_is_complex(self, complex_number):
        assert True is core.complex__Q__(complex_number)

    def test_real_is_not_complex(self, real_number):
        assert False is core.complex__Q__(real_number)

    def test_fraction_is_not_complex(self, fraction):
        assert False is core.complex__Q__(fraction)

    def test_decimal_is_not_complex(self, decimal):
        assert False is core.complex__Q__(decimal)

    def test_is_decimal(self, decimal):
        assert True is core.decimal__Q__(decimal)

    def test_real_is_not_decimal(self, real_number):
        assert False is core.decimal__Q__(real_number)

    def test_fraction_is_not_decimal(self, fraction):
        assert False is core.decimal__Q__(fraction)

    def test_complex_is_not_decimal(self, complex_number):
        assert False is core.decimal__Q__(complex_number)

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

    def test_even_nums_are_even(self, even_int):
        assert True is core.even__Q__(even_int)

    def test_even_pred_throws_on_floats(self, even_float):
        with pytest.raises(TypeError):
            core.even__Q__(even_float)

    def test_odd_nums_are_not_even(self, odd_int):
        assert False is core.even__Q__(odd_int)

    def test_is_int(self, int_number):
        assert True is core.int__Q__(int_number)

    def test_decimal_is_not_int(self, decimal):
        assert False is core.int__Q__(decimal)

    def test_double_is_not_int(self, float_number):
        assert False is core.int__Q__(float_number)

    def test_fraction_is_not_int(self, fraction):
        assert False is core.int__Q__(fraction)

    def test_complex_is_not_int(self, complex_number):
        assert False is core.int__Q__(complex_number)

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
        assert False is core.neg__Q__(v)

    @pytest.mark.parametrize("v", [-1, -100])
    def test_is_neg_int(self, v):
        assert True is core.neg_int__Q__(v)

    @pytest.mark.parametrize(
        "v", [0, 1, 100, 1.0, 9_999_839.874_394, -1.0, -9_999_839.874_394]
    )
    def test_is_not_neg_int(self, v):
        assert False is core.neg_int__Q__(v)

    @pytest.mark.parametrize("v", [0, 1, 100])
    def test_is_nat_int(self, v):
        assert True is core.nat_int__Q__(v)

    @pytest.mark.parametrize(
        "v", [-1, -100, -1.0, -9_999_839.874_394, 4.6, 3.14, 0.111]
    )
    def test_is_not_nat_int(self, v):
        assert False is core.nat_int__Q__(v)

    def test_is_real_number(self, real_number):
        assert True is core.real_number__Q__(real_number)

    def test_is_not_real_number(self, complex_number):
        assert False is core.real_number__Q__(complex_number)

    def test_is_fraction(self, fraction):
        assert True is core.ratio__Q__(fraction)

    def test_decimal_is_not_fraction(self, decimal):
        assert False is core.ratio__Q__(decimal)

    def test_double_is_not_fraction(self, float_number):
        assert False is core.ratio__Q__(float_number)

    def test_int_is_not_fraction(self, int_number):
        assert False is core.ratio__Q__(int_number)

    def test_complex_is_not_fraction(self, complex_number):
        assert False is core.ratio__Q__(complex_number)

    def test_odd_nums_are_odd(self, odd_int):
        assert True is core.odd__Q__(odd_int)

    def test_even_nums_are_not_odd(self, even_int):
        assert False is core.odd__Q__(even_int)

    def test_odd_pred_throws_on_floats(self, odd_float):
        with pytest.raises(TypeError):
            core.odd__Q__(odd_float)

    def test_fraction_is_rational(self, fraction):
        assert True is core.rational__Q__(fraction)

    def test_decimal_is_rational(self, decimal):
        assert True is core.rational__Q__(decimal)

    def test_int_is_rational(self, int_number):
        assert True is core.rational__Q__(int_number)

    def test_double_is_not_rational(self, float_number):
        assert False is core.rational__Q__(float_number)

    def test_complex_is_not_rational(self, complex_number):
        assert False is core.rational__Q__(complex_number)

    @pytest.mark.parametrize("v", [float("inf"), float("-inf")])
    def test_is_infinite(self, v):
        assert True is core.infinite__Q__(v)

    def test_is_not_infinite(self, real_number):
        assert False is core.infinite__Q__(real_number)

    def test_is_nan(self):
        assert True is core.NaN__Q__(float("nan"))

    def test_is_not_nan(self, real_number):
        assert False is core.infinite__Q__(real_number)


class TestIsPy:
    @pytest.mark.parametrize("v", [{}, {"a": "b"}])
    def test_is_py_dict(self, v):
        assert True is core.py_dict__Q__(v)

    @pytest.mark.parametrize("v", [lmap.EMPTY, lmap.map({"a": "b"})])
    def test_is_not_py_dict(self, v):
        assert False is core.py_dict__Q__(v)

    @pytest.mark.parametrize("v", [frozenset(), frozenset(["a", "b"])])
    def test_is_py_frozenset(self, v):
        assert True is core.py_frozenset__Q__(v)

    @pytest.mark.parametrize("v", [lset.EMPTY, lset.s("a", "b")])
    def test_is_not_py_frozenset(self, v):
        assert False is core.py_frozenset__Q__(v)

    @pytest.mark.parametrize("v", [[], ["a", "b"]])
    def test_is_py_list(self, v):
        assert True is core.py_list__Q__(v)

    @pytest.mark.parametrize("v", [vec.EMPTY, vec.v("a", "b")])
    def test_is_not_py_list(self, v):
        assert False is core.py_list__Q__(v)

    @pytest.mark.parametrize("v", [set(), {"a", "b"}])
    def test_is_py_set(self, v):
        assert True is core.py_set__Q__(v)

    @pytest.mark.parametrize("v", [lset.EMPTY, lset.s("a", "b")])
    def test_is_not_py_set(self, v):
        assert False is core.py_set__Q__(v)

    @pytest.mark.parametrize("v", [(), ("a", "b")])
    def test_is_py_tuple(self, v):
        assert True is core.py_tuple__Q__(v)

    @pytest.mark.parametrize("v", [llist.EMPTY, llist.l("a", "b")])
    def test_is_not_py_tuple(self, v):
        assert False is core.py_tuple__Q__(v)


class TestIsSome:
    def test_nil_values_are_not_some(self, nil_value):
        assert False is core.some__Q__(nil_value)

    def test_non_nil_values_are_some(self, non_nil_value):
        assert True is core.some__Q__(non_nil_value)


class TestIsUUID:
    def test_is_uuid(self):
        assert True is core.uuid__Q__(UUID("1a937d1b-6d58-4a4b-9b61-64b1bf488125"))

    @pytest.mark.parametrize(
        "v",
        [
            "1a937d1b-6d58-4a4b-9b61-64b1bf488125",
            226_621_546_944_545_983_927_518_395_183_087_914_867,
            b"\xb7\x1a\xb0\xafk\xbcDS\xa3\xc7\x85\x17\xa4b\xe1\xeb",
            (1_939_259_628, 18526, 17139, 160, 63, 61_716_288_539_780),
            vec.v(1_939_259_628, 18526, 17139, 160, 63, 61_716_288_539_780),
        ],
    )
    def test_is_not_uuid(self, v):
        assert False is core.uuid__Q__(v)

    @pytest.mark.parametrize(
        "v",
        [
            UUID("1a937d1b-6d58-4a4b-9b61-64b1bf488125"),
            "1a937d1b-6d58-4a4b-9b61-64b1bf488125",
            226_621_546_944_545_983_927_518_395_183_087_914_867,
            b"\xb7\x1a\xb0\xafk\xbcDS\xa3\xc7\x85\x17\xa4b\xe1\xeb",
            (1_939_259_628, 18526, 17139, 160, 63, 61_716_288_539_780),
            # vec.v(1_939_259_628, 18526, 17139, 160, 63, 61_716_288_539_780),
        ],
    )
    def test_is_uuid_like(self, v):
        assert True is core.uuid_like__Q__(v)

    @pytest.mark.parametrize(
        "v",
        [
            "1a91b-6d58-4a-961-64b1bf488125",
            226_621_546_944_545_983_927_518_395_867,
            b"\xb7\x1a\xb0\xafk\xbcDS\x85\x17\xa4b\xe1\xeb",
            (1_939_259_628, 18526, 17139, 160, 61_716_288_539_780),
            vec.v(1_939_259_628, 18_526_160, 63, 61_716_288_539_780),
        ],
    )
    def test_is_not_uuid_like(self, v):
        assert False is core.uuid_like__Q__(lisp_value)


class TestExceptionData:
    def test_ex_cause_for_non_exception(self):
        assert None is core.ex_cause("a string")

    def test_ex_has_no_cause(self):
        assert None is core.ex_cause(Exception())

    def test_ex_has_cause(self):
        try:
            raise ExceptionInfo("Exception Message", lmap.map({"a": "b"}))
        except Exception as cause:
            try:
                raise Exception("Hi") from cause
            except Exception as outer:
                inner = core.ex_cause(outer)
                assert inner is cause
                assert isinstance(inner, IExceptionInfo)
                assert "Exception Message" == inner.message
                assert lmap.map({"a": "b"}) == inner.data

    def test_ex_has_contextual_cause(self):
        try:
            raise ExceptionInfo("Exception Message", lmap.map({"a": "b"}))
        except Exception as cause:
            try:
                raise Exception("Hi")
            except Exception as outer:
                inner = core.ex_cause(outer)
                assert inner is cause
                assert isinstance(inner, IExceptionInfo)
                assert "Exception Message" == inner.message
                assert lmap.map({"a": "b"}) == inner.data

    def test_ex_data_for_non_exception(self):
        assert None is core.ex_data("a string")

    def test_ex_data_iexceptioninfo(self):
        try:
            raise ExceptionInfo("Exception Message", lmap.map({"a": "b"}))
        except IExceptionInfo as e:
            assert lmap.map({"a": "b"}) == core.ex_data(e)

    def test_ex_data_standard_exception(self):
        try:
            raise Exception("Exception Message")
        except Exception as e:
            assert None is core.ex_data(e)

    def test_ex_message_for_non_exception(self):
        assert None is core.ex_message("a string")

    def test_ex_message_iexceptioninfo(self):
        try:
            raise ExceptionInfo("Exception Message", lmap.map({"a": "b"}))
        except IExceptionInfo as e:
            assert "Exception Message" == core.ex_message(e)

    def test_ex_message_standard_exception(self):
        try:
            raise Exception("Exception Message")
        except Exception as e:
            assert "Exception Message" == core.ex_message(e)


class TestBitManipulation:
    def test_bit_and(self):
        assert 8 == core.bit_and(12, 9)
        assert 195 == core.bit_and(235, 199)

    def test_bit_or(self):
        assert 13 == core.bit_or(12, 9)
        assert 239 == core.bit_or(235, 199)

    def test_bit_not(self):
        assert -13 == core.bit_not(12)
        assert -236 == core.bit_not(235)

    def test_bit_shift_left(self):
        assert 1024 == core.bit_shift_left(1, 10)
        assert 360 == core.bit_shift_left(45, 3)

    def test_bit_shift_right(self):
        assert 1 == core.bit_shift_right(1024, 10)
        assert 5 == core.bit_shift_right(45, 3)

    def test_bit_xor(self):
        assert 5 == core.bit_xor(12, 9)
        assert 44 == core.bit_xor(235, 199)

    def test_bit_clear(self):
        assert 3 == core.bit_clear(11, 3)
        assert 0 == core.bit_clear(1024, 10)

    def test_bit_flip(self):
        assert 11 == core.bit_flip(15, 2)
        assert 1025 == core.bit_flip(1024, 0)

    def test_bit_set(self):
        assert 15 == core.bit_set(11, 2)
        assert 9_223_372_036_854_775_808 == core.bit_set(0, 63)

    def test_bit_test(self):
        assert core.bit_test(9, 0)
        assert not core.bit_test(9, 1)
        assert not core.bit_test(9, 7)


class TestAssociativeFunctions:
    def test_contains(self):
        assert False is core.contains__Q__(None, "a")
        assert True is core.contains__Q__(lmap.map({"a": 1}), "a")
        assert False is core.contains__Q__(lmap.map({"a": 1}), "b")
        assert True is core.contains__Q__(lmap.map({"a": 1}).to_transient(), "a")
        assert False is core.contains__Q__(lmap.map({"a": 1}).to_transient(), "b")
        assert True is core.contains__Q__(vec.v(1, 2, 3), 0)
        assert True is core.contains__Q__(vec.v(1, 2, 3), 1)
        assert True is core.contains__Q__(vec.v(1, 2, 3), 2)
        assert False is core.contains__Q__(vec.v(1, 2, 3), 3)
        assert False is core.contains__Q__(vec.v(1, 2, 3), -1)
        assert True is core.contains__Q__(vec.v(1, 2, 3).to_transient(), 0)
        assert True is core.contains__Q__(vec.v(1, 2, 3).to_transient(), 1)
        assert True is core.contains__Q__(vec.v(1, 2, 3).to_transient(), 2)
        assert False is core.contains__Q__(vec.v(1, 2, 3).to_transient(), 3)
        assert False is core.contains__Q__(vec.v(1, 2, 3).to_transient(), -1)

        assert False is core.contains__Q__("", "a")
        assert False is core.contains__Q__("", 0)
        assert False is core.contains__Q__("", 1)
        assert True is core.contains__Q__("a", "a")
        assert True is core.contains__Q__("a", 0)
        assert False is core.contains__Q__("a", 1)
        assert True is core.contains__Q__("abc", "a")
        assert True is core.contains__Q__("abc", 0)
        assert True is core.contains__Q__("abc", 1)
        assert False is core.contains__Q__("abc", 4)

        with pytest.raises(TypeError):
            core.contains__Q__("abc", None)

    def test_disj(self):
        assert None is core.disj(None)
        assert None is core.disj(None, "a")
        assert None is core.disj(None, "a", "b", "c")
        assert lset.EMPTY == core.disj(lset.EMPTY, "a")
        assert lset.EMPTY == core.disj(lset.s("a"), "a")
        assert lset.s("b", "d") == core.disj(lset.s("a", "b", "c", "d"), "a", "c", "e")

    def test_dissoc(self):
        assert None is core.dissoc(None)
        assert None is core.dissoc(None, "a")
        assert None is core.dissoc(None, "a", "b", "c")
        assert lmap.EMPTY == core.dissoc(lmap.map({"a": 1}), "a", "c")
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

    def test_find(self):
        assert None is core.find(None, "a")
        assert core.map_entry("a", 1) == core.find(lmap.map({"a": 1}), "a")
        assert None is core.find(lmap.map({"a": 1}), "b")
        assert core.map_entry(0, 1) == core.find(vec.v(1, 2, 3), 0)
        assert core.map_entry(1, 2) == core.find(vec.v(1, 2, 3), 1)
        assert core.map_entry(2, 3) == core.find(vec.v(1, 2, 3), 2)
        assert None is core.find(vec.v(1, 2, 3), 3)

    def test_assoc_in(self):
        assert lmap.map({"a": 1}) == core.assoc_in(None, vec.v("a"), 1)
        assert lmap.map({"a": 8}) == core.assoc_in(lmap.map({"a": 1}), vec.v("a"), 8)
        assert lmap.map({"a": 1, "b": "string"}) == core.assoc_in(
            lmap.map({"a": 1}), vec.v("b"), "string"
        )

        assert lmap.map({"a": lmap.map({"b": 3})}) == core.assoc_in(
            lmap.map({"a": lmap.map({"b": lmap.map({"c": 3})})}), vec.v("a", "b"), 3
        )
        assert lmap.map({"a": lmap.map({"b": lmap.map({"c": 4})})}) == core.assoc_in(
            lmap.map({"a": lmap.map({"b": lmap.map({"c": 3})})}),
            vec.v("a", "b", "c"),
            4,
        )
        assert lmap.map(
            {"a": lmap.map({"b": lmap.map({"c": 3, "f": 6})})}
        ) == core.assoc_in(
            lmap.map({"a": lmap.map({"b": lmap.map({"c": 3})})}),
            vec.v("a", "b", "f"),
            6,
        )
        assert lmap.map(
            {"a": lmap.map({"b": lmap.map({"c": 3}), "e": lmap.map({"f": 6})})}
        ) == core.assoc_in(
            lmap.map({"a": lmap.map({"b": lmap.map({"c": 3})})}),
            vec.v("a", "e"),
            lmap.map({"f": 6}),
        )

        assert vec.v("a") == core.assoc_in(vec.EMPTY, vec.v(0), "a")
        assert vec.v("c", "b") == core.assoc_in(vec.v("a", "b"), vec.v(0), "c")
        assert vec.v("a", "c") == core.assoc_in(vec.v("a", "b"), vec.v(1), "c")
        assert vec.v("a", "d", "c") == core.assoc_in(
            vec.v("a", "b", "c"), vec.v(1), "d"
        )

        assert vec.v("a", vec.v("q", "r", "s", "w"), "c") == core.assoc_in(
            vec.v("a", vec.v("q", "r", "s", "t"), "c"), vec.v(1, 3), "w"
        )
        assert vec.v(
            "a", vec.v("q", "r", "s", lmap.map({"w": "y"})), "c"
        ) == core.assoc_in(
            vec.v("a", vec.v("q", "r", "s", lmap.map({"w": "x"})), "c"),
            vec.v(1, 3, "w"),
            "y",
        )
        assert vec.v(
            "a", vec.v("q", "r", "s", lmap.map({"w": "x", "v": "u"})), "c"
        ) == core.assoc_in(
            vec.v("a", vec.v("q", "r", "s", lmap.map({"w": "x"})), "c"),
            vec.v(1, 3, "v"),
            "u",
        )

    def test_get_in(self):
        assert None is core.get_in(None, vec.v())
        assert lmap.map({"a": 1}) == core.get_in(lmap.map({"a": 1}), None)
        assert lmap.map({"a": 1}) == core.get_in(lmap.map({"a": 1}), vec.EMPTY)

        assert "default" == core.get_in(lmap.EMPTY, vec.v(None), "default")

        assert 1 == core.get_in(lmap.map({"a": 1}), vec.v("a"))
        assert None is core.get_in(lmap.map({"a": 1}), vec.v("b"))
        assert 2 == core.get_in(lmap.map({"a": 1}), vec.v("b"), 2)

        assert lmap.map({"b": lmap.map({"c": 3})}) == core.get_in(
            lmap.map({"a": lmap.map({"b": lmap.map({"c": 3})})}), vec.v("a")
        )
        assert lmap.map({"c": 3}) == core.get_in(
            lmap.map({"a": lmap.map({"b": lmap.map({"c": 3})})}), vec.v("a", "b")
        )
        assert 3 == core.get_in(
            lmap.map({"a": lmap.map({"b": lmap.map({"c": 3})})}), vec.v("a", "b", "c")
        )
        assert None is core.get_in(
            lmap.map({"a": lmap.map({"b": lmap.map({"c": 3})})}), vec.v("a", "b", "f")
        )
        assert None is core.get_in(
            lmap.map({"a": lmap.map({"b": lmap.map({"c": 3})})}), vec.v("a", "e", "c")
        )
        assert "Not Found" == core.get_in(
            lmap.map({"a": lmap.map({"b": lmap.map({"c": 3})})}),
            vec.v("a", "b", "f"),
            "Not Found",
        )
        assert "Not Found" == core.get_in(
            lmap.map({"a": lmap.map({"b": lmap.map({"c": 3})})}),
            vec.v("a", "e", "c"),
            "Not Found",
        )

        assert "b" == core.get_in(vec.v("a", "b", "c"), vec.v(1))
        assert "t" == core.get_in(
            vec.v("a", vec.v("q", "r", "s", "t"), "c"), vec.v(1, 3)
        )
        assert "x" == core.get_in(
            vec.v("a", vec.v("q", "r", "s", lmap.map({"w": "x"})), "c"),
            vec.v(1, 3, "w"),
        )
        assert None is core.get_in(
            vec.v("a", vec.v("q", "r", "s", lmap.map({"w": "x"})), "c"),
            vec.v(1, 3, "v"),
        )

    def test_update_in(self):
        assert lmap.map({"a": 2}) == core.update_in(
            lmap.map({"a": 1}), vec.v("a"), core.inc
        )
        assert lmap.map({"a": 1, "b": lmap.map({"c": 3})}) == core.update_in(
            lmap.map({"a": 1}), vec.v("b"), core.assoc, "c", 3
        )

        assert lmap.map({"a": lmap.map({"b": lmap.map({"c": 2})})}) == core.update_in(
            lmap.map({"a": lmap.map({"b": lmap.map({"c": 3})})}),
            vec.v("a", "b", "c"),
            core.dec,
        )
        assert lmap.map({"a": lmap.map({"b": lmap.map({"c": 3})})}) == core.update_in(
            lmap.map({"a": lmap.map({"b": lmap.map({"c": 3, "f": 6})})}),
            vec.v("a", "b"),
            core.dissoc,
            "f",
        )
        assert lmap.map(
            {"a": lmap.map({"b": lmap.map({"c": 3}), "e": lmap.map({"f": 6})})}
        ) == core.update_in(
            lmap.map({"a": lmap.map({"b": lmap.map({"c": 3})})}),
            vec.v("a"),
            core.assoc,
            "e",
            lmap.map({"f": 6}),
        )

        assert vec.v(0, 2) == core.update_in(vec.v(1, 2), vec.v(0), core.dec)
        assert vec.v(1, 3) == core.update_in(vec.v(1, 2), vec.v(1), core.inc)
        assert vec.v("a", "B", "c") == core.update_in(
            vec.v("a", "b", "c"), vec.v(1), lambda s: s.upper()
        )

        assert vec.v("a", vec.v("q", "r", "s", "T"), "c") == core.update_in(
            vec.v("a", vec.v("q", "r", "s", "t"), "c"), vec.v(1, 3), lambda s: s.upper()
        )
        assert vec.v(
            "a", vec.v("q", "r", "s", lmap.map({"w": "X"})), "c"
        ) == core.update_in(
            vec.v("a", vec.v("q", "r", "s", lmap.map({"w": "x"})), "c"),
            vec.v(1, 3, "w"),
            lambda s: s.upper(),
        )
        assert vec.v(
            "a",
            vec.v("q", "r", "s", lmap.map({"w": "x", "v": lmap.map({"t": "u"})})),
            "c",
        ) == core.update_in(
            vec.v("a", vec.v("q", "r", "s", lmap.map({"w": "x"})), "c"),
            vec.v(1, 3, "v"),
            core.assoc,
            "t",
            "u",
        )

    def test_select_keys(self):
        assert lmap.EMPTY == core.select_keys(lmap.EMPTY, vec.EMPTY)
        assert lmap.EMPTY == core.select_keys(
            lmap.EMPTY, vec.v(kw.keyword("a"), kw.keyword("b"))
        )
        assert lmap.map(
            {kw.keyword("a"): "a", kw.keyword("b"): "b"}
        ) == core.select_keys(
            lmap.map(
                {kw.keyword("a"): "a", kw.keyword("b"): "b", kw.keyword("c"): "c"}
            ),
            vec.v(kw.keyword("a"), kw.keyword("b")),
        )


def test_sort():
    assert llist.l(1) == core.sort(vec.v(1))
    assert llist.l(1, 2, 3) == core.sort(vec.v(1, 2, 3))
    assert llist.l(1, 2, 3, 4, 5) == core.sort(vec.v(5, 3, 1, 2, 4))


def test_is_map_entry():
    assert True is core.map_entry__Q__(lmap.MapEntry.of("a", "b"))
    assert False is core.map_entry__Q__(vec.EMPTY)
    assert False is core.map_entry__Q__(vec.v("a", "b"))
    assert False is core.map_entry__Q__(vec.v("a", "b", "c"))


def test_range():
    assert llist.EMPTY == core.range_(1, 1)
    assert llist.l(1, 2, 3, 4, 5) == core.range_(1, 6)
    assert llist.l(1, 3, 5, 7, 9) == core.range_(1, 11, 2)
    assert llist.l(1, -1, -3, -5, -7, -9) == core.range_(1, -10, -2)
    assert 9999 == len(core.vec(core.range_(1, 10000)))


def test_partial_kw():
    assert {"value": 3} == core.partial_kw(dict)(value=3)
    assert {"value": 82} == core.partial_kw(dict, lmap.map({kw.keyword("value"): 3}))(
        value=82
    )
    assert {
        "value": 82,
        "other-value": "some string",
        "other_value": "a string",
    } == core.partial_kw(
        dict, lmap.map({kw.keyword("value"): 3, "other-value": "some string"})
    )(
        value=82, other_value="a string"
    )
    assert {"value": 82, "other_value": "a string"} == core.partial_kw(
        dict, kw.keyword("value"), 3, kw.keyword("other-value"), "some string"
    )(value=82, other_value="a string")


class TestIsEvery:
    @pytest.mark.parametrize("coll", [vec.EMPTY, vec.v(3), vec.v(3, 5, 7, 9, 11)])
    def test_is_every(self, coll):
        assert True is core.every__Q__(core.odd__Q__, coll)

    @pytest.mark.parametrize(
        "coll",
        [vec.v(2), vec.v(3, 5, 7, 9, 2), vec.v(2, 3, 5, 7, 9), vec.v(3, 5, 2, 7, 9)],
    )
    def test_is_not_every(self, coll):
        assert False is core.every__Q__(core.odd__Q__, coll)


class TestIsNotEvery:
    @pytest.mark.parametrize("coll", [vec.EMPTY, vec.v(3), vec.v(3, 5, 7, 9, 11)])
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

    @pytest.mark.parametrize("coll", [vec.EMPTY, vec.v(2), vec.v(2, 4, 6, 8, 10)])
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

    @pytest.mark.parametrize("coll", [vec.EMPTY, vec.v(2), vec.v(2, 4, 6, 8, 10)])
    def test_not_is_not_any(self, coll):
        assert True is core.not_any__Q__(core.odd__Q__, coll)
