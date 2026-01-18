import decimal
import fractions
import functools
import math
from fractions import Fraction
from typing import Callable, TypeVar

from basilisp.lang.typing import LispNumber

T_num = TypeVar("T_num", bound=LispNumber)


def _normalize_fraction_result(
    f: Callable[[T_num, LispNumber], LispNumber],
) -> Callable[[T_num, LispNumber], LispNumber]:
    """
    Decorator for arithmetic operations to simplify `fractions.Fraction` values with
    a denominator of 1 to an integer.
    """

    @functools.wraps(f)
    def _normalize(x: T_num, y: LispNumber) -> LispNumber:
        result = f(x, y)
        # fractions.Fraction.is_integer() wasn't added until 3.12
        return (
            result.numerator
            if isinstance(result, fractions.Fraction) and result.denominator == 1
            else result
        )

    return _normalize


def _to_decimal(x: LispNumber) -> decimal.Decimal:
    """Coerce the input Lisp number to a `decimal.Decimal`.

    `fractions.Fraction` types are not accepted as direct inputs, so this is a utility
    to simplify that coercion.."""
    if isinstance(x, Fraction):
        numerator, denominator = x.as_integer_ratio()
        return decimal.Decimal(numerator) / decimal.Decimal(denominator)
    return decimal.Decimal(x)


# All the arithmetic helpers below downcast `decimal.Decimal` values down to floats
# in any binary arithmetic operation which involves one `float` and one `decimal.Decimal`.
# This perhaps peculiar behavior, but it is what Clojure does. I suspect that is due
# to the potential loss of precision with any calculation between these two types, so
# Clojure errs on the side of returning the less precise type to indicate the potential
# lossiness of the calculation.


@functools.singledispatch
@_normalize_fraction_result
def add(x: LispNumber, y: LispNumber) -> LispNumber:
    """Add two numbers together and return the result."""
    return x + y  # type: ignore[operator]


@add.register(float)
@_normalize_fraction_result
def _add_float(x: float, y: LispNumber) -> LispNumber:
    if isinstance(y, decimal.Decimal):
        return float(decimal.Decimal(x) + y)
    return x + y


@add.register(decimal.Decimal)
@_normalize_fraction_result
def _add_decimal(x: decimal.Decimal, y: LispNumber) -> LispNumber:
    v = x + _to_decimal(y)
    return float(v) if isinstance(y, float) else v


@add.register(Fraction)
@_normalize_fraction_result
def _add_fraction(x: Fraction, y: LispNumber) -> LispNumber:
    if isinstance(y, decimal.Decimal):
        return _to_decimal(x) + y
    return x + y


@functools.singledispatch
@_normalize_fraction_result
def subtract(x: LispNumber, y: LispNumber) -> LispNumber:
    """Subtract `y` from `x` and return the result."""
    return x - y  # type: ignore[operator]


@subtract.register(float)
@_normalize_fraction_result
def _subtract_float(x: float, y: LispNumber) -> LispNumber:
    if isinstance(y, decimal.Decimal):
        return float(decimal.Decimal(x) - y)
    return x - y


@subtract.register(decimal.Decimal)
@_normalize_fraction_result
def _subtract_decimal(x: decimal.Decimal, y: LispNumber) -> LispNumber:
    v = x - _to_decimal(y)
    return float(v) if isinstance(y, float) else v


@subtract.register(Fraction)
@_normalize_fraction_result
def _subtract_fraction(x: Fraction, y: LispNumber) -> LispNumber:
    if isinstance(y, decimal.Decimal):
        return _to_decimal(x) - y
    return x - y


@functools.singledispatch
@_normalize_fraction_result
def divide(x: LispNumber, y: LispNumber) -> LispNumber:
    """Division reducer. If both arguments are integers, return a Fraction.
    Otherwise, return the true division of x and y."""
    return x / y  # type: ignore[operator]


@divide.register(int)
@_normalize_fraction_result
def _divide_ints(x: int, y: LispNumber) -> LispNumber:
    if isinstance(y, int):
        return Fraction(x, y)
    return x / y


@divide.register(float)
@_normalize_fraction_result
def _divide_float(x: float, y: LispNumber) -> LispNumber:
    if isinstance(y, decimal.Decimal):
        return float(decimal.Decimal(x) / y)
    try:
        return x / y
    except ZeroDivisionError:
        if math.isnan(x):
            return math.nan
        elif x >= 0:
            return math.inf
        else:
            return -math.inf


@divide.register(decimal.Decimal)
@_normalize_fraction_result
def _divide_decimal(x: decimal.Decimal, y: LispNumber) -> LispNumber:
    v = x / _to_decimal(y)
    return float(v) if isinstance(y, float) else v


@divide.register(Fraction)
@_normalize_fraction_result
def _divide_fraction(x: Fraction, y: LispNumber) -> LispNumber:
    if isinstance(y, decimal.Decimal):
        return _to_decimal(x) / y
    return x / y


@functools.singledispatch
@_normalize_fraction_result
def multiply(x: LispNumber, y: LispNumber) -> LispNumber:
    """Multiply two numbers together and return the result."""
    return x * y  # type: ignore[operator]


@multiply.register(float)
@_normalize_fraction_result
def _multiply_float(x: float, y: LispNumber) -> LispNumber:
    if isinstance(y, decimal.Decimal):
        return float(decimal.Decimal(x) * y)
    return x * y


@multiply.register(decimal.Decimal)
@_normalize_fraction_result
def _multiply_decimal(x: decimal.Decimal, y: LispNumber) -> LispNumber:
    v = x * _to_decimal(y)
    return float(v) if isinstance(y, float) else v


@multiply.register(Fraction)
@_normalize_fraction_result
def _multiply_fraction(x: Fraction, y: LispNumber) -> LispNumber:
    if isinstance(y, decimal.Decimal):
        return _to_decimal(x) * y
    return x * y


@functools.singledispatch
def trunc(x: LispNumber) -> LispNumber:
    """Truncate any fractional part of the input value, preserving the input type.

    Truncation is effectively rounding towards 0."""
    return math.trunc(x)


@trunc.register(float)
def _trunc_float(x: float) -> LispNumber:
    return float(math.trunc(x))


@trunc.register(decimal.Decimal)
def _trunc_decimal(x: decimal.Decimal) -> LispNumber:
    return decimal.Decimal(math.trunc(x))


@trunc.register(Fraction)
def _trunc_fraction(x: Fraction) -> LispNumber:
    v = fractions.Fraction(math.trunc(x))
    return v.numerator if v.denominator == 1 else v
