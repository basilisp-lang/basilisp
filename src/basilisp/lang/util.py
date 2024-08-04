import builtins
import datetime
import inspect
import keyword
import re
import uuid
from decimal import Decimal
from fractions import Fraction
from typing import Iterable, Match, Pattern, Type

from dateutil import parser as dateparser

from basilisp.lang import atom as atom

_DOUBLE_DOT = ".."
_DOUBLE_DOT_REPLACEMENT = "__DOT_DOT__"

_MUNGE_REPLACEMENTS = {
    "+": "__PLUS__",
    "-": "_",
    "*": "__STAR__",
    "/": "__DIV__",
    ">": "__GT__",
    "<": "__LT__",
    "!": "__BANG__",
    "=": "__EQ__",
    "?": "__Q__",
    "\\": "__IDIV__",
    "&": "__AMP__",
    "$": "__DOLLAR__",
    "%": "__PCT__",
}
_MUNGE_TRANSLATE_TABLE = str.maketrans(_MUNGE_REPLACEMENTS)


def count(seq: Iterable) -> int:
    return sum(1 for _ in seq)


def munge(s: str, allow_builtins: bool = False) -> str:
    """Replace characters which are not valid in Python symbols
    with valid replacement strings."""
    new_s = s.translate(_MUNGE_TRANSLATE_TABLE)

    if new_s == _DOUBLE_DOT:
        return _DOUBLE_DOT_REPLACEMENT

    if keyword.iskeyword(new_s):
        return f"{new_s}_"

    if not allow_builtins and new_s in builtins.__dict__:
        return f"{new_s}_"

    return new_s


_DEMUNGE_PATTERN = re.compile(r"(__[A-Z]+__)")
_DEMUNGE_REPLACEMENTS = {v: k for k, v in _MUNGE_REPLACEMENTS.items()}


def demunge(s: str) -> str:
    """Replace munged string components with their original
    representation."""

    def demunge_replacer(match: Match) -> str:
        full_match = match.group(0)
        replacement = _DEMUNGE_REPLACEMENTS.get(full_match, None)
        if replacement:
            return replacement
        return full_match

    return re.sub(_DEMUNGE_PATTERN, demunge_replacer, s).replace("_", "-")


def is_abstract(tp: Type) -> bool:
    """Return True if tp is an abstract class.

    The builtin `inspect.isabstract` returns False for marker abstract classes
    which do not define any abstract members."""
    if inspect.isabstract(tp):
        return True
    return (
        inspect.isclass(tp)
        and hasattr(tp, "__abstractmethods__")
        and tp.__abstractmethods__ == frozenset()
    )


# Trimmed list of __dunder__ methods generated by using this shell command:
#
#     curl https://raw.githubusercontent.com/python/cpython/main/Doc/reference/datamodel.rst \
#       | egrep -oh '__[a-z_][A-Za-z_0-9]*__' \
#       | sort \
#       | uniq
#
# Running the above command will yield a list of dunders from the Python 'Data Model'
# documentation page, but many of those matches will be false positives. Many hits
# are object properties (such as __doc__ and __name__) and others are not dunders
# at all (such as the unfortunately named __future__ module).
#
# Unfortunately we can't introspect this list from Python itself, so it's going to be
# a moving target as the data model changes over time.
#
# Note that some Python standard library modules and packages define their own dunder
# methods which are not documented in the data model documentation which are included
# in separate sets below.
OBJECT_DUNDER_METHODS = frozenset(
    {
        "__abs__",
        "__add__",
        "__aenter__",
        "__aexit__",
        "__aiter__",
        "__and__",
        "__anext__",
        "__await__",
        "__bool__",
        "__buffer__",
        "__bytes__",
        "__call__",
        "__ceil__",
        "__class_getitem__",
        "__complex__",
        "__contains__",
        "__del__",
        "__delattr__",
        "__delete__",
        "__delitem__",
        "__dict__",
        "__dir__",
        "__divmod__",
        "__enter__",
        "__eq__",
        "__exit__",
        "__float__",
        "__floor__",
        "__floordiv__",
        "__format__",
        "__ge__",
        "__get__",
        "__getattr__",
        "__getattribute__",
        "__getitem__",
        "__gt__",
        "__hash__",
        "__iadd__",
        "__iand__",
        "__ifloordiv__",
        "__ilshift__",
        "__imatmul__",
        "__imod__",
        "__imul__",
        "__index__",
        "__init_subclass__",
        "__instancecheck__",
        "__int__",
        "__invert__",
        "__ior__",
        "__ipow__",
        "__irshift__",
        "__isub__",
        "__iter__",
        "__itruediv__",
        "__ixor__",
        "__le__",
        "__len__",
        "__length_hint__",
        "__lshift__",
        "__lt__",
        "__match_args__",
        "__matmul__",
        "__missing__",
        "__mod__",
        "__mro_entries__",
        "__mul__",
        "__ne__",
        "__neg__",
        "__new__",
        "__next__",
        "__or__",
        "__pos__",
        "__pow__",
        "__prepare__",
        "__radd__",
        "__rand__",
        "__rdivmod__",
        "__release_buffer__",
        "__repr__",
        "__reversed__",
        "__rfloordiv__",
        "__rlshift__",
        "__rmatmul__",
        "__rmod__",
        "__rmul__",
        "__ror__",
        "__round__",
        "__rpow__",
        "__rrshift__",
        "__rshift__",
        "__rsub__",
        "__rtruediv__",
        "__rxor__",
        "__set__",
        "__set_name__",
        "__setattr__",
        "__setitem__",
        "__sizeof__",
        "__slots__",
        "__str__",
        "__sub__",
        "__subclasscheck__",
        "__truediv__",
        "__trunc__",
        "__xor__",
    }
    |
    # Support for ABCs
    {"__subclasshook__"}
    |
    # Support for pickling
    {
        "__getnewargs_ex__",
        "__getnewargs__",
        "__getstate__",
        "__reduce__",
        "__reduce_ex__",
        "__setstate__",
    }
)


# Use an atomically incremented integer as a suffix for all
# user-defined function and variable names compiled into Python
# code so no conflicts occur
_NAME_COUNTER = atom.Atom(1)


def next_name_id() -> int:
    """Increment the name counter and return the next value."""
    return _NAME_COUNTER.swap(lambda x: x + 1)


def genname(prefix: str) -> str:
    """Generate a unique function name with the given prefix."""
    i = next_name_id()
    return f"{prefix}_{i}"


def decimal_from_str(decimal_str: str) -> Decimal:
    """Create a Decimal from a numeric string."""
    return Decimal(decimal_str)


def fraction(numerator: int, denominator: int) -> Fraction:
    """Create a Fraction from a numerator and denominator."""
    return Fraction(numerator=numerator, denominator=denominator)


def inst_from_str(inst_str: str) -> datetime.datetime:
    """Create a datetime instance from an RFC 3339 formatted date string."""
    return dateparser.parse(inst_str)


def regex_from_str(regex_str: str) -> Pattern:
    """Create a new regex pattern from the input string."""
    return re.compile(regex_str)


def uuid_from_str(uuid_str: str) -> uuid.UUID:
    """Create a new UUID instance from the canonical string representation
    of a UUID."""
    return uuid.UUID(f"{{{uuid_str}}}")
