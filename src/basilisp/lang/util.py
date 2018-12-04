import builtins
import datetime
import keyword
import re
import uuid
from decimal import Decimal
from fractions import Fraction
from typing import Pattern, Match

import dateutil.parser as dateparser

import basilisp.lang.atom as atom

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
}


def munge(s: str, allow_builtins: bool = False) -> str:
    """Replace characters which are not valid in Python symbols
    with valid replacement strings."""
    new_str = []
    for c in s:
        new_str.append(_MUNGE_REPLACEMENTS.get(c, c))

    new_s = "".join(new_str)

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
