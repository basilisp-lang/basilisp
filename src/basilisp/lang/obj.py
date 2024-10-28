import datetime
import math
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable
from decimal import Decimal
from fractions import Fraction
from functools import singledispatch
from itertools import islice
from pathlib import Path
from re import Pattern
from typing import Any, Union, cast

from typing_extensions import TypedDict, Unpack

PrintCountSetting = Union[bool, int, None]

SURPASSED_PRINT_LENGTH = "..."
SURPASSED_PRINT_LEVEL = "#"

PRINT_DUP = False
PRINT_LENGTH: PrintCountSetting = None
PRINT_LEVEL: PrintCountSetting = None
PRINT_META = False
PRINT_NAMESPACE_MAPS = False
PRINT_READABLY = True
PRINT_SEPARATOR = " "


class PrintSettings(TypedDict, total=False):
    human_readable: bool
    print_dup: bool
    print_length: PrintCountSetting
    print_level: PrintCountSetting
    print_meta: bool
    print_namespace_maps: bool
    print_readably: bool


def _dec_print_level(lvl: PrintCountSetting) -> PrintCountSetting:
    """Decrement the print level if it is numeric."""
    if isinstance(lvl, int):
        return lvl - 1
    return lvl


def process_lrepr_kwargs(**kwargs: Unpack[PrintSettings]) -> PrintSettings:
    """Process keyword arguments, decreasing the print-level. Should be called
    after examining the print level for the current level."""
    return cast(
        PrintSettings, dict(kwargs, print_level=_dec_print_level(kwargs["print_level"]))
    )


class LispObject(ABC):
    """Abstract base class for Lisp objects which would like to customize their
    ``__str__`` and Python ``__repr__`` representation.

    .. note::

       Callers should use :py:class:`basilisp.lang.interfaces.ILispObject` as their
       main interface. This interface is defined here so it may be used in
       ``isinstance`` checks below without a circular dependency."""

    __slots__ = ()

    def __repr__(self):
        return self.lrepr()

    def __str__(self):
        return self.lrepr(human_readable=True)

    @abstractmethod
    def _lrepr(self, **kwargs: Unpack[PrintSettings]) -> str:
        """Private Lisp representation method. Callers (including object
        internal callers) should not call this method directly, but instead
        should use the module function :py:meth:`lrepr` ."""
        raise NotImplementedError()

    def lrepr(self, **kwargs: Unpack[PrintSettings]) -> str:
        """Return a string representation of this Lisp object which can be
        read by the reader."""
        return lrepr(self, **kwargs)


def seq_lrepr(
    iterable: Iterable[Any],
    start: str,
    end: str,
    meta=None,
    **kwargs: Unpack[PrintSettings],
) -> str:
    """Produce a Lisp representation of a sequential collection, bookended
    with the start and end string supplied. The keyword arguments will be
    passed along to lrepr for the sequence elements."""
    print_level = kwargs["print_level"]
    if isinstance(print_level, int) and print_level < 1:
        return SURPASSED_PRINT_LEVEL

    kwargs = process_lrepr_kwargs(**kwargs)

    trailer = []
    print_dup = kwargs["print_dup"]
    print_length = kwargs["print_length"]
    if not print_dup and isinstance(print_length, int):
        items = list(islice(iterable, print_length + 1))
        if len(items) > print_length:
            items.pop()
            trailer.append(SURPASSED_PRINT_LENGTH)
    else:
        items = iterable  # type: ignore

    kw_items = kwargs.copy()
    kw_items["human_readable"] = False
    items = list(map(lambda o: lrepr(o, **kw_items), items))
    seq_lrepr = PRINT_SEPARATOR.join(items + trailer)

    print_meta = kwargs["print_meta"]
    if print_meta and meta:
        return f"^{lrepr(meta, **kwargs)} {start}{seq_lrepr}{end}"

    return f"{start}{seq_lrepr}{end}"


# pylint: disable=unused-argument
@singledispatch
def lrepr(  # pylint: disable=too-many-arguments
    o: Any,
    human_readable: bool = False,
    print_dup: bool = PRINT_DUP,
    print_length: PrintCountSetting = PRINT_LENGTH,
    print_level: PrintCountSetting = PRINT_LEVEL,
    print_meta: bool = PRINT_META,
    print_namespace_maps: bool = PRINT_NAMESPACE_MAPS,
    print_readably: bool = PRINT_READABLY,
) -> str:
    """Return a string representation of a Lisp object.

    Permissible keyword arguments are:
    - human_readable: if logical True, print strings without quotations or
                      escape sequences (default: false)
    - print_dup: if logical true, print objects in a way that preserves their
                 types (default: false)
    - print_length: the number of items in a collection which will be printed,
                    or no limit if bound to a logical falsey value (default: 50)
    - print_level: the depth of the object graph to print, starting with 0, or
                   no limit if bound to a logical falsey value (default: nil)
    - print_namespace_maps: if logical true, and the object is a map consisting
                            with keys belonging to the same namespace, print the
                            namespace at the beginning of the map instead of
                            beside the keys (default: false)
    - print_meta: if logical true, print objects meta in a way that can be
                  read back by the reader (default: false)
    - print_readably: if logical false, print strings and characters with
                      non-alphanumeric characters converted to escape sequences
                      (default: true)

    Note that this function is not capable of capturing the values bound at
    runtime to the basilisp.core dynamic variables which correspond to each
    of the keyword arguments to this function. To use a version of lrepr
    which does capture those values, call basilisp.lang.runtime.lrepr directly."""
    return repr(o)


@lrepr.register(LispObject)
def _lrepr_lisp_obj(  # pylint: disable=too-many-arguments
    o: Any,
    human_readable: bool = False,
    print_dup: bool = PRINT_DUP,
    print_length: PrintCountSetting = PRINT_LENGTH,
    print_level: PrintCountSetting = PRINT_LEVEL,
    print_meta: bool = PRINT_META,
    print_namespace_maps: bool = PRINT_NAMESPACE_MAPS,
    print_readably: bool = PRINT_READABLY,
) -> str:  # pragma: no cover
    return o._lrepr(
        human_readable=human_readable,
        print_dup=print_dup,
        print_length=print_length,
        print_level=print_level,
        print_meta=print_meta,
        print_namespace_maps=print_namespace_maps,
        print_readably=print_readably,
    )


@lrepr.register(bool)
def _lrepr_bool(o: bool, **_) -> str:
    return repr(o).lower()


@lrepr.register(bytes)
def _lrepr_bytes(o: bytes, **_) -> str:
    v = repr(o)
    return f'#b "{v[2:-1]}"'


@lrepr.register(type(None))
def _lrepr_nil(_: None, **__) -> str:
    return "nil"


@lrepr.register(str)
def _lrepr_str(
    o: str, human_readable: bool = False, print_readably: bool = PRINT_READABLY, **_
) -> str:
    if human_readable:
        return o
    if print_readably is None or print_readably is False:
        return o
    escaped = o.encode("unicode_escape").replace(b'"', rb"\"").decode("utf-8")
    return f'"{escaped}"'


@lrepr.register(list)
def _lrepr_py_list(o: list, **kwargs: Unpack[PrintSettings]) -> str:
    return f"#py {seq_lrepr(o, '[', ']', **kwargs)}"


@lrepr.register(set)
def _lrepr_py_set(o: set, **kwargs: Unpack[PrintSettings]) -> str:
    return f"#py {seq_lrepr(o, '#{', '}', **kwargs)}"


@lrepr.register(tuple)
def _lrepr_py_tuple(o: tuple, **kwargs: Unpack[PrintSettings]) -> str:
    return f"#py {seq_lrepr(o, '(', ')', **kwargs)}"


@lrepr.register(complex)
def _lrepr_complex(o: complex, **_) -> str:
    return repr(o).upper()


@lrepr.register(float)
def _lrepr_float(o: float, **_) -> str:
    if math.isinf(o):
        return "##Inf" if o > 0 else "##-Inf"
    if math.isnan(o):
        return "##NaN"
    return repr(o)


@lrepr.register(datetime.datetime)
def _lrepr_datetime(o: datetime.datetime, **_) -> str:
    return f'#inst "{o.isoformat()}"'


@lrepr.register(Decimal)
def _lrepr_decimal(o: Decimal, print_dup: bool = PRINT_DUP, **_) -> str:
    if print_dup:
        return f"{str(o)}M"
    return str(o)


@lrepr.register(Fraction)
def _lrepr_fraction(o: Fraction, **_) -> str:
    return f"{o.numerator}/{o.denominator}"


@lrepr.register(Path)
def _lrepr_path(o: Path, **_) -> str:
    return str(o)


@lrepr.register(type(re.compile("")))
def _lrepr_pattern(o: Pattern, **_) -> str:
    return f'#"{o.pattern}"'


@lrepr.register(uuid.UUID)
def _lrepr_uuid(o: uuid.UUID, human_readable: bool = False, **_) -> str:
    if human_readable:
        return str(o)
    return f'#uuid "{str(o)}"'
