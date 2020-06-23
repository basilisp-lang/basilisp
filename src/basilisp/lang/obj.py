import datetime
import math
import re
import uuid
from abc import ABC, abstractmethod
from decimal import Decimal
from fractions import Fraction
from functools import singledispatch
from typing import Any, Callable, Iterable, Pattern, Tuple, Union

from basilisp.util import take

PrintCountSetting = Union[bool, int, None]

SURPASSED_PRINT_LENGTH = "..."
SURPASSED_PRINT_LEVEL = "#"

PRINT_DUP = False
PRINT_LENGTH: PrintCountSetting = 50
PRINT_LEVEL: PrintCountSetting = None
PRINT_META = False
PRINT_READABLY = True
PRINT_SEPARATOR = " "


def _dec_print_level(lvl: PrintCountSetting):
    """Decrement the print level if it is numeric."""
    if isinstance(lvl, int):
        return lvl - 1
    return lvl


def _process_kwargs(**kwargs):
    """Process keyword arguments, decreasing the print-level. Should be called
    after examining the print level for the current level."""
    return dict(kwargs, print_level=_dec_print_level(kwargs["print_level"]))


class LispObject(ABC):
    """Abstract base class for Lisp objects which would like to customize
    their `str` and Python `repr` representation.

    Callers should use `ILispObject` in `basilisp.lang.interfaces` as their
    main interface. This interface is defined here so it may be used in
    `isinstance` checks below."""

    __slots__ = ()

    def __repr__(self):
        return self.lrepr()

    def __str__(self):
        return self.lrepr(human_readable=True)

    @abstractmethod
    def _lrepr(self, **kwargs) -> str:
        """Private Lisp representation method. Callers (including object
        internal callers) should not call this method directly, but instead
        should use the module function .lrepr()."""
        raise NotImplementedError()

    def lrepr(self, **kwargs) -> str:
        """Return a string representation of this Lisp object which can be
        read by the reader."""
        return lrepr(self, **kwargs)


def map_lrepr(
    entries: Callable[[], Iterable[Tuple[Any, Any]]],
    start: str,
    end: str,
    meta=None,
    **kwargs,
) -> str:
    """Produce a Lisp representation of an associative collection, bookended
    with the start and end string supplied. The entries argument must be a
    callable which will produce tuples of key-value pairs.

    The keyword arguments will be passed along to lrepr for the sequence
    elements."""
    print_level = kwargs["print_level"]
    if isinstance(print_level, int) and print_level < 1:
        return SURPASSED_PRINT_LEVEL

    kwargs = _process_kwargs(**kwargs)

    def entry_reprs():
        for k, v in entries():
            yield "{k} {v}".format(k=lrepr(k, **kwargs), v=lrepr(v, **kwargs))

    trailer = []
    print_dup = kwargs["print_dup"]
    print_length = kwargs["print_length"]
    if not print_dup and isinstance(print_length, int):
        items = list(take(entry_reprs(), print_length + 1))
        if len(items) > print_length:
            items.pop()
            trailer.append(SURPASSED_PRINT_LENGTH)
    else:
        items = list(entry_reprs())

    seq_lrepr = PRINT_SEPARATOR.join(items + trailer)

    print_meta = kwargs["print_meta"]
    if print_meta and meta:
        return f"^{lrepr(meta, **kwargs)} {start}{seq_lrepr}{end}"

    return f"{start}{seq_lrepr}{end}"


def seq_lrepr(
    iterable: Iterable[Any], start: str, end: str, meta=None, **kwargs
) -> str:
    """Produce a Lisp representation of a sequential collection, bookended
    with the start and end string supplied. The keyword arguments will be
    passed along to lrepr for the sequence elements."""
    print_level = kwargs["print_level"]
    if isinstance(print_level, int) and print_level < 1:
        return SURPASSED_PRINT_LEVEL

    kwargs = _process_kwargs(**kwargs)

    trailer = []
    print_dup = kwargs["print_dup"]
    print_length = kwargs["print_length"]
    if not print_dup and isinstance(print_length, int):
        items = list(take(iterable, print_length + 1))
        if len(items) > print_length:
            items.pop()
            trailer.append(SURPASSED_PRINT_LENGTH)
    else:
        items = iterable  # type: ignore

    items = list(map(lambda o: lrepr(o, **kwargs), items))
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
    print_readably: bool = PRINT_READABLY,
) -> str:  # pragma: no cover
    return o._lrepr(
        human_readable=human_readable,
        print_dup=print_dup,
        print_length=print_length,
        print_level=print_level,
        print_meta=print_meta,
        print_readably=print_readably,
    )


@lrepr.register(bool)
def _lrepr_bool(o: bool, **_) -> str:
    return repr(o).lower()


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
        return f'"{o}"'
    return f'"{o.encode("unicode_escape").decode("utf-8")}"'


@lrepr.register(list)
def _lrepr_py_list(o: list, **kwargs) -> str:
    return f"#py {seq_lrepr(o, '[', ']', **kwargs)}"


@lrepr.register(dict)
def _lrepr_py_dict(o: dict, **kwargs) -> str:
    return f"#py {map_lrepr(o.items, '{', '}', **kwargs)}"


@lrepr.register(set)
def _lrepr_py_set(o: set, **kwargs) -> str:
    return f"#py {seq_lrepr(o, '#{', '}', **kwargs)}"


@lrepr.register(tuple)
def _lrepr_py_tuple(o: tuple, **kwargs) -> str:
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


@lrepr.register(type(re.compile("")))
def _lrepr_pattern(o: Pattern, **_) -> str:
    return f'#"{o.pattern}"'


@lrepr.register(uuid.UUID)
def _lrepr_uuid(o: uuid.UUID, **_) -> str:
    return f'#uuid "{str(o)}"'
