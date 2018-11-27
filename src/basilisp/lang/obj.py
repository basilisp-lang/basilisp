import datetime
import uuid
from abc import ABC, abstractmethod
from decimal import Decimal
from fractions import Fraction
from typing import Union, Any, Pattern, Iterable

from functional import seq
from pyrsistent import pmap

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


class LispObject(ABC):
    __slots__ = ()

    @abstractmethod
    def _lrepr(self, **kwargs) -> str:
        """Private Lisp representation method. Callers should not call
        Object._lrepr directly, but instead should use the module function
        lrepr(obj) directly."""
        raise NotImplementedError()

    def lrepr(self, **kwargs) -> str:
        """Return a string representation of this Lisp object."""
        return lrepr(self, **kwargs)

    @staticmethod
    def seq_lrepr(
        iterable: Iterable[Any], start: str, end: str, meta=None, **kwargs
    ) -> str:
        """Produce a Lisp representation of a sequential collection, bookended
        with the start and end string supplied. The keyword arguments will be
        passed along to lrepr for the sequence elements."""
        print_level = kwargs["print_level"]
        if isinstance(print_level, int) and print_level < 0:
            return SURPASSED_PRINT_LEVEL

        kwargs = pmap(initial=kwargs).transform(["print_level"], _dec_print_level)

        print_length = kwargs["print_length"]
        if isinstance(print_length, int):
            items = seq(iterable).take(print_length + 1).to_list()
            if len(items) > print_length:
                items.pop()
                items.append(SURPASSED_PRINT_LENGTH)
        else:
            items = iterable

        items = seq(items).map(lambda o: lrepr(o, **kwargs)).to_list()
        seq_lrepr = PRINT_SEPARATOR.join(items)

        print_meta = kwargs["print_meta"]
        if print_meta and meta:
            return f"^{LispObject.lrepr(meta, **kwargs)} {start}{seq_lrepr}{end}"

        return f"{start}{seq_lrepr}{end}"


def lrepr(  # pylint: disable=too-many-arguments
    o: Any,
    print_dup: bool = PRINT_DUP,
    print_length: PrintCountSetting = PRINT_LENGTH,
    print_level: PrintCountSetting = PRINT_LEVEL,
    print_meta: bool = PRINT_META,
    print_readably: bool = PRINT_READABLY,
) -> str:
    """Return a string representation of a Lisp object.

    Permissible keyword arguments are:
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
    kwargs = pmap(
        {
            "print_dup": print_dup,
            "print_length": print_length,
            "print_level": print_level,
            "print_meta": print_meta,
            "print_readably": print_readably,
        }
    )
    if isinstance(o, LispObject):
        return o._lrepr(**kwargs)
    elif o is True:
        return "true"
    elif o is False:
        return "false"
    elif o is None:
        return "nil"
    elif isinstance(o, str):
        if print_readably is None or print_readably is False:
            return f'"{o}"'
        return f'"{o}"'
    elif isinstance(o, complex):
        return repr(o).upper()
    elif isinstance(o, datetime.datetime):
        inst_str = o.isoformat()
        return f'#inst "{inst_str}"'
    elif isinstance(o, Decimal):
        return str(o)
    elif isinstance(o, Fraction):
        return f"{o.numerator}/{o.denominator}"
    elif isinstance(o, Pattern):
        return f'#"{o.pattern}"'
    elif isinstance(o, uuid.UUID):
        uuid_str = str(o)
        return f'#uuid "{uuid_str}"'
    else:
        return repr(o)
