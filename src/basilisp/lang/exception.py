import functools
import sys
import traceback
import types
from types import TracebackType
from typing import Callable, List, Optional, Type

import attr

from basilisp.lang.interfaces import IExceptionInfo, IPersistentMap
from basilisp.lang.obj import lrepr


@attr.define(repr=False, str=False)
class ExceptionInfo(IExceptionInfo):
    message: str
    data: IPersistentMap

    def __repr__(self):
        return (
            f"basilisp.lang.exception.ExceptionInfo({self.message}, {lrepr(self.data)})"
        )

    def __str__(self):
        return f"{self.message} {lrepr(self.data)}"


ExceptionPrinter = Callable[
    [Type[BaseException], BaseException, Optional[types.TracebackType]], None
]


@functools.singledispatch
def format_exception(
    e: Optional[BaseException],
    tp: Optional[Type[BaseException]] = None,
    tb: Optional[TracebackType] = None,
) -> List[str]:
    """Format an exception into something readable, returning a list of newline
    terminated strings.

    For the majority of Python exceptions, this will just be the result from calling
    `traceback.format_exception`. For Basilisp specific compilation errors, a custom
    output will be returned."""
    if isinstance(e, BaseException):  # pragma: no cover
        if tp is None:
            tp = type(e)
        if tb is None:
            tb = e.__traceback__
    return traceback.format_exception(tp, e, tb)


def print_exception(
    tp: Optional[Type[BaseException]],
    e: Optional[BaseException],
    tb: Optional[TracebackType],
) -> None:
    """Print the given exception `e` using Basilisp's own exception formatting.

    For the majority of exception types, this should be identical to the base Python
    traceback formatting. `basilisp.lang.compiler.CompilerException` and
    `basilisp.lang.reader.SyntaxError` have special handling to print useful information
    on exceptions."""
    print("".join(format_exception(e, tp, tb)), file=sys.stderr)
