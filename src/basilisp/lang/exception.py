import functools
import sys
import traceback
from types import TracebackType
from typing import Optional

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


@functools.singledispatch
def format_exception(  # pylint: disable=unused-argument
    e: Optional[BaseException],
    tp: Optional[type[BaseException]] = None,
    tb: Optional[TracebackType] = None,
    disable_color: Optional[bool] = None,
) -> list[str]:
    """Format an exception into something readable, returning a list of newline
    terminated strings.

    For the majority of Python exceptions, this will just be the result from calling
    `traceback.format_exception`. For Basilisp specific compilation errors, a custom
    output will be returned.

    If `disable_color` is True, no color formatting should be applied to the source
    code."""
    if isinstance(e, BaseException):
        if tp is None:
            tp = type(e)
        if tb is None:
            tb = e.__traceback__
    return traceback.format_exception(tp, e, tb)


def print_exception(
    e: Optional[BaseException],
    tp: Optional[type[BaseException]] = None,
    tb: Optional[TracebackType] = None,
) -> None:
    """Print the given exception `e` using Basilisp's own exception formatting.

    For the majority of exception types, this should be identical to the base Python
    traceback formatting. `basilisp.lang.compiler.CompilerException` and
    `basilisp.lang.reader.SyntaxError` have special handling to print useful information
    on exceptions."""
    print("".join(format_exception(e, tp, tb)), file=sys.stderr)
