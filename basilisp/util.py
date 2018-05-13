import functools
import inspect
import os.path

from functional import seq

from basilisp.lang.util import lrepr


def drop_last(s, n=1):
    """Drop the last n items in the sequence s."""
    return seq(s).drop_right(n)


def trace(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        calling_frame = inspect.getframeinfo(inspect.stack()[1][0])
        filename = os.path.relpath(calling_frame.filename)
        lineno = calling_frame.lineno
        strargs = ', '.join(map(repr, args))
        strkwargs = ', '.join(
            [f'{lrepr(k)}={lrepr(v)}' for k, v in kwargs.items()])

        try:
            ret = f(*args, **kwargs)
            print(
                f"[{filename}:{lineno}] {f.__name__}({strargs}, {strkwargs}) => {ret}"
            )
            return ret
        except Exception as e:
            print(
                f"[{filename}:{lineno}] {f.__name__}({strargs}, {strkwargs}) => raised {type(e)}"
            )
            raise e

    return wrapper
