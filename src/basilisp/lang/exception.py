import attr

import basilisp.lang.map as lmap
from basilisp.lang.obj import lrepr


@attr.s(auto_attribs=True, cmp=False, repr=False, slots=True, str=False)
class ExceptionInfo(Exception):
    message: str
    data: lmap.Map

    def __repr__(self):
        return (
            f"basilisp.lang.exception.ExceptionInfo({self.message}, {lrepr(self.data)})"
        )

    def __str__(self):
        return f"{self.message} {lrepr(self.data)}"
