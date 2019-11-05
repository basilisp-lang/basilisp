import attr

from basilisp.lang.interfaces import IExceptionInfo, IPersistentMap
from basilisp.lang.obj import lrepr

_CMP_OFF = getattr(attr, "__version_info__", (0,)) >= (19, 2)


@attr.s(  # type: ignore
    auto_attribs=True,
    **({"eq": True} if _CMP_OFF else {"cmp": False}),
    repr=False,
    slots=True,
    str=False,
)
class ExceptionInfo(IExceptionInfo):
    message: str
    data: IPersistentMap

    def __repr__(self):
        return (
            f"basilisp.lang.exception.ExceptionInfo({self.message}, {lrepr(self.data)})"
        )

    def __str__(self):
        return f"{self.message} {lrepr(self.data)}"
