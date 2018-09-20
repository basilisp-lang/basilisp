import basilisp.lang.map as lmap
from basilisp.lang.util import lrepr


class ExceptionInfo(Exception):
    __slots__ = ('_msg', '_data',)

    def __init__(self, message: str, data: lmap.Map) -> None:
        super().__init__()
        self._msg = message
        self._data = data

    def __repr__(self):
        return f"basilisp.lang.exception.ExceptionInfo({self._msg}, {lrepr(self._data)})"

    def __str__(self):
        return f"{self._msg} {lrepr(self._data)}"

    @property
    def data(self):
        return self._data

    @property
    def message(self):
        return self._data
