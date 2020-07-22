import threading
from typing import Any, Callable, Generic, Optional, TypeVar

from basilisp.lang import map as lmap
from basilisp.lang import symbol as sym
from basilisp.lang.interfaces import IPersistentMap
from basilisp.util import Maybe

T = TypeVar("T")
DispatchFunction = Callable[..., T]
Method = Callable[..., Any]


class MultiFunction(Generic[T]):
    __slots__ = ("_name", "_default", "_dispatch", "_lock", "_methods")

    # pylint:disable=assigning-non-slot
    def __init__(
        self, name: sym.Symbol, dispatch: DispatchFunction, default: T
    ) -> None:
        self._name = name
        self._default = default
        self._dispatch = dispatch
        self._lock = threading.Lock()
        self._methods: IPersistentMap[T, Method] = lmap.PersistentMap.empty()

    def __call__(self, *args, **kwargs):
        key = self._dispatch(*args, **kwargs)
        method = self.get_method(key)
        if method is not None:
            return method(*args, **kwargs)
        raise NotImplementedError

    def add_method(self, key: T, method: Method) -> None:
        """Add a new method to this function which will respond for
        key returned from the dispatch function."""
        with self._lock:
            self._methods = self._methods.assoc(key, method)

    def get_method(self, key: T) -> Optional[Method]:
        """Return the method which would handle this dispatch key or
        None if no method defined for this key and no default."""
        method_cache = self._methods
        # The 'type: ignore' comment below silences a spurious MyPy error
        # about having a return statement in a method which does not return.
        return Maybe(method_cache.val_at(key, None)).or_else(
            lambda: method_cache.val_at(self._default, None)  # type: ignore
        )

    def remove_method(self, key: T) -> Optional[Method]:
        """Remove the method defined for this key and return it."""
        with self._lock:
            method = self._methods.val_at(key, None)
            if method:
                self._methods = self._methods.dissoc(key)
            return method

    def remove_all_methods(self) -> None:
        """Remove all methods defined for this multi-function."""
        with self._lock:
            self._methods = lmap.PersistentMap.empty()

    @property
    def default(self) -> T:
        return self._default

    @property
    def methods(self) -> IPersistentMap[T, Method]:
        return self._methods


def multifn(dispatch: DispatchFunction, default=None) -> MultiFunction[T]:
    """Decorator function which can be used to make Python multi functions."""
    name = sym.symbol(dispatch.__qualname__, ns=dispatch.__module__)
    return MultiFunction(name, dispatch, default)
