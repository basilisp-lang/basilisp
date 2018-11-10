from typing import TypeVar, Generic, Callable, Any, Optional

import basilisp.lang.atom as atom
import basilisp.lang.map as lmap
import basilisp.lang.symbol as sym
from basilisp.util import Maybe

T = TypeVar("T")
DispatchFunction = Callable[..., T]
Method = Callable[..., Any]


class MultiFunction(Generic[T]):
    __slots__ = ("_name", "_default", "_dispatch", "_methods")

    def __init__(
        self, name: sym.Symbol, dispatch: DispatchFunction, default: T
    ) -> None:
        self._name = name  # pylint:disable=assigning-non-slot
        self._default = default  # pylint:disable=assigning-non-slot
        self._dispatch = dispatch  # pylint:disable=assigning-non-slot
        self._methods: atom.Atom = atom.Atom(  # pylint:disable=assigning-non-slot
            lmap.Map.empty()
        )

    def __call__(self, *args, **kwargs):
        key = self._dispatch(*args, **kwargs)
        method_cache = self.methods
        method = Maybe(method_cache.entry(key, None)).or_else(
            lambda: method_cache.entry(self._default, None)
        )
        if method:
            return method(*args, **kwargs)
        raise NotImplementedError

    @staticmethod
    def __add_method(m: lmap.Map, key: T, method: Method) -> lmap.Map:
        """Swap the methods atom to include method with key."""
        return m.assoc(key, method)

    def add_method(self, key: T, method: Method) -> None:
        """Add a new method to this function which will respond for
        key returned from the dispatch function."""
        self._methods.swap(MultiFunction.__add_method, key, method)

    def get_method(self, key: T) -> Optional[Method]:
        """Return the method which would handle this dispatch key or
        None if no method defined for this key and no default."""
        method_cache = self.methods
        # The 'type: ignore' comment below silences a spurious MyPy error
        # about having a return statement in a method which does not return.
        return Maybe(method_cache.entry(key, None)).or_else(
            lambda: method_cache.entry(self._default, None)  # type: ignore
        )

    @staticmethod
    def __remove_method(m: lmap.Map, key: T) -> lmap.Map:
        """Swap the methods atom to remove method with key."""
        return m.dissoc(key)

    def remove_method(self, key: T) -> Optional[Method]:
        """Remove the method defined for this key and return it."""
        method = self.methods.entry(key, None)
        if method:
            self._methods.swap(MultiFunction.__remove_method, key)
        return method

    def remove_all_methods(self) -> None:
        """Remove all methods defined for this multi-function."""
        self._methods.reset(lmap.Map.empty())

    @property
    def default(self) -> T:
        return self._default

    @property
    def methods(self) -> lmap.Map:
        return self._methods.deref()


def multifn(dispatch: DispatchFunction, default=None) -> MultiFunction[T]:
    """Decorator function which can be used to make Python multi functions."""
    name = sym.symbol(dispatch.__qualname__, ns=dispatch.__module__)
    return MultiFunction(name, dispatch, default)
