import threading
from typing import Any, Callable, Generic, Optional, TypeVar

from basilisp.lang import map as lmap
from basilisp.lang import runtime
from basilisp.lang import set as lset
from basilisp.lang import symbol as sym
from basilisp.lang.interfaces import IPersistentMap, IPersistentSet, IRef

T = TypeVar("T")
DispatchFunction = Callable[..., T]
Method = Callable[..., Any]


_GLOBAL_HIERARCHY_SYM = sym.symbol("global-hierarchy", ns=runtime.CORE_NS)
_ISA_SYM = sym.symbol("isa?", ns=runtime.CORE_NS)


class MultiFunction(Generic[T]):
    __slots__ = (
        "_name",
        "_default",
        "_dispatch",
        "_lock",
        "_methods",
        "_cache",
        "_prefers",
        "_hierarchy",
        "_cached_hierarchy",
        "_isa",
    )

    def __init__(
        self,
        name: sym.Symbol,
        dispatch: DispatchFunction,
        default: T,
        hierarchy: Optional[IRef] = None,
    ) -> None:
        self._name = name
        self._default = default
        self._dispatch = dispatch
        self._lock = threading.Lock()
        self._methods: IPersistentMap[T, Method] = lmap.EMPTY
        self._cache: IPersistentMap[T, Method] = lmap.EMPTY
        self._prefers: IPersistentMap[T, IPersistentSet[T]] = lmap.EMPTY

        # Fetch some items from basilisp.core that we need to compute the final
        # dispatch method. These cannot be imported statically because that would
        # produce a circular reference between basilisp.core and this module.
        self._isa = runtime.Var.find_safe(_ISA_SYM)
        self._hierarchy: IRef[IPersistentMap] = hierarchy or runtime.Var.find_safe(
            _GLOBAL_HIERARCHY_SYM
        )

        if not isinstance(self._hierarchy, IRef):
            raise runtime.RuntimeException(
                f"Expected IRef type for :hierarchy; got {type(hierarchy)}"
            )

        # Maintain a cache of the hierarchy value to detect when the hierarchy
        # has changed. If the hierarchy changes, we need to reset the internal
        # caches.
        self._cached_hierarchy = self._hierarchy.deref()

    def __call__(self, *args, **kwargs):
        key = self._dispatch(*args, **kwargs)
        method = self.get_method(key)
        if method is not None:
            return method(*args, **kwargs)
        raise NotImplementedError

    def _reset_cache(self):
        """Reset the local cache to the base method mapping.

        Should be called after methods are added or removed or after preferences are
        altered."""
        # Does not use a lock to avoid lock reentrance
        self._cache = self._methods
        self._cached_hierarchy = self._hierarchy.deref()

    def _is_a(self, tag: T, parent: T) -> bool:
        """Return True if `tag` can be considered a `parent` type using `isa?`."""
        return bool(self._isa.value(self._hierarchy.deref(), tag, parent))

    def _has_preference(self, preferred_key: T, other_key: T) -> bool:
        """Return True if this multimethod has `preferred_key` listed as a preference
        over `other_key`."""
        others = self._prefers.val_at(preferred_key)
        return others is not None and other_key in others

    def _precedes(self, tag: T, parent: T) -> bool:
        """Return True if `tag` should be considered ahead of `parent` for method
        selection."""
        return self._has_preference(tag, parent) or self._is_a(tag, parent)

    def add_method(self, key: T, method: Method) -> None:
        """Add a new method to this function which will respond for key returned from
        the dispatch function."""
        with self._lock:
            self._methods = self._methods.assoc(key, method)
            self._reset_cache()

    def _find_and_cache_method(self, key: T) -> Optional[Method]:
        """Find and cache the best method for dispatch value `key`."""
        with self._lock:
            best_key: Optional[T] = None
            best_method: Optional[Method] = None
            for method_key, method in self._methods.items():
                if self._is_a(key, method_key):
                    if best_key is None or self._precedes(method_key, best_key):
                        best_key, best_method = method_key, method
                    if not self._precedes(best_key, method_key):
                        raise runtime.RuntimeException(
                            "Cannot resolve a unique method for dispatch value "
                            f"'{key}'; '{best_key}' and '{method_key}' both match and "
                            "neither is preferred"
                        )

            if best_method is None:
                best_method = self._methods.val_at(self._default)

            if best_method is not None:
                self._cache = self._cache.assoc(key, best_method)

            return best_method

    def get_method(self, key: T) -> Optional[Method]:
        """Return the method which would handle this dispatch key or None if no method
        defined for this key and no default."""
        if self._cached_hierarchy != self._hierarchy.deref():
            self._reset_cache()

        cached_val = self._cache.val_at(key)
        if cached_val is not None:
            return cached_val

        return self._find_and_cache_method(key)

    def prefer_method(self, preferred_key: T, other_key: T) -> None:
        """Update the multimethod to prefer `preferred_key` over `other_key` in cases
        where method selection might be ambiguous between two values."""
        with self._lock:
            if self._has_preference(  # pylint: disable=arguments-out-of-order
                other_key, preferred_key
            ):
                raise runtime.RuntimeException(
                    f"Cannot set preference for '{preferred_key}' over '{other_key}' "
                    f"due to existing preference for '{other_key}' over "
                    f"'{preferred_key}'"
                )
            existing = self._prefers.val_at(preferred_key, lset.EMPTY)
            assert existing is not None
            self._prefers = self._prefers.assoc(preferred_key, existing.cons(other_key))
            self._reset_cache()

    @property
    def prefers(self):
        """Return a mapping of preferred values to the set of other values."""
        return self._prefers

    def remove_method(self, key: T) -> Optional[Method]:
        """Remove the method defined for this key and return it."""
        with self._lock:
            method = self._methods.val_at(key, None)
            if method:
                self._methods = self._methods.dissoc(key)
            self._reset_cache()
            return method

    def remove_all_methods(self) -> None:
        """Remove all methods defined for this multimethod1."""
        with self._lock:
            self._methods = lmap.EMPTY
            self._reset_cache()

    @property
    def default(self) -> T:
        return self._default

    @property
    def methods(self) -> IPersistentMap[T, Method]:
        return self._methods
