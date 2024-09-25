import threading
from typing import Any, Callable, Optional, TypeVar

from typing_extensions import Concatenate, ParamSpec

from basilisp.lang import keyword as kw
from basilisp.lang import map as lmap
from basilisp.lang.exception import ExceptionInfo
from basilisp.lang.interfaces import (
    IPersistentMap,
    IRef,
    IReference,
    RefValidator,
    RefWatcher,
    RefWatchKey,
)

P = ParamSpec("P")
AlterMeta = Callable[Concatenate[Optional[IPersistentMap], P], Optional[IPersistentMap]]


class ReferenceBase(IReference):
    """Mixin for IReference classes to define the full IReference interface.

    `basilisp.lang.runtime.Namespace` objects are the only objects which are
    `IReference` objects without also being `IRef` objects.

    Implementers must have the `_lock` and `_meta` properties defined."""

    _lock: threading.RLock
    _meta: Optional[IPersistentMap]

    @property
    def meta(self) -> Optional[IPersistentMap]:
        with self._lock:
            return self._meta

    def alter_meta(self, f: AlterMeta, *args) -> Optional[IPersistentMap]:
        with self._lock:
            self._meta = f(self._meta, *args)
            return self._meta

    def reset_meta(self, meta: Optional[IPersistentMap]) -> Optional[IPersistentMap]:
        with self._lock:
            self._meta = meta
            return meta


T = TypeVar("T")


class RefBase(IRef[T], ReferenceBase):
    """
    Mixin for IRef classes to define the full IRef interface.

    `IRef` objects are generally shared, mutable state objects such as Atoms and
    Vars.

    Implementers must have the `_validators` and `_watches` properties defined.
    """

    _validator: Optional[RefValidator[T]]
    _watches: IPersistentMap[RefWatchKey, RefWatcher[T]]

    def add_watch(self, k: RefWatchKey, wf: RefWatcher[T]) -> "RefBase[T]":
        with self._lock:
            self._watches = self._watches.assoc(k, wf)
            return self

    def _notify_watches(self, old: T, new: T) -> None:
        for k, wf in self._watches.items():
            wf(k, self, old, new)

    def remove_watch(self, k: RefWatchKey) -> "RefBase[T]":
        with self._lock:
            self._watches = self._watches.dissoc(k)
            return self

    def get_validator(self) -> Optional[RefValidator[T]]:
        return self._validator

    def set_validator(self, vf: Optional[RefValidator[T]] = None) -> None:
        with self._lock:
            if vf is not None:
                self._validate(self.deref(), vf=vf)
            self._validator = vf

    def _validate(self, val: Any, vf: Optional[RefValidator[T]] = None) -> None:
        vf = vf or self._validator
        if vf is not None:
            try:
                res = vf(val)
            except Exception:  # pylint: disable=broad-exception-caught
                res = False

            if not res:
                raise ExceptionInfo(  # pylint: disable=abstract-class-instantiated
                    "Invalid reference state",
                    lmap.map({kw.keyword("data"): val, kw.keyword("validator"): vf}),
                )
