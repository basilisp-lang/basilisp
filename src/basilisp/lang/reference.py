from typing import Any, Callable, Optional, TypeVar

from readerwriterlock.rwlock import Lockable

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

try:
    from typing import Protocol
except ImportError:
    AlterMeta = Callable[..., Optional[IPersistentMap]]
else:

    class AlterMeta(Protocol):  # type: ignore [no-redef]
        def __call__(
            self, meta: Optional[IPersistentMap], *args
        ) -> Optional[IPersistentMap]:
            ...  # pylint: disable=pointless-statement


class ReferenceBase(IReference):
    """Mixin for IReference classes to define the full IReference interface.

    `basilisp.lang.runtime.Namespace` objects are the only objects which are
    `IReference` objects without also being `IRef` objects.

    Implementers must have the `_rlock`, `_wlock`, and `_meta` properties defined."""

    _rlock: Lockable
    _wlock: Lockable
    _meta: Optional[IPersistentMap]

    @property
    def meta(self) -> Optional[IPersistentMap]:
        with self._rlock:
            return self._meta

    def alter_meta(self, f: AlterMeta, *args) -> Optional[IPersistentMap]:
        with self._wlock:
            self._meta = f(self._meta, *args)
            return self._meta

    def reset_meta(self, meta: Optional[IPersistentMap]) -> Optional[IPersistentMap]:
        with self._wlock:
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

    _validator: Optional[RefValidator]
    _watches: IPersistentMap

    def add_watch(self, k: RefWatchKey, wf: RefWatcher) -> "RefBase[T]":
        with self._wlock:
            self._watches = self._watches.assoc(k, wf)
            return self

    def _notify_watches(self, old: Any, new: Any):
        for k, wf in self._watches.items():
            wf(k, self, old, new)

    def remove_watch(self, k: RefWatchKey) -> "RefBase[T]":
        with self._wlock:
            self._watches = self._watches.dissoc(k)
            return self

    def get_validator(self) -> Optional[RefValidator]:
        return self._validator

    def set_validator(self, vf: Optional[RefValidator] = None) -> None:
        # We cannot use a write lock here since we're calling `self.deref()` which
        # attempts to acquire the read lock for the Ref and will deadlock if the
        # lock is not reentrant.
        #
        # There are no guarantees that the Ref lock is reentrant and the default
        # locks for Atoms and Vars are not).
        #
        # This is probably ok for most cases since we expect contention is low or
        # non-existent while setting a validator function.
        if vf is not None:
            self._validate(self.deref(), vf=vf)
        self._validator = vf

    def _validate(self, val: Any, vf: Optional[RefValidator] = None):
        vf = vf or self._validator
        if vf is not None:
            if not vf(val):
                raise ExceptionInfo(
                    "Invalid reference state",
                    lmap.map({kw.keyword("data"): val, kw.keyword("validator"): vf}),
                )
