from typing import Callable, Optional

from readerwriterlock.rwlock import Lockable

from basilisp.lang.interfaces import IPersistentMap, IReference

try:
    from typing import Protocol
except ImportError:
    AlterMeta = Callable[..., IPersistentMap]
else:

    class AlterMeta(Protocol):  # type: ignore [no-redef]
        def __call__(
            self, meta: Optional[IPersistentMap], *args
        ) -> Optional[IPersistentMap]:
            ...  # pylint: disable=pointless-statement


class ReferenceBase(IReference):
    """Mixin for IReference classes to define the full IReference interface.

    Consumers must have a `_lock` and `_meta` property defined."""

    _rlock: Lockable
    _wlock: Lockable
    _meta: Optional[IPersistentMap]

    @property
    def meta(self) -> Optional[IPersistentMap]:
        with self._rlock:
            return self._meta

    def alter_meta(self, f: AlterMeta, *args) -> IPersistentMap:
        with self._wlock:
            self._meta = f(self._meta, *args)
            return self._meta

    def reset_meta(self, meta: IPersistentMap) -> IPersistentMap:
        with self._wlock:
            self._meta = meta
            return meta
