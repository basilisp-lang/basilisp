from typing import Callable, Generic, Optional, TypeVar

from readerwriterlock.rwlock import RWLockFair

from basilisp.lang.interfaces import IDeref, IPersistentMap
from basilisp.lang.reference import ReferenceBase

T = TypeVar("T")


class Atom(IDeref[T], ReferenceBase, Generic[T]):
    __slots__ = ("_meta", "_state", "_rlock", "_wlock")

    # pylint: disable=assigning-non-slot
    def __init__(self, state: T) -> None:
        self._meta: Optional[IPersistentMap] = None
        self._state = state
        lock = RWLockFair()
        self._rlock = lock.gen_rlock()
        self._wlock = lock.gen_wlock()

    def compare_and_set(self, old: T, new: T) -> bool:
        """Compare the current state of the Atom to `old`. If the value is the same,
        atomically set the value of the state of Atom to `new`. Return True if the
        value was swapped. Return False otherwise."""
        with self._wlock:
            if self._state != old:
                return False
            self._state = new
            return True

    def deref(self) -> T:
        """Return the state stored within the Atom."""
        with self._rlock:
            return self._state

    def reset(self, v: T) -> T:
        """Reset the state of the Atom to `v` without regard to the current value."""
        with self._wlock:
            self._state = v
            return v

    def swap(self, f: Callable[..., T], *args, **kwargs) -> T:
        """Atomically swap the state of the Atom to the return value of
        `f(old, *args, **kwargs)`, returning the new value."""
        with self._wlock:
            newval = f(self._state, *args, **kwargs)
            self._state = newval
            return newval
