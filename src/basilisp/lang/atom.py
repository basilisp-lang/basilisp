from typing import Callable, Generic, Optional, TypeVar

from readerwriterlock.rwlock import RWLockFair

from basilisp.lang.interfaces import IPersistentMap
from basilisp.lang.map import PersistentMap
from basilisp.lang.reference import RefBase

T = TypeVar("T")


class Atom(RefBase[T], Generic[T]):
    __slots__ = ("_meta", "_state", "_rlock", "_wlock", "_watches", "_validator")

    def __init__(self, state: T) -> None:
        self._meta: Optional[IPersistentMap] = None
        self._state = state
        lock = RWLockFair()
        self._rlock = lock.gen_rlock()
        self._wlock = lock.gen_wlock()
        self._watches = PersistentMap.empty()
        self._validator = None

    def _compare_and_set(self, old: T, new: T) -> bool:
        with self._wlock:
            if self._state != old:
                return False
            self._state = new
            return True

    def compare_and_set(self, old: T, new: T) -> bool:
        """Compare the current state of the Atom to `old`. If the value is the same,
        atomically set the value of the state of Atom to `new`. Return True if the
        value was swapped. Return False otherwise."""
        self._validate(new)
        if self._compare_and_set(old, new):
            self._notify_watches(old, new)
            return True
        return False

    def deref(self) -> T:
        """Return the state stored within the Atom."""
        with self._rlock:
            return self._state

    def reset(self, v: T) -> T:
        """Reset the state of the Atom to `v` without regard to the current value."""
        while True:
            oldval = self._state
            self._validate(v)
            if self._compare_and_set(oldval, v):
                self._notify_watches(oldval, v)
                return v

    def swap(self, f: Callable[..., T], *args, **kwargs) -> T:
        """Atomically swap the state of the Atom to the return value of
        `f(old, *args, **kwargs)`, returning the new value."""
        while True:
            oldval = self._state
            newval = f(oldval, *args, **kwargs)
            self._validate(newval)
            if self._compare_and_set(oldval, newval):
                self._notify_watches(oldval, newval)
                return newval
