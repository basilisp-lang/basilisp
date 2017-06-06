import threading


class Atom:
    def __init__(self, v):
        self._lock = threading.Lock()
        self._v = v

    def swap(self, f):
        with self._lock:
            self._v = f(self._v)

    def reset(self, v):
        with self._lock:
            self._v = v

    def deref(self):
        with self._lock:
            return self._v
