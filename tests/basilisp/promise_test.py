import threading
import time

from basilisp.lang import vector as vec
from basilisp.lang.promise import Promise


def test_promise():
    v = vec.v(1, 2, 3)

    p = Promise()
    assert False is p.is_realized

    def set_promise():
        p.deliver(v)

    assert "not set yet" == p.deref(timeout=0.2, timeout_val="not set yet")
    assert False is p.is_realized

    t = threading.Thread(target=set_promise)
    t.start()
    t.join()

    assert v == p.deref()
    assert True is p.is_realized

    p.deliver("another value")

    assert v == p.deref()
    assert True is p.is_realized


def test_promise_timeout():
    """Tests that a promise can be delivered in another thread before the timeout expires."""
    v = vec.v(1, 2, 3)

    p = Promise()

    def set_promise():
        # Short sleep to allow the main thread to resume before
        # delivering the promise.
        time.sleep(0.1)
        p.deliver(v)

    t = threading.Thread(target=set_promise, daemon=True)
    t.start()

    # Dereferencing the promise should be near instantaneous.
    start = time.time()
    dv = p.deref(timeout=2)
    assert (time.time() - start) < 1

    assert v == dv
    assert True is p.is_realized

    t.join()
