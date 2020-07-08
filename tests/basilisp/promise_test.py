import threading

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
