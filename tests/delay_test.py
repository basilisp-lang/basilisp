import basilisp.lang.delay as delay
import basilisp.lang.vector as vec


def test_delay(capsys):
    d = delay.Delay(lambda: vec.v(print("In Delay Now"), 1, 2, 3))

    assert vec.v(None, 1, 2, 3) == d.deref()
    captured = capsys.readouterr()
    assert "In Delay Now\n" == captured.out

    assert vec.v(None, 1, 2, 3) == d.deref()
    captured = capsys.readouterr()
    assert "" == captured.out
