import pytest

from basilisp.util import Maybe


def test_maybe_is_present():
    assert Maybe(None).is_present is False
    assert Maybe("Something").is_present


def test_maybe_value():
    assert Maybe(None).value is None
    assert Maybe("Something").value == "Something"


def test_maybe_equals():
    assert Maybe("Something") == "Something"
    assert Maybe("Something") == Maybe("Something")
    assert Maybe(None) == Maybe(None)
    assert Maybe(None) != Maybe("Something")


def test_maybe_or_else():
    assert Maybe(None).or_else(lambda: "Not None") == "Not None"
    assert Maybe("Something").or_else(lambda: "Nothing") == "Something"


def test_maybe_or_else_get():
    assert Maybe(None).or_else_get("Not None") == "Not None"
    assert Maybe("Something").or_else_get("Nothing") == "Something"


def test_maybe_or_else_raise():
    with pytest.raises(ValueError):
        assert Maybe(None).or_else_raise(lambda: ValueError("No value"))

    assert (
        Maybe("Something").or_else_raise(lambda: ValueError("No value")) == "Something"
    )


def test_maybe_map():
    l = Maybe("lower")
    lmapped = l.map(lambda s: s.upper())
    assert lmapped == Maybe("LOWER")
    assert lmapped == "LOWER"
    assert lmapped.or_else_get("Nothing") == "LOWER"
    assert lmapped.or_else(lambda: "Nothing") == "LOWER"
    assert lmapped.or_else_raise(lambda: ValueError("Nothing!")) == "LOWER"


def test_maybe_stream():
    assert len(Maybe(None).stream().map(lambda v: v.upper()).to_list()) == 0
    m = Maybe("lower").stream().map(lambda v: v.upper()).to_list()
    assert len(m) == 1
    assert m[0] == "LOWER"
