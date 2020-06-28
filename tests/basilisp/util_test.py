import pytest

from basilisp.util import Maybe, partition


class TestMaybe:
    def test_maybe_is_present(self):
        assert Maybe(None).is_present is False
        assert Maybe("Something").is_present

    def test_maybe_value(self):
        assert Maybe(None).value is None
        assert Maybe("Something").value == "Something"

    def test_maybe_equals(self):
        assert Maybe("Something") == "Something"
        assert Maybe("Something") == Maybe("Something")
        assert Maybe(None) == Maybe(None)
        assert Maybe(None) != Maybe("Something")

    def test_maybe_or_else(self):
        assert Maybe(None).or_else(lambda: "Not None") == "Not None"
        assert Maybe("Something").or_else(lambda: "Nothing") == "Something"

    def test_maybe_or_else_get(self):
        assert Maybe(None).or_else_get("Not None") == "Not None"
        assert Maybe("Something").or_else_get("Nothing") == "Something"

    def test_maybe_or_else_raise(self):
        with pytest.raises(ValueError):
            assert Maybe(None).or_else_raise(lambda: ValueError("No value"))

        assert (
            Maybe("Something").or_else_raise(lambda: ValueError("No value"))
            == "Something"
        )

    def test_maybe_map(self):
        l = Maybe("lower")
        lmapped = l.map(lambda s: s.upper())
        assert lmapped == Maybe("LOWER")
        assert lmapped == "LOWER"
        assert lmapped.or_else_get("Nothing") == "LOWER"
        assert lmapped.or_else(lambda: "Nothing") == "LOWER"
        assert lmapped.or_else_raise(lambda: ValueError("Nothing!")) == "LOWER"


def test_partition():
    assert [(1, 2, 3, 4)] == list(partition([1, 2, 3, 4], 5))
    assert [(1, 2, 3, 4)] == list(partition([1, 2, 3, 4], 4))
    assert [(1, 2, 3), (4,)] == list(partition([1, 2, 3, 4], 3))
    assert [(1, 2), (3, 4)] == list(partition([1, 2, 3, 4], 2))
    assert [(1,), (2,), (3,), (4,)] == list(partition([1, 2, 3, 4], 1))
