import builtins
import keyword

import pytest

from basilisp.lang.util import _MUNGE_REPLACEMENTS, demunge, munge


def test_demunge():
    for v, munged in _MUNGE_REPLACEMENTS.items():
        assert demunge(munged) == v

    assert "-->--" == demunge("____GT____")
    assert "--init--" == demunge("__init__")
    assert "random--V--" == demunge("random__V__")
    assert "hi-how-are-you?" == demunge("hi_how_are_you__Q__")
    assert "hi-how-are-you----" == demunge("hi_how_are_you____")


@pytest.mark.parametrize(
    "expected,input",
    [
        ("__PLUS__", "+"),
        ("_", "-"),
        ("__STAR__ns__STAR__", "*ns*"),
        ("__DIV__", "/"),
        ("__LT__", "<"),
        ("__GT__", ">"),
        ("send__BANG__", "send!"),
        ("__EQ__", "="),
        ("string__Q__", "string?"),
        ("__IDIV__", "\\"),
        ("__AMP__form", "&form"),
    ],
)
def test_munge_disallows_syms(expected, input):
    assert expected == munge(input)


def test_munge_disallows_python_builtins():
    for name in builtins.__dict__.keys():
        assert f"{name}_" == munge(name)


def test_munge_disallows_python_kws():
    for kw in keyword.kwlist:
        assert f"{kw}_" == munge(kw)
