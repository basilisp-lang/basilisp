from basilisp.lang.util import _MUNGE_REPLACEMENTS, demunge


def test_demunge():
    for v, munged in _MUNGE_REPLACEMENTS.items():
        assert demunge(munged) == v

    assert "-->--" == demunge("____GT____")
    assert "--init--" == demunge("__init__")
    assert "random--V--" == demunge("random__V__")
    assert "hi-how-are-you?" == demunge("hi_how_are_you__Q__")
    assert "hi-how-are-you----" == demunge("hi_how_are_you____")
