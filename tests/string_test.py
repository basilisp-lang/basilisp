import re

import basilisp.lang.vector as vec
from basilisp.main import init

init()

import basilisp.string as lstr


def test_alpha__Q__():
    assert False is lstr.alpha__Q__("")
    assert False is lstr.alpha__Q__("?")
    assert False is lstr.alpha__Q__("1")
    assert True is lstr.alpha__Q__("abcdef")


def test_alphanumeric__Q__():
    assert False is lstr.alphanumeric__Q__("")
    assert False is lstr.alphanumeric__Q__("?")
    assert True is lstr.alphanumeric__Q__("1")
    assert True is lstr.alphanumeric__Q__("abcdef")


def test_digits__Q__():
    assert False is lstr.digits__Q__("")
    assert False is lstr.digits__Q__("?")
    assert False is lstr.digits__Q__("abcdef")
    assert True is lstr.digits__Q__("1")
    assert True is lstr.digits__Q__("1375234723984")


def test_blank__Q__():
    assert None is lstr.blank__Q__(None)
    assert True is lstr.blank__Q__("")
    assert True is lstr.blank__Q__("                    ")
    assert True is lstr.blank__Q__("\n\r")
    assert False is lstr.blank__Q__("rn")


def test_capitalize():
    assert "" == lstr.capitalize("")
    assert "Chris" == lstr.capitalize("chris")
    assert "Chris" == lstr.capitalize("Chris")
    assert "Chris" == lstr.capitalize("CHRIS")


def test_title_case():
    assert "" == lstr.title_case("")
    assert "Chris" == lstr.title_case("chris")
    assert "Chris" == lstr.title_case("Chris")
    assert "Chris" == lstr.title_case("CHRIS")
    assert "Chris Crink" == lstr.title_case("chris crink")
    assert "Chris Crink" == lstr.title_case("chris Crink")
    assert "Chris Crink" == lstr.title_case("chris CRINK")


def test_lower_case():
    assert "" == lstr.lower_case("")
    assert "chris" == lstr.lower_case("chris")
    assert "chris" == lstr.lower_case("Chris")
    assert "chris" == lstr.lower_case("CHRIS")


def test_upper_case():
    assert "" == lstr.upper_case("")
    assert "CHRIS" == lstr.upper_case("chris")
    assert "CHRIS" == lstr.upper_case("Chris")
    assert "CHRIS" == lstr.upper_case("CHRIS")


def test_ends_with__Q__():
    assert False is lstr.ends_with__Q__("", "something")
    assert True is lstr.ends_with__Q__("Chris", "hris")
    assert False is lstr.ends_with__Q__("Chris", "ohn")


def test_starts_with__Q__():
    assert False is lstr.starts_with__Q__("", "something")
    assert True is lstr.starts_with__Q__("Chris", "Chri")
    assert False is lstr.starts_with__Q__("Chris", "Joh")


def test_includes__Q__():
    assert False is lstr.includes__Q__("", "something")
    assert True is lstr.includes__Q__("Chris", "hri")
    assert False is lstr.includes__Q__("Chris", "oh")


def test_index_of():
    assert None is lstr.index_of("", "hi")
    assert 1 == lstr.index_of("Chris", "hri")
    assert 3 == lstr.index_of("Chris", "is")
    assert None is lstr.index_of("Chris", "oh")

    assert 10 == lstr.index_of("Chris is thrice my favorite", "hri", 5)
    assert None is lstr.index_of("Chris is thrice my favorite", "hri", 15)
    assert 6 == lstr.index_of("Chris is thrice my favorite", "is", 5)
    assert None is lstr.index_of("Chris is thrice my favorite", "is", 15)


def test_last_index_of():
    assert None is lstr.last_index_of("", "hi")
    assert 1 == lstr.last_index_of("Chris", "hri")
    assert 3 == lstr.last_index_of("Chris", "is")
    assert None is lstr.last_index_of("Chris", "oh")

    assert 1 == lstr.last_index_of("Chris is thrice my favorite", "hri", 5)
    assert None is lstr.last_index_of("Chris is thrice my favorite", "hri", 1)
    assert 3 == lstr.last_index_of("Chris is thrice my favorite", "is", 5)
    assert None is lstr.last_index_of("Chris is thrice my favorite", "is", 3)


def test_join():
    assert "" == lstr.join(vec.Vector.empty())
    assert "hi" == lstr.join(vec.v("hi"))
    assert "123" == lstr.join(vec.v(1, 2, 3))
    assert "1,2,3" == lstr.join(",", vec.v(1, 2, 3))


def test_reverse():
    assert "" == lstr.reverse("")
    assert "ih" == lstr.reverse("hi")
    assert "hi there" == lstr.reverse("ereht ih")


def test_split():
    assert vec.v("Basilisp", "is", "awesome!") == lstr.split("Basilisp is awesome!", re.compile(" "))
    assert vec.v("q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "") == lstr.split(
        "q1w2e3r4t5y6u7i8o9p0", re.compile("\d+"))
    assert vec.v("q", "w", "e", "r", "t5y6u7i8o9p0") == lstr.split(
        "q1w2e3r4t5y6u7i8o9p0", re.compile("\d+"), 5)

    assert vec.v(" ", "q", "1", "w", "2", " ") == lstr.split(" q1w2 ", re.compile(""))
    assert vec.v(" ", "q", "1", "w", "2", " ") == lstr.split(" q1w2 ", "")

    assert vec.v("a") == lstr.split("a", re.compile("b"))
    assert vec.v("a") == lstr.split("a", "b")

    assert vec.v("") == lstr.split("", re.compile("b"))
    assert vec.v("") == lstr.split("", "b")

    assert vec.Vector.empty() == lstr.split("", re.compile(""))
    assert vec.Vector.empty() == lstr.split("", "")

    assert vec.v("", "") == lstr.split("a", re.compile("a"))
    assert vec.v("", "") == lstr.split("a", "a")


def test_split_lines():
    assert vec.Vector.empty() == lstr.split_lines("")
    assert vec.v("Hello, my name is Chris.") == lstr.split_lines("Hello, my name is Chris.")
    assert vec.v("Hello,", " my name is Chris.") == lstr.split_lines("Hello,\n my name is Chris.")
    assert vec.v("Hello,", " my name ", " is Chris.") == lstr.split_lines("Hello,\n my name \r is Chris.")
