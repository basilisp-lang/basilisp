import math
import re
import uuid
from fractions import Fraction

import dateutil.parser as dateparser
import pytest

import basilisp.lang.keyword as kw
from tests.basilisp.helpers import CompileFn


@pytest.fixture
def test_ns() -> str:
    return "basilisp.lrepr-test"


@pytest.fixture
def compiler_file_path() -> str:
    return "lrepr_test"


@pytest.mark.parametrize(
    "s,code", [("1.6M", "(binding [*print-dup* true] (pr-str 1.6M))"),]
)
def test_print_dup(lcompile: CompileFn, s: str, code: str):
    assert s == lcompile(code)


@pytest.mark.parametrize(
    "s,code",
    [
        ("[]", "(binding [*print-length* 0] (pr-str []))"),
        ("[]", "(binding [*print-length* nil] (pr-str []))"),
        ("[]", "(binding [*print-length* 10] (pr-str []))"),
        ("[...]", "(binding [*print-length* 0] (pr-str [1 2 3]))"),
        ("[1 2 3]", "(binding [*print-length* nil] (pr-str [1 2 3]))"),
        ("[1 2 ...]", "(binding [*print-length* 2] (pr-str [1 2 3]))"),
        ("[1 2 3]", "(binding [*print-length* 10] (pr-str [1 2 3]))"),
        ("{}", "(binding [*print-length* 0] (pr-str {}))"),
        ("{}", "(binding [*print-length* nil] (pr-str {}))"),
        ("{}", "(binding [*print-length* 10] (pr-str {}))"),
        ("{...}", "(binding [*print-length* 0] (pr-str {:a 1 :b 2 :c 3}))"),
    ],
)
def test_print_length(lcompile: CompileFn, s: str, code: str):
    assert s == lcompile(code)


def test_print_length_maps(lcompile: CompileFn):
    assert lcompile("(binding [*print-length* 1] (pr-str {:a 1 :b 2}))") in {
        "{:a 1 ...}",
        "{:b 2 ...}",
    }


@pytest.mark.parametrize(
    "s,code",
    [
        ("[1 2 3]", "(binding [*print-level* 1] (pr-str [1 2 3]))"),
        ("[#]", "(binding [*print-level* 1] (pr-str [[1 2 3]]))"),
        ("[#]", "(binding [*print-level* 1] (pr-str [[[1 2 3]]]))"),
        ("[[#]]", "(binding [*print-level* 2] (pr-str [[[1 2 3]]]))"),
        ("[[[1 2 3]]]", "(binding [*print-level* 3] (pr-str [[[1 2 3]]]))"),
        ("[[[1 2 3]]]", "(binding [*print-level* 4] (pr-str [[[1 2 3]]]))"),
        ("[[[1 2 3]]]", "(binding [*print-level* nil] (pr-str [[[1 2 3]]]))"),
        ("#py [1 2 3]", "(binding [*print-level* 1] (pr-str #py [1 2 3]))"),
        ("#py [#]", "(binding [*print-level* 1] (pr-str #py [[1 2 3]]))"),
        ("#py [#]", "(binding [*print-level* 1] (pr-str #py [[[1 2 3]]]))"),
        ("#py [[#]]", "(binding [*print-level* 2] (pr-str #py [[[1 2 3]]]))"),
        ("#py [[[1 2 3]]]", "(binding [*print-level* 3] (pr-str #py [[[1 2 3]]]))"),
        ("#py [[[1 2 3]]]", "(binding [*print-level* 4] (pr-str #py [[[1 2 3]]]))"),
        ("#py [[[1 2 3]]]", "(binding [*print-level* nil] (pr-str #py [[[1 2 3]]]))"),
        ("{:a #}", "(binding [*print-level* 1] (pr-str {:a {:b {:c :d}}}))"),
        ("{:a {:b #}}", "(binding [*print-level* 2] (pr-str {:a {:b {:c :d}}}))"),
        ("{:a {:b {:c :d}}}", "(binding [*print-level* 3] (pr-str {:a {:b {:c :d}}}))"),
        ("{:a {:b {:c :d}}}", "(binding [*print-level* 4] (pr-str {:a {:b {:c :d}}}))"),
        (
            "{:a {:b {:c :d}}}",
            "(binding [*print-level* nil] (pr-str {:a {:b {:c :d}}}))",
        ),
        ("#py {:a #}", "(binding [*print-level* 1] (pr-str #py {:a {:b {:c :d}}}))"),
        (
            "#py {:a {:b #}}",
            "(binding [*print-level* 2] (pr-str #py {:a {:b {:c :d}}}))",
        ),
        (
            "#py {:a {:b {:c :d}}}",
            "(binding [*print-level* 3] (pr-str #py {:a {:b {:c :d}}}))",
        ),
        (
            "#py {:a {:b {:c :d}}}",
            "(binding [*print-level* 4] (pr-str #py {:a {:b {:c :d}}}))",
        ),
        (
            "#py {:a {:b {:c :d}}}",
            "(binding [*print-level* nil] (pr-str #py {:a {:b {:c :d}}}))",
        ),
    ],
)
def test_print_level(lcompile: CompileFn, s: str, code: str):
    assert s == lcompile(code)


@pytest.mark.parametrize(
    "s,code",
    [
        ("s", "(binding [*print-meta* true] (pr-str 's))"),
        ("ns/s", "(binding [*print-meta* true] (pr-str 'ns/s))"),
        ("[]", "(binding [*print-meta* true] (pr-str []))"),
        ("[:a :b :c]", "(binding [*print-meta* true] (pr-str [:a :b :c]))"),
        ("()", "(binding [*print-meta* true] (pr-str '()))"),
        ("(:a :b :c)", "(binding [*print-meta* true] (pr-str '(:a :b :c)))"),
        ("#{}", "(binding [*print-meta* true] (pr-str #{}))"),
        ("#{:a}", "(binding [*print-meta* true] (pr-str #{:a}))"),
        ("{}", "(binding [*print-meta* true] (pr-str {}))"),
        ("{:a 1}", "(binding [*print-meta* true] (pr-str {:a 1}))"),
        ("^{:redef true} s", "(binding [*print-meta* true] (pr-str '^:redef s))"),
        ("^{:redef true} ns/s", "(binding [*print-meta* true] (pr-str '^:redef ns/s))"),
        ("^{:empty true} []", "(binding [*print-meta* true] (pr-str ^:empty []))"),
        (
            "^{:empty false} [:a :b :c]",
            "(binding [*print-meta* true] (pr-str ^{:empty false} [:a :b :c]))",
        ),
        ("^{:empty true} ()", "(binding [*print-meta* true] (pr-str '^:empty ()))"),
        (
            "^{:empty false} (:a :b :c)",
            "(binding [*print-meta* true] (pr-str '^{:empty false}(:a :b :c)))",
        ),
        ("^{:empty true} #{}", "(binding [*print-meta* true] (pr-str ^:empty #{}))"),
        (
            "^{:empty false} #{:a}",
            "(binding [*print-meta* true] (pr-str ^{:empty false} #{:a}))",
        ),
        ("^{:empty true} {}", "(binding [*print-meta* true] (pr-str ^:empty {}))"),
        (
            "^{:empty false} {:a 1}",
            "(binding [*print-meta* true] (pr-str ^{:empty false} {:a 1}))",
        ),
    ],
)
def test_print_meta(lcompile: CompileFn, s: str, code: str):
    assert s == lcompile(code)


def test_print_readably(lcompile: CompileFn):
    assert '"Hello\nworld!"' == lcompile(
        '(binding [*print-readably* false] (pr-str "Hello\\nworld!"))'
    )


@pytest.mark.parametrize(
    "repr,code",
    [
        ("true", "(pr-str true)"),
        ("false", "(pr-str false)"),
        ("nil", "(pr-str nil)"),
        ("4J", "(pr-str 4J)"),
        ("37.8J", "(pr-str 37.8J)"),
        ("8837", "(pr-str 8837)"),
        ("0.64", "(pr-str 0.64)"),
        ("3.14", "(pr-str 3.14M)"),
        ("22/7", "(pr-str 22/7)"),
        ("##NaN", "(pr-str ##NaN)"),
        ("##Inf", "(pr-str ##Inf)"),
        ("##-Inf", "(pr-str ##-Inf)"),
        ('"hi"', '(pr-str "hi")'),
        ('"Hello\\nworld!"', '(pr-str "Hello\nworld!")'),
        (
            '#uuid "81f35603-0408-4b3d-bbc0-462e3702747f"',
            '(pr-str #uuid "81f35603-0408-4b3d-bbc0-462e3702747f")',
        ),
        ('#"\\s"', '(pr-str #"\\s")'),
        (
            '#inst "2018-11-28T12:43:25.477000+00:00"',
            '(pr-str #inst "2018-11-28T12:43:25.477-00:00")',
        ),
        ("#py {}", "(pr-str #py {})"),
        ("#py {:a 1}", "(pr-str #py {:a 1})"),
        ("#py []", "(pr-str #py [])"),
        ('#py [:a 1 "s"]', '(pr-str #py [:a 1 "s"])'),
        ("#py #{}", "(pr-str #py #{})"),
        ("#py #{:a}", "(pr-str #py #{:a})"),
        ("#py ()", "(pr-str #py ())"),
        ('#py (:a 1 "s")', '(pr-str #py (:a 1 "s"))'),
    ],
)
def test_lrepr(lcompile: CompileFn, repr: str, code: str):
    assert repr == lcompile(code)


@pytest.mark.parametrize(
    "o,code",
    [
        (4j, "(read-string (pr-str 4J))"),
        (37.8j, "(read-string (pr-str 37.8J))"),
        (8837, "(read-string (pr-str 8837))"),
        (0.64, "(read-string (pr-str 0.64))"),
        (3.14, "(read-string (pr-str 3.14M))"),
        (Fraction(22, 7), "(read-string (pr-str 22/7))"),
        (float("inf"), "(read-string (pr-str ##Inf))"),
        (-float("inf"), "(read-string (pr-str ##-Inf))"),
        ("hi", '(read-string (pr-str "hi"))'),
        ("Hello\nworld!", '(read-string (pr-str "Hello\nworld!"))'),
        (
            uuid.UUID("81f35603-0408-4b3d-bbc0-462e3702747f"),
            '(read-string (pr-str #uuid "81f35603-0408-4b3d-bbc0-462e3702747f"))',
        ),
        (re.compile(r"\s"), '(read-string (pr-str #"\\s"))'),
        (
            dateparser.parse("2018-11-28T12:43:25.477000+00:00"),
            '(read-string (pr-str #inst "2018-11-28T12:43:25.477-00:00"))',
        ),
        ({}, "(read-string (pr-str #py {}))"),
        ({kw.keyword("a"): 1}, "(read-string (pr-str #py {:a 1}))"),
        ([], "(read-string (pr-str #py []))"),
        ([kw.keyword("a"), 1, "s"], '(read-string (pr-str #py [:a 1 "s"]))'),
        (set(), "(read-string (pr-str #py #{}))"),
        ({kw.keyword("a"), 1, "a"}, '(read-string (pr-str #py #{:a 1 "a"}))'),
        ((), "(read-string (pr-str #py ()))"),
        ((kw.keyword("a"), 1, "s"), '(read-string (pr-str #py (:a 1 "s")))'),
    ],
)
def test_lrepr_round_trip(lcompile: CompileFn, o, code: str):
    assert o == lcompile(code)


def test_lrepr_round_trip_special_cases(lcompile: CompileFn):
    assert True is lcompile("(read-string (pr-str true))")
    assert False is lcompile("(read-string (pr-str false))")
    assert None is lcompile("(read-string (pr-str nil))")
    assert math.isnan(lcompile("(read-string (pr-str ##NaN))"))


@pytest.mark.parametrize(
    "s,code",
    [
        ("true", "(print-str true)"),
        ("false", "(print-str false)"),
        ("nil", "(print-str nil)"),
        ("4J", "(print-str 4J)"),
        ("37.8J", "(print-str 37.8J)"),
        ("37.8J", "(print-str 37.8J)"),
        ("8837", "(print-str 8837)"),
        ("0.64", "(print-str 0.64)"),
        ("3.14", "(print-str 3.14M)"),
        ("22/7", "(print-str 22/7)"),
        ("##NaN", "(print-str ##NaN)"),
        ("##Inf", "(print-str ##Inf)"),
        ("##-Inf", "(print-str ##-Inf)"),
        ("hi", '(print-str "hi")'),
        ("Hello\nworld!", '(print-str "Hello\nworld!")'),
        (
            '#uuid "81f35603-0408-4b3d-bbc0-462e3702747f"',
            '(print-str #uuid "81f35603-0408-4b3d-bbc0-462e3702747f")',
        ),
        ('#"\\s"', '(print-str #"\\s")'),
        (
            '#inst "2018-11-28T12:43:25.477000+00:00"',
            '(print-str #inst "2018-11-28T12:43:25.477-00:00")',
        ),
        ("#py []", "(print-str #py [])"),
        ("#py ()", "(print-str #py ())"),
        ("#py {}", "(print-str #py {})"),
        ("#py #{}", "(print-str #py #{})"),
    ],
)
def test_lstr(lcompile: CompileFn, s: str, code: str):
    assert s == lcompile(code)
