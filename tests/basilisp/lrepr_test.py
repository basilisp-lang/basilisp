import re
import uuid
from fractions import Fraction
from typing import Any, Callable

import dateutil.parser as dateparser
import pytest

import basilisp.lang.compiler as compiler
import basilisp.lang.keyword as kw
import basilisp.lang.reader as reader
import basilisp.lang.runtime as runtime
import basilisp.lang.symbol as sym
from tests.basilisp.helpers import get_or_create_ns

COMPILER_FILE_PATH = "lrepr_test"


@pytest.fixture
def test_ns() -> str:
    return "basilisp.lrepr-test"


@pytest.fixture
def test_ns_sym(test_ns: str) -> sym.Symbol:
    return sym.symbol(test_ns)


CompileFn = Callable[[str], Any]


@pytest.fixture
def lcompile(test_ns: str, test_ns_sym: sym.Symbol) -> CompileFn:
    get_or_create_ns(test_ns_sym)

    with runtime.ns_bindings(test_ns) as ns:

        def _lcompile(s: str,):
            """Compile and execute the code in the input string.

            Return the resulting expression."""
            ctx = compiler.CompilerContext(COMPILER_FILE_PATH)

            last = None
            for form in reader.read_str(s):
                last = compiler.compile_and_exec_form(form, ctx, ns)

            return last

        try:
            yield _lcompile
        finally:
            runtime.Namespace.remove(test_ns_sym)


def test_print_dup(lcompile: CompileFn):
    assert "1.6M" == lcompile("(binding [*print-dup* true] (pr-str 1.6M))")


def test_print_length(lcompile: CompileFn):
    assert "[]" == lcompile("(binding [*print-length* 0] (pr-str []))")
    assert "[]" == lcompile("(binding [*print-length* nil] (pr-str []))")
    assert "[]" == lcompile("(binding [*print-length* 10] (pr-str []))")
    assert "[...]" == lcompile("(binding [*print-length* 0] (pr-str [1 2 3]))")
    assert "[1 2 3]" == lcompile("(binding [*print-length* nil] (pr-str [1 2 3]))")
    assert "[1 2 ...]" == lcompile("(binding [*print-length* 2] (pr-str [1 2 3]))")
    assert "[1 2 3]" == lcompile("(binding [*print-length* 10] (pr-str [1 2 3]))")

    assert "{}" == lcompile("(binding [*print-length* 0] (pr-str {}))")
    assert "{}" == lcompile("(binding [*print-length* nil] (pr-str {}))")
    assert "{}" == lcompile("(binding [*print-length* 10] (pr-str {}))")
    assert "{...}" == lcompile("(binding [*print-length* 0] (pr-str {:a 1 :b 2 :c 3}))")
    assert lcompile("(binding [*print-length* 1] (pr-str {:a 1 :b 2}))") in {
        "{:a 1 ...}",
        "{:b 2 ...}",
    }


def test_print_level(lcompile: CompileFn):
    assert "[1 2 3]" == lcompile("(binding [*print-level* 1] (pr-str [1 2 3]))")
    assert "[#]" == lcompile("(binding [*print-level* 1] (pr-str [[1 2 3]]))")
    assert "[#]" == lcompile("(binding [*print-level* 1] (pr-str [[[1 2 3]]]))")
    assert "[[#]]" == lcompile("(binding [*print-level* 2] (pr-str [[[1 2 3]]]))")
    assert "[[[1 2 3]]]" == lcompile("(binding [*print-level* 3] (pr-str [[[1 2 3]]]))")
    assert "[[[1 2 3]]]" == lcompile("(binding [*print-level* 4] (pr-str [[[1 2 3]]]))")
    assert "[[[1 2 3]]]" == lcompile(
        "(binding [*print-level* nil] (pr-str [[[1 2 3]]]))"
    )

    assert "#py [1 2 3]" == lcompile("(binding [*print-level* 1] (pr-str #py [1 2 3]))")
    assert "#py [#]" == lcompile("(binding [*print-level* 1] (pr-str #py [[1 2 3]]))")
    assert "#py [#]" == lcompile("(binding [*print-level* 1] (pr-str #py [[[1 2 3]]]))")
    assert "#py [[#]]" == lcompile(
        "(binding [*print-level* 2] (pr-str #py [[[1 2 3]]]))"
    )
    assert "#py [[[1 2 3]]]" == lcompile(
        "(binding [*print-level* 3] (pr-str #py [[[1 2 3]]]))"
    )
    assert "#py [[[1 2 3]]]" == lcompile(
        "(binding [*print-level* 4] (pr-str #py [[[1 2 3]]]))"
    )
    assert "#py [[[1 2 3]]]" == lcompile(
        "(binding [*print-level* nil] (pr-str #py [[[1 2 3]]]))"
    )

    assert "{:a #}" == lcompile(
        "(binding [*print-level* 1] (pr-str {:a {:b {:c :d}}}))"
    )
    assert "{:a {:b #}}" == lcompile(
        "(binding [*print-level* 2] (pr-str {:a {:b {:c :d}}}))"
    )
    assert "{:a {:b {:c :d}}}" == lcompile(
        "(binding [*print-level* 3] (pr-str {:a {:b {:c :d}}}))"
    )
    assert "{:a {:b {:c :d}}}" == lcompile(
        "(binding [*print-level* 4] (pr-str {:a {:b {:c :d}}}))"
    )
    assert "{:a {:b {:c :d}}}" == lcompile(
        "(binding [*print-level* nil] (pr-str {:a {:b {:c :d}}}))"
    )

    assert "#py {:a #}" == lcompile(
        "(binding [*print-level* 1] (pr-str #py {:a {:b {:c :d}}}))"
    )
    assert "#py {:a {:b #}}" == lcompile(
        "(binding [*print-level* 2] (pr-str #py {:a {:b {:c :d}}}))"
    )
    assert "#py {:a {:b {:c :d}}}" == lcompile(
        "(binding [*print-level* 3] (pr-str #py {:a {:b {:c :d}}}))"
    )
    assert "#py {:a {:b {:c :d}}}" == lcompile(
        "(binding [*print-level* 4] (pr-str #py {:a {:b {:c :d}}}))"
    )
    assert "#py {:a {:b {:c :d}}}" == lcompile(
        "(binding [*print-level* nil] (pr-str #py {:a {:b {:c :d}}}))"
    )


def test_print_meta(lcompile: CompileFn):
    assert "s" == lcompile("(binding [*print-meta* true] (pr-str 's))")
    assert "ns/s" == lcompile("(binding [*print-meta* true] (pr-str 'ns/s))")
    assert "[]" == lcompile("(binding [*print-meta* true] (pr-str []))")
    assert "[:a :b :c]" == lcompile("(binding [*print-meta* true] (pr-str [:a :b :c]))")
    assert "()" == lcompile("(binding [*print-meta* true] (pr-str '()))")
    assert "(:a :b :c)" == lcompile(
        "(binding [*print-meta* true] (pr-str '(:a :b :c)))"
    )
    assert "#{}" == lcompile("(binding [*print-meta* true] (pr-str #{}))")
    assert "#{:a}" == lcompile("(binding [*print-meta* true] (pr-str #{:a}))")
    assert "{}" == lcompile("(binding [*print-meta* true] (pr-str {}))")
    assert "{:a 1}" == lcompile("(binding [*print-meta* true] (pr-str {:a 1}))")

    assert "^{:redef true} s" == lcompile(
        "(binding [*print-meta* true] (pr-str '^:redef s))"
    )
    assert "^{:redef true} ns/s" == lcompile(
        "(binding [*print-meta* true] (pr-str '^:redef ns/s))"
    )
    assert "^{:empty true} []" == lcompile(
        "(binding [*print-meta* true] (pr-str ^:empty []))"
    )
    assert "^{:empty false} [:a :b :c]" == lcompile(
        "(binding [*print-meta* true] (pr-str ^{:empty false} [:a :b :c]))"
    )
    assert "^{:empty true} ()" == lcompile(
        "(binding [*print-meta* true] (pr-str '^:empty ()))"
    )
    assert "^{:empty false} (:a :b :c)" == lcompile(
        "(binding [*print-meta* true] (pr-str '^{:empty false}(:a :b :c)))"
    )
    assert "^{:empty true} #{}" == lcompile(
        "(binding [*print-meta* true] (pr-str ^:empty #{}))"
    )
    assert "^{:empty false} #{:a}" == lcompile(
        "(binding [*print-meta* true] (pr-str ^{:empty false} #{:a}))"
    )
    assert "^{:empty true} {}" == lcompile(
        "(binding [*print-meta* true] (pr-str ^:empty {}))"
    )
    assert "^{:empty false} {:a 1}" == lcompile(
        "(binding [*print-meta* true] (pr-str ^{:empty false} {:a 1}))"
    )


def test_print_readably(lcompile: CompileFn):
    assert '"Hello\nworld!"' == lcompile(
        '(binding [*print-readably* false] (pr-str "Hello\\nworld!"))'
    )


def test_lrepr(lcompile: CompileFn):
    assert "true" == lcompile("(pr-str true)")
    assert "false" == lcompile("(pr-str false)")
    assert "nil" == lcompile("(pr-str nil)")
    assert "4J" == lcompile("(pr-str 4J)")
    assert "37.8J" == lcompile("(pr-str 37.8J)")
    assert "8837" == lcompile("(pr-str 8837)")
    assert "0.64" == lcompile("(pr-str 0.64)")
    assert "3.14" == lcompile("(pr-str 3.14M)")
    assert "22/7" == lcompile("(pr-str 22/7)")
    assert '"hi"' == lcompile('(pr-str "hi")')
    assert '"Hello\\nworld!"' == lcompile('(pr-str "Hello\nworld!")')
    assert '#uuid "81f35603-0408-4b3d-bbc0-462e3702747f"' == lcompile(
        '(pr-str #uuid "81f35603-0408-4b3d-bbc0-462e3702747f")'
    )
    assert '#"\\s"' == lcompile('(pr-str #"\\s")')
    assert '#inst "2018-11-28T12:43:25.477000+00:00"' == lcompile(
        '(pr-str #inst "2018-11-28T12:43:25.477-00:00")'
    )

    assert "#py {}" == lcompile("(pr-str #py {})")
    assert "#py {:a 1}" == lcompile("(pr-str #py {:a 1})")

    assert "#py []" == lcompile("(pr-str #py [])")
    assert '#py [:a 1 "s"]' == lcompile('(pr-str #py [:a 1 "s"])')

    assert "#py #{}" == lcompile("(pr-str #py #{})")
    assert "#py #{:a}" == lcompile("(pr-str #py #{:a})")

    assert "#py ()" == lcompile("(pr-str #py ())")
    assert '#py (:a 1 "s")' == lcompile('(pr-str #py (:a 1 "s"))')


def test_lrepr_round_trip(lcompile: CompileFn):
    assert True is lcompile("(read-string (pr-str true))")
    assert False is lcompile("(read-string (pr-str false))")
    assert None is lcompile("(read-string (pr-str nil))")
    assert 4j == lcompile("(read-string (pr-str 4J))")
    assert 37.8j == lcompile("(read-string (pr-str 37.8J))")
    assert 8837 == lcompile("(read-string (pr-str 8837))")
    assert 0.64 == lcompile("(read-string (pr-str 0.64))")
    assert 3.14 == lcompile("(read-string (pr-str 3.14M))")
    assert Fraction(22, 7) == lcompile("(read-string (pr-str 22/7))")
    assert "hi" == lcompile('(read-string (pr-str "hi"))')
    assert "Hello\nworld!" == lcompile('(read-string (pr-str "Hello\nworld!"))')
    assert uuid.UUID("81f35603-0408-4b3d-bbc0-462e3702747f") == lcompile(
        '(read-string (pr-str #uuid "81f35603-0408-4b3d-bbc0-462e3702747f"))'
    )
    assert re.compile(r"\s") == lcompile('(read-string (pr-str #"\\s"))')
    assert dateparser.parse("2018-11-28T12:43:25.477000+00:00") == lcompile(
        '(read-string (pr-str #inst "2018-11-28T12:43:25.477-00:00"))'
    )

    assert {} == lcompile("(read-string (pr-str #py {}))")
    assert {kw.keyword("a"): 1} == lcompile("(read-string (pr-str #py {:a 1}))")

    assert [] == lcompile("(read-string (pr-str #py []))")
    assert [kw.keyword("a"), 1, "s"] == lcompile(
        '(read-string (pr-str #py [:a 1 "s"]))'
    )

    assert set() == lcompile("(read-string (pr-str #py #{}))")
    assert {kw.keyword("a"), 1, "a"} == lcompile(
        '(read-string (pr-str #py #{:a 1 "a"}))'
    )

    assert () == lcompile("(read-string (pr-str #py ()))")
    assert (kw.keyword("a"), 1, "s") == lcompile(
        '(read-string (pr-str #py (:a 1 "s")))'
    )


def test_lstr(lcompile: CompileFn):
    assert "true" == lcompile("(print-str true)")
    assert "false" == lcompile("(print-str false)")
    assert "nil" == lcompile("(print-str nil)")
    assert "4J" == lcompile("(print-str 4J)")
    assert "37.8J" == lcompile("(print-str 37.8J)")
    assert "37.8J" == lcompile("(print-str 37.8J)")
    assert "8837" == lcompile("(print-str 8837)")
    assert "0.64" == lcompile("(print-str 0.64)")
    assert "3.14" == lcompile("(print-str 3.14M)")
    assert "22/7" == lcompile("(print-str 22/7)")
    assert "hi" == lcompile('(print-str "hi")')
    assert "Hello\nworld!" == lcompile('(print-str "Hello\nworld!")')
    assert '#uuid "81f35603-0408-4b3d-bbc0-462e3702747f"' == lcompile(
        '(print-str #uuid "81f35603-0408-4b3d-bbc0-462e3702747f")'
    )
    assert '#"\\s"' == lcompile('(print-str #"\\s")')
    assert '#inst "2018-11-28T12:43:25.477000+00:00"' == lcompile(
        '(print-str #inst "2018-11-28T12:43:25.477-00:00")'
    )
    assert "#py []" == lcompile("(print-str #py [])")
    assert "#py ()" == lcompile("(print-str #py ())")
    assert "#py {}" == lcompile("(print-str #py {})")
    assert "#py #{}" == lcompile("(print-str #py #{})")
