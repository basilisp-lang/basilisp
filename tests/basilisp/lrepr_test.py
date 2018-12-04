import basilisp.lang.compiler as compiler
import basilisp.lang.reader as reader
import basilisp.lang.runtime as runtime


def lcompile(s: str):
    """Compile and execute the code in the input string.

    Return the resulting expression."""
    ctx = compiler.CompilerContext()
    mod = runtime._new_module("lrepr_test")

    last = None
    for form in reader.read_str(s):
        last = compiler.compile_and_exec_form(form, ctx, mod)

    return last


def test_print_length():
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


def test_print_level():
    assert "[1 2 3]" == lcompile("(binding [*print-level* 1] (pr-str [1 2 3]))")
    assert "[#]" == lcompile("(binding [*print-level* 1] (pr-str [[1 2 3]]))")
    assert "[#]" == lcompile("(binding [*print-level* 1] (pr-str [[[1 2 3]]]))")
    assert "[[#]]" == lcompile("(binding [*print-level* 2] (pr-str [[[1 2 3]]]))")
    assert "[[[1 2 3]]]" == lcompile("(binding [*print-level* 3] (pr-str [[[1 2 3]]]))")
    assert "[[[1 2 3]]]" == lcompile("(binding [*print-level* 4] (pr-str [[[1 2 3]]]))")
    assert "[[[1 2 3]]]" == lcompile(
        "(binding [*print-level* nil] (pr-str [[[1 2 3]]]))"
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


def test_print_meta():
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


def test_lrepr():
    assert "true" == lcompile("(pr-str true)")
    assert "false" == lcompile("(pr-str false)")
    assert "nil" == lcompile("(pr-str nil)")
    assert "4J" == lcompile("(pr-str 4J)")
    assert "37.8J" == lcompile("(pr-str 37.8J)")
    assert "37.8J" == lcompile("(pr-str 37.8J)")
    assert "8837" == lcompile("(pr-str 8837)")
    assert "0.64" == lcompile("(pr-str 0.64)")
    assert "3.14" == lcompile("(pr-str 3.14M)")
    assert "22/7" == lcompile("(pr-str 22/7)")
    assert '"hi"' == lcompile('(pr-str "hi")')
    assert '#uuid "81f35603-0408-4b3d-bbc0-462e3702747f"' == lcompile(
        '(pr-str #uuid "81f35603-0408-4b3d-bbc0-462e3702747f")'
    )
    assert '#"\\s"' == lcompile('(pr-str #"\\s")')
    assert '#inst "2018-11-28T12:43:25.477000+00:00"' == lcompile(
        '(pr-str #inst "2018-11-28T12:43:25.477-00:00")'
    )


def test_lstr():
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
    assert '#uuid "81f35603-0408-4b3d-bbc0-462e3702747f"' == lcompile(
        '(print-str #uuid "81f35603-0408-4b3d-bbc0-462e3702747f")'
    )
    assert '#"\\s"' == lcompile('(print-str #"\\s")')
    assert '#inst "2018-11-28T12:43:25.477000+00:00"' == lcompile(
        '(print-str #inst "2018-11-28T12:43:25.477-00:00")'
    )
