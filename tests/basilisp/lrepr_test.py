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
