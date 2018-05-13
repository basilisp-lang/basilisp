import dateutil.parser as dateparser
import re
import uuid
import pytest
import basilisp.compiler as compiler
import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.namespace as namespace
import basilisp.lang.set as lset
import basilisp.lang.runtime as runtime
import basilisp.lang.symbol as sym
import basilisp.lang.var as var
import basilisp.lang.vector as vec


@pytest.fixture
def test_ns() -> str:
    return "test"


@pytest.fixture
def ns_var(test_ns: str):
    runtime.init_ns_var(which_ns=runtime._CORE_NS)
    yield runtime.set_current_ns(test_ns)
    namespace.remove(sym.symbol(runtime._CORE_NS))


def lcompile(s: str):
    """Compile and execute the code in the input string.

    Return the resulting expression."""
    ast = compiler.compile_str(s)
    return compiler.exec_ast(ast)


def test_string():
    assert lcompile('"some string"') == "some string"
    assert lcompile('""') == ""


def test_int():
    assert lcompile('1') == 1
    assert lcompile('100') == 100
    assert lcompile('99927273') == 99927273
    assert lcompile('0') == 0
    assert lcompile('-1') == -1
    assert lcompile('-538282') == -538282


def test_float():
    assert lcompile('0.0') == 0.0
    assert lcompile('0.09387372') == 0.09387372
    assert lcompile('1.0') == 1.0
    assert lcompile('1.332') == 1.332
    assert lcompile('-1.332') == -1.332
    assert lcompile('-1.0') == -1.0
    assert lcompile('-0.332') == -0.332


def test_kw():
    assert lcompile(":kw") == kw.keyword("kw")
    assert lcompile(":ns/kw") == kw.keyword("kw", ns="ns")
    assert lcompile(":qualified.ns/kw") == kw.keyword("kw", ns="qualified.ns")


def test_literals():
    assert lcompile("nil") is None
    assert lcompile("true") is True
    assert lcompile("false") is False


def test_quoted_symbol():
    assert lcompile("'sym") == sym.symbol('sym')
    assert lcompile("'ns/sym") == sym.symbol('sym', ns='ns')
    assert lcompile("'qualified.ns/sym") == sym.symbol(
        'sym', ns='qualified.ns')


def test_map():
    assert lcompile("{}") == lmap.m()
    assert lcompile('{:a "string"}') == lmap.map({kw.keyword("a"): "string"})
    assert lcompile('{:a "string" 45 :my-age}') == lmap.map({
        kw.keyword("a"):
        "string",
        45:
        kw.keyword("my-age")
    })


def test_set():
    assert lcompile("#{}") == lset.s()
    assert lcompile("#{:a}") == lset.s(kw.keyword("a"))
    assert lcompile("#{:a 1}") == lset.s(kw.keyword("a"), 1)


def test_vec():
    assert lcompile("[]") == vec.v()
    assert lcompile("[:a]") == vec.v(kw.keyword("a"))
    assert lcompile("[:a 1]") == vec.v(kw.keyword("a"), 1)


def test_def(ns_var: var.Var):
    ns_name = ns_var.value.name
    assert lcompile("(def a :some-val)") == var.find_in_ns(
        sym.symbol(ns_name), sym.symbol('a'))
    assert lcompile('(def beep "a sound a robot makes")') == var.find_in_ns(
        sym.symbol(ns_name), sym.symbol('beep'))
    assert lcompile("a") == kw.keyword("some-val")
    assert lcompile("beep") == "a sound a robot makes"


def test_do(ns_var: var.Var):
    code = """
    (do
      (def first-name :Darth)
      (def last-name "Vader"))
    """
    ns_name = ns_var.value.name
    assert lcompile(code) == var.find_in_ns(
        sym.symbol(ns_name), sym.symbol('last-name'))
    assert lcompile("first-name") == kw.keyword("Darth")
    assert lcompile("last-name") == "Vader"


def test_fn(ns_var: var.Var):
    code = """
    (def string-upper (fn* [s] (.upper s)))
    """
    ns_name = ns_var.value.name
    fvar = lcompile(code)
    assert fvar == var.find_in_ns(
        sym.symbol(ns_name), sym.symbol('string-upper'))
    assert callable(fvar.value)
    assert fvar.value("lower") == "LOWER"

    code = """
    (def string-lower #(.lower %))
    """
    ns_name = ns_var.value.name
    fvar = lcompile(code)
    assert fvar == var.find_in_ns(
        sym.symbol(ns_name), sym.symbol('string-lower'))
    assert callable(fvar.value)
    assert fvar.value("UPPER") == "upper"


def test_if():
    assert lcompile("(if true :a :b)") == kw.keyword("a")
    assert lcompile("(if false :a :b)") == kw.keyword("b")
    assert lcompile("(if nil :a :b)") == kw.keyword("b")
    assert lcompile("(if true (if false :a :c) :b)") == kw.keyword("c")


def test_interop_call():
    assert lcompile('(. "ALL-UPPER" lower)') == "all-upper"
    assert lcompile('(.upper "lower-string")') == "LOWER-STRING"
    assert lcompile('(.strip "www.example.com" "cmowz.")') == "example"


def test_interop_prop():
    assert lcompile("(.-ns 'some.ns/sym)") == "some.ns"
    assert lcompile("(. 'some.ns/sym -ns)") == "some.ns"
    assert lcompile("(.-name 'some.ns/sym)") == "sym"
    assert lcompile("(. 'some.ns/sym -name)") == "sym"

    with pytest.raises(AttributeError):
        lcompile("(.-fake 'some.ns/sym)")


def test_quoted_list():
    assert lcompile("'(str)") == llist.l(sym.symbol('str'))
    assert lcompile("'(str 3)") == llist.l(sym.symbol('str'), 3)
    assert lcompile("'(str 3 :feet-deep)") == llist.l(
        sym.symbol('str'), 3, kw.keyword("feet-deep"))


def test_try_catch(capsys):
    code = """
      (try
        (.fake-lower "UPPER")
        (catch AttributeError _ "lower"))
    """
    assert lcompile(code) == "lower"

    code = """
      (try
        (.fake-lower "UPPER")
        (catch TypeError _ "lower")
        (catch AttributeError _ "mIxEd"))
    """
    assert lcompile(code) == "mIxEd"

    code = """
      (import* builtins)
      (try
        (.fake-lower "UPPER")
        (catch TypeError _ "lower")
        (catch AttributeError _ "mIxEd")
        (finally (builtins/print "neither")))
    """
    assert lcompile(code) == "mIxEd"
    captured = capsys.readouterr()
    assert captured.out == "neither\n"


def test_var(ns_var: var.Var):
    code = """
    (def some-var "a value")

    (var test/some-var)"""

    ns_name = ns_var.value.name
    v = lcompile(code)
    assert v == var.find_in_ns(sym.symbol(ns_name), sym.symbol('some-var'))
    assert v.value == "a value"

    code = """
    (def some-var "a value")

    #'test/some-var"""

    ns_name = ns_var.value.name
    v = lcompile(code)
    assert v == var.find_in_ns(sym.symbol(ns_name), sym.symbol('some-var'))
    assert v.value == "a value"


def test_inst():
    assert lcompile('#inst "2018-01-18T03:26:57.296-00:00"'
                    ) == dateparser.parse('2018-01-18T03:26:57.296-00:00')


def test_regex():
    assert lcompile('#"\s"') == re.compile('\s')


def test_uuid():
    assert lcompile('#uuid "0366f074-a8c5-4764-b340-6a5576afd2e8"'
                    ) == uuid.UUID('{0366f074-a8c5-4764-b340-6a5576afd2e8}')
