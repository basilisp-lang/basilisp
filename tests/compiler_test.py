import re
import types
import uuid
from typing import Optional
from unittest.mock import Mock

import dateutil.parser as dateparser
import pytest

import basilisp.compiler as compiler
import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.runtime as runtime
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
import basilisp.reader as reader
from basilisp.lang.runtime import Namespace, Var
from basilisp.main import init
from basilisp.util import Maybe

init()

# Cache the initial state of the `print_generated_python` flag.
__PRINT_GENERATED_PYTHON_FN = runtime.print_generated_python


def setup_module(module):
    """Disable the `print_generated_python` flag so we can safely capture
    stderr and stdout for tests which require those facilities."""
    runtime.print_generated_python = Mock(return_value=False)


def teardown_module(module):
    """Restore the `print_generated_python` flag after we finish running tests."""
    runtime.print_generated_python = __PRINT_GENERATED_PYTHON_FN


@pytest.fixture
def test_ns() -> str:
    return "test"


@pytest.fixture
def ns_var(test_ns: str):
    runtime.init_ns_var(which_ns=runtime._CORE_NS)
    yield runtime.set_current_ns(test_ns)
    Namespace.remove(sym.symbol(runtime._CORE_NS))


@pytest.fixture
def resolver() -> reader.Resolver:
    return runtime.resolve_alias


def lcompile(s: str,
             resolver: Optional[reader.Resolver] = None,
             ctx: Optional[compiler.CompilerContext] = None,
             mod: Optional[types.ModuleType] = None):
    """Compile and execute the code in the input string.

    Return the resulting expression."""
    ctx = Maybe(ctx).or_else(lambda: compiler.CompilerContext())
    mod = Maybe(mod).or_else(lambda: runtime._new_module('compiler_test'))

    last = None
    for form in reader.read_str(s, resolver=resolver):
        last = compiler.compile_and_exec_form(form, ctx, mod)

    return last


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


def test_def(ns_var: Var):
    ns_name = ns_var.value.name
    assert lcompile("(def a :some-val)") == Var.find_in_ns(
        sym.symbol(ns_name), sym.symbol('a'))
    assert lcompile('(def beep "a sound a robot makes")') == Var.find_in_ns(
        sym.symbol(ns_name), sym.symbol('beep'))
    assert lcompile("a") == kw.keyword("some-val")
    assert lcompile("beep") == "a sound a robot makes"


def test_do(ns_var: Var):
    code = """
    (do
      (def first-name :Darth)
      (def last-name "Vader"))
    """
    ns_name = ns_var.value.name
    assert lcompile(code) == Var.find_in_ns(
        sym.symbol(ns_name), sym.symbol('last-name'))
    assert lcompile("first-name") == kw.keyword("Darth")
    assert lcompile("last-name") == "Vader"


def test_fn(ns_var: Var):
    code = """
    (def string-upper (fn* string-upper [s] (.upper s)))
    """
    ns_name = ns_var.value.name
    fvar = lcompile(code)
    assert fvar == Var.find_in_ns(
        sym.symbol(ns_name), sym.symbol('string-upper'))
    assert callable(fvar.value)
    assert fvar.value("lower") == "LOWER"

    code = """
    (def string-lower #(.lower %))
    """
    ns_name = ns_var.value.name
    fvar = lcompile(code)
    assert fvar == Var.find_in_ns(
        sym.symbol(ns_name), sym.symbol('string-lower'))
    assert callable(fvar.value)
    assert fvar.value("UPPER") == "upper"


def test_fn_call(ns_var: Var):
    code = """
    (def string-upper (fn* [s] (.upper s)))

    (string-upper "bloop")
    """
    fvar = lcompile(code)
    assert fvar == "BLOOP"

    code = """
    (def string-lower #(.lower %))

    (string-lower "BLEEP")
    """
    fvar = lcompile(code)
    assert fvar == "bleep"


def test_if(ns_var: Var):
    assert lcompile("(if true :a :b)") == kw.keyword("a")
    assert lcompile("(if false :a :b)") == kw.keyword("b")
    assert lcompile("(if nil :a :b)") == kw.keyword("b")
    assert lcompile("(if true (if false :a :c) :b)") == kw.keyword("c")


def test_interop_call(ns_var: Var):
    assert lcompile('(. "ALL-UPPER" lower)') == "all-upper"
    assert lcompile('(.upper "lower-string")') == "LOWER-STRING"
    assert lcompile('(.strip "www.example.com" "cmowz.")') == "example"


def test_interop_prop(ns_var: Var):
    assert lcompile("(.-ns 'some.ns/sym)") == "some.ns"
    assert lcompile("(. 'some.ns/sym -ns)") == "some.ns"
    assert lcompile("(.-name 'some.ns/sym)") == "sym"
    assert lcompile("(. 'some.ns/sym -name)") == "sym"

    with pytest.raises(AttributeError):
        lcompile("(.-fake 'some.ns/sym)")


def test_let(ns_var: Var):
    assert lcompile("(let* [a 1] a)") == 1
    assert lcompile("(let* [a :keyword b \"string\"] a)") == kw.keyword('keyword')
    assert lcompile("(let* [a :value b a] b)") == kw.keyword('value')
    assert lcompile("(let* [a 1 b :length c {b a} a 4] c)") == lmap.map({kw.keyword('length'): 1})
    assert lcompile("(let* [a 1 b :length c {b a} a 4] a)") == 4
    assert lcompile("(let* [a \"lower\"] (.upper a))") == "LOWER"

    with pytest.raises(AttributeError):
        lcompile("(let* [a 'sym] c)")

    with pytest.raises(compiler.CompilerException):
        lcompile("(let* [] \"string\")")


def test_quoted_list(ns_var: Var):
    assert lcompile("'()") == llist.l()
    assert lcompile("'(str)") == llist.l(sym.symbol('str'))
    assert lcompile("'(str 3)") == llist.l(sym.symbol('str'), 3)
    assert lcompile("'(str 3 :feet-deep)") == llist.l(
        sym.symbol('str'), 3, kw.keyword("feet-deep"))


def test_syntax_quoting(test_ns: str, ns_var: Var, resolver: reader.Resolver):
    code = """
    (def some-val \"some value!\")

    `(some-val)"""
    assert llist.l(sym.symbol('some-val', ns=test_ns)) == lcompile(code, resolver=resolver)

    code = """
    (def second-val \"some value!\")

    `(other-val)"""
    assert llist.l(sym.symbol('other-val')) == lcompile(code)

    code = """
    (def a-str \"a definite string\")
    (def a-number 1583)

    `(a-str ~a-number)"""
    assert llist.l(sym.symbol('a-str', ns=test_ns), 1583) == lcompile(code, resolver=resolver)

    code = """
    (def whatever \"yes, whatever\")
    (def ssss \"a snake\")

    `(whatever ~@[ssss 45])"""
    assert llist.l(sym.symbol('whatever', ns=test_ns), "a snake", 45) == lcompile(code, resolver=resolver)

    assert llist.l(sym.symbol('my-symbol', ns=test_ns)) == lcompile("`(my-symbol)", resolver)


def test_throw(ns_var):
    with pytest.raises(AttributeError):
        lcompile("(throw (builtins/AttributeError))")

    with pytest.raises(TypeError):
        lcompile("(throw (builtins/TypeError))")

    with pytest.raises(ValueError):
        lcompile("(throw (builtins/ValueError))")


def test_try_catch(capsys, ns_var):
    code = """
      (try
        (.fake-lower "UPPER")
        (catch AttributeError _ "lower"))
    """
    assert "lower" == lcompile(code)

    code = """
      (try
        (.fake-lower "UPPER")
        (catch TypeError _ "lower")
        (catch AttributeError _ "mIxEd"))
    """
    assert "mIxEd" == lcompile(code)

    # If you hit an error here, do yourself a favor
    # and look in the import code first.
    code = """
      (import* builtins)
      (try
        (.fake-lower "UPPER")
        (catch TypeError _ "lower")
        (catch AttributeError _ "mIxEd")
        (finally (builtins/print "neither")))
    """
    assert "mIxEd" == lcompile(code)
    captured = capsys.readouterr()
    assert "neither\n" == captured.out


def test_unquote(ns_var: Var):
    with pytest.raises(AttributeError):
        lcompile("~s")

    assert llist.l(sym.symbol('s')) == lcompile('`(s)')

    with pytest.raises(AttributeError):
        lcompile("`(~s)")


def test_unquote_splicing(ns_var: Var, resolver: reader.Resolver):
    with pytest.raises(AttributeError):
        lcompile("~@[1 2 3]")

    assert llist.l(1, 2, 3) == lcompile("`(~@[1 2 3])")

    assert llist.l(sym.symbol('print', ns='basilisp.core'), 1, 2, 3) == lcompile(
        "`(print ~@[1 2 3])", resolver=resolver)

    assert llist.l(llist.l(reader._UNQUOTE_SPLICING, 53233)) == lcompile("'(~@53233)")


def test_var(ns_var: Var):
    code = """
    (def some-var "a value")

    (var test/some-var)"""

    ns_name = ns_var.value.name
    v = lcompile(code)
    assert v == Var.find_in_ns(sym.symbol(ns_name), sym.symbol('some-var'))
    assert v.value == "a value"

    code = """
    (def some-var "a value")

    #'test/some-var"""

    ns_name = ns_var.value.name
    v = lcompile(code)
    assert v == Var.find_in_ns(sym.symbol(ns_name), sym.symbol('some-var'))
    assert v.value == "a value"


def test_inst(ns_var: Var):
    assert lcompile('#inst "2018-01-18T03:26:57.296-00:00"'
                    ) == dateparser.parse('2018-01-18T03:26:57.296-00:00')


def test_regex(ns_var: Var):
    assert lcompile('#"\s"') == re.compile('\s')


def test_uuid(ns_var: Var):
    assert lcompile('#uuid "0366f074-a8c5-4764-b340-6a5576afd2e8"'
                    ) == uuid.UUID('{0366f074-a8c5-4764-b340-6a5576afd2e8}')
