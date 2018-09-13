import decimal
import re
import types
import uuid
from fractions import Fraction
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
    assert 1 == lcompile('1')
    assert 100 == lcompile('100')
    assert 99927273 == lcompile('99927273')
    assert 0 == lcompile('0')
    assert -1 == lcompile('-1')
    assert -538282 == lcompile('-538282')

    assert 1 == lcompile('1N')
    assert 100 == lcompile('100N')
    assert 99927273 == lcompile('99927273N')
    assert 0 == lcompile('0N')
    assert -1 == lcompile('-1N')
    assert -538282 == lcompile('-538282N')


def test_decimal():
    assert decimal.Decimal('0.0') == lcompile('0.0M')
    assert decimal.Decimal('0.09387372') == lcompile('0.09387372M')
    assert decimal.Decimal('1.0') == lcompile('1.0M')
    assert decimal.Decimal('1.332') == lcompile('1.332M')
    assert decimal.Decimal('-1.332') == lcompile('-1.332M')
    assert decimal.Decimal('-1.0') == lcompile('-1.0M')
    assert decimal.Decimal('-0.332') == lcompile('-0.332M')
    assert decimal.Decimal('3.14') == lcompile('3.14M')


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


def test_def_dynamic(ns_var: Var):
    v: Var = lcompile("(def ^:dynamic *a-dynamic-var* 1)")
    assert v.dynamic is True
    lcompile("(.push-bindings #'*a-dynamic-var* :hi)")
    assert kw.keyword("hi") == lcompile("*a-dynamic-var*")
    assert kw.keyword("hi") == lcompile("(.pop-bindings #'*a-dynamic-var*)")
    assert 1 == lcompile("*a-dynamic-var*")

    v: Var = lcompile("(def a-regular-var 1)")
    assert v.dynamic is False
    lcompile("(.push-bindings #'a-regular-var :hi)")
    assert 1 == lcompile("a-regular-var")
    assert kw.keyword("hi") == lcompile("(.pop-bindings #'a-regular-var)")
    assert 1 == lcompile("a-regular-var")


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


def test_single_arity_fn(ns_var: Var):
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
    (def string-upper (fn* string-upper ([s] (.upper s))))
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


def test_multi_arity_fn(ns_var: Var):
    with pytest.raises(compiler.CompilerException):
        lcompile('(fn f)')

    with pytest.raises(compiler.CompilerException):
        lcompile("""
            (def f
              (fn* f
                ([] :no-args)
                ([] :also-no-args)))
            """)

    with pytest.raises(compiler.CompilerException):
        lcompile("""
            (def f
              (fn* f
                ([& args] (concat [:no-starter] args))
                ([s & args] (concat [s] args))))
            """)

    with pytest.raises(compiler.CompilerException):
        lcompile("""
            (def f
              (fn* f
                ([s] (concat [s] :one-arg))
                ([& args] (concat [:rest-params] args))))
            """)

    code = """
    (def multi-fn
      (fn* multi-fn
        ([] :no-args)
        ([s] s)
        ([s & args] (concat [s] args))))
    """
    ns_name = ns_var.value.name
    fvar = lcompile(code)
    assert fvar == Var.find_in_ns(
        sym.symbol(ns_name), sym.symbol('multi-fn'))
    assert callable(fvar.value)
    assert fvar.value() == kw.keyword('no-args')
    assert fvar.value('STRING') == 'STRING'
    assert fvar.value(kw.keyword('first-arg'), 'second-arg', 3) == llist.l(kw.keyword('first-arg'), 'second-arg', 3)

    with pytest.raises(runtime.RuntimeException):
        code = """
            (def angry-multi-fn
              (fn* angry-multi-fn
                ([] :send-me-an-arg!)
                ([i] i)
                ([i j] (concat [i] [j]))))
            """
        ns_name = ns_var.value.name
        fvar = lcompile(code)
        fvar.value(1, 2, 3)


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


def test_macro_expansion(ns_var: Var):
    assert llist.l(1, 2, 3) == lcompile("((fn [] '(1 2 3)))")


def test_if(ns_var: Var):
    assert lcompile("(if true :a :b)") == kw.keyword("a")
    assert lcompile("(if false :a :b)") == kw.keyword("b")
    assert lcompile("(if nil :a :b)") == kw.keyword("b")
    assert lcompile("(if true (if false :a :c) :b)") == kw.keyword("c")

    code = """
    (def f (fn* [s] s))

    (f (if true \"YELLING\" \"whispering\"))
    """
    assert "YELLING" == lcompile(code)


def test_truthiness(ns_var: Var):
    # Valid false values
    assert kw.keyword("b") == lcompile("(if false :a :b)")
    assert kw.keyword("b") == lcompile("(if nil :a :b)")

    # Everything else is true
    assert kw.keyword("a") == lcompile("(if true :a :b)")

    assert kw.keyword("a") == lcompile("(if 's :a :b)")
    assert kw.keyword("a") == lcompile("(if 'ns/s :a :b)")

    assert kw.keyword("a") == lcompile("(if :kw :a :b)")
    assert kw.keyword("a") == lcompile("(if :ns/kw :a :b)")

    assert kw.keyword("a") == lcompile("(if \"\" :a :b)")
    assert kw.keyword("a") == lcompile("(if \"not empty\" :a :b)")

    assert kw.keyword("a") == lcompile("(if 0 :a :b)")
    assert kw.keyword("a") == lcompile("(if 1 :a :b)")
    assert kw.keyword("a") == lcompile("(if -1 :a :b)")
    assert kw.keyword("a") == lcompile("(if 1.0 :a :b)")
    assert kw.keyword("a") == lcompile("(if 0.0 :a :b)")
    assert kw.keyword("a") == lcompile("(if -1.0 :a :b)")

    assert kw.keyword("a") == lcompile("(if () :a :b)")
    assert kw.keyword("a") == lcompile("(if '(0) :a :b)")
    assert kw.keyword("a") == lcompile("(if '(false) :a :b)")
    assert kw.keyword("a") == lcompile("(if '(true) :a :b)")

    assert kw.keyword("a") == lcompile("(if [] :a :b)")
    assert kw.keyword("a") == lcompile("(if [0] :a :b)")
    assert kw.keyword("a") == lcompile("(if '(false) :a :b)")
    assert kw.keyword("a") == lcompile("(if '(true) :a :b)")

    assert kw.keyword("a") == lcompile("(if {} :a :b)")
    assert kw.keyword("a") == lcompile("(if {0 0} :a :b)")
    assert kw.keyword("a") == lcompile("(if {false false} :a :b)")
    assert kw.keyword("a") == lcompile("(if {true true} :a :b)")

    assert kw.keyword("a") == lcompile("(if #{} :a :b)")
    assert kw.keyword("a") == lcompile("(if #{0} :a :b)")
    assert kw.keyword("a") == lcompile("(if #{false} :a :b)")
    assert kw.keyword("a") == lcompile("(if #{true} :a :b)")


def test_interop_new(ns_var: Var):
    assert "hi" == lcompile('(builtins.str. "hi")')
    assert "1" == lcompile('(builtins.str. 1)')
    assert sym.symbol('hi') == lcompile('(basilisp.lang.symbol.Symbol. "hi")')

    with pytest.raises(AttributeError):
        lcompile('(builtins.str "hi")')


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


def test_let_lazy_evaluation(ns_var: Var):
    code = """
    (if false
      (let [n  (.-name :value)
            ns (.-ns "string")]  ;; this line would fail if we eagerly evaluated
        :true)
      :false)
    """
    assert kw.keyword("false") == lcompile(code)


def test_quoted_list(ns_var: Var):
    assert lcompile("'()") == llist.l()
    assert lcompile("'(str)") == llist.l(sym.symbol('str'))
    assert lcompile("'(str 3)") == llist.l(sym.symbol('str'), 3)
    assert lcompile("'(str 3 :feet-deep)") == llist.l(
        sym.symbol('str'), 3, kw.keyword("feet-deep"))


def test_recur(ns_var: Var):
    code = """
    (def last
      (fn [s]
        (if (seq (rest s))
          (recur (rest s))
          (first s))))
    """

    lcompile(code)

    assert None is lcompile("(last '())")
    assert 1 == lcompile("(last '(1))")
    assert 2 == lcompile("(last '(1 2))")
    assert 3 == lcompile("(last '(1 2 3))")

    code = """
    (def last
      (fn [s]
        (let [r (rest s)]
          (if (seq r)
            (recur r)
            (first s)))))
    """

    lcompile(code)

    assert None is lcompile("(last '())")
    assert 1 == lcompile("(last '(1))")
    assert 2 == lcompile("(last '(1 2))")
    assert 3 == lcompile("(last '(1 2 3))")

    code = """
    (def rev-str
      (fn rev-str [s & args]
        (let [coerce (fn [in out]
                       (if (seq (rest in))
                         (recur (rest in) (cons (builtins/str (first in)) out))
                         (cons (builtins/str (first in)) out)))]
         (.join \"\" (coerce (cons s args) '())))))
     """

    lcompile(code)

    assert "a" == lcompile("(rev-str \"a\")")
    assert "ba" == lcompile("(rev-str \"a\" :b)")
    assert "3ba" == lcompile("(rev-str \"a\" :b 3)")


def test_recur_arity(ns_var: Var):
    # Single arity function
    code = """
    (def ++
      (fn ++ [x & args]
        (if (seq (rest args))
          (recur (operator/add x (first args)) (rest args))
          (operator/add x (first args)))))
    """

    lcompile(code)

    assert 3 == lcompile("(++ 1 2)")
    assert 6 == lcompile("(++ 1 2 3)")
    assert 10 == lcompile("(++ 1 2 3 4)")
    assert 15 == lcompile("(++ 1 2 3 4 5)")

    # Multi-arity function
    code = """
    (def +++
      (fn +++ 
        ([] 0)
        ([x] x)
        ([x & args]
          (if (seq (rest args))
            (recur (operator/add x (first args)) (rest args))
            (operator/add x (first args))))))
    """

    lcompile(code)

    assert 0 == lcompile("(+++)")
    assert 1 == lcompile("(+++ 1)")
    assert 3 == lcompile("(+++ 1 2)")
    assert 6 == lcompile("(+++ 1 2 3)")
    assert 10 == lcompile("(+++ 1 2 3 4)")
    assert 15 == lcompile("(+++ 1 2 3 4 5)")


def test_disallow_recur_in_special_forms(ns_var: Var):
    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (def b (recur \"a\")))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (import* (recur \"a\")))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (.join \"\" (recur \"a\")))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (.-p (recur \"a\")))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (throw (recur \"a\"))))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (var (recur \"a\"))))")


def test_disallow_recur_outside_tail(ns_var: Var):
    with pytest.raises(compiler.CompilerException):
        lcompile("(recur)")

    with pytest.raises(compiler.CompilerException):
        lcompile("(do (recur))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(if true (recur) :b)")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (do (recur \"a\") :b))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (if (recur \"a\") :a :b))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (if (recur \"a\") :a))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (let [a (recur \"a\")] a))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (let [a (do (recur \"a\"))] a))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (let [a (do :b (recur \"a\"))] a))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (let [a (do (recur \"a\") :c)] a))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (let [a \"a\"] (recur a) a))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (try (do (recur a) :b) (catch AttributeError _ nil)))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (try :b (catch AttributeError _ (do (recur :a) :c))))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (try :b (finally (do (recur :a) :c))))")


def test_named_anonymous_fn_recursion(ns_var: Var):
    code = """
    (let [compute-sum (fn sum [n]
                        (if (operator/eq 0 n)
                          0
                          (operator/add n (sum (operator/sub n 1)))))]
      (compute-sum 5))
    """
    assert 15 == lcompile(code)

    code = """
    (let [compute-sum (fn sum
                        ([] 0)
                        ([n]
                         (if (operator/eq 0 n)
                           0
                           (operator/add n (sum (operator/sub n 1))))))]
      (compute-sum 5))
    """
    assert 15 == lcompile(code)


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


def test_fraction(ns_var: Var):
    assert Fraction('22/7') == lcompile('22/7')


def test_inst(ns_var: Var):
    assert dateparser.parse('2018-01-18T03:26:57.296-00:00') == lcompile(
        '#inst "2018-01-18T03:26:57.296-00:00"')


def test_regex(ns_var: Var):
    assert lcompile(r'#"\s"') == re.compile(r'\s')


def test_uuid(ns_var: Var):
    assert uuid.UUID('{0366f074-a8c5-4764-b340-6a5576afd2e8}') == lcompile(
        '#uuid "0366f074-a8c5-4764-b340-6a5576afd2e8"')
