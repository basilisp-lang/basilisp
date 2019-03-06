import decimal
import logging
import re
import types
import uuid
from fractions import Fraction
from typing import Optional, Dict
from unittest.mock import Mock

import dateutil.parser as dateparser
import pytest

import basilisp.lang.compiler as compiler
import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.reader as reader
import basilisp.lang.runtime as runtime
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
from basilisp.lang.runtime import Var
from basilisp.main import init
from basilisp.util import Maybe

COMPILER_FILE_PATH = "compiler_test"


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """Disable the `print_generated_python` flag so we can safely capture
    stderr and stdout for tests which require those facilities."""
    init()
    orig = runtime.print_generated_python
    runtime.print_generated_python = Mock(return_value=False)
    yield
    runtime.print_generated_python = orig


@pytest.fixture
def test_ns() -> str:
    return "test"


@pytest.fixture
def ns(test_ns: str) -> runtime.Namespace:
    runtime.init_ns_var(which_ns=runtime._CORE_NS)
    with runtime.ns_bindings(test_ns) as ns:
        yield ns


@pytest.fixture
def resolver() -> reader.Resolver:
    return runtime.resolve_alias


def assert_no_logs(caplog):
    __tracebackhide__ = True
    log_records = caplog.record_tuples
    if len(log_records) != 0:
        pytest.fail(f"At least one log message found: {log_records}")


def lcompile(
    s: str,
    resolver: Optional[reader.Resolver] = None,
    opts: Optional[Dict[str, bool]] = None,
    mod: Optional[types.ModuleType] = None,
):
    """Compile and execute the code in the input string.

    Return the resulting expression."""
    ctx = compiler.CompilerContext(COMPILER_FILE_PATH, opts=opts)
    mod = Maybe(mod).or_else(lambda: runtime._new_module(COMPILER_FILE_PATH))

    last = None
    for form in reader.read_str(s, resolver=resolver):
        last = compiler.compile_and_exec_form(form, ctx, mod)

    return last


class TestLiterals:
    def test_string(self):
        assert lcompile('"some string"') == "some string"
        assert lcompile('""') == ""

    def test_int(self):
        assert 1 == lcompile("1")
        assert 100 == lcompile("100")
        assert 99_927_273 == lcompile("99927273")
        assert 0 == lcompile("0")
        assert -1 == lcompile("-1")
        assert -538_282 == lcompile("-538282")

        assert 1 == lcompile("1N")
        assert 100 == lcompile("100N")
        assert 99_927_273 == lcompile("99927273N")
        assert 0 == lcompile("0N")
        assert -1 == lcompile("-1N")
        assert -538_282 == lcompile("-538282N")

    def test_decimal(self):
        assert decimal.Decimal("0.0") == lcompile("0.0M")
        assert decimal.Decimal("0.09387372") == lcompile("0.09387372M")
        assert decimal.Decimal("1.0") == lcompile("1.0M")
        assert decimal.Decimal("1.332") == lcompile("1.332M")
        assert decimal.Decimal("-1.332") == lcompile("-1.332M")
        assert decimal.Decimal("-1.0") == lcompile("-1.0M")
        assert decimal.Decimal("-0.332") == lcompile("-0.332M")
        assert decimal.Decimal("3.14") == lcompile("3.14M")

    def test_float(self):
        assert lcompile("0.0") == 0.0
        assert lcompile("0.09387372") == 0.093_873_72
        assert lcompile("1.0") == 1.0
        assert lcompile("1.332") == 1.332
        assert lcompile("-1.332") == -1.332
        assert lcompile("-1.0") == -1.0
        assert lcompile("-0.332") == -0.332

    def test_kw(self):
        assert lcompile(":kw") == kw.keyword("kw")
        assert lcompile(":ns/kw") == kw.keyword("kw", ns="ns")
        assert lcompile(":qualified.ns/kw") == kw.keyword("kw", ns="qualified.ns")

    def test_literals(self):
        assert lcompile("nil") is None
        assert lcompile("true") is True
        assert lcompile("false") is False

    def test_quoted_symbol(self):
        assert lcompile("'sym") == sym.symbol("sym")
        assert lcompile("'ns/sym") == sym.symbol("sym", ns="ns")
        assert lcompile("'qualified.ns/sym") == sym.symbol("sym", ns="qualified.ns")

    def test_map(self):
        assert lcompile("{}") == lmap.m()
        assert lcompile('{:a "string"}') == lmap.map({kw.keyword("a"): "string"})
        assert lcompile('{:a "string" 45 :my-age}') == lmap.map(
            {kw.keyword("a"): "string", 45: kw.keyword("my-age")}
        )

    def test_set(self):
        assert lcompile("#{}") == lset.s()
        assert lcompile("#{:a}") == lset.s(kw.keyword("a"))
        assert lcompile("#{:a 1}") == lset.s(kw.keyword("a"), 1)

    def test_vec(self):
        assert lcompile("[]") == vec.v()
        assert lcompile("[:a]") == vec.v(kw.keyword("a"))
        assert lcompile("[:a 1]") == vec.v(kw.keyword("a"), 1)


class TestDef:
    def test_def(self, ns: runtime.Namespace):
        ns_name = ns.name
        assert lcompile("(def a :some-val)") == Var.find_in_ns(
            sym.symbol(ns_name), sym.symbol("a")
        )
        assert lcompile('(def beep "a sound a robot makes")') == Var.find_in_ns(
            sym.symbol(ns_name), sym.symbol("beep")
        )
        assert lcompile("a") == kw.keyword("some-val")
        assert lcompile("beep") == "a sound a robot makes"

    def test_def_unbound(self, ns: runtime.Namespace):
        lcompile("(def a)")
        var = Var.find_in_ns(sym.symbol(ns.name), sym.symbol("a"))
        assert var.root is None
        # TODO: fix this
        # assert not var.is_bound

    def test_def_number_of_elems(self, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile("(def)")

        with pytest.raises(compiler.CompilerException):
            lcompile('(def a "docstring" :b :c)')

    def test_def_name_is_symbol(self, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile("(def :a)")

    def test_def_docstring_is_string(self, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile("(def a :not-a-docstring :a)")

    def test_compiler_metadata(self, ns: runtime.Namespace):
        lcompile('(def ^{:doc "Super cool docstring"} unique-oeuene :a)')

        var = ns.find(sym.symbol("unique-oeuene"))
        meta = var.meta

        assert 1 == meta.entry(kw.keyword("line"))
        assert COMPILER_FILE_PATH == meta.entry(kw.keyword("file"))
        assert 1 == meta.entry(kw.keyword("col"))
        assert sym.symbol("unique-oeuene") == meta.entry(kw.keyword("name"))
        assert ns == meta.entry(kw.keyword("ns"))
        assert "Super cool docstring" == meta.entry(kw.keyword("doc"))

    def test_no_warn_on_redef_meta(self, ns: runtime.Namespace, caplog):
        lcompile(
            """
        (def unique-zhddkd :a)
        (def ^:no-warn-on-redef unique-zhddkd :b)
        """
        )
        assert_no_logs(caplog)

    def test_warn_on_redef_if_warn_on_redef_meta_missing(
        self, ns: runtime.Namespace, caplog
    ):
        lcompile(
            """
        (def unique-djhvyz :a)
        (def unique-djhvyz :b)
        """
        )
        assert (
            "basilisp.lang.compiler.generator",
            logging.WARNING,
            f"redefining local Python name 'unique_djhvyz' in module '{ns.name}'",
        ) in caplog.record_tuples

    def test_redef_vars(self, ns: runtime.Namespace, caplog):
        assert kw.keyword("b") == lcompile(
            """
        (def ^:redef orig :a)
        (def redef-check (fn* [] orig))
        (def orig :b)
        (redef-check)
        """
        )
        assert (
            f"redefining local Python name 'orig' in module '{ns.name}'"
        ) not in caplog.messages

    def test_def_dynamic(self, ns: runtime.Namespace):
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


def test_do(ns: runtime.Namespace):
    code = """
    (do
      (def first-name :Darth)
      (def last-name "Vader"))
    """
    ns_name = ns.name
    assert lcompile(code) == Var.find_in_ns(
        sym.symbol(ns_name), sym.symbol("last-name")
    )
    assert lcompile("first-name") == kw.keyword("Darth")
    assert lcompile("last-name") == "Vader"


class TestFunctionShadowName:
    def test_single_arity_fn_no_log_if_warning_disabled(
        self, ns: runtime.Namespace, caplog
    ):
        lcompile("(fn [v] (fn [v] v))")
        assert ("name 'v' shadows name from outer scope") not in caplog.messages

    def test_multi_arity_fn_no_log_if_warning_disabled(
        self, ns: runtime.Namespace, caplog
    ):
        lcompile(
            """
        (fn
          ([] :a)
          ([v] (fn [v] v)))
        """
        )
        assert ("name 'v' shadows name from outer scope") not in caplog.messages

    def test_single_arity_fn_log_if_warning_enabled(
        self, ns: runtime.Namespace, caplog
    ):
        lcompile("(fn [v] (fn [v] v))", opts={compiler.WARN_ON_SHADOWED_NAME: True})
        assert (
            "basilisp.lang.compiler.parser",
            logging.WARNING,
            "name 'v' shadows name from outer scope",
        ) in caplog.record_tuples

    def test_multi_arity_fn_log_if_warning_enabled(self, ns: runtime.Namespace, caplog):
        code = """
        (fn
          ([] :a)
          ([v] (fn [v] v)))
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_NAME: True})
        assert (
            "basilisp.lang.compiler.parser",
            logging.WARNING,
            "name 'v' shadows name from outer scope",
        ) in caplog.record_tuples

    def test_single_arity_fn_log_shadows_var_if_warning_enabled(
        self, ns: runtime.Namespace, caplog
    ):
        code = """
        (def unique-bljzndd :a)
        (fn [unique-bljzndd] unique-bljzndd)
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_NAME: True})
        assert (
            "basilisp.lang.compiler.parser",
            logging.WARNING,
            "name 'unique-bljzndd' shadows def'ed Var from outer scope",
        ) in caplog.record_tuples

    def test_multi_arity_fn_log_shadows_var_if_warning_enabled(
        self, ns: runtime.Namespace, caplog
    ):
        code = """
        (def unique-yezddid :a)
        (fn
          ([] :b)
          ([unique-yezddid] unique-yezddid))
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_NAME: True})
        assert (
            "basilisp.lang.compiler.parser",
            logging.WARNING,
            "name 'unique-yezddid' shadows def'ed Var from outer scope",
        ) in caplog.record_tuples


class TestFunctionShadowVar:
    def test_single_arity_fn_no_log_if_warning_disabled(
        self, ns: runtime.Namespace, caplog
    ):
        code = """
        (def unique-vfsdhsk :a)
        (fn [unique-vfsdhsk] unique-vfsdhsk)
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_VAR: False})
        assert_no_logs(caplog)

    def test_multi_arity_fn_no_log_if_warning_disabled(
        self, ns: runtime.Namespace, caplog
    ):
        code = """
        (def unique-mmndheee :a)
        (fn
          ([] :b)
          ([unique-mmndheee] unique-mmndheee))
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_VAR: False})
        assert_no_logs(caplog)

    def test_single_arity_fn_log_if_warning_enabled(
        self, ns: runtime.Namespace, caplog
    ):
        code = """
        (def unique-kuieeid :a)
        (fn [unique-kuieeid] unique-kuieeid)
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_VAR: True})
        assert (
            "basilisp.lang.compiler.parser",
            logging.WARNING,
            "name 'unique-kuieeid' shadows def'ed Var from outer scope",
        ) in caplog.record_tuples

    def test_multi_arity_fn_log_if_warning_enabled(self, ns: runtime.Namespace, caplog):
        code = """
        (def unique-peuudcdf :a)
        (fn
          ([] :b)
          ([unique-peuudcdf] unique-peuudcdf))
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_VAR: True})
        assert (
            "basilisp.lang.compiler.parser",
            logging.WARNING,
            "name 'unique-peuudcdf' shadows def'ed Var from outer scope",
        ) in caplog.record_tuples


class TestFunctionWarnUnusedName:
    def test_single_arity_fn_no_log_if_warning_disabled(
        self, ns: runtime.Namespace, caplog
    ):
        lcompile("(fn [v] (fn [v] v))", opts={compiler.WARN_ON_UNUSED_NAMES: False})
        assert_no_logs(caplog)

    def test_multi_arity_fn_no_log_if_warning_disabled(
        self, ns: runtime.Namespace, caplog
    ):
        lcompile(
            """
        (fn
          ([] :a)
          ([v] (fn [v] v)))
        """,
            opts={compiler.WARN_ON_UNUSED_NAMES: False},
        )
        assert_no_logs(caplog)

    def test_single_arity_fn_log_if_warning_enabled(
        self, ns: runtime.Namespace, caplog
    ):
        lcompile("(fn [v] (fn [v] v))", opts={compiler.WARN_ON_UNUSED_NAMES: True})
        assert (
            "basilisp.lang.compiler.parser",
            logging.WARNING,
            f"symbol 'v' defined but not used ({ns}: 1)",
        ) in caplog.record_tuples

    def test_multi_arity_fn_log_if_warning_enabled(self, ns: runtime.Namespace, caplog):
        lcompile(
            """
        (fn
          ([] :a)
          ([v] (fn [v] v)))
        """,
            opts={compiler.WARN_ON_UNUSED_NAMES: True},
        )
        assert (
            "basilisp.lang.compiler.parser",
            logging.WARNING,
            f"symbol 'v' defined but not used ({ns}: 3)",
        ) in caplog.record_tuples


class TestFunctionDef:
    def test_fn_with_no_args_throws(self, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* a)")

    def test_fn_with_invalid_name_throws(self, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* :a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* :a [])")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* :a ([] :a) ([a] a))")

    def test_variadic_arity_fn_has_variadic_argument(self, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* [m &] m)")

    def test_variadic_arity_fn_method_has_variadic_argument(
        self, ns: runtime.Namespace
    ):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* ([] :a) ([m &] m))")

    def test_fn_argument_vector_is_vector(self, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* () :a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* (a) a)")

    def test_fn_method_argument_vector_is_vector(self, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* (() :a) ((a) a))")

    def test_fn_arg_is_symbol(self, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* [:a] :a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* [a :b] :a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* [a b & :c] :a)")

    def test_fn_method_arg_is_symbol(self, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* ([a] a) ([a :b] a))")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* ([a] a) ([a & :b] a))")

    def test_fn_allows_empty_body(self, ns: runtime.Namespace):
        ns_name = ns.name
        fvar = lcompile("(def empty-single (fn* empty-single []))")
        assert Var.find_in_ns(sym.symbol(ns_name), sym.symbol("empty-single")) == fvar
        assert callable(fvar.value)
        assert None is fvar.value()

    def test_fn_method_allows_empty_body(self, ns: runtime.Namespace):
        ns_name = ns.name
        fvar = lcompile("(def empty-single (fn* empty-single ([]) ([a] :a)))")
        assert Var.find_in_ns(sym.symbol(ns_name), sym.symbol("empty-single")) == fvar
        assert callable(fvar.value)
        assert None is fvar.value()

    def test_single_arity_fn(self, ns: runtime.Namespace):
        code = """
        (def string-upper (fn* string-upper [s] (.upper s)))
        """
        ns_name = ns.name
        fvar = lcompile(code)
        assert Var.find_in_ns(sym.symbol(ns_name), sym.symbol("string-upper")) == fvar
        assert callable(fvar.value)
        assert "LOWER" == fvar.value("lower")

        code = """
        (def string-upper (fn* string-upper ([s] (.upper s))))
        """
        ns_name = ns.name
        fvar = lcompile(code)
        assert Var.find_in_ns(sym.symbol(ns_name), sym.symbol("string-upper")) == fvar
        assert callable(fvar.value)
        assert "LOWER" == fvar.value("lower")

        code = """
        (def string-lower #(.lower %))
        """
        ns_name = ns.name
        fvar = lcompile(code)
        assert Var.find_in_ns(sym.symbol(ns_name), sym.symbol("string-lower")) == fvar
        assert callable(fvar.value)
        assert "upper" == fvar.value("UPPER")

    def test_no_fn_method_has_same_fixed_arity(self, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
                (def f
                  (fn* f
                    ([] :no-args)
                    ([] :also-no-args)))
                """
            )

        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
                (def f
                  (fn* f
                    ([s] :one-arg)
                    ([s] :also-one-arg)))
                """
            )

        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
                (def f
                  (fn* f
                    ([] :no-args)
                    ([s] :one-arg)
                    ([a b] [a b])
                    ([s3] :also-one-arg)))
                """
            )

    def test_multi_arity_fn_cannot_have_two_variadic_methods(
        self, ns: runtime.Namespace
    ):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
                (def f
                  (fn* f
                    ([& args] (concat [:no-starter] args))
                    ([s & args] (concat [s] args))))
                """
            )

    def test_variadic_method_cannot_have_lower_fixed_arity_than_other_methods(
        self, ns: runtime.Namespace
    ):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
                (def f
                  (fn* f
                    ([s] (concat [s] :one-arg))
                    ([& args] (concat [:rest-params] args))))
                """
            )

    def test_multi_arity_fn_dispatches_properly(self, ns: runtime.Namespace):
        code = """
        (def empty-multi-fn
          (fn* empty-multi-fn
            ([])
            ([s])))
        """
        ns_name = ns.name
        fvar = lcompile(code)
        assert Var.find_in_ns(sym.symbol(ns_name), sym.symbol("empty-multi-fn")) == fvar
        assert callable(fvar.value)
        assert None is fvar.value()
        assert None is fvar.value("STRING")

        code = """
        (def multi-fn
          (fn* multi-fn
            ([] :no-args)
            ([s] s)
            ([s & args] (concat [s] args))))
        """
        ns_name = ns.name
        fvar = lcompile(code)
        assert fvar == Var.find_in_ns(sym.symbol(ns_name), sym.symbol("multi-fn"))
        assert callable(fvar.value)
        assert fvar.value() == kw.keyword("no-args")
        assert fvar.value("STRING") == "STRING"
        assert fvar.value(kw.keyword("first-arg"), "second-arg", 3) == llist.l(
            kw.keyword("first-arg"), "second-arg", 3
        )

    def test_multi_arity_fn_call_fails_if_no_valid_arity(self, ns: runtime.Namespace):
        with pytest.raises(runtime.RuntimeException):
            fvar = lcompile(
                """
                (def angry-multi-fn
                  (fn* angry-multi-fn
                    ([] :send-me-an-arg!)
                    ([i] i)
                    ([i j] (concat [i] [j]))))
                """
            )
            fvar.value(1, 2, 3)


def test_fn_call(ns: runtime.Namespace):
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


def test_macro_expansion(ns: runtime.Namespace):
    assert llist.l(1, 2, 3) == lcompile("((fn [] '(1 2 3)))")


def test_if(ns: runtime.Namespace):
    assert lcompile("(if true :a :b)") == kw.keyword("a")
    assert lcompile("(if false :a :b)") == kw.keyword("b")
    assert lcompile("(if nil :a :b)") == kw.keyword("b")
    assert lcompile("(if true (if false :a :c) :b)") == kw.keyword("c")

    code = """
    (def f (fn* [s] s))

    (f (if true \"YELLING\" \"whispering\"))
    """
    assert "YELLING" == lcompile(code)


def test_truthiness(ns: runtime.Namespace):
    # Valid false values
    assert kw.keyword("b") == lcompile("(if false :a :b)")
    assert kw.keyword("b") == lcompile("(if nil :a :b)")

    # Everything else is true
    assert kw.keyword("a") == lcompile("(if true :a :b)")

    assert kw.keyword("a") == lcompile("(if 's :a :b)")
    assert kw.keyword("a") == lcompile("(if 'ns/s :a :b)")

    assert kw.keyword("a") == lcompile("(if :kw :a :b)")
    assert kw.keyword("a") == lcompile("(if :ns/kw :a :b)")

    assert kw.keyword("a") == lcompile('(if "" :a :b)')
    assert kw.keyword("a") == lcompile('(if "not empty" :a :b)')

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


def test_import(ns: runtime.Namespace):
    with pytest.raises(compiler.CompilerException):
        lcompile("(import* :time)")

    with pytest.raises(compiler.CompilerException):
        lcompile('(import* "time")')

    with pytest.raises(ImportError):
        lcompile("(import* real.fake.module)")

    import time

    assert time.perf_counter == lcompile("(import* time) time/perf-counter")
    assert time.perf_counter == lcompile(
        "(import* [time :as py-time]) py-time/perf-counter"
    )


def test_interop_new(ns: runtime.Namespace):
    assert "hi" == lcompile('(builtins.str. "hi")')
    assert "1" == lcompile("(builtins.str. 1)")
    assert sym.symbol("hi") == lcompile('(basilisp.lang.symbol.Symbol. "hi")')

    with pytest.raises(compiler.CompilerException):
        lcompile('(builtins.str "hi")')


def test_interop_call(ns: runtime.Namespace):
    assert lcompile('(. "ALL-UPPER" lower)') == "all-upper"
    assert lcompile('(.upper "lower-string")') == "LOWER-STRING"
    assert lcompile('(.strip "www.example.com" "cmowz.")') == "example"


def test_interop_prop(ns: runtime.Namespace):
    assert lcompile("(.-ns 'some.ns/sym)") == "some.ns"
    assert lcompile("(.- 'some.ns/sym ns)") == "some.ns"
    assert lcompile("(.-name 'some.ns/sym)") == "sym"
    assert lcompile("(.- 'some.ns/sym name)") == "sym"

    with pytest.raises(AttributeError):
        lcompile("(.-fake 'some.ns/sym)")


def test_interop_quoted(ns: runtime.Namespace):
    assert lcompile("'(.match pattern)") == llist.l(
        sym.symbol(".match"), sym.symbol("pattern")
    )
    assert lcompile("'(.-pattern regex)") == llist.l(
        sym.symbol(".-pattern"), sym.symbol("regex")
    )


def test_let(ns: runtime.Namespace):
    assert lcompile("(let* [a 1] a)") == 1
    assert lcompile('(let* [a :keyword b "string"] a)') == kw.keyword("keyword")
    assert lcompile("(let* [a :value b a] b)") == kw.keyword("value")
    assert lcompile("(let* [a 1 b :length c {b a} a 4] c)") == lmap.map(
        {kw.keyword("length"): 1}
    )
    assert lcompile("(let* [a 1 b :length c {b a} a 4] a)") == 4
    assert lcompile('(let* [a "lower"] (.upper a))') == "LOWER"

    with pytest.raises(compiler.CompilerException):
        lcompile("(let* [a 'sym] c)")

    with pytest.raises(compiler.CompilerException):
        lcompile('(let* [] "string")')


def test_let_lazy_evaluation(ns: runtime.Namespace):
    code = """
    (if false
      (let [n  (.-name :value)
            ns (.-ns "string")]  ;; this line would fail if we eagerly evaluated
        :true)
      :false)
    """
    assert kw.keyword("false") == lcompile(code)


class TestLetShadowName:
    def test_no_warning_if_no_shadowing_and_warning_disabled(
        self, ns: runtime.Namespace, caplog
    ):
        lcompile("(let [m 3] m)")
        assert_no_logs(caplog)

    def test_no_warning_if_warning_disabled(self, ns: runtime.Namespace, caplog):
        lcompile(
            "(let [m 3] (let [m 4] m))", opts={compiler.WARN_ON_UNUSED_NAMES: False}
        )
        assert_no_logs(caplog)

    def test_no_warning_if_no_shadowing_and_warning_enabled(
        self, ns: runtime.Namespace, caplog
    ):
        lcompile("(let [m 3] m)", opts={compiler.WARN_ON_SHADOWED_NAME: True})
        assert_no_logs(caplog)

    def test_warning_if_warning_enabled(self, ns: runtime.Namespace, caplog):
        lcompile(
            "(let [m 3] (let [m 4] m))", opts={compiler.WARN_ON_SHADOWED_NAME: True}
        )
        assert (
            "basilisp.lang.compiler.parser",
            logging.WARNING,
            "name 'm' shadows name from outer scope",
        ) in caplog.record_tuples

    def test_warning_if_shadowing_var_and_warning_enabled(
        self, ns: runtime.Namespace, caplog
    ):
        code = """
        (def unique-yyenfvhj :a)
        (let [unique-yyenfvhj 3] unique-yyenfvhj)
        """

        lcompile(code, opts={compiler.WARN_ON_SHADOWED_NAME: True})
        assert (
            "basilisp.lang.compiler.parser",
            logging.WARNING,
            "name 'unique-yyenfvhj' shadows def'ed Var from outer scope",
        ) in caplog.record_tuples


class TestLetShadowVar:
    def test_no_warning_if_warning_disabled(self, ns: runtime.Namespace, caplog):
        code = """
        (def unique-gghdjeeh :a)
        (let [unique-gghdjeeh 3] unique-gghdjeeh)
        """

        lcompile(code, opts={compiler.WARN_ON_SHADOWED_VAR: False})
        assert_no_logs(caplog)

    def test_warning_if_warning_enabled(self, ns: runtime.Namespace, caplog):
        code = """
        (def unique-uoieyqq :a)
        (let [unique-uoieyqq 3] unique-uoieyqq)
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_VAR: True})
        assert (
            "basilisp.lang.compiler.parser",
            logging.WARNING,
            "name 'unique-uoieyqq' shadows def'ed Var from outer scope",
        ) in caplog.record_tuples


class TestLetUnusedNames:
    def test_warning_if_warning_enabled(self, ns: runtime.Namespace, caplog):
        lcompile("(let [v 4] :a)", opts={compiler.WARN_ON_UNUSED_NAMES: True})
        assert (
            "basilisp.lang.compiler.parser",
            logging.WARNING,
            f"symbol 'v' defined but not used ({ns}: 1)",
        ) in caplog.record_tuples

    def test_no_warning_if_warning_disabled(self, ns: runtime.Namespace, caplog):
        lcompile("(let [v 4] :a)", opts={compiler.WARN_ON_UNUSED_NAMES: False})
        assert f"symbol 'v' defined but not used ({ns}: 1)" not in caplog.messages

    def test_warning_for_nested_let_if_warning_enabled(
        self, ns: runtime.Namespace, caplog
    ):
        lcompile(
            """
        (let [v 4]
          (let [v 5]
            v))
        """,
            opts={compiler.WARN_ON_UNUSED_NAMES: True},
        )
        assert (
            "basilisp.lang.compiler.parser",
            logging.WARNING,
            f"symbol 'v' defined but not used ({ns}: 1)",
        ) in caplog.record_tuples

    def test_no_warning_for_nested_let_if_warning_disabled(
        self, ns: runtime.Namespace, caplog
    ):
        lcompile(
            """
        (let [v 4]
          (let [v 5]
            v))
        """,
            opts={compiler.WARN_ON_UNUSED_NAMES: False},
        )
        assert f"symbol 'v' defined but not used ({ns}: 1)" not in caplog.messages


def test_loop(ns: runtime.Namespace):
    assert 1 == lcompile("(loop* [a 1] a)")
    assert kw.keyword("keyword") == lcompile('(loop* [a :keyword b "string"] a)')
    assert kw.keyword("value") == lcompile("(loop* [a :value b a] b)")
    assert lmap.map({kw.keyword("length"): 1}) == lcompile(
        "(loop* [a 1 b :length c {b a} a 4] c)"
    )
    assert 4 == lcompile("(loop* [a 1 b :length c {b a} a 4] a)")
    assert "LOWER" == lcompile('(loop* [a "lower"] (.upper a))')
    assert "string" == lcompile('(loop* [] "string")')

    with pytest.raises(compiler.CompilerException):
        lcompile("(loop* [a 'sym] c)")

    code = """
    (import* io)
    (let* [reader (io/StringIO "string")
           writer (io/StringIO)]
      (loop* []
        (let* [c (.read reader 1)]
          (if (not= c "")
            (do
              (.write writer c)
              (recur))
            (.getvalue writer)))))"""
    assert "string" == lcompile(code)

    code = """
    (loop* [s     "tester"
            accum []]
      (if (seq s)
        (recur (rest s) (conj accum (first s)))
        (apply str accum)))"""
    assert "tester" == lcompile(code)


def test_quoted_list(ns: runtime.Namespace):
    assert lcompile("'()") == llist.l()
    assert lcompile("'(str)") == llist.l(sym.symbol("str"))
    assert lcompile("'(str 3)") == llist.l(sym.symbol("str"), 3)
    assert lcompile("'(str 3 :feet-deep)") == llist.l(
        sym.symbol("str"), 3, kw.keyword("feet-deep")
    )


def test_recur(ns: runtime.Namespace):
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

    assert "a" == lcompile('(rev-str "a")')
    assert ":ba" == lcompile('(rev-str "a" :b)')
    assert "3:ba" == lcompile('(rev-str "a" :b 3)')


def test_recur_arity(ns: runtime.Namespace):
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


def test_disallow_recur_in_special_forms(ns: runtime.Namespace):
    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (def b (recur "a")))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (import* (recur "a")))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (.join "" (recur "a")))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (.-p (recur "a")))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (throw (recur "a"))))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (var (recur "a"))))')


def test_disallow_recur_outside_tail(ns: runtime.Namespace):
    with pytest.raises(compiler.CompilerException):
        lcompile("(recur)")

    with pytest.raises(compiler.CompilerException):
        lcompile("(do (recur))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(if true (recur) :b)")

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (do (recur "a") :b))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (if (recur "a") :a :b))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (if (recur "a") :a))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (let [a (recur "a")] a))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (let [a (do (recur "a"))] a))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (let [a (do :b (recur "a"))] a))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (let [a (do (recur "a") :c)] a))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (let [a "a"] (recur a) a))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (loop* [a (recur "a")] a))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (loop* [a (do (recur "a"))] a))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (loop* [a (do :b (recur "a"))] a))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (loop* [a (do (recur "a") :c)] a))')

    with pytest.raises(compiler.CompilerException):
        lcompile('(fn [a] (loop* [a "a"] (recur a) a))')

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (try (do (recur a) :b) (catch AttributeError _ nil)))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (try :b (catch AttributeError _ (do (recur :a) :c))))")

    with pytest.raises(compiler.CompilerException):
        lcompile("(fn [a] (try :b (finally (do (recur :a) :c))))")


def test_named_anonymous_fn_recursion(ns: runtime.Namespace):
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


def test_syntax_quoting(test_ns: str, ns: runtime.Namespace, resolver: reader.Resolver):
    code = """
    (def some-val \"some value!\")

    `(some-val)"""
    assert llist.l(sym.symbol("some-val", ns=test_ns)) == lcompile(
        code, resolver=resolver
    )

    code = """
    (def second-val \"some value!\")

    `(other-val)"""
    assert llist.l(sym.symbol("other-val")) == lcompile(code)

    code = """
    (def a-str \"a definite string\")
    (def a-number 1583)

    `(a-str ~a-number)"""
    assert llist.l(sym.symbol("a-str", ns=test_ns), 1583) == lcompile(
        code, resolver=resolver
    )

    code = """
    (def whatever \"yes, whatever\")
    (def ssss \"a snake\")

    `(whatever ~@[ssss 45])"""
    assert llist.l(sym.symbol("whatever", ns=test_ns), "a snake", 45) == lcompile(
        code, resolver=resolver
    )

    assert llist.l(sym.symbol("my-symbol", ns=test_ns)) == lcompile(
        "`(my-symbol)", resolver
    )


def test_throw(ns: runtime.Namespace):
    with pytest.raises(AttributeError):
        lcompile("(throw (builtins/AttributeError))")

    with pytest.raises(TypeError):
        lcompile("(throw (builtins/TypeError))")

    with pytest.raises(ValueError):
        lcompile("(throw (builtins/ValueError))")


def test_try_catch(capsys, ns: runtime.Namespace):
    code = """
      (try
        (.fake-lower "UPPER")
        (catch AttributeError _ "lower"))
    """
    assert "lower" == lcompile(code)

    code = """
      (try
        (.fake-lower "UPPER")
        (catch builtins/AttributeError e (.-args e)))
    """
    assert ("'str' object has no attribute 'fake_lower'",) == lcompile(code)

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


def test_unquote(ns: runtime.Namespace):
    with pytest.raises(compiler.CompilerException):
        lcompile("~s")

    assert llist.l(sym.symbol("s")) == lcompile("`(s)")

    with pytest.raises(compiler.CompilerException):
        lcompile("`(~s)")


def test_unquote_splicing(ns: runtime.Namespace, resolver: reader.Resolver):
    with pytest.raises(TypeError):
        lcompile("~@[1 2 3]")

    assert llist.l(1, 2, 3) == lcompile("`(~@[1 2 3])")

    assert llist.l(sym.symbol("print", ns="basilisp.core"), 1, 2, 3) == lcompile(
        "`(print ~@[1 2 3])", resolver=resolver
    )

    assert llist.l(llist.l(reader._UNQUOTE_SPLICING, 53233)) == lcompile("'(~@53233)")


def test_aliased_macro_symbol_resolution(ns: runtime.Namespace):
    current_ns: runtime.Namespace = ns
    other_ns_name = sym.symbol("other.ns")
    try:
        other_ns = runtime.Namespace.get_or_create(other_ns_name)
        current_ns.add_alias(other_ns_name, other_ns)
        current_ns.add_alias(sym.symbol("other"), other_ns)

        with runtime.ns_bindings(other_ns_name.name):
            lcompile("(def ^:macro m (fn* [&form v] v))")

        with runtime.ns_bindings(current_ns.name):
            assert kw.keyword("z") == lcompile("(other.ns/m :z)")
            assert kw.keyword("a") == lcompile("(other/m :a)")
    finally:
        runtime.Namespace.remove(other_ns_name)


class TestWarnOnVarIndirection:
    @pytest.fixture
    def other_ns(self, ns: runtime.Namespace):
        current_ns: runtime.Namespace = ns
        other_ns_name = sym.symbol("other.ns")
        try:
            other_ns = runtime.Namespace.get_or_create(other_ns_name)
            Var.intern(other_ns_name, sym.symbol("m"), lambda x: x)
            current_ns.add_alias(other_ns_name, other_ns)
            current_ns.add_alias(sym.symbol("other"), other_ns)

            with runtime.ns_bindings(current_ns.name):
                yield
        finally:
            runtime.Namespace.remove(other_ns_name)

    def test_warning_for_cross_ns_reference(self, other_ns, caplog):
        lcompile(
            "(fn [] (other.ns/m :z))", opts={compiler.WARN_ON_VAR_INDIRECTION: True}
        )
        assert (
            "basilisp.lang.compiler.generator",
            logging.WARNING,
            "could not resolve a direct link to Var 'm'",
        ) in caplog.record_tuples

    def test_no_warning_for_cross_ns_reference_if_warning_disabled(
        self, other_ns, caplog
    ):
        lcompile(
            "(fn [] (other.ns/m :z))", opts={compiler.WARN_ON_VAR_INDIRECTION: False}
        )
        assert ("could not resolve a direct link to Var 'm'") not in caplog.messages

    def test_warning_for_cross_ns_alias_reference(self, other_ns, caplog):
        lcompile("(fn [] (other/m :z))", opts={compiler.WARN_ON_VAR_INDIRECTION: True})
        assert (
            "basilisp.lang.compiler.generator",
            logging.WARNING,
            "could not resolve a direct link to Var 'm'",
        ) in caplog.record_tuples

    def test_no_warning_for_cross_ns_alias_reference_if_warning_disabled(
        self, other_ns, caplog
    ):
        lcompile("(fn [] (other/m :z))", opts={compiler.WARN_ON_VAR_INDIRECTION: False})
        assert ("could not resolve a direct link to Var 'm'") not in caplog.messages

    def test_warning_on_imported_name(self, ns: runtime.Namespace, caplog):
        """Basilisp should be able to directly resolve a link to cross-namespace
        imports, so no warning should be raised."""
        ns.add_import(sym.symbol("string"), __import__("string"))

        with runtime.ns_bindings(ns.name):
            lcompile(
                '(fn [] (string/capwords "capitalize this"))',
                opts={compiler.WARN_ON_VAR_INDIRECTION: True},
            )
            assert (
                "could not resolve a direct link to Python variable 'string/m'"
            ) not in caplog.messages

    def test_exception_raised_for_nonexistent_imported_name(
        self, ns: runtime.Namespace, caplog
    ):
        """If a name does not exist, then a CompilerException will be raised."""
        ns.add_import(sym.symbol("string"), __import__("string"))

        with runtime.ns_bindings(ns.name), pytest.raises(compiler.CompilerException):
            lcompile(
                "(fn [] (string/m :z))", opts={compiler.WARN_ON_VAR_INDIRECTION: True}
            )

    def test_exception_raised_for_nonexistent_var_name(self, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn [] m)", opts={compiler.WARN_ON_VAR_INDIRECTION: True})


def test_var(ns: runtime.Namespace):
    code = """
    (def some-var "a value")

    (var test/some-var)"""

    ns_name = ns.name
    v = lcompile(code)
    assert v == Var.find_in_ns(sym.symbol(ns_name), sym.symbol("some-var"))
    assert v.value == "a value"

    code = """
    (def some-var "a value")

    #'test/some-var"""

    ns_name = ns.name
    v = lcompile(code)
    assert v == Var.find_in_ns(sym.symbol(ns_name), sym.symbol("some-var"))
    assert v.value == "a value"


def test_fraction(ns: runtime.Namespace):
    assert Fraction("22/7") == lcompile("22/7")


def test_inst(ns: runtime.Namespace):
    assert dateparser.parse("2018-01-18T03:26:57.296-00:00") == lcompile(
        '#inst "2018-01-18T03:26:57.296-00:00"'
    )


def test_regex(ns: runtime.Namespace):
    assert lcompile(r'#"\s"') == re.compile(r"\s")


def test_uuid(ns: runtime.Namespace):
    assert uuid.UUID("{0366f074-a8c5-4764-b340-6a5576afd2e8}") == lcompile(
        '#uuid "0366f074-a8c5-4764-b340-6a5576afd2e8"'
    )
