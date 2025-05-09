import asyncio
import datetime
import decimal
import importlib
import inspect
import logging
import os
import pathlib
import re
import sys
import textwrap
import typing
import uuid
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pytest

from basilisp.lang import compiler as compiler
from basilisp.lang import keyword as kw
from basilisp.lang import list as llist
from basilisp.lang import map as lmap
from basilisp.lang import queue as lqueue
from basilisp.lang import reader as reader
from basilisp.lang import runtime as runtime
from basilisp.lang import set as lset
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec
from basilisp.lang.compiler.constants import SYM_INLINE_META_KW, SYM_PRIVATE_META_KEY
from basilisp.lang.exception import format_exception
from basilisp.lang.interfaces import IRecord, IType, IWithMeta
from basilisp.lang.runtime import Var
from basilisp.lang.util import demunge
from tests.basilisp.helpers import CompileFn, get_or_create_ns


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """Disable the `print_generated_python` flag, so we can safely capture
    stderr and stdout for tests which require those facilities."""
    orig = runtime.print_generated_python
    runtime.print_generated_python = Mock(return_value=False)
    yield
    runtime.print_generated_python = orig


@pytest.fixture
def test_ns() -> str:
    return "basilisp.compiler_test"


@pytest.fixture
def compiler_file_path() -> str:
    return "<Compiler Test>"


@pytest.fixture
def resolver() -> reader.Resolver:
    return runtime.resolve_alias


@pytest.fixture
def assert_no_logs(caplog):
    def _assert_no_logs() -> None:
        __tracebackhide__ = True
        log_records = caplog.record_tuples
        if len(log_records) != 0:
            pytest.fail(f"At least one log message found: {log_records}")

    return _assert_no_logs


@pytest.fixture
def assert_matching_logs(
    caplog,
) -> typing.Callable[[str, int, typing.Union[typing.Pattern, str]], None]:
    def _assert_matching_logs(
        logger_name: str, log_level: int, pattern: typing.Union[typing.Pattern, str]
    ) -> None:
        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        __tracebackhide__ = True
        for t in caplog.record_tuples:
            if (
                t[0] == logger_name
                and t[1] == log_level
                and pattern.match(t[2]) is not None
            ):
                return

        pytest.fail(
            "\n".join(
                (
                    f"Expected log to match ({logger_name}, {log_level}, {pattern}), but found none in:",
                    *[f" - {v}" for v in caplog.record_tuples],
                )
            )
        )

    return _assert_matching_logs


@pytest.fixture
def assert_no_matching_logs(
    caplog,
) -> typing.Callable[[str, int, typing.Union[typing.Pattern, str]], None]:
    def _assert_no_matching_logs(
        logger_name: str, log_level: int, pattern: typing.Union[typing.Pattern, str]
    ) -> None:
        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        __tracebackhide__ = True
        for t in caplog.record_tuples:
            if t[0] != logger_name or t[1] != log_level:
                continue
            if pattern.match(t[2]) is not None:
                pytest.fail(
                    "\n".join(
                        (
                            "Unexpected log line found:",
                            f"  Logger:  {logger_name} != {t[0]}",
                            f"  Level:   {log_level} != {t[1]}"
                            f"  Message: '{pattern}' does not match '{t[2]}'",
                        )
                    )
                )

    return _assert_no_matching_logs


def async_to_sync(asyncf, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(asyncf(*args, **kwargs))


class TestExceptionFormat:
    def test_no_cause_exception(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException) as e:
            lcompile("(let* [a 3] b)")

        assert [
            f"{os.linesep}",
            f"  exception: <class 'basilisp.lang.compiler.exception.CompilerException'>{os.linesep}",
            f"      phase: :analyzing{os.linesep}",
            f"    message: unable to resolve symbol 'b' in this context{os.linesep}",
            f"       form: b{os.linesep}",
            f"   location: <Compiler Test>:1{os.linesep}",
        ] == format_exception(e.value)

    def test_exception_with_cause(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException) as e:
            lcompile("(fn*)")

        assert [
            f"{os.linesep}",
            f"  exception: <class 'IndexError'> from <class 'basilisp.lang.compiler.exception.CompilerException'>{os.linesep}",
            f"      phase: :analyzing{os.linesep}",
            f"    message: fn form must match: (fn* name? [arg*] body*) or (fn* name? method*): Index 1 out of bounds{os.linesep}",
            f"       form: (fn*){os.linesep}",
            f"   location: <Compiler Test>:1{os.linesep}",
        ] == format_exception(e.value)

    def test_macroexpansion_exception_with_cause(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException) as e:
            lcompile("(let [a 3]\nb)")

        assert [
            f"{os.linesep}",
            f"  exception: <class 'basilisp.lang.compiler.exception.CompilerException'> from <class 'basilisp.lang.compiler.exception.CompilerException'>{os.linesep}",
            f"      phase: :macroexpansion{os.linesep}",
            f"    message: error occurred during macroexpansion: unable to resolve symbol 'b' in this context{os.linesep}",
            f"       form: (let [a 3] b){os.linesep}",
            f"   location: <Compiler Test>:1-2{os.linesep}",
        ] == format_exception(e.value)

    def test_macroexpansion_non_compiler_error(self, lcompile: CompileFn):
        lcompile("(defmacro badmacro [arg] `[~(.append arg)])")

        with pytest.raises(compiler.CompilerException) as e:
            lcompile("(badmacro :kw)")

        assert [
            f"{os.linesep}",
            f"  exception: <class 'AttributeError'> from <class 'basilisp.lang.compiler.exception.CompilerException'>{os.linesep}",
            f"      phase: :macroexpansion{os.linesep}",
            f"    message: error occurred during macroexpansion: 'Keyword' object has no attribute 'append'{os.linesep}",
            f"       form: (badmacro :kw){os.linesep}",
            f"   location: <Compiler Test>:1{os.linesep}",
        ] == format_exception(e.value)

    class TestExceptionsWithSourceContext:
        @pytest.fixture
        def source_file(self, tmp_path: Path) -> Path:
            return tmp_path / "compiler_test.lpy"

        @pytest.fixture
        def compiler_file_path(self, source_file: Path) -> str:
            return str(source_file)

        @pytest.mark.parametrize("include_newline", [True, False])
        def test_shows_source_context(
            self,
            monkeypatch,
            source_file: Path,
            lcompile: CompileFn,
            include_newline: bool,
        ):
            source_file.write_text(
                textwrap.dedent(
                    """
                    (ns compiler-test)
                
                    (let [a :b]
                      b)
                    """
                ).lstrip()
                + ("\n" if include_newline else "")
            )
            monkeypatch.syspath_prepend(source_file.parent)

            with pytest.raises(compiler.CompilerException) as e:
                lcompile("(require 'compiler-test)")

            formatted = format_exception(e.value, disable_color=True)
            assert re.match(
                (
                    rf"{os.linesep}"
                    rf"  exception: <class 'basilisp.lang.compiler.exception.CompilerException'> from <class 'basilisp.lang.compiler.exception.CompilerException'>{os.linesep}"
                    rf"      phase: :macroexpansion{os.linesep}"
                    rf"    message: error occurred during macroexpansion: unable to resolve symbol 'b' in this context{os.linesep}"
                    rf"       form: \(let \[a :b\] b\){os.linesep}"
                    rf"   location: (?:\w:)?[^:]*:3-4{os.linesep}"
                    rf"    context:{os.linesep}"
                    rf"{os.linesep}"
                    rf" 1   \| \(ns compiler-test\){os.linesep}"
                    rf" 2   \| {os.linesep}"
                    rf" 3 > \| \(let \[a :b]{os.linesep}"
                    rf" 4 > \|   b\){os.linesep}"
                ),
                "".join(formatted),
            )


class TestLiterals:
    def test_nil(self, lcompile: CompileFn):
        assert None is lcompile("nil")

    def test_string(self, lcompile: CompileFn):
        assert lcompile('"some string"') == "some string"
        assert lcompile('""') == ""

    def test_int(self, lcompile: CompileFn):
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

    def test_decimal(self, lcompile: CompileFn):
        assert decimal.Decimal("0.0") == lcompile("0.0M")
        assert decimal.Decimal("0.09387372") == lcompile("0.09387372M")
        assert decimal.Decimal("1.0") == lcompile("1.0M")
        assert decimal.Decimal("1.332") == lcompile("1.332M")
        assert decimal.Decimal("-1.332") == lcompile("-1.332M")
        assert decimal.Decimal("-1.0") == lcompile("-1.0M")
        assert decimal.Decimal("-0.332") == lcompile("-0.332M")
        assert decimal.Decimal("3.14") == lcompile("3.14M")

    def test_float(self, lcompile: CompileFn):
        assert lcompile("0.0") == 0.0
        assert lcompile("0.09387372") == 0.093_873_72
        assert lcompile("1.0") == 1.0
        assert lcompile("1.332") == 1.332
        assert lcompile("-1.332") == -1.332
        assert lcompile("-1.0") == -1.0
        assert lcompile("-0.332") == -0.332

    def test_kw(self, lcompile: CompileFn):
        assert lcompile(":kw") == kw.keyword("kw")
        assert lcompile(":ns/kw") == kw.keyword("kw", ns="ns")
        assert lcompile(":qualified.ns/kw") == kw.keyword("kw", ns="qualified.ns")

    def test_literals(self, lcompile: CompileFn):
        assert lcompile("nil") is None
        assert lcompile("true") is True
        assert lcompile("false") is False

    def test_quoted_symbol(self, lcompile: CompileFn):
        assert lcompile("'sym") == sym.symbol("sym")
        assert lcompile("'ns/sym") == sym.symbol("sym", ns="ns")
        assert lcompile("'qualified.ns/sym") == sym.symbol("sym", ns="qualified.ns")

    def test_map(self, lcompile: CompileFn):
        assert lcompile("{}") == lmap.m()
        assert lcompile('{:a "string"}') == lmap.map({kw.keyword("a"): "string"})
        assert lcompile('{:a "string" 45 :my-age}') == lmap.map(
            {kw.keyword("a"): "string", 45: kw.keyword("my-age")}
        )

    def test_set(self, lcompile: CompileFn):
        assert lcompile("#{}") == lset.s()
        assert lcompile("#{:a}") == lset.s(kw.keyword("a"))
        assert lcompile("#{:a 1}") == lset.s(kw.keyword("a"), 1)

    def test_vec(self, lcompile: CompileFn):
        assert lcompile("[]") == vec.v()
        assert lcompile("[:a]") == vec.v(kw.keyword("a"))
        assert lcompile("[:a 1]") == vec.v(kw.keyword("a"), 1)

    def test_fraction(self, lcompile: CompileFn):
        assert Fraction("22/7") == lcompile("22/7")

    def test_inst(self, lcompile: CompileFn):
        assert (
            datetime.datetime.fromisoformat("2018-01-18T03:26:57.296-00:00")
            == lcompile('#inst "2018-01-18T03:26:57.296-00:00"')
            == datetime.datetime(
                2018, 1, 18, 3, 26, 57, 296000, tzinfo=datetime.timezone.utc
            )
        )

    def test_queue(self, lcompile: CompileFn):
        assert lcompile("#queue ()") == lqueue.EMPTY
        assert lcompile("#queue (1 2 3)") == lqueue.q(1, 2, 3)
        q = lcompile("^:has-meta #queue ()")
        assert q.meta == lmap.map({kw.keyword("has-meta"): True})

    def test_regex(self, lcompile: CompileFn):
        assert lcompile(r'#"\s"') == re.compile(r"\s")

    def test_uuid(self, lcompile: CompileFn):
        assert uuid.UUID("{0366f074-a8c5-4764-b340-6a5576afd2e8}") == lcompile(
            '#uuid "0366f074-a8c5-4764-b340-6a5576afd2e8"'
        )

    def test_py_dict(self, lcompile: CompileFn):
        assert isinstance(lcompile("#py {}"), dict)
        assert {} == lcompile("#py {}")
        assert {kw.keyword("a"): 1, "b": "str"} == lcompile('#py {:a 1 "b" "str"}')

    def test_py_list(self, lcompile: CompileFn):
        assert isinstance(lcompile("#py []"), list)
        assert [] == lcompile("#py []")
        assert [1, kw.keyword("a"), "str"] == lcompile('#py [1 :a "str"]')

    def test_py_set(self, lcompile: CompileFn):
        assert isinstance(lcompile("#py #{}"), set)
        assert set() == lcompile("#py #{}")
        assert {1, kw.keyword("a"), "str"} == lcompile('#py #{1 :a "str"}')

    def test_py_tuple(self, lcompile: CompileFn):
        assert isinstance(lcompile("#py ()"), tuple)
        assert tuple() == lcompile("#py ()")
        assert (1, kw.keyword("a"), "str") == lcompile('#py (1 :a "str")')


class TestAwait:
    def test_await_must_appear_in_async_def(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn [] (await :a))")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn test [] (await :a))")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn ^:async test [] (fn [] (await :a)))")

        lcompile(
            """
        (fn ^:async test []
          (fn []
            (fn ^:async inner [] (await :a))))
        """
        )

    def test_await_number_of_elems(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn ^:async test [] (await))")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn ^:async test [] (await :a :b))")

    def test_await(self, lcompile: CompileFn):
        awaiter_var: runtime.Var = lcompile(
            """
        (def unique-tywend
          (fn ^:async unique-tywend
            []
            :await-result))

        (def unique-jkeddd
          (fn ^:async unique-jkeddd
            []
            (await (unique-tywend))))
        """
        )

        awaiter = awaiter_var.value
        assert kw.keyword("await-result") == async_to_sync(awaiter)


class TestDef:
    def test_def(self, lcompile: CompileFn, ns: runtime.Namespace):
        ns_name = ns.name
        assert lcompile("(def a :some-val)") == Var.find_in_ns(
            sym.symbol(ns_name), sym.symbol("a")
        )
        assert lcompile('(def beep "a sound a robot makes")') == Var.find_in_ns(
            sym.symbol(ns_name), sym.symbol("beep")
        )
        assert lcompile("a") == kw.keyword("some-val")
        assert lcompile("beep") == "a sound a robot makes"

    def test_def_with_docstring(self, lcompile: CompileFn, ns: runtime.Namespace):
        ns_name = ns.name
        assert lcompile('(def z "this is a docstring" :some-val)') == Var.find_in_ns(
            sym.symbol(ns_name), sym.symbol("z")
        )
        assert lcompile("z") == kw.keyword("some-val")
        var = Var.find_in_ns(sym.symbol(ns.name), sym.symbol("z"))
        assert "this is a docstring" == var.meta.val_at(kw.keyword("doc"))

    def test_def_unbound(self, lcompile: CompileFn, ns: runtime.Namespace):
        lcompile("(def a)")
        var = Var.find_in_ns(sym.symbol(ns.name), sym.symbol("a"))
        assert var.root == runtime.Unbound(var)
        assert not var.is_bound

    def test_def_name_with_no_metadata(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        var = lcompile(
            """
        (defmacro defxyz [v] `(def ~(symbol "xyz") ~v))
        (defxyz :val)
        """
        )
        assert var.root == kw.keyword("val")

    def test_recursive_def(self, lcompile: CompileFn, ns: runtime.Namespace):
        lcompile("(def a a)")
        var = Var.find_in_ns(sym.symbol(ns.name), sym.symbol("a"))
        assert var.root == runtime.Unbound(var)
        assert var.is_bound

    def test_def_number_of_elems(self, lcompile: CompileFn, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile("(def)")

        with pytest.raises(compiler.CompilerException):
            lcompile('(def a "docstring" :b :c)')

    def test_def_name_is_symbol(self, lcompile: CompileFn, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile("(def :a)")

    def test_def_docstring_is_string(self, lcompile: CompileFn, ns: runtime.Namespace):
        with pytest.raises(compiler.CompilerException):
            lcompile("(def a :not-a-docstring :a)")

    def test_compiler_metadata(
        self, lcompile: CompileFn, ns: runtime.Namespace, compiler_file_path: str
    ):
        lcompile('(def ^{:doc "Super cool docstring"} unique-oeuene :a)')

        var = ns.find(sym.symbol("unique-oeuene"))
        meta = var.meta

        assert 1 == meta.val_at(kw.keyword("line"))
        assert compiler_file_path == meta.val_at(kw.keyword("file"))
        assert 0 == meta.val_at(kw.keyword("col"))
        assert sym.symbol("unique-oeuene") == meta.val_at(kw.keyword("name"))
        assert ns == meta.val_at(kw.keyword("ns"))
        assert "Super cool docstring" == meta.val_at(kw.keyword("doc"))

    def test_def_sets_tag_ast(self, lcompile: CompileFn, ns: runtime.Namespace):
        lcompile('(def ^python/str s "a string")')
        assert typing.get_type_hints(ns.module)["s"] == str
        assert ns.find(sym.symbol("s")).meta.val_at(kw.keyword("tag")) == str

    def test_def_tag_ast_can_be_complex(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        lcompile('(def ^{:tag #(.lower "TAG FN")} s "a string")')
        assert typing.get_type_hints(ns.module)["s"]() == "tag fn"
        assert ns.find(sym.symbol("s")).meta.val_at(kw.keyword("tag"))() == "tag fn"

    def test_may_only_provide_tag_metadata_on_def_symbol_or_fn(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        with pytest.raises(compiler.CompilerException):
            lcompile("(def ^python/str f (fn ^python/int [a] a))")

    def test_def_allows_fn_return_annotation_with_no_tag(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        var = lcompile("(def f (fn ^python/int [a] a))")
        assert var.meta.val_at(kw.keyword("tag")) is None
        sig = inspect.signature(var.value)
        assert sig.return_annotation == int

    def test_def_allows_tag_with_no_fn_return_annotation(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        var = lcompile("(def ^python/int f (fn [a] a))")
        assert var.meta.val_at(kw.keyword("tag")) == int
        sig = inspect.signature(var.value)
        assert sig.return_annotation == inspect.Parameter.empty

    def test_no_warn_on_redef_meta(
        self, lcompile: CompileFn, ns: runtime.Namespace, assert_no_logs
    ):
        lcompile(
            """
        (def unique-zhddkd :a)
        (def ^:no-warn-on-redef unique-zhddkd :b)
        """
        )
        assert_no_logs()

    def test_warn_on_redef_if_warn_on_redef_meta_missing(
        self,
        lcompile: CompileFn,
        ns: runtime.Namespace,
        assert_matching_logs,
    ):
        lcompile(
            """
        (def unique-djhvyz :a)
        (def unique-djhvyz :b)
        """
        )
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            r"redefining Var 'unique-djhvyz' in namespace [\w_\-\.]+:\d+",
        )

    def test_redef_vars(
        self, lcompile: CompileFn, ns: runtime.Namespace, assert_no_matching_logs
    ):
        assert kw.keyword("b") == lcompile(
            """
        (def ^:redef orig :a)
        (def redef-check (fn* [] orig))
        (def orig :b)
        (redef-check)
        """
        )
        assert_no_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            r"redefining Var 'orig' in namespace [\w_\-\.]+:\d+",
        )

    def test_def_dynamic(self, lcompile: CompileFn):
        v: Var = lcompile("(def ^:dynamic *a-dynamic-var* 1)")
        assert v.dynamic is True
        lcompile("(.push-bindings #'*a-dynamic-var* :hi)")
        assert kw.keyword("hi") == lcompile("*a-dynamic-var*")
        assert kw.keyword("hi") == lcompile("(.pop-bindings #'*a-dynamic-var*)")
        assert 1 == lcompile("*a-dynamic-var*")

        v: Var = lcompile("(def a-regular-var 1)")
        assert v.dynamic is False

        with pytest.raises(runtime.RuntimeException):
            lcompile("(.push-bindings #'a-regular-var :hi)")

        assert 1 == lcompile("a-regular-var")

        with pytest.raises(runtime.RuntimeException):
            assert kw.keyword("hi") == lcompile("(.pop-bindings #'a-regular-var)")

        assert 1 == lcompile("a-regular-var")

    def test_def_fn_with_meta(self, lcompile: CompileFn):
        v: Var = lcompile(
            "(def with-meta-fn-node ^:meta-kw (fn* [] :fn-with-meta-node))"
        )
        assert hasattr(v.value, "meta")
        assert hasattr(v.value, "with_meta")
        assert lmap.map({kw.keyword("meta-kw"): True}) == v.value.meta
        assert kw.keyword("fn-with-meta-node") == v.value()

    def test_redef_unbound_var(self, lcompile: CompileFn):
        v1: Var = lcompile("(def unbound-var)")
        assert runtime.Unbound(v1) == v1.root

        v2: Var = lcompile("(def unbound-var :a)")
        assert kw.keyword("a") == v2.root
        assert v2.is_bound

        assert v1 is v2

    def test_def_unbound_does_not_clear_var_root(self, lcompile: CompileFn):
        v1: Var = lcompile("(def bound-var :a)")
        assert kw.keyword("a") == v1.root
        assert v1.is_bound

        v2: Var = lcompile("(def bound-var)")
        assert kw.keyword("a") == v2.root
        assert v2.is_bound

        assert v1 is v2


class TestDefType:
    @pytest.mark.parametrize("code", ["(deftype*)", "(deftype* Point)"])
    def test_deftype_number_of_elems(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code", ["(deftype* :Point [x y])", '(deftype* "Point" [x y])']
    )
    def test_deftype_name_is_sym(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_deftype_fields_is_vec(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(deftype* Point (x y))")

    @pytest.mark.parametrize(
        "code",
        [
            "(deftype* Point [:x y])",
            "(deftype* Point [x :y])",
            '(deftype* Point [x y "z"])',
        ],
    )
    def test_deftype_fields_are_syms(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_deftype_has_implements_kw(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
            (deftype* Point [x y]
              [collections.abc/Sized]
              (__len__ [this] 2))
            """
            )

    @pytest.mark.parametrize(
        "code",
        [
            """
            (deftype* Point [x y]
              :implements (collections.abc/Sized)
              (__len__ [this] 2))
            """,
            """
            (deftype* Point [x y]
              :implements collections.abc/Sized
              (__len__ [this] 2))
            """,
        ],
    )
    def test_deftype_implements_is_vector(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_deftype_must_declare_implements(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
                (deftype* Point [x y z]
                  (--call-- [this] [x y z]))
                """
            )

    @pytest.mark.parametrize(
        "code",
        [
            """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable])
            """,
            """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable collections.abc/Sized]
              (--call-- [this] [x y z]))
            """,
            """
            (deftype* Point [x y z]
              (--call-- [this] [x y z]))
            """,
        ],
    )
    def test_deftype_impls_must_match_defined_interfaces(
        self, lcompile: CompileFn, code: str
    ):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable collections.abc/Callable]
              (--call-- [this] [x y z])
              (--call-- [this] 1))
            """,
            """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable collections.abc/Sized collections.abc/Callable]
              (--call-- [this] [x y z])
              (--len-- [this] 1)
              (--call-- [this] 1))
             """,
            """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable collections.abc/Callable collections.abc/Sized]
              (--call-- [this] [x y z])
              (--call-- [this] 1)
              (--len-- [this] 1))
            """,
        ],
    )
    def test_deftype_prohibit_duplicate_interface(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            """
            (deftype* Point [x y z]
              :implements [:collections.abc/Callable]
              (--call-- [this] [x y z]))
            """,
            """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable]
              [--call-- [this] [x y z]])
            """,
            """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable :collections.abc/Sized]
              [--call-- [this] [x y z]])
             """,
        ],
    )
    def test_deftype_impls_must_be_sym_or_list(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_deftype_interface_must_be_host_form(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
            (let [a :kw]
              (deftype* Point [x y z]
                :implements [a]
                (--call-- [this] [x y z])))
            """
            )

    @pytest.mark.parametrize(
        "code,ExceptionType",
        [
            (
                """
            (import* collections)
            (deftype* Point [x y z]
              :implements [collections/OrderedDict]
              (keys [this] [x y z]))""",
                compiler.CompilerException,
            ),
            (
                """
            (do
              (def Shape (python/type "Shape" #py () #py {}))
              (deftype* Circle [radius]
                :implements [Shape]))""",
                compiler.CompilerException,
            ),
            (
                """
            ((fn []
                (def Shape (python/type "Shape" #py () #py {}))
                (deftype* Circle [radius]
                  :implements [Shape])))""",
                runtime.RuntimeException,
            ),
        ],
    )
    def test_deftype_interface_must_be_abstract(
        self, lcompile: CompileFn, code: str, ExceptionType
    ):
        with pytest.raises(ExceptionType):
            lcompile(code)

    def test_deftype_name_can_contain_dashes(self, lcompile: CompileFn):
        cls = lcompile("(deftype* three-axis-point [x y z])")
        o = lcompile("(three-axis-point. 1 2 3)")
        assert isinstance(o, cls)
        assert (o.x, o.y, o.z) == (1, 2, 3)

    def test_deftype_allows_empty_abstract_interface(self, lcompile: CompileFn):
        Point = lcompile(
            """
            (deftype* Point [x y z]
              :implements [basilisp.lang.interfaces/IType])"""
        )
        pt = Point(1, 2, 3)
        assert isinstance(pt, IType)

    def test_deftype_allows_empty_dynamic_abstract_interface(self, lcompile: CompileFn):
        Circle = lcompile(
            # TODO: it's currently a bug for the `(import* abc)` to appear
            #       in the same (do ...) block as the rest of this code.
            """
            (import* abc)
            (do
              (def Shape (python/type "Shape" #py (abc/ABC) #py {}))
              (deftype* Circle [radius]
                :implements [Shape]))"""
        )
        c = Circle(1)
        assert c.radius == 1

    @pytest.mark.parametrize(
        "code,ExceptionType",
        [
            (
                """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Collection]
              (--len-- [this] 3))
            """,
                compiler.CompilerException,
            ),
            (
                """
            (do
              (import* abc)
              (def Shape
                (python/type "Shape"
                             #py (abc/ABC)
                             #py {"area"
                                   (abc/abstractmethod
                                    (fn []))}))
              (deftype* Circle [radius]
                :implements [Shape]))
            """,
                compiler.CompilerException,
            ),
            (
                """
                ((fn []
                  (import* abc)
                  (def Shape
                    (python/type "Shape"
                                 #py (abc/ABC)
                                 #py {"area"
                                       (abc/abstractmethod
                                        (fn []))}))
                  (deftype* Circle [radius]
                    :implements [Shape])))
                """,
                runtime.RuntimeException,
            ),
        ],
    )
    def test_deftype_interface_must_implement_all_abstract_methods(
        self,
        lcompile: CompileFn,
        code: str,
        ExceptionType,
    ):
        with pytest.raises(ExceptionType):
            lcompile(code)

    @pytest.mark.parametrize(
        "code,ExceptionType",
        [
            (
                """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Sized]
              (--len-- [this] 3)
              (call [this] :called))
            """,
                compiler.CompilerException,
            ),
            (
                """
            (do
              (import* abc collections.abc)
              (def Shape
                (python/type "Shape"
                             #py (abc/ABC)
                             #py {"area"
                                   (abc/abstractmethod
                                    (fn []))}))
              (deftype* Circle [radius]
                :implements [Shape]
                (area [this] (* 2 radius radius))
                (call [this] :called)))
            """,
                compiler.CompilerException,
            ),
            (
                """
                ((fn []
                  (import* abc collections.abc)
                  (def Shape
                    (python/type "Shape"
                                 #py (abc/ABC)
                                 #py {"area"
                                       (abc/abstractmethod
                                        (fn []))}))
                  (deftype* Circle [radius]
                    :implements [Shape]
                    (area [this] (* 2 radius radius))
                    (call [this] :called))))
                """,
                runtime.RuntimeException,
            ),
        ],
    )
    def test_deftype_may_not_add_extra_methods_to_interface(
        self,
        lcompile: CompileFn,
        code: str,
        ExceptionType,
    ):
        with pytest.raises(ExceptionType):
            lcompile(code)

    @pytest.mark.parametrize(
        "code", ["(deftype* Shape [])", "(deftype* Shape [] :implements [])"]
    )
    def test_deftype_interface_may_have_no_fields_or_methods(
        self,
        lcompile: CompileFn,
        code: str,
    ):
        lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            """
        (deftype* Point [x y z]
          :implements []
          (__str__ [this]
            (python/repr #py ("Point" x y z))))
        """,
            """
        (deftype* Point [x y z]
          :implements [python/object]
          (__str__ [this]
            (python/repr #py ("Point" x y z))))
        """,
            """
        (do
          (def PyObject python/object)
          (deftype* Point [x y z]
          :implements [PyObject]
          (__str__ [this]
            (python/repr #py ("Point" x y z)))))
        """,
        ],
    )
    def test_deftype_interface_may_implement_only_some_object_methods(
        self, lcompile: CompileFn, code: str
    ):
        Point = lcompile(code)
        pt = Point(1, 2, 3)
        assert "('Point', 1, 2, 3)" == str(pt)

    @pytest.mark.skipif(
        sys.version_info < (3, 10), reason="Fails for versions of Python before 3.10"
    )
    def test_deftype_field_tag_annotations(self, lcompile: CompileFn):
        Point = lcompile("(deftype* Rectangle [^python/int x y])")
        hints = typing.get_type_hints(Point)
        assert hints["x"] == int
        assert "y" not in hints

    @pytest.mark.skipif(
        sys.version_info > (3, 9), reason="This version is intended for 3.9"
    )
    def test_deftype_field_tag_annotations_pre310(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        Point = lcompile("(deftype* Rectangle [^python/int x y])")
        hints = typing.get_type_hints(Point, globalns=ns.module.__dict__)
        assert hints["x"] == int
        assert "y" not in hints

    def test_deftype_field_with_complex_tag_annotations(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                '(deftype* Rectangle [^python/int x ^{:tag #(.upper "a float")} y])'
            )

    @pytest.mark.parametrize(
        "code",
        [
            """
        (deftype* Point [x y z]
          :implements [WithProp]
          (^:property prop [this] [x y z])
          (^:classmethod prop [cls] cls))
        """,
            """
        (deftype* Point [x y z]
          :implements [WithProp]
          (^:classmethod prop [cls] cls)
          (^:property prop [this] [x y z]))
        """,
        ],
    )
    def test_deftype_property_and_method_names_cannot_overlap(
        self, lcompile: CompileFn, code: str
    ):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                f"""
        (import* abc)
        (def WithProp
          (python/type "WithProp"
                         #py (abc/ABC)
                         #py {{"prop"
                              (python/property
                               (abc/abstractmethod
                                (fn [self])))}}))
        (def WithCls
              (python/type "WithCls"
                             #py (abc/ABC)
                             #py {{"prop"
                                  (python/classmethod
                                   (abc/abstractmethod
                                    (fn [cls])))}}))
        {code}
        """
            )

    class TestDefTypeBases:
        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* argparse)
                (deftype* SomeAction []
                  :implements [^:abstract argparse/Action]
                  (__call__ [this]))""",
                """
                (def AABase
                  (python/type "AABase" #py () #py {"some_method" (fn [this])}))
                (deftype* SomeAction []
                  :implements [^:abstract AABase]
                  (some-method [this]))""",
                """
                (do
                  (import* argparse)
                  (deftype* SomeAction []
                    :implements [^:abstract argparse/Action]
                    (__call__ [this])))""",
                """
                (do
                  (def AABase
                    (python/type "AABase" #py () #py {"some_method" (fn [this])}))
                  (deftype* SomeAction []
                    :implements [^:abstract AABase]
                    (some-method [this])))""",
            ],
        )
        def test_deftype_allows_artificially_abstract_super_type(
            self, lcompile: CompileFn, code: str
        ):
            lcompile(code)

        @pytest.mark.parametrize(
            "code,ExceptionType",
            [
                (
                    """
                    (import* argparse)
                    (deftype* SomeAction []
                      :implements [^:abstract argparse/Action]
                      (__call__ [this])
                      (do-action [this]))""",
                    compiler.CompilerException,
                ),
                (
                    """
                    (def AABase
                      (python/type "AABase" #py () #py {"some_method" (fn [this])}))
                    (deftype* SomeAction []
                      :implements [^:abstract AABase]
                      (some-method [this])
                      (other-method [this]))""",
                    compiler.CompilerException,
                ),
                (
                    """
                    (do
                      (import* argparse)
                      (deftype* SomeAction []
                        :implements [^:abstract argparse/Action]
                        (__call__ [this])
                        (do-action [this])))""",
                    compiler.CompilerException,
                ),
                (
                    """
                    (do
                      (def AABase
                        (python/type "AABase" #py () #py {"some_method" (fn [this])}))
                      (deftype* SomeAction []
                        :implements [^:abstract AABase]
                        (some-method [this])
                        (other-method [this])))""",
                    compiler.CompilerException,
                ),
                (
                    """
                    ((fn []
                      (def AABase
                        (python/type "AABase" #py () #py {"some_method" (fn [this])}))
                      (deftype* SomeAction []
                        :implements [^:abstract AABase]
                        (some-method [this])
                        (other-method [this]))))""",
                    runtime.RuntimeException,
                ),
            ],
        )
        def test_deftype_disallows_extra_methods_if_not_in_aa_super_type(
            self, lcompile: CompileFn, code: str, ExceptionType
        ):
            with pytest.raises(ExceptionType):
                lcompile(code)

        @pytest.mark.parametrize(
            "code,ExceptionType",
            [
                (
                    """
                    (import* io)
                    (deftype* SomeReader []
                      :implements [^:abstract ^{:abstract-members '(:read)} io/IOBase]
                      (read [this v]))""",
                    compiler.CompilerException,
                ),
                (
                    """
                    (import* io)
                    (deftype* SomeReader []
                      :implements [^:abstract ^{:abstract-members [:read]} io/IOBase]
                      (read [this v]))""",
                    compiler.CompilerException,
                ),
                (
                    """
                    (import* io)
                    (deftype* SomeReader []
                      :implements [^:abstract ^{:abstract-members #py [:read]} io/IOBase]
                      (read [this v]))""",
                    compiler.CompilerException,
                ),
            ],
        )
        def test_deftype_abstract_members_must_be_a_set(
            self, lcompile: CompileFn, code: str, ExceptionType
        ):
            with pytest.raises(ExceptionType):
                lcompile(code)

        def test_deftype_abstract_members_must_have_elements(self, lcompile: CompileFn):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (import* io)
                    (deftype* SomeReader []
                      :implements [^:abstract ^{:abstract-members #{}} io/IOBase]
                      (read [this v]))
                    """
                )

        def test_deftype_abstract_should_not_have_namespaces(
            self, lcompile: CompileFn, assert_matching_logs
        ):
            lcompile(
                """
                (import* io)
                (deftype* SomeReader []
                      :implements [^:abstract ^{:abstract-members #{:io/read}} io/IOBase]
                  (read [this v]))
                """
            )
            assert_matching_logs(
                "basilisp.lang.compiler.analyzer",
                logging.WARNING,
                "Unexpected namespace for artificially abstract member to deftype*",
            )

        def test_deftype_abstract_members_names_must_be_str_keyword_or_symbol(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (import* io)
                    (deftype* SomeReader []
                      :implements [^:abstract ^{:abstract-members #{1 :read}} io/IOBase]
                      (read [this v]))
                    """,
                )

        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* io)
                (deftype* SomeReader []
                  :implements [^:abstract ^{:abstract-members #{:read :write}} io/IOBase]
                  (read [this n])
                  (write [this v]))
                """,
                """
                (import* io)
                (deftype* SomeReader []
                  :implements [^:abstract ^{:abstract-members #{read write}} io/IOBase]
                  (read [this n])
                  (write [this v]))
                """,
                """
                (import* io)
                (deftype* SomeReader []
                  :implements [^:abstract ^{:abstract-members #{"read" "write"}} io/IOBase]
                  (read [this n])
                  (write [this v]))
                """,
            ],
        )
        def test_deftype_artificially_abstract_members(
            self,
            lcompile: CompileFn,
            code: str,
        ):
            instance = lcompile(code)
            assert instance.read is not None
            assert instance.write is not None

    class TestDefTypeFields:
        def test_deftype_fields(self, lcompile: CompileFn):
            Point = lcompile("(deftype* Point [x y z])")
            pt = Point(1, 2, 3)
            assert (1, 2, 3) == (pt.x, pt.y, pt.z)

        def test_deftype_mutable_field(self, lcompile: CompileFn):
            Point = lcompile(
                """
            (import* collections.abc)
            (deftype* Point [^:mutable x y z]
              :implements [collections.abc/Callable]
              (--call-- [this new-x]
                (set! x new-x)))
            """
            )
            pt = Point(1, 2, 3)
            assert (1, 2, 3) == (pt.x, pt.y, pt.z)
            pt(4)
            assert (4, 2, 3) == (pt.x, pt.y, pt.z)

        def test_deftype_cannot_set_immutable_field(self, lcompile: CompileFn):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (import* collections.abc)
                (deftype* Point [^:mutable x y z]
                  :implements [collections.abc/Callable]
                  (--call-- [this new-y]
                    (set! y new-y)))
                """
                )

        def test_deftype_allow_default_fields(self, lcompile: CompileFn):
            Point = lcompile("(deftype* Point [x ^{:default 2} y ^{:default 3} z])")
            pt = Point(1)
            assert (1, 2, 3) == (pt.x, pt.y, pt.z)
            pt1 = Point(1, 4)
            assert (1, 4, 3) == (pt1.x, pt1.y, pt1.z)
            pt2 = Point(1, 4, 5)
            assert (1, 4, 5) == (pt2.x, pt2.y, pt2.z)

        @pytest.mark.parametrize(
            "code",
            [
                "(deftype* Point [^{:default 1} x y z])",
                "(deftype* Point [x ^{:default 2} y z])",
            ],
        )
        def test_deftype_disallow_non_default_fields_after_default(
            self, lcompile: CompileFn, code: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

    class TestDefTypeMember:
        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* collections.abc)
                (deftype* Point [x y z]
                  :implements [collections.abc/Callable]
                  (:--call-- [this] [x y z]))
                """,
                """
                (import* collections.abc)
                (deftype* Point [x y z]
                  collections.abc/Callable
                  (\"--call--\" [this] [x y z]))
                """,
            ],
        )
        def test_deftype_member_is_named_by_sym(self, lcompile: CompileFn, code: str):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

        def test_deftype_member_args_are_vec(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (import* collections.abc)
                (deftype* Point [x y z]
                  :implements [collections.abc/Callable]
                  (--call-- (this) [x y z]))
                """
                )

        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* collections.abc)
                (deftype* Point [x y z]
                  :implements [collections.abc/Callable]
                  (^:property ^:staticmethod __call__ [this]
                    [x y z]))
                """,
                """
                (import* collections.abc)
                (deftype* Point [x y z]
                  collections.abc/Callable
                  (^:classmethod ^:property __call__ [this]
                  [x y z]))
                """,
                """
                (import* collections.abc)
                (deftype* Point [x y z]
                  collections.abc/Callable
                  (^:classmethod ^:staticmethod __call__ [this]
                  [x y z]))
                """,
            ],
        )
        def test_deftype_member_may_not_be_multiple_types(
            self, lcompile: CompileFn, code: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

    class TestDefTypeClassMethod:
        @pytest.fixture(autouse=True)
        def class_interface(self, lcompile: CompileFn):
            return lcompile(
                """
            (import* abc)
            (def WithCls
              (python/type "WithCls"
                             #py (abc/ABC)
                             #py {"create"
                                  (python/classmethod
                                   (abc/abstractmethod
                                    (fn [cls])))}))
            """
            )

        @pytest.mark.parametrize(
            "code,ExceptionType",
            [
                (
                    """
                    (deftype* Point [x y z]
                      :implements [WithCls])
                    """,
                    compiler.CompilerException,
                ),
                (
                    """
                    (do
                      (import* abc)
                      (def WithClassMethod
                        (python/type "WithCls"
                                     #py (abc/ABC)
                                     #py {"make"
                                          (python/classmethod
                                           (abc/abstractmethod
                                            (fn [cls])))}))
                      (deftype* Point [x y z]
                        :implements [WithClassMethod]))
                    """,
                    compiler.CompilerException,
                ),
                (
                    """
                    ((fn []
                      (import* abc)
                      (def WithClassMethod
                        (python/type "WithCls"
                                     #py (abc/ABC)
                                     #py {"make"
                                          (python/classmethod
                                           (abc/abstractmethod
                                            (fn [cls])))}))
                      (deftype* Point [x y z]
                        :implements [WithClassMethod])))
                    """,
                    runtime.RuntimeException,
                ),
            ],
        )
        def test_deftype_must_implement_interface_classmethod(
            self, lcompile: CompileFn, code: str, ExceptionType
        ):
            with pytest.raises(ExceptionType):
                lcompile(code)

        @pytest.mark.parametrize(
            "code",
            [
                """
            (deftype* Point [x y z]
              :implements [WithCls]
              (^:classmethod create [:cls]
                [x y z]))
              """,
                """
            (deftype* Point [x y z]
              :implements [WithCls]
              (^:classmethod create [cls :arg2]
                [x y z]))
              """,
            ],
        )
        def test_deftype_classmethod_args_are_syms(
            self, lcompile: CompileFn, code: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

        def test_deftype_classmethod_may_not_reference_fields(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (deftype* Point [x y z]
                  :implements [WithCls]
                  (^:classmethod create [cls]
                    [x y z]))"""
                )

        def test_deftype_classmethod_args_includes_cls(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (deftype* Point [x y z]
                  :implements [WithCls]
                  (^:classmethod create []))
                    """
                )

        def test_deftype_classmethod_disallows_recur(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (deftype* Point [x]
                  :implements [WithCls]
                  (^:classmethod create [cls]
                    (recur)))
                """
                )

        def test_deftype_can_have_classmethod(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
            (deftype* Point [x y z]
              :implements [WithCls]
              (^:classmethod create [cls x y z]
                (cls x y z))
              (__eq__ [this other]
                (operator/eq
                 [x y z] 
                 [(.-x other) (.-y other) (.-z other)])))"""
            )
            assert Point(1, 2, 3) == Point.create(1, 2, 3)

        def test_deftype_symboltable_is_restored_after_classmethod(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
            (deftype* Point [x y z]
              :implements [WithCls]
              (^:classmethod create [cls x y z]
                (cls x y z))
              (__str__ [this]
                (python/str [x y z])))"""
            )
            pt = Point.create(1, 2, 3)
            assert "[1 2 3]" == str(pt)

        def test_deftype_empty_classmethod_body(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
            (deftype* Point [x y z]
              :implements [WithCls]
              (^:classmethod create [cls]))"""
            )
            assert None is Point.create()

        def test_deftype_classmethod_returns_value(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
            (import* types)
            (do
              (def a (types/SimpleNamespace))
              (deftype* Point [x]
                :implements [WithCls]
                (^:classmethod create [cls x]
                  (set! (.-val a) x)))
              Point)"""
            )
            assert kw.keyword("a") is Point.create(kw.keyword("a"))

        def test_deftype_classmethod_only_support_valid_kwarg_strategies(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (deftype* Point [x y z]
                  :implements [WithCls]
                  (^:classmethod ^{:kwargs :kwarg-it} create [cls]))"""
                )

        def test_deftype_classmethod_apply_kwargs(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
                (deftype* Point [x y z]
                  :implements [WithCls]
                  (^:classmethod ^{:kwargs :apply} create
                    [cls & args]
                    (let [m (apply hash-map args)]
                      (cls (:x m) (:y m) (:z m)))))"""
            )

            pt = Point.create(x=1, y=2, z=3)
            assert (1, 2, 3) == (pt.x, pt.y, pt.z)

        def test_deftype_classmethod_collect_kwargs(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
                (deftype* Point [x y z]
                  :implements [WithCls]
                  (^:classmethod ^{:kwargs :collect} create
                    [cls x y m]
                    (cls x y (:z m))))"""
            )

            pt = Point.create(1, 2, z=3)
            assert (1, 2, 3) == (pt.x, pt.y, pt.z)

    class TestDefTypeMethod:
        def test_deftype_fields_and_methods(self, lcompile: CompileFn):
            Point = lcompile(
                """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable collections.abc/Sized]
              (--len-- [this] 1)
              (--call-- [this] [x y z]))
            """
            )
            pt = Point(1, 2, 3)
            assert 1 == len(pt)
            assert vec.v(1, 2, 3) == pt()
            assert (1, 2, 3) == (pt.x, pt.y, pt.z)

        def test_deftype_method_with_args(self, lcompile: CompileFn):
            Point = lcompile(
                """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable]
              (--call-- [this i j k] [x i y j z k]))
            """
            )
            pt = Point(1, 2, 3)
            assert vec.v(1, 4, 2, 5, 3, 6) == pt(4, 5, 6)
            assert (1, 2, 3) == (pt.x, pt.y, pt.z)

        @pytest.mark.parametrize(
            "code",
            [
                """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable]
              (--call-- [this &]))
            """,
                """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable]
              (--call-- [this & :args]))
            """,
            ],
        )
        def test_deftype_method_with_varargs_malformed(
            self, lcompile: CompileFn, code: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

        def test_deftype_method_with_varargs(self, lcompile: CompileFn):
            Mirror = lcompile(
                """
            (import* collections.abc)
            (deftype* Mirror [x]
              :implements [collections.abc/Callable]
              (--call-- [this & args] [x args]))
            """
            )
            mirror = Mirror("Beauty is in the eye of the beholder")
            assert vec.v(
                "Beauty is in the eye of the beholder", llist.l(1, 2, 3)
            ) == mirror(1, 2, 3)

        def test_deftype_can_refer_to_type_within_methods(self, lcompile: CompileFn):
            Point = lcompile(
                """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable]
              (--call-- [this i j k]
                (Point i j k)))
            """
            )
            pt = Point(1, 2, 3)
            assert (1, 2, 3) == (pt.x, pt.y, pt.z)
            pt2 = pt(4, 5, 6)
            assert (4, 5, 6) == (pt2.x, pt2.y, pt2.z)

        def test_deftype_empty_method_body(self, lcompile: CompileFn):
            Point = lcompile(
                """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable]
              (--call-- [this]))
            """
            )
            pt = Point(1, 2, 3)
            assert None is pt()
            assert (1, 2, 3) == (pt.x, pt.y, pt.z)

        def test_deftype_method_allows_recur(self, lcompile: CompileFn):
            Point = lcompile(
                """
            (import* collections.abc operator)
            (deftype* Point [x]
              :implements [collections.abc/Callable]
              (--call-- [this sum start]
                (if (operator/gt start 0)
                  (recur (operator/add sum start) (operator/sub start 1))
                  (operator/add sum x))))
            """
            )
            pt = Point(7)
            assert 22 == pt(0, 5)

        def test_deftype_method_args_vec_includes_this(self, lcompile: CompileFn):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (import* collections.abc)
                (deftype* Point [x y z]
                  :implements [collections.abc/Callable]
                  (--call-- [] [x y z]))
                """
                )

        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* collections.abc)
                (deftype* Point [x y z]
                  :implements [collections.abc/Callable]
                  (--call-- [\"this\"] [x y z]))
                """,
                """
                (import* collections.abc)
                (deftype* Point [x y z]
                  :implements [collections.abc/Callable]
                  (--call-- [this :new] [x y z]))
                """,
            ],
        )
        def test_deftype_method_args_are_syms(self, lcompile: CompileFn, code: str):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

        def test_deftype_method_returns_value(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
            (import* collections.abc)
            (deftype* Point [^:mutable x]
              :implements [collections.abc/Callable]
              (--call-- [this new-val]
                (set! x new-val)))"""
            )
            pt = Point(1)
            assert pt.x == 1
            assert 5 == pt(5)
            assert pt.x == 5

        def test_deftype_method_only_support_valid_kwarg_strategies(
            self, lcompile: CompileFn
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (import* collections.abc)
                (deftype* Point [x y z]
                  :implements [collections.abc/Callable]
                  (^{:kwargs :kwarg-it} --call-- [this]))"""
                )

        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* collections.abc)
                (deftype* Point [x y z]
                  :implements [collections.abc/Callable]
                  (^{:kwargs :apply} --call--
                    [this & args]
                    (merge {:x x :y y :z z} (apply hash-map args))))""",
                """
                (import* collections.abc)
                (deftype* Point [x y z]
                  :implements [collections.abc/Callable]
                  (^{:kwargs :collect} --call--
                    [this kwargs]
                    (merge {:x x :y y :z z} kwargs)))""",
            ],
        )
        def test_deftype_method_kwargs(self, lcompile: CompileFn, code: str):
            Point = lcompile(code)

            pt = Point(1, 2, 3)
            assert lmap.map(
                {
                    kw.keyword("w"): 2,
                    kw.keyword("x"): 1,
                    kw.keyword("y"): 4,
                    kw.keyword("z"): 3,
                }
            ) == pt(w=2, y=4)

        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* collections.abc)
                (deftype* Point [x y z]
                  :implements [collections.abc/Callable]
                  (--call-- [this]
                    :no-args)
                  (--call-- [this]
                    :also-no-args))
            """,
                """
                (import* collections.abc)
                (deftype* Point [x y z]
                  :implements [collections.abc/Callable]
                  (--call-- [this s]
                    :one-arg)
                  (--call-- [this s]
                    :also-one-arg))
            """,
                """
                (import* collections.abc)
                (deftype* Point [x y z]
                  :implements [collections.abc/Callable]
                  (--call-- [this]
                    :no-args)
                  (--call-- [this s]
                    :one-arg)
                  (--call-- [this a b]
                    [a b])
                  (--call-- [this s3]
                    :also-one-arg))
            """,
            ],
        )
        def test_no_deftype_method_arity_has_same_fixed_arity(
            self, lcompile: CompileFn, code: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

        @pytest.mark.parametrize(
            "code",
            [
                """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable]
              (--call-- [this & args]
                (concat [:no-starter] args))
              (--call-- [this s & args]
                (concat [s] args)))
            """,
                """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable]
              (--call-- [this s & args]
                (concat [s] args))
              (--call-- [this & args]
                (concat [:no-starter] args)))
            """,
            ],
        )
        def test_deftype_method_cannot_have_two_variadic_arities(
            self, lcompile: CompileFn, code: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

        def test_deftype_method_variadic_method_cannot_have_lower_fixed_arity_than_other_methods(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (import* collections.abc)
                    (deftype* Point [x y z]
                      :implements [collections.abc/Callable]
                      (--call-- [this a b]
                        [a b])
                      (--call-- [this & args]
                        (concat [:no-starter] args)))
                    """
                )

        @pytest.mark.parametrize(
            "code",
            [
                """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable]
              (--call-- [this s] s)
              (^{:kwargs :collect} --call-- [this s kwargs]
                (concat [s] kwargs)))
            """,
                """
            (import* collections.abc)
            (deftype* Point [x y z]
              :implements [collections.abc/Callable]
              (^{:kwargs :collect} --call-- [this kwargs] kwargs)
              (^{:kwargs :apply} --call-- [thi shead & kwargs]
                (apply hash-map :first head kwargs)))
            """,
            ],
        )
        def test_deftype_method_does_not_support_kwargs(
            self, lcompile: CompileFn, code: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

        def test_multi_arity_deftype_method_dispatches_properly(
            self,
            lcompile: CompileFn,
            ns: runtime.Namespace,
        ):
            code = """
            (import* abc)
            (def DoubleTrouble
              (python/type "DoubleTrouble"
                           #py (abc/ABC)
                           #py {"_double_up_arity0" (abc/abstractmethod (fn [self]))
                                "_double_up_arity1" (abc/abstractmethod (fn [self arg1]))
                                "double_up"         (abc/abstractmethod (fn [& args]))}))
            (deftype* Point [x y z]
              :implements [DoubleTrouble]
              (double-up [this] :a)
              (double-up [this s] [:a s]))
            """
            Point = lcompile(code)
            assert callable(Point(1, 2, 3).double_up)
            assert kw.keyword("a") == Point(1, 2, 3).double_up()
            assert vec.v(kw.keyword("a"), kw.keyword("c")) == Point(1, 2, 3).double_up(
                kw.keyword("c")
            )

            code = """
            (import* abc)
            (def InTriplicate
              (python/type "InTriplicate"
                           #py (abc/ABC)
                           #py {"_triple_up_arity0"     (abc/abstractmethod (fn [self]))
                                "_triple_up_arity1"     (abc/abstractmethod (fn [self arg1]))
                                "_triple_up_arity_rest" (abc/abstractmethod (fn [self arg1 & args]))
                                "triple_up"             (abc/abstractmethod (fn [& args]))}))
            (deftype* Point [x y z]
              :implements [InTriplicate]
              (triple-up [this] :no-args)
              (triple-up [this s] s)
              (triple-up [this s & args]
                (concat [s] args)))
            """
            Point = lcompile(code)
            assert callable(Point(1, 2, 3).triple_up)
            assert Point(1, 2, 3).triple_up() == kw.keyword("no-args")
            assert Point(1, 2, 3).triple_up("STRING") == "STRING"
            assert Point(1, 2, 3).triple_up(
                kw.keyword("first-arg"), "second-arg", 3
            ) == llist.l(kw.keyword("first-arg"), "second-arg", 3)

        def test_multi_arity_deftype_method_call_fails_if_no_valid_arity(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
                (import* abc)
                (def InTriplicate
                  (python/type "InTriplicate"
                               #py (abc/ABC)
                               #py {"_triple_up_arity0" (abc/abstractmethod (fn [self]))
                                    "_triple_up_arity1" (abc/abstractmethod (fn [self arg1]))
                                    "_triple_up_arity2" (abc/abstractmethod (fn [self arg1 arg2]))
                                    "triple_up"         (abc/abstractmethod (fn [& args]))}))
                (deftype* Point [x y z]
                  :implements [InTriplicate]
                  (triple-up [this] :send-me-an-arg!)
                  (triple-up [this i] i)
                  (triple-up [this i j] (concat [i] [j])))
                """
            )

            with pytest.raises(runtime.RuntimeException):
                Point(1, 2, 3).triple_up(4, 5, 6)

    class TestDefTypeProperty:
        @pytest.fixture(autouse=True)
        def property_interface(self, lcompile: CompileFn):
            return lcompile(
                """
            (import* abc)
            (def WithProp
              (python/type "WithProp"
                             #py (abc/ABC)
                             #py {"prop"
                                  (python/property
                                   (abc/abstractmethod
                                    (fn [self])))}))
            """
            )

        @pytest.mark.parametrize(
            "code,ExceptionType",
            [
                (
                    """
                (deftype* Point [x y z]
                  :implements [WithProp])
                  """,
                    compiler.CompilerException,
                ),
                (
                    """
                    (do
                      (import* abc)
                      (def WithProperty
                        (python/type "WithProp"
                                     #py (abc/ABC)
                                     #py {"a_property"
                                          (python/property
                                           (abc/abstractmethod
                                            (fn [self])))}))
                      (deftype* Point [x y z]
                        :implements [WithProperty]))
                    """,
                    compiler.CompilerException,
                ),
                (
                    """
                    ((fn []
                      (import* abc)
                      (def WithProperty
                        (python/type "WithProp"
                                     #py (abc/ABC)
                                     #py {"a_property"
                                          (python/property
                                           (abc/abstractmethod
                                            (fn [self])))}))
                      (deftype* Point [x y z]
                        :implements [WithProperty])))
                    """,
                    runtime.RuntimeException,
                ),
            ],
        )
        def test_deftype_must_implement_interface_property(
            self, lcompile: CompileFn, code: str, ExceptionType
        ):
            with pytest.raises(ExceptionType):
                lcompile(code)

        def test_deftype_property_includes_this(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (deftype* Point [x y z]
                  :implements [WithProp]
                  (^:property prop [] [x y z]))
                  """
                )

        def test_deftype_property_args_are_syms(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (deftype* Point [x y z]
                      :implements [WithProp]
                      (^:property prop [:this] [x y z]))
                      """
                )

        def test_deftype_property_may_not_have_args(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (deftype* Point [x y z]
                  :implements [WithProp]
                  (^:property prop [this and-that] [x y z]))
                  """
                )

        def test_deftype_property_disallows_recur(self, lcompile: CompileFn):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (deftype* Point [x]
                  :implements [WithProp]
                  (^:property prop [this]
                    (recur)))
                """
                )

        def test_deftype_field_can_be_property(
            self,
            lcompile: CompileFn,
        ):
            Item = lcompile("(deftype* Item [prop] :implements [WithProp])")
            assert "prop" == Item("prop").prop

        def test_deftype_can_have_property(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
            (deftype* Point [x y z]
              :implements [WithProp]
              (^:property prop [this] [x y z]))"""
            )
            assert vec.v(1, 2, 3) == Point(1, 2, 3).prop

        def test_deftype_empty_property_body(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
            (deftype* Point [x y z]
              :implements [WithProp]
              (^:property prop [this]))"""
            )
            assert None is Point(1, 2, 3).prop

        def test_deftype_property_returns_value(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
            (do
              (deftype* Point [^:mutable x]
                :implements [WithProp]
                (^:property prop [this]
                  (set! x (inc x))))
              Point)"""
            )
            pt = Point(1)
            assert pt.x == 1
            assert pt.prop == 2
            assert pt.x == 2
            assert pt.prop == 3
            assert pt.x == 3

        @pytest.mark.parametrize("kwarg_support", [":apply", ":collect", ":kwarg-it"])
        def test_deftype_property_does_not_support_kwargs(
            self, lcompile: CompileFn, kwarg_support: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    f"""
                (deftype* Point [x y z]
                  :implements [WithProp]
                  (^:property ^{{:kwargs {kwarg_support}}} prop [this]))"""
                )

        def test_deftype_property_may_not_be_multi_arity(self, lcompile: CompileFn):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (deftype* Point [x]
                  :implements [WithProp]
                  (^:property prop [this] :a)
                  (^:property prop [this] :b))
                """
                )

    class TestDefTypeStaticMethod:
        @pytest.fixture(autouse=True)
        def static_interface(self, lcompile: CompileFn):
            return lcompile(
                """
            (import* abc)
            (def WithStatic
              (python/type "WithStatic"
                             #py (abc/ABC)
                             #py {"dostatic"
                                  (python/staticmethod
                                   (abc/abstractmethod
                                    (fn [])))}))
            """
            )

        @pytest.mark.parametrize(
            "code,ExceptionType",
            [
                (
                    """
                    (deftype* Point [x y z]
                      :implements [WithCls])
                    """,
                    compiler.CompilerException,
                ),
                (
                    """
                    (do
                      (import* abc)
                      (def WithStaticMethod
                        (python/type "WithStatic"
                                     #py (abc/ABC)
                                     #py {"do_static_method"
                                          (python/staticmethod
                                           (abc/abstractmethod
                                            (fn [])))}))
                      (deftype* Point [x y z]
                        :implements [WithStaticMethod]))
                    """,
                    compiler.CompilerException,
                ),
                (
                    """
                    ((fn []
                      (import* abc)
                      (def WithStaticMethod
                        (python/type "WithStatic"
                                     #py (abc/ABC)
                                     #py {"do_static_method"
                                          (python/staticmethod
                                           (abc/abstractmethod
                                            (fn [])))}))
                      (deftype* Point [x y z]
                        :implements [WithStaticMethod])))
                    """,
                    runtime.RuntimeException,
                ),
            ],
        )
        def test_deftype_must_implement_interface_staticmethod(
            self,
            lcompile: CompileFn,
            code: str,
            ExceptionType,
        ):
            with pytest.raises(ExceptionType):
                lcompile(code)

        @pytest.mark.parametrize(
            "code",
            [
                """
            (deftype* Point [x y z]
              :implements [WithStatic]
              (^:staticmethod dostatic [:arg]
                [x y z]))
              """,
                """
            (deftype* Point [x y z]
              :implements [WithStatic]
              (^:staticmethod dostatic [arg1 :arg2]
                [x y z]))
              """,
            ],
        )
        def test_deftype_staticmethod_args_are_syms(
            self, lcompile: CompileFn, code: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

        def test_deftype_staticmethod_may_not_reference_fields(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (deftype* Point [x y z]
                  :implements [WithStatic]
                  (^:staticmethod dostatic []
                    [x y z]))"""
                )

        def test_deftype_staticmethod_may_have_no_args(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
            (deftype* Point [x y z]
              :implements [WithStatic]
              (^:staticmethod dostatic []))
              """
            )
            assert None is Point.dostatic()

        def test_deftype_staticmethod_disallows_recur(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (deftype* Point [x]
                  :implements [WithStatic]
                  (^:staticmethod dostatic []
                    (recur)))
                    """
                )

        def test_deftype_can_have_staticmethod(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
            (deftype* Point [x y z]
              :implements [WithStatic]
              (^:staticmethod dostatic [x y z]
                [x y z]))"""
            )
            assert vec.v(1, 2, 3) == Point.dostatic(1, 2, 3)

        def test_deftype_symboltable_is_restored_after_staticmethod(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
            (deftype* Point [x y z]
              :implements [WithStatic]
              (^:staticmethod dostatic [x y z]
                [x y z])
              (__str__ [this]
                (python/str [x y z])))"""
            )
            assert vec.v(1, 2, 3) == Point.dostatic(1, 2, 3)
            assert "[1 2 3]" == str(Point(1, 2, 3))

        def test_deftype_empty_staticmethod_body(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
            (deftype* Point [x y z]
              :implements [WithStatic]
              (^:staticmethod dostatic [arg1 arg2]))"""
            )
            assert None is Point.dostatic("x", "y")

        def test_deftype_staticmethod_returns_value(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
            (import* types)
            (do
              (def a (types/SimpleNamespace))
              (deftype* Point [x]
                :implements [WithStatic]
                (^:staticmethod dostatic [x]
                  (set! (.-val a) x)))
              Point)"""
            )
            assert kw.keyword("a") is Point.dostatic(kw.keyword("a"))

        def test_deftype_staticmethod_only_support_valid_kwarg_strategies(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                (deftype* Point [x y z]
                  :implements [WithStatic]
                  (^:staticmethod ^{:kwargs :kwarg-it} dostatic [cls]))"""
                )

        @pytest.mark.parametrize(
            "code",
            [
                """
            (deftype* Point [x y z]
              :implements [WithStatic]
              (^:staticmethod ^{:kwargs :apply} dostatic
                [& args]
                (apply hash-map args)))""",
                """
            (deftype* Point [x y z]
              :implements [WithStatic]
              (^:staticmethod ^{:kwargs :collect} dostatic
                [m]
                m))""",
            ],
        )
        def test_deftype_staticmethod_kwargs(
            self,
            lcompile: CompileFn,
            code: str,
        ):
            Point = lcompile(code)
            assert lmap.map(
                {kw.keyword("x"): 1, kw.keyword("y"): 2, kw.keyword("z"): 3}
            ) == Point.dostatic(x=1, y=2, z=3)

    class TestDefTypeReaderForm:
        def test_ns_does_not_exist(self, lcompile: CompileFn, test_ns: str):
            with pytest.raises(reader.SyntaxError):
                lcompile(f"#{test_ns}_other.NewType[1 2 3]")

        def test_type_does_not_exist(self, lcompile: CompileFn, test_ns: str):
            with pytest.raises(reader.SyntaxError):
                lcompile(f"#{test_ns}.NewType[1 2 3]")

        def test_type_is_not_itype(
            self, lcompile: CompileFn, test_ns: str, ns: runtime.Namespace
        ):
            # Set the Type in the namespace module manually, because
            # our repeatedly recycled test namespace module does not
            # report to contain NewType with standard deftype*
            setattr(ns.module, "NewType", type("NewType", (object,), {}))
            with pytest.raises(reader.SyntaxError):
                lcompile(f"#{test_ns}.NewType[1 2])")

        def test_type_is_not_irecord(
            self, lcompile: CompileFn, test_ns: str, ns: runtime.Namespace
        ):
            # Set the Type in the namespace module manually, because
            # our repeatedly recycled test namespace module does not
            # report to contain NewType with standard deftype*
            setattr(ns.module, "NewType", type("NewType", (IType, object), {}))
            with pytest.raises(reader.SyntaxError):
                lcompile(f"#{test_ns}.NewType{{:a 1 :b 2}}")


class TestDo:
    def test_do(self, lcompile: CompileFn, ns: runtime.Namespace):
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

    def test_do_returning_name(self, lcompile: CompileFn, ns: runtime.Namespace):
        assert (
            lcompile(
                """
        (let* [a :a]
          (do 
            (def some-value a)
            a))
        """
            )
            == kw.keyword("a")
        )


class TestFunctionShadowName:
    def test_single_arity_fn_no_log_if_warning_disabled(
        self, lcompile: CompileFn, assert_no_matching_logs
    ):
        lcompile("(fn [v] (fn [v] v))")
        assert_no_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            "name 'v' shadows name from outer scope",
        )

    def test_multi_arity_fn_no_log_if_warning_disabled(
        self, lcompile: CompileFn, assert_no_matching_logs
    ):
        lcompile(
            """
        (fn
          ([] :a)
          ([v] (fn [v] v)))
        """
        )
        assert_no_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            "name 'v' shadows name from outer scope",
        )

    def test_single_arity_fn_log_if_warning_enabled(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        lcompile("(fn [v] (fn [v] v))", opts={compiler.WARN_ON_SHADOWED_NAME: True})
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            "name 'v' shadows name from outer scope",
        )

    def test_multi_arity_fn_log_if_warning_enabled(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        code = """
        (fn
          ([] :a)
          ([v] (fn [v] v)))
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_NAME: True})
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            "name 'v' shadows name from outer scope",
        )

    def test_single_arity_fn_log_shadows_var_if_warning_enabled(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        code = """
        (def unique-bljzndd :a)
        (fn [unique-bljzndd] unique-bljzndd)
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_NAME: True})
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            "name 'unique-bljzndd' shadows def'ed Var from outer scope",
        )

    def test_multi_arity_fn_log_shadows_var_if_warning_enabled(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        code = """
        (def unique-yezddid :a)
        (fn
          ([] :b)
          ([unique-yezddid] unique-yezddid))
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_NAME: True})
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            "name 'unique-yezddid' shadows def'ed Var from outer scope",
        )


class TestFunctionShadowVar:
    def test_single_arity_fn_no_log_if_warning_disabled(
        self, lcompile: CompileFn, assert_no_logs
    ):
        code = """
        (def unique-vfsdhsk :a)
        (fn [unique-vfsdhsk] unique-vfsdhsk)
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_VAR: False})
        assert_no_logs()

    def test_multi_arity_fn_no_log_if_warning_disabled(
        self, lcompile: CompileFn, assert_no_logs
    ):
        code = """
        (def unique-mmndheee :a)
        (fn
          ([] :b)
          ([unique-mmndheee] unique-mmndheee))
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_VAR: False})
        assert_no_logs()

    def test_single_arity_fn_log_if_warning_enabled(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        code = """
        (def unique-kuieeid :a)
        (fn [unique-kuieeid] unique-kuieeid)
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_VAR: True})
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            "name 'unique-kuieeid' shadows def'ed Var from outer scope",
        )

    def test_multi_arity_fn_log_if_warning_enabled(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        code = """
        (def unique-peuudcdf :a)
        (fn
          ([] :b)
          ([unique-peuudcdf] unique-peuudcdf))
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_VAR: True})
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            "name 'unique-peuudcdf' shadows def'ed Var from outer scope",
        )


class TestFunctionWarnUnusedName:
    def test_single_arity_fn_no_log_if_warning_disabled(
        self, lcompile: CompileFn, assert_no_logs
    ):
        lcompile("(fn [v] (fn [v] v))", opts={compiler.WARN_ON_UNUSED_NAMES: False})
        assert_no_logs()

    def test_multi_arity_fn_no_log_if_warning_disabled(
        self, lcompile: CompileFn, assert_no_logs
    ):
        lcompile(
            """
        (fn
          ([] :a)
          ([v] (fn [v] v)))
        """,
            opts={compiler.WARN_ON_UNUSED_NAMES: False},
        )
        assert_no_logs()

    def test_single_arity_fn_log_if_warning_enabled(
        self, lcompile: CompileFn, ns: runtime.Namespace, assert_matching_logs
    ):
        lcompile("(fn [v] (fn [v] v))", opts={compiler.WARN_ON_UNUSED_NAMES: True})
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"symbol 'v' defined but not used \({ns}: \d+\)",
        )

    def test_single_arity_fn_underscore_log_if_warning_enabled(
        self, lcompile: CompileFn, ns: runtime.Namespace, assert_no_logs
    ):
        lcompile("(fn [_v] (fn [v] v))", opts={compiler.WARN_ON_UNUSED_NAMES: True})
        assert_no_logs()

    def test_multi_arity_fn_log_if_warning_enabled(
        self, lcompile: CompileFn, ns: runtime.Namespace, assert_matching_logs
    ):
        lcompile(
            """
        (fn
          ([] :a)
          ([v] (fn [v] v)))
        """,
            opts={compiler.WARN_ON_UNUSED_NAMES: True},
        )
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"symbol 'v' defined but not used \({ns}: \d+\)",
        )


class TestFunctionDef:
    def test_fn_with_no_name_or_args(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn*)")

    def test_fn_with_no_args_throws(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* a)")

    def test_fn_with_invalid_name_throws(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* :a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* :a [])")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* :a ([] :a) ([a] a))")

    def test_variadic_arity_fn_has_variadic_argument(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* [m &] m)")

    def test_variadic_arity_fn_method_has_variadic_argument(
        self,
        lcompile: CompileFn,
    ):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* ([] :a) ([m &] m))")

    def test_variadic_arity_fn_collects_args_correctly(self, lcompile: CompileFn):
        f = lcompile("(fn* [& args] args)")
        assert f() is None
        assert f(1) == llist.l(1)
        assert f(1, 2, 3) == llist.l(1, 2, 3)

    def test_fn_argument_vector_is_vector(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* () :a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* (a) a)")

    def test_fn_method_argument_vector_is_vector(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* (() :a) ((a) a))")

    def test_fn_arg_is_symbol(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* [:a] :a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* [a :b] :a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* [a b & :c] :a)")

    def test_fn_method_arg_is_symbol(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* ([a] a) ([a :b] a))")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* ([a] a) ([a & :b] a))")

    def test_fn_has_arity_or_arg(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn* a :a)")

    def test_fn_allows_empty_body(self, lcompile: CompileFn, ns: runtime.Namespace):
        ns_name = ns.name
        fvar = lcompile("(def empty-single (fn* empty-single []))")
        assert Var.find_in_ns(sym.symbol(ns_name), sym.symbol("empty-single")) == fvar
        assert callable(fvar.value)
        assert None is fvar.value()

    def test_fn_method_allows_empty_body(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        ns_name = ns.name
        fvar = lcompile("(def empty-single (fn* empty-single ([]) ([a] :a)))")
        assert Var.find_in_ns(sym.symbol(ns_name), sym.symbol("empty-single")) == fvar
        assert callable(fvar.value)
        assert None is fvar.value()

    @pytest.mark.parametrize(
        "code,args",
        [
            ("(defn test_dfn1a [a b] 5)", ["a", "b"]),
            ("(def test_dfn2 (fn [a b & c] 5))", ["a", "b", "c"]),
            ("(def test_dfn3 (fn [a b {:as c}] 5))", ["a", "b", "c"]),
        ],
    )
    def test_fn_argument_names_unchanged(
        self, lcompile: CompileFn, code: str, args: list[str]
    ):
        fvar = lcompile(code)
        fn_args = list(inspect.signature(fvar.deref()).parameters.keys())
        assert fn_args == args

    @pytest.mark.parametrize(
        "code",
        [
            "(defn ^:safe-py-params test_dfn0 [a b] 5)",
            "(def ^:safe-py-params test_dfn2 (fn [a b & c] 5))",
            "(def test_dfn2 ^:safe-py-params (fn [a b & c] 5))",
            "(def test_dfn2 ^:safe-py-params (fn [a b {:as c}] 5))",
        ],
    )
    def test_fn_argument_names_globally_unique(self, lcompile: CompileFn, code: str):
        fvar = lcompile(code)
        args = list(inspect.signature(fvar.deref()).parameters.keys())
        assert all(
            re.fullmatch(r"[a-z]_\d+", arg) for arg in args
        ), f"unexpected argument names {args}"

    def test_fn_argument_names_unchanged_except_ignored_argument(
        self, lcompile: CompileFn
    ):
        code = "(fn [a b _ _] 5)"
        fvar = lcompile(code)
        fn_args = set(inspect.signature(fvar).parameters.keys())
        assert len(fn_args) == 4
        assert {"a", "b"}.issubset(fn_args)
        remaining_args = fn_args.difference({"a", "b"})
        assert all(
            re.fullmatch(r"__\d+", arg) for arg in remaining_args
        ), f"unexpected argument names {remaining_args}"

    def test_single_arity_fn(self, lcompile: CompileFn, ns: runtime.Namespace):
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

    def test_single_arity_fn_sets_tag_ast(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        f = lcompile("(fn* ^python/str [^python/int i f] (str i f))")
        sig = inspect.signature(f)
        assert sig.return_annotation == str
        params = {k.split("_")[0]: v for k, v in sig.parameters.items()}
        assert params["i"].annotation == int
        assert params["f"].annotation == inspect.Parameter.empty

    def test_single_arity_fn_tag_ast_can_be_complex(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        f = lcompile(
            '(fn* ^{:tag #(.lower "RETURN ANNO")} [^python/int i ^{:tag #(.upper "param anno")} f] (str i f))'
        )
        sig = inspect.signature(f)
        assert sig.return_annotation() == "return anno"
        params = {k.split("_")[0]: v for k, v in sig.parameters.items()}
        assert params["i"].annotation == int
        assert params["f"].annotation() == "PARAM ANNO"

    class TestFunctionKeywordArgSupport:
        def test_only_valid_kwarg_support_strategy(self, lcompile: CompileFn):
            with pytest.raises(compiler.CompilerException):
                lcompile("^{:kwargs :kwarg-it} (fn* [& args] args)")

        def test_single_arity_apply_kwargs(self, lcompile: CompileFn):
            f = lcompile("^{:kwargs :apply} (fn* [& args] args)")

            kwargs = {"some_value": 32, "value": "a string"}
            assert lmap.map(
                {kw.keyword(demunge(k)): v for k, v in kwargs.items()}
            ) == lmap.hash_map(*f(**kwargs))

        def test_single_arity_collect_kwargs(self, lcompile: CompileFn):
            f = lcompile("^{:kwargs :collect} (fn* [a b c kwargs] [a b c kwargs])")

            kwargs = {"some_value": 32, "value": "a string"}
            assert vec.v(
                1,
                "2",
                kw.keyword("three"),
                lmap.map({kw.keyword(demunge(k)): v for k, v in kwargs.items()}),
            ) == f(1, "2", kw.keyword("three"), **kwargs)

        @pytest.mark.parametrize("kwarg_support", [":apply", ":collect"])
        def test_multi_arity_fns_do_not_support_kwargs(
            self, lcompile: CompileFn, kwarg_support: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    f"^{{:kwargs {kwarg_support}}} (fn* ([arg] arg) ([arg kwargs] [arg kwargs]))"
                )

    @pytest.mark.parametrize(
        "code",
        [
            """
        (def f
          (fn* f
            ([] :no-args)
            ([] :also-no-args)))
        """,
            """
        (def f
          (fn* f
            ([s] :one-arg)
            ([s] :also-one-arg)))
        """,
            """
        (def f
          (fn* f
            ([] :no-args)
            ([s] :one-arg)
            ([a b] [a b])
            ([s3] :also-one-arg)))
        """,
        ],
    )
    def test_no_fn_method_has_same_fixed_arity(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            """
        (def f
          (fn* f
            ([& args] (concat [:no-starter] args))
            ([s & args] (concat [s] args))))
        """,
            """
        (def f
          (fn* f
            ([s & args] (concat [s] args))
            ([& args] (concat [:no-starter] args))))
        """,
        ],
    )
    def test_multi_arity_fn_cannot_have_two_variadic_methods(
        self, lcompile: CompileFn, code: str
    ):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            """
            (def f
              (fn* f
                ([s] (concat [s] :one-arg))
                ([& args] (concat [:rest-params] args))))""",
            """
            (def f
              (fn* f
                ([& args] (concat [:rest-params] args))
                ([s] (concat [s] :one-arg))))""",
        ],
    )
    def test_variadic_method_cannot_have_lower_fixed_arity_than_other_methods(
        self, lcompile: CompileFn, code: str
    ):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_multi_arity_fn_dispatches_properly(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
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

    def test_multi_arity_fn_call_fails_if_no_valid_arity(self, lcompile: CompileFn):
        fvar = lcompile(
            """
            (def angry-multi-fn
              (fn* angry-multi-fn
                ([] :send-me-an-arg!)
                ([i] i)
                ([i j] (concat [i] [j]))))
            """
        )
        with pytest.raises(runtime.RuntimeException):
            fvar.value(1, 2, 3)

    def test_multi_arity_fn_sets_tag_ast(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        f = lcompile('(fn* (^python/str [] "s") (^python/int [i] i))')
        sig = inspect.signature(f)
        assert sig.return_annotation == typing.Union[str, int]

    def test_async_single_arity(self, lcompile: CompileFn):
        awaiter_var: runtime.Var = lcompile(
            """
        (def unique-kdghii
          (fn ^:async unique-kdghii
            []
            :await-result))

        (def unique-pqekee
          (fn ^:async unique-pqekee
            []
            (await (unique-kdghii))))
        """
        )

        awaiter = awaiter_var.value
        assert kw.keyword("await-result") == async_to_sync(awaiter)

    def test_async_multi_arity(self, lcompile: CompileFn):
        awaiter_var: runtime.Var = lcompile(
            """
        (def unique-wywbddd
          (fn ^:async unique-wywbddd
            ([]
             :await-result-0)
            ([^:no-warn-when-unused arg]
             :await-result-1)))

        (def unique-hdhene
          (fn ^:async unique-hdhene
            []
            [(await (unique-wywbddd)) (await (unique-wywbddd :arg1))]))
        """
        )

        awaiter = awaiter_var.value
        assert vec.v(
            kw.keyword("await-result-0"), kw.keyword("await-result-1")
        ) == async_to_sync(awaiter)

    def test_async_generator_single_statement(self, lcompile: CompileFn):
        awaiter_fn: runtime.Var = lcompile(
            """
            (fn ^:async unique-kdghii
              []
              (yield :async-generator))
            """
        )

        async def get():
            ret = None
            async for val in awaiter_fn():
                ret = val
            return ret

        assert kw.keyword("async-generator") == async_to_sync(get)

    def test_async_generator_multi_statement(self, lcompile: CompileFn):
        awaiter_fn: runtime.Var = lcompile(
            """
            (fn ^:async unique-kdghii
              []
              (yield :async-generator-1)
              (yield :async-generator-2))
            """
        )

        async def get():
            vals = []
            async for val in awaiter_fn():
                vals.append(val)
            return vals

        assert [
            kw.keyword("async-generator-1"),
            kw.keyword("async-generator-2"),
        ] == async_to_sync(get)

    def test_fn_with_meta_must_be_map(self, lcompile: CompileFn):
        f = lcompile("^:meta-kw (fn* [] :super-unique-kw)")
        with pytest.raises(TypeError):
            f.with_meta(None)

    def test_single_arity_meta(self, lcompile: CompileFn):
        f = lcompile("^:meta-kw (fn* [] :super-unique-kw)")
        assert hasattr(f, "meta")
        assert hasattr(f, "with_meta")
        assert lmap.map({kw.keyword("meta-kw"): True}) == f.meta
        assert kw.keyword("super-unique-kw") == f()

    def test_single_arity_with_meta(self, lcompile: CompileFn):
        f = lcompile(
            """
        (with-meta
          ^:meta-kw (fn* [] :super-unique-kw)
          {:meta-kw false :other-meta "True"})
        """
        )
        assert hasattr(f, "meta")
        assert hasattr(f, "with_meta")
        assert (
            lmap.map({kw.keyword("meta-kw"): False, kw.keyword("other-meta"): "True"})
            == f.meta
        )
        assert kw.keyword("super-unique-kw") == f()

    def test_multi_arity_meta(self, lcompile: CompileFn):
        f = lcompile(
            """
        ^:meta-kw (fn* ([] :arity-0-kw) ([a] [a :arity-1-kw]))
        """
        )
        assert hasattr(f, "meta")
        assert hasattr(f, "with_meta")
        assert lmap.map({kw.keyword("meta-kw"): True}) == f.meta
        assert kw.keyword("arity-0-kw") == f()
        assert vec.v(kw.keyword("jabberwocky"), kw.keyword("arity-1-kw")) == f(
            kw.keyword("jabberwocky")
        )

    def test_multi_arity_with_meta(self, lcompile: CompileFn):
        f = lcompile(
            """
        (with-meta
          ^:meta-kw (fn* ([] :arity-0-kw) ([a] [a :arity-1-kw]))
          {:meta-kw false :other-meta "True"})
        """
        )
        assert hasattr(f, "meta")
        assert hasattr(f, "with_meta")
        assert (
            lmap.map({kw.keyword("meta-kw"): False, kw.keyword("other-meta"): "True"})
            == f.meta
        )
        assert kw.keyword("arity-0-kw") == f()
        assert vec.v(kw.keyword("jabberwocky"), kw.keyword("arity-1-kw")) == f(
            kw.keyword("jabberwocky")
        )

    def test_async_with_meta(self, lcompile: CompileFn):
        f = lcompile(
            """
        (with-meta
          ^:async (fn* [] :super-unique-kw)
          {:meta-kw true})
        """
        )
        assert hasattr(f, "meta")
        assert hasattr(f, "with_meta")
        assert (
            lmap.map({kw.keyword("meta-kw"): True, kw.keyword("async"): True}) == f.meta
        )
        assert kw.keyword("super-unique-kw") == async_to_sync(f)


def test_fn_call(lcompile: CompileFn):
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


class TestFunctionInlining:
    def test_cannot_inline_variadic_fn(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(defn ^:inline f [& args] args)")

    def test_cannot_inline_multi_arity_fn(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
                (defn ^:inline f
                  ([a] a)
                  ([a b] [a b]))
                """
            )

    def test_cannot_inline_multi_expression_fn(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
                (defn ^:inline f
                  [a b]
                  :extra-expression
                  [a b])
                """
            )

    def test_cannot_inline_unimported_module(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
                (defn ^{:inline (fn [s] `(json/loads ~s))} loads
                  [s]
                  s)

                (loads "{}")
                """
            )

    def test_non_boolean_inline_metadata_ignored(self, lcompile: CompileFn):
        v = lcompile("(defn ^{:inline :yes-please} f [a] a)")
        assert v.meta is not None
        assert v.meta.val_at(SYM_INLINE_META_KW) == kw.keyword("yes-please")

    def test_no_inlining_occurs_if_no_inline_meta_specified(self, lcompile: CompileFn):
        val = lcompile(
            """
            (defn ^{:inline (fn [] :a)} f
              []
              :b)

            [(f) ^:no-inline (f)]
            """
        )
        assert val == vec.v(kw.keyword("a"), kw.keyword("b"))

    def test_no_inlining_occurs_if_inline_not_callable(self, lcompile: CompileFn):
        val = lcompile(
            """
            (defn ^{:inline 1} f [] :b)

            [(f) ^:no-inline (f)]
            """
        )
        assert val == vec.v(kw.keyword("b"), kw.keyword("b"))

    @pytest.mark.parametrize("inline_functions", [True, False])
    def test_function_result_is_same_with_or_without_inline(
        self, lcompile: CompileFn, inline_functions: bool
    ):
        val = lcompile(
            """
            (defn ^{:inline (fn [a b] `(+ 1 (* ~a ~b)))} f
              [a b]
              (operator/add 1 (operator/mul a b)))

            (f 2 3)
            """,
            opts={
                compiler.INLINE_FUNCTIONS: inline_functions,
            },
        )
        assert val == 7

    @pytest.mark.parametrize(
        "should_generate_inlines,inline_functions",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_function_result_is_same_with_or_without_auto_inline(
        self, lcompile: CompileFn, should_generate_inlines: bool, inline_functions: bool
    ):
        val = lcompile(
            """
            (defn ^:inline f
              [a b]
              (operator/add 1 (operator/mul a b)))

            (f 2 3)
            """,
            opts={
                compiler.GENERATE_AUTO_INLINES: should_generate_inlines,
                compiler.INLINE_FUNCTIONS: inline_functions,
            },
        )
        assert val == 7


class TestMacroexpandFunctions:
    @pytest.fixture
    def example_macro(self, lcompile: CompileFn):
        lcompile(
            "(defmacro parent [] `(defmacro ~'child [] (fn [])))",
            resolver=runtime.resolve_alias,
        )

    def test_macroexpand_1(self, lcompile: CompileFn, example_macro):
        assert llist.l(
            sym.symbol("defmacro", ns="basilisp.core"),
            sym.symbol("child"),
            vec.EMPTY,
            llist.l(sym.symbol("fn", ns="basilisp.core"), vec.EMPTY),
        ) == compiler.macroexpand_1(llist.l(sym.symbol("parent")))

        assert llist.l(
            sym.symbol("add", ns="operator"), 1, 2
        ) == compiler.macroexpand_1(llist.l(sym.symbol("add", ns="operator"), 1, 2))
        assert sym.symbol("map") == compiler.macroexpand_1(sym.symbol("map"))
        assert llist.l(sym.symbol("map")) == compiler.macroexpand_1(
            llist.l(sym.symbol("map"))
        )
        assert vec.EMPTY == compiler.macroexpand_1(vec.EMPTY)

        assert sym.symbol("non-existent-symbol") == compiler.macroexpand_1(
            sym.symbol("non-existent-symbol")
        )

    def test_macroexpand(self, lcompile: CompileFn, example_macro):
        meta = lmap.map(
            {
                reader.READER_LINE_KW: 1,
                reader.READER_COL_KW: 1,
                reader.READER_END_LINE_KW: 1,
                reader.READER_END_COL_KW: 1,
            }
        )

        assert llist.l(
            sym.symbol("def"),
            sym.symbol("child"),
            llist.l(
                sym.symbol("fn", ns="basilisp.core"),
                sym.symbol("child"),
                vec.v(sym.symbol("&env"), sym.symbol("&form")),
                llist.l(
                    sym.symbol("fn", ns="basilisp.core"),
                    vec.EMPTY,
                ),
            ),
        ) == compiler.macroexpand(llist.l(sym.symbol("parent"), meta=meta))

        assert llist.l(sym.symbol("add", ns="operator"), 1, 2) == compiler.macroexpand(
            llist.l(sym.symbol("add", ns="operator"), 1, 2, meta=meta)
        )
        assert sym.symbol("map") == compiler.macroexpand(sym.symbol("map"))
        assert llist.l(sym.symbol("map")) == compiler.macroexpand(
            llist.l(sym.symbol("map"), meta=meta)
        )
        assert vec.EMPTY == compiler.macroexpand(vec.EMPTY.with_meta(meta))

        assert sym.symbol("non-existent-symbol") == compiler.macroexpand(
            sym.symbol("non-existent-symbol")
        )

        assert sym.symbol(
            "non-existent-symbol", ns="ns.second"
        ) == compiler.macroexpand(sym.symbol("non-existent-symbol", ns="ns.second"))


class TestIf:
    def test_if_number_of_elems(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(if)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(if true)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(if true :true :false :other)")

    def test_if(self, lcompile: CompileFn):
        assert lcompile("(if true :a :b)") == kw.keyword("a")
        assert lcompile("(if false :a :b)") == kw.keyword("b")
        assert lcompile("(if nil :a :b)") == kw.keyword("b")
        assert lcompile("(if true (if false :a :c) :b)") == kw.keyword("c")

        code = """
        (def f (fn* [s] s))

        (f (if true \"YELLING\" \"whispering\"))
        """
        assert "YELLING" == lcompile(code)

    def test_if_with_name(self, lcompile: CompileFn):
        assert lcompile("(let* [a true] (if a :a :b))") == kw.keyword("a")

    def test_truthiness(self, lcompile: CompileFn):
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


class TestImport:
    def test_import_must_have_at_least_one_module(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(import*)")

    @pytest.mark.parametrize(
        "code",
        [
            "(import* :time)",
            '(import* "time")',
            "(import* string :time)",
            '(import* string "time")',
        ],
    )
    def test_import_module_must_be_symbol(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_import_alias_may_not_contain_dot(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(import* [re :as reg.expr])")

    @pytest.mark.parametrize(
        "code",
        [
            "(import* [])",
            "(import* [:time :as py-time])",
            "(import* [time py-time])",
            "(import* [time :as :py-time])",
            "(import* [time :as])",
            "(import* [time :named py-time])",
            "(import* [time :named py time])",
        ],
    )
    def test_import_aliased_module_format(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            "(import* [math :refer ()])",
            "(import* [math :refer {}])",
            "(import* [math :refer #{}])",
        ],
    )
    def test_import_refer_name_collection_must_be_a_vector(
        self, lcompile: CompileFn, code: str
    ):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_import_refer_names_must_have_one_name(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(import* [math :refer []])")

    @pytest.mark.parametrize(
        "code", ["(import* [math :refer [:sqrt]])", '(import* [math :refer ["sqrt"]])']
    )
    def test_import_refer_names_must_by_symbols(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_import_refer_multiple_names(self, lcompile: CompileFn):
        import math

        lcompile("(import* [math :refer [pi sqrt]])")
        refers = lcompile("[pi sqrt]")
        assert refers[0] == math.pi
        assert refers[1] == math.sqrt

    def test_import_refer_all(self, lcompile: CompileFn, ns: runtime.Namespace):
        import math

        lcompile("(import* [math :refer :all])")

        import_refers = {name: val for name, val in ns.import_refers.items()}
        assert import_refers == {
            sym.symbol(name): val
            for name, val in vars(math).items()
            if not runtime._PRIVATE_NAME_PATTERN.fullmatch(name)
        }

    def test_import_module_must_exist(self, lcompile: CompileFn):
        with pytest.raises(ImportError):
            lcompile("(import* real.fake.module)")

    def test_import_resolves_within_do_block(self, lcompile: CompileFn):
        import time

        assert time.perf_counter == lcompile("(do (import* time)) time/perf-counter")
        assert time.perf_counter == lcompile(
            """
            (do (import* [time :as py-time]))
            py-time/perf-counter
            """
        )

    def test_single_import(self, lcompile: CompileFn):
        import time

        assert time.perf_counter == lcompile("(import* time) time/perf-counter")
        assert time.perf_counter == lcompile(
            "(import* [time :as py-time]) py-time/perf-counter"
        )

    def test_multi_import(self, lcompile: CompileFn):
        import string
        import time

        assert [time.perf_counter, string.capwords] == list(
            lcompile(
                "(import* [string :as pystr] time) [time/perf-counter pystr/capwords]"
            )
        )
        assert [string.capwords, time.perf_counter] == list(
            lcompile(
                "(import* string [time :as py-time]) [string/capwords py-time/perf-counter]"
            )
        )

    def test_nested_imports_visible_with_parent(self, lcompile: CompileFn):
        import collections.abc

        assert [collections.OrderedDict, collections.abc.Sized] == lcompile(
            """
        (import* collections collections.abc)
        #py [collections/OrderedDict collections.abc/Sized]
        """
        )

    def test_aliased_nested_import_refers_to_child(self, lcompile: CompileFn):
        import os.path

        assert os.path.exists is lcompile("(import [os.path :as path]) path/exists")

    def test_warn_on_duplicated_import_name(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        lcompile("(import os.path os.path)")
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            r"duplicate name or alias 'os.path' in import",
        )

    def test_warn_on_duplicated_import_name_with_alias(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        lcompile("(import abc [collections.abc :as abc])")
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            r"duplicate name or alias 'abc' in import",
        )

    def test_warn_on_shadowing_by_existing_import(
        self, ns: runtime.Namespace, lcompile: CompileFn, assert_matching_logs
    ):
        lcompile("(import abc) (import [collections.abc :as abc])")
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"name 'abc' may be shadowed by an existing import in '{ns}'",
        )

    def test_warn_on_shadowing_by_existing_import_alias(
        self, ns: runtime.Namespace, lcompile: CompileFn, assert_matching_logs
    ):
        lcompile("(import [collections.abc :as abc]) (import abc)")
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"name 'abc' may be shadowed by an existing import alias in '{ns}'",
        )

    def test_warn_on_shadowing_by_existing_namespace_alias(
        self, ns: runtime.Namespace, lcompile: CompileFn, assert_matching_logs
    ):
        lcompile("(require '[basilisp.string :as abc]) (import abc)")
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"name 'abc' may shadow an existing alias in '{ns}'",
        )


class TestInvoke:
    @pytest.mark.parametrize(
        "code,v",
        [
            ("(python/dict **)", {}),
            ("(python/dict ** :value 1.2)", {"value": 1.2}),
            ('(python/dict ** "value" 1.2)', {"value": 1.2}),
            (
                '(python/dict ** :value 1.2 "other-value" "a string")',
                {"value": 1.2, "other_value": "a string"},
            ),
            (
                '(python/dict ** "value" 1.2 :other-value "a string")',
                {"value": 1.2, "other_value": "a string"},
            ),
        ],
    )
    def test_call_with_kwargs(self, lcompile: CompileFn, code: str, v):
        assert v == lcompile(code)

    @pytest.mark.parametrize(
        "code,v",
        [
            (
                "(python/dict {:value 3.14} ** :value 82)",
                {kw.keyword("value"): 3.14, "value": 82},
            ),
            ('(python/dict {"value" 3.14} ** :value 82)', {"value": 82}),
            ('(python/dict {"value" 3.14} ** "value" 82)', {"value": 82}),
        ],
    )
    def test_kwargs_are_always_strings(self, lcompile: CompileFn, code: str, v):
        assert v == lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            "(python/dict ** **)",
            "(python/dict ** :value ** 3.14)",
            '(python/dict ** :value 3.14 ** :other-value "a string")',
            '(python/dict ** :value 3.14 :other-value "a string" **)',
        ],
    )
    def test_call_with_multiple_kwarg_markers_fails(
        self, lcompile: CompileFn, code: str
    ):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            '(python/dict ** :value 3.14 "value" "a string")',
            '(python/dict ** :value 3.14 :value "a string")',
            '(python/dict ** :value 3.14 :other-value "a string" :value :some-kw)',
            '(python/dict ** "value" 3.14 :other-value "a string" "value" :some-kw)',
        ],
    )
    def test_call_with_duplicate_keys_fails(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            "(python/dict ** :value)",
            '(python/dict ** "value")',
            "(python/dict ** :value 3.14 :other-value)",
            '(python/dict ** "value" :a-keyword :other-key)',
        ],
    )
    def test_call_with_kwargs_and_only_key_fails(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            "(python/dict ** value 1.2)",
            '(python/dict ** :value 1.2 other-value "some string")',
            '(python/dict ** value 1.2 :other-value "some string")',
        ],
    )
    def test_call_with_invalid_key_type_fails(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)


class TestPythonInterop:
    def test_interop_is_valid_type(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile('(. :kw "str")')

        with pytest.raises(compiler.CompilerException):
            lcompile("(. :kw [:vec :of :kws])")

        with pytest.raises(compiler.CompilerException):
            lcompile("(. :kw 1)")

    def test_interop_new(self, lcompile: CompileFn):
        assert "hi" == lcompile('(python.str. "hi")')
        assert "1" == lcompile("(python.str. 1)")
        assert sym.symbol("hi") == lcompile('(basilisp.lang.symbol.Symbol. "hi")')

        with pytest.raises(compiler.CompilerException):
            lcompile('(python.str "hi")')

    def test_interop_new_with_import(self, lcompile: CompileFn, ns: runtime.Namespace):
        import builtins

        ns.add_import(sym.symbol("builtins"), builtins)
        assert "hi" == lcompile('(builtins.str. "hi")')
        assert "1" == lcompile("(builtins.str. 1)")

        with pytest.raises(compiler.CompilerException):
            lcompile('(builtins.str "hi")')

    def test_interop_call_num_elems(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(.upper)")

    def test_interop_prop_method_is_symbol(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile('(. "ALL-UPPER" (:lower))')

        with pytest.raises(compiler.CompilerException):
            lcompile('(. "ALL-UPPER" ("lower"))')

    def test_interop_call(self, lcompile: CompileFn):
        assert "all-upper" == lcompile('(. "ALL-UPPER" lower)')

        assert "LOWER-STRING" == lcompile('(.upper "lower-string")')
        assert "LOWER-STRING" == lcompile('(. "lower-string" (upper))')

        assert "example" == lcompile('(.strip "www.example.com" "cmowz.")')
        assert "example" == lcompile('(. "www.example.com" (strip "cmowz."))')
        assert "example" == lcompile('(. "www.example.com" strip "cmowz.")')

    @pytest.mark.parametrize(
        "code,v",
        [
            ('(python.str/split "a b c")', ["a", "b", "c"]),
            ('(python.str/.split "a b c")', ["a", "b", "c"]),
            (
                '(python.int/from_bytes #b"\\x00\\x10" #?@(:lpy310- [** :byteorder "big"]))',
                16,
            ),
            (
                '(python.int/.from_bytes #b"\\x00\\x10" #?@(:lpy310- [** :byteorder "big"]))',
                16,
            ),
        ],
    )
    def test_interop_qualified_method(self, lcompile: CompileFn, code: str, v):
        assert v == lcompile(code)

    def test_interop_prop_field_is_symbol(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(.- 'some.ns/sym :ns)")

        with pytest.raises(compiler.CompilerException):
            lcompile('(.- \'some.ns/sym "ns")')

    def test_interop_prop_num_elems(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(.- 'some.ns/sym)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(.- 'some.ns/sym ns :argument)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(.-ns 'some.ns/sym :argument)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(. 'some.ns/sym -ns :argument)")

    def test_interop_prop(self, lcompile: CompileFn):
        assert "some.ns" == lcompile("(.-ns 'some.ns/sym)")
        assert "some.ns" == lcompile("(.- 'some.ns/sym ns)")
        assert "some.ns" == lcompile("(. 'some.ns/sym -ns)")
        assert "sym" == lcompile("(.-name 'some.ns/sym)")
        assert "sym" == lcompile("(.- 'some.ns/sym name)")
        assert "sym" == lcompile("(. 'some.ns/sym -name)")
        assert 5 == lcompile(
            '(.-abc (python/type "ip-test" (python/tuple) #py {"abc" 5}))'
        )
        # when prop name matches a bultins name
        assert 6 == lcompile(
            '(.-str (python/type "ip-test" (python/tuple) #py {"str" 6}))'
        )

        with pytest.raises(AttributeError):
            lcompile("(.-fake 'some.ns/sym)")

    def test_interop_quoted(self, lcompile: CompileFn):
        assert lcompile("'(.match pattern)") == llist.l(
            sym.symbol(".match"), sym.symbol("pattern")
        )
        assert lcompile("'(.-pattern regex)") == llist.l(
            sym.symbol(".-pattern"), sym.symbol("regex")
        )


class TestLet:
    def test_let_num_elems(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(let*)")

    def test_let_may_have_empty_bindings(self, lcompile: CompileFn):
        assert None is lcompile("(let* [])")
        assert kw.keyword("kw") == lcompile("(let* [] :kw)")

    def test_let_bindings_must_be_vector(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(let* () :kw)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(let* (a kw) a)")

    def test_let_bindings_must_have_name_and_value(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(let* [a :kw b] a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(let* [a :kw b :other-kw c] a)")

    def test_let_binding_name_must_be_symbol(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(let* [:a :kw] a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(let* [a :kw :b :other-kw] a)")

    def test_let_name_does_not_resolve(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(let* [a 'sym] c)")

    def test_let_may_have_empty_body(self, lcompile: CompileFn):
        assert None is lcompile("(let* [])")
        assert None is lcompile("(let* [a :kw])")

    def test_let_return_bound_name(self, lcompile: CompileFn):
        assert lcompile("(let* [a :a b a] b)") == kw.keyword("a")

    def test_let_bindings_get_tag_ast(self, lcompile: CompileFn, ns: runtime.Namespace):
        lcompile('(let [^python/str s "a string"] s)')
        hints = typing.get_type_hints(ns.module)
        let_var = next(iter(k for k in hints.keys() if k.startswith("s_")), None)
        assert hints[let_var] == str

    def test_let_binding_tag_ast_can_be_complex(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        lcompile('(let [^{:tag #(.lower "TAG FN")} s "a string"] s)')
        hints = typing.get_type_hints(ns.module)
        let_var = next(iter(k for k in hints.keys() if k.startswith("s_")), None)
        assert hints[let_var]() == "tag fn"

    def test_let(self, lcompile: CompileFn):
        assert lcompile("(let* [a 1] a)") == 1
        assert lcompile('(let* [a :keyword b "string"] a)') == kw.keyword("keyword")
        assert lcompile("(let* [a :value b a] b)") == kw.keyword("value")
        assert lcompile("(let* [a 1 b :length c {b a} a 4] c)") == lmap.map(
            {kw.keyword("length"): 1}
        )
        assert lcompile("(let* [a 1 b :length c {b a} a 4] a)") == 4
        assert lcompile('(let* [a "lower"] (.upper a))') == "LOWER"

    def test_let_lazy_evaluation(self, lcompile: CompileFn):
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
        self, lcompile: CompileFn, assert_no_logs
    ):
        lcompile("(let [m 3] m)")
        assert_no_logs()

    def test_no_warning_if_warning_disabled(self, lcompile: CompileFn, assert_no_logs):
        lcompile(
            "(let [m 3] (let [m 4] m))", opts={compiler.WARN_ON_UNUSED_NAMES: False}
        )
        assert_no_logs()

    def test_no_warning_if_no_shadowing_and_warning_enabled(
        self, lcompile: CompileFn, assert_no_logs
    ):
        lcompile("(let [m 3] m)", opts={compiler.WARN_ON_SHADOWED_NAME: True})
        assert_no_logs()

    def test_warning_if_warning_enabled(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        lcompile(
            "(let [m 3] (let [m 4] m))", opts={compiler.WARN_ON_SHADOWED_NAME: True}
        )
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            "name 'm' shadows name from outer scope",
        )

    def test_warning_if_shadowing_var_and_warning_enabled(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        code = """
        (def unique-yyenfvhj :a)
        (let [unique-yyenfvhj 3] unique-yyenfvhj)
        """

        lcompile(code, opts={compiler.WARN_ON_SHADOWED_NAME: True})
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            "name 'unique-yyenfvhj' shadows def'ed Var from outer scope",
        )


class TestLetShadowVar:
    def test_no_warning_if_warning_disabled(self, lcompile: CompileFn, assert_no_logs):
        code = """
        (def unique-gghdjeeh :a)
        (let [unique-gghdjeeh 3] unique-gghdjeeh)
        """

        lcompile(code, opts={compiler.WARN_ON_SHADOWED_VAR: False})
        assert_no_logs()

    def test_warning_if_warning_enabled(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        code = """
        (def unique-uoieyqq :a)
        (let [unique-uoieyqq 3] unique-uoieyqq)
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_VAR: True})
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            "name 'unique-uoieyqq' shadows def'ed Var from outer scope",
        )


class TestLetUnusedNames:
    def test_warning_if_warning_enabled(
        self, lcompile: CompileFn, ns: runtime.Namespace, assert_matching_logs
    ):
        lcompile("(let [v 4] :a)", opts={compiler.WARN_ON_UNUSED_NAMES: True})
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"symbol 'v' defined but not used \({ns}: \d+\)",
        )

    def test_no_warning_if_warning_disabled(
        self, lcompile: CompileFn, ns: runtime.Namespace, assert_no_matching_logs
    ):
        lcompile("(let [v 4] :a)", opts={compiler.WARN_ON_UNUSED_NAMES: False})
        assert_no_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"symbol 'v' defined but not used \({ns}: \d+\)",
        )

    def test_warning_for_nested_let_if_warning_enabled(
        self, lcompile: CompileFn, ns: runtime.Namespace, assert_matching_logs
    ):
        lcompile(
            """
        (let [v 4]
          (let [v 5]
            v))
        """,
            opts={compiler.WARN_ON_UNUSED_NAMES: True},
        )
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"symbol 'v' defined but not used \({ns}: \d+\)",
        )

    def test_no_warning_for_nested_let_if_warning_disabled(
        self, lcompile: CompileFn, ns: runtime.Namespace, assert_no_matching_logs
    ):
        lcompile(
            """
        (let [v 4]
          (let [v 5]
            v))
        """,
            opts={compiler.WARN_ON_UNUSED_NAMES: False},
        )
        assert_no_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"symbol 'v' defined but not used \({ns}: \d+\)",
        )


class TestLetFn:
    def test_letfn_num_elems(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(letfn*)")

    def test_letfn_may_have_empty_bindings(self, lcompile: CompileFn):
        assert None is lcompile("(letfn* [])")
        assert kw.keyword("kw") == lcompile("(letfn* [] :kw)")

    def test_letfn_bindings_must_be_vector(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(letfn* () :kw)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(letfn* (f (fn* [])) f)")

    def test_let_bindings_must_have_name_and_value(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(letfn* [a (fn* []) b] a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(letfn* [a (fn* [] :kw) b (fn* [] :other-kw) c] a)")

    def test_letfn_binding_fns_must_be_list(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(letfn* [f [fn* []]] f)")

    def test_letfn_binding_name_must_be_symbol(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(letfn* [:a (fn* a [])] a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(letfn* [a (fn* a [] :a) :b (fn* :b [] :b)] a)")

    def test_letfn_binding_value_must_be_function(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(letfn* [a :a] a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(letfn* [a (fn* a [] :a) b (vector 1 2 3)] a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(letfn* [a (fn* a [] :a) b (map odd? [1 2 3])] a)")

    def test_letfn_name_does_not_resolve(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(letfn* [a (fn* a [] 'sym)] c)")

    def test_letfn_may_have_empty_body(self, lcompile: CompileFn):
        assert None is lcompile("(letfn* [])")
        assert None is lcompile("(letfn* [a (fn* a [])])")

    def test_let_return_bound_name(self, lcompile: CompileFn):
        f = lcompile("(letfn* [a (fn* a [])] a)")
        assert f() is None

    def test_letfn(self, lcompile: CompileFn):
        assert lcompile("(letfn* [a (fn* a [] 1)] (a))") == 1
        assert lcompile("(letfn* [a (fn* a [] 1) b (fn* b [] 2)] (b))") == 2
        assert lcompile('(letfn* [a (fn* a [] "lower")] (.upper (a)))') == "LOWER"
        assert lcompile(
            "(letfn* [a (fn* a [] :value) b (fn* b [] (a))] (b))"
        ) == kw.keyword("value")
        assert lcompile(
            "(letfn* [a (fn* a [] (b)) b (fn* b [] :value)] (b))"
        ) == kw.keyword("value")

    @pytest.mark.parametrize(
        "v,exp", [(0, True), (1, False), (2, True), (3, False), (4, True)]
    )
    def test_letfn_mutual_recursion(self, lcompile: CompileFn, v: int, exp: bool):
        assert exp is lcompile(
            f"""
        (letfn* [neven? (fn* neven? [n] (if (zero? n) true (nodd? (dec n))))
                 nodd?  (fn* nodd? [n] (if (zero? n) false (neven? (dec n))))]
          (neven? {v}))
        """
        )


class TestLetFnShadowName:
    def test_no_warning_if_no_shadowing_and_warning_disabled(
        self, lcompile: CompileFn, assert_no_logs
    ):
        lcompile("(letfn* [m (fn* m [] 3)] (m))")
        assert_no_logs()

    def test_no_warning_if_warning_disabled(self, lcompile: CompileFn, assert_no_logs):
        lcompile(
            "(letfn* [m (fn* m [] 3)] (letfn* [m (fn* m [] 4)] (m)))",
            opts={compiler.WARN_ON_UNUSED_NAMES: False},
        )
        assert_no_logs()

    def test_no_warning_if_no_shadowing_and_warning_enabled(
        self, lcompile: CompileFn, assert_no_logs
    ):
        lcompile(
            "(letfn* [m (fn* m [] 3)] (m))", opts={compiler.WARN_ON_SHADOWED_NAME: True}
        )
        assert_no_logs()

    def test_warning_if_warning_enabled(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        lcompile(
            "(letfn* [m (fn* m [] 3)] (letfn* [m (fn* m [] 4)] (m)))",
            opts={compiler.WARN_ON_SHADOWED_NAME: True},
        )
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            "name 'm' shadows name from outer scope",
        )

    def test_warning_if_shadowing_var_and_warning_enabled(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        code = """
        (def unique-kdhenne :a)
        (letfn* [unique-kdhenne (fn* unique-kdhenne [] 3)] (unique-kdhenne))
        """

        lcompile(code, opts={compiler.WARN_ON_SHADOWED_NAME: True})
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            "name 'unique-kdhenne' shadows def'ed Var from outer scope",
        )


class TestLetFnShadowVar:
    def test_no_warning_if_warning_disabled(self, lcompile: CompileFn, assert_no_logs):
        code = """
        (def unique-qpkdmdkd :a)
        (letfn* [unique-qpkdmdkd (fn* unique-qpkdmdkd [] 3)] (unique-qpkdmdkd))
        """

        lcompile(code, opts={compiler.WARN_ON_SHADOWED_VAR: False})
        assert_no_logs()

    def test_warning_if_warning_enabled(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        code = """
        (def unique-bdddnda :a)
        (letfn* [unique-bdddnda (fn* unique-bdddnda [] 3)] (unique-bdddnda))
        """
        lcompile(code, opts={compiler.WARN_ON_SHADOWED_VAR: True})
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            "name 'unique-bdddnda' shadows def'ed Var from outer scope",
        )


class TestLetFnUnusedNames:
    def test_warning_if_warning_enabled(
        self, lcompile: CompileFn, ns: runtime.Namespace, assert_matching_logs
    ):
        lcompile(
            "(letfn* [v (fn* v [] 4)] :a)", opts={compiler.WARN_ON_UNUSED_NAMES: True}
        )
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"symbol 'v' defined but not used \({ns}: \d+\)",
        )

    def test_no_warning_if_warning_disabled(
        self, lcompile: CompileFn, ns: runtime.Namespace, assert_no_matching_logs
    ):
        lcompile(
            "(letfn* [v (fn* v [] 4)] :a)", opts={compiler.WARN_ON_UNUSED_NAMES: False}
        )
        assert_no_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"symbol 'v' defined but not used \({ns}: \d+\)",
        )

    def test_warning_for_nested_let_if_warning_enabled(
        self, lcompile: CompileFn, ns: runtime.Namespace, assert_matching_logs
    ):
        lcompile(
            """
        (letfn* [v (fn* v [] 4)]
          (letfn* [v (fn* v [] 5)]
            (v)))
        """,
            opts={compiler.WARN_ON_UNUSED_NAMES: True},
        )
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"symbol 'v' defined but not used \({ns}: \d+\)",
        )

    def test_no_warning_for_nested_let_if_warning_disabled(
        self, lcompile: CompileFn, ns: runtime.Namespace, assert_no_matching_logs
    ):
        lcompile(
            """
        (letfn* [v (fn* v [] 4)]
          (letfn* [v (fn* v [] 5)]
            (v)))
        """,
            opts={compiler.WARN_ON_UNUSED_NAMES: False},
        )
        assert_no_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"symbol 'v' defined but not used \({ns}: \d+\)",
        )


class TestLoop:
    def test_loop_num_elems(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(loop*)")

    def test_loop_may_have_empty_bindings(self, lcompile: CompileFn):
        assert None is lcompile("(loop* [])")
        assert kw.keyword("kw") == lcompile("(loop* [] :kw)")

    def test_loop_bindings_must_be_vector(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(loop* () a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(loop* (a kw) a)")

    def test_loop_bindings_must_have_name_and_value(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(loop* [a :kw b] a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(loop* [a :kw b :other-kw c] a)")

    def test_loop_binding_name_must_be_symbol(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(loop* [:a :kw] a)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(loop* [a :kw :b :other-kw] a)")

    def test_let_name_does_not_resolve(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(loop* [a 'sym] c)")

    def test_loop_may_have_empty_body(self, lcompile: CompileFn):
        assert None is lcompile("(loop* [])")
        assert None is lcompile("(loop* [a :kw])")

    def test_loop_without_recur(self, lcompile: CompileFn):
        assert 1 == lcompile("(loop* [a 1] a)")
        assert kw.keyword("keyword") == lcompile('(loop* [a :keyword b "string"] a)')
        assert kw.keyword("value") == lcompile("(loop* [a :value b a] b)")
        assert lmap.map({kw.keyword("length"): 1}) == lcompile(
            "(loop* [a 1 b :length c {b a} a 4] c)"
        )
        assert 4 == lcompile("(loop* [a 1 b :length c {b a} a 4] a)")
        assert "LOWER" == lcompile('(loop* [a "lower"] (.upper a))')
        assert "string" == lcompile('(loop* [] "string")')

    def test_loop_with_recur(self, lcompile: CompileFn):
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
        (import* io)
        (let* [writer (io/StringIO)]
          (loop* [s "string"]
            (if (seq s)
              (do
                (.write writer (first s))
                (recur (rest s)))
              (.getvalue writer))))"""
        assert "string" == lcompile(code)

        code = """
        (loop* [s     "tester"
                accum []]
          (if (seq s)
            (recur (rest s) (conj accum (first s)))
            (apply str accum)))"""
        assert "tester" == lcompile(code)


class TestMacros:
    def test_macro_expansion(self, lcompile: CompileFn):
        assert llist.l(1, 2, 3) == lcompile("((fn [] '(1 2 3)))")

    def test_syntax_quoting(
        self, test_ns: str, lcompile: CompileFn, resolver: reader.Resolver
    ):
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
            "`(my-symbol)", resolver=resolver
        )

    def test_syntax_quoting_anonymous_fns_with_single_arg(
        self, test_ns: str, lcompile: CompileFn, resolver: reader.Resolver
    ):
        single_arg_fn = lcompile("`#(println %)", resolver=resolver)
        assert single_arg_fn.first == sym.symbol("fn*")
        single_arg_vec = runtime.nth(single_arg_fn, 1)
        assert isinstance(single_arg_vec, vec.PersistentVector)
        single_arg = single_arg_vec[0]
        assert isinstance(single_arg, sym.Symbol)
        assert re.match(r"arg-1_\d+", single_arg.name) is not None
        println_call = runtime.nth(single_arg_fn, 2)
        assert runtime.nth(println_call, 1) == single_arg

    def test_syntax_quoting_anonymous_fns_with_multiple_args(
        self, test_ns: str, lcompile: CompileFn, resolver: reader.Resolver
    ):
        multi_arg_fn = lcompile("`#(vector %1 %2 %3)", resolver=resolver)
        assert multi_arg_fn.first == sym.symbol("fn*")
        multi_arg_vec = runtime.nth(multi_arg_fn, 1)
        assert isinstance(multi_arg_vec, vec.PersistentVector)

        for arg in multi_arg_vec:
            assert isinstance(arg, sym.Symbol)
            assert re.match(r"arg-\d_\d+", arg.name) is not None

        vector_call = runtime.nth(multi_arg_fn, 2)
        assert vec.vector(runtime.nthrest(vector_call, 1)) == multi_arg_vec

    def test_syntax_quoting_anonymous_fns_with_rest_arg(
        self, test_ns: str, lcompile: CompileFn, resolver: reader.Resolver
    ):
        rest_arg_fn = lcompile("`#(vec %&)", resolver=resolver)
        assert rest_arg_fn.first == sym.symbol("fn*")
        rest_arg_vec = runtime.nth(rest_arg_fn, 1)
        assert isinstance(rest_arg_vec, vec.PersistentVector)
        assert rest_arg_vec[0] == sym.symbol("&")
        rest_arg = rest_arg_vec[1]
        assert isinstance(rest_arg, sym.Symbol)
        assert re.match(r"arg-rest_\d+", rest_arg.name) is not None
        vec_call = runtime.nth(rest_arg_fn, 2)
        assert runtime.nth(vec_call, 1) == rest_arg

    @pytest.mark.parametrize("code,v", [("`(s)", llist.l(sym.symbol("s")))])
    def test_unquote(self, lcompile: CompileFn, code: str, v):
        assert v == lcompile(code)

    def test_unquote_arbitrary_deftypes(self, lcompile: CompileFn):
        v = lcompile(
            """
        (deftype Point [a b c])

        (defmacro make-point
          [a b c]
          `(identity ~(Point. a b c)))

        (make-point 1 2 3)
        """
        )

        assert isinstance(v, IType)
        assert (1, 2, 3) == (v.a, v.b, v.c)

    def test_unquote_arbitrary_defrecords(self, lcompile: CompileFn):
        v = lcompile(
            """
        (defrecord Point [a b c])

        (defmacro make-point
          [a b c]
          `(identity ~(Point. a b c)))

        (make-point 1 2 3)
        """
        )

        assert isinstance(v, IRecord)
        assert (1, 2, 3) == (v.a, v.b, v.c)

    @pytest.mark.parametrize("code", ["~s", "`(~s)"])
    def test_invalid_unquote(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code,v",
        [
            ("`(~@[1 2 3])", llist.l(1, 2, 3)),
            ("'(~@53233)", llist.l(llist.l(reader._UNQUOTE_SPLICING, 53233))),
        ],
    )
    def test_unquote_splicing(self, lcompile: CompileFn, code: str, v):
        assert v == lcompile(code)

    @pytest.mark.parametrize(
        "code,v",
        [
            (
                "`(print ~@[1 2 3])",
                llist.l(sym.symbol("print", ns="basilisp.core"), 1, 2, 3),
            )
        ],
    )
    def test_unquote_splicing_with_resolver(
        self, lcompile: CompileFn, resolver: reader.Resolver, code: str, v
    ):
        assert v == lcompile(code, resolver=resolver)

    @pytest.mark.parametrize("code", ["~@[1 2 3]"])
    def test_invalid_unquote_splicing(self, lcompile: CompileFn, code: str):
        with pytest.raises(TypeError):
            lcompile(code)


class TestQuote:
    @pytest.mark.parametrize("code", ["(quote)", "(quote form other-form)"])
    def test_quote_num_elems(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_quoted_list(self, lcompile: CompileFn):
        assert lcompile("'()") == llist.l()
        assert lcompile("'(str)") == llist.l(sym.symbol("str"))
        assert lcompile("'(str 3)") == llist.l(sym.symbol("str"), 3)
        assert lcompile("'(str 3 :feet-deep)") == llist.l(
            sym.symbol("str"), 3, kw.keyword("feet-deep")
        )

    def test_quoted_map(self, lcompile: CompileFn):
        assert lcompile("'{}") == lmap.EMPTY
        assert lcompile("'{:a 2}") == lmap.map({kw.keyword("a"): 2})
        assert lcompile('\'{:a 2 "str" s}') == lmap.map(
            {kw.keyword("a"): 2, "str": sym.symbol("s")}
        )

    def test_quoted_queue(self, lcompile: CompileFn):
        assert lcompile("'#queue ()") == lqueue.EMPTY
        assert lcompile('\'#queue (s :a "d")') == lqueue.q(
            sym.symbol("s"), kw.keyword("a"), "d"
        )
        assert lcompile("'#queue (1 2 3)") == lqueue.q(1, 2, 3)

    def test_quoted_set(self, lcompile: CompileFn):
        assert lcompile("'#{}") == lset.EMPTY
        assert lcompile("'#{:a 2}") == lset.s(kw.keyword("a"), 2)
        assert lcompile('\'#{:a 2 "str"}') == lset.s(kw.keyword("a"), 2, "str")

    def test_quoted_inst(self, lcompile: CompileFn):
        assert (
            datetime.datetime.fromisoformat("2018-01-18T03:26:57.296-00:00")
            == lcompile('(quote #inst "2018-01-18T03:26:57.296-00:00")')
            == datetime.datetime(
                2018, 1, 18, 3, 26, 57, 296000, tzinfo=datetime.timezone.utc
            )
        )

    def test_regex(self, lcompile: CompileFn):
        assert lcompile(r'(quote #"\s")') == re.compile(r"\s")

    def test_uuid(self, lcompile: CompileFn):
        assert uuid.UUID("{0366f074-a8c5-4764-b340-6a5576afd2e8}") == lcompile(
            '(quote #uuid "0366f074-a8c5-4764-b340-6a5576afd2e8")'
        )

    def test_py_dict(self, lcompile: CompileFn):
        assert isinstance(lcompile("'#py {}"), dict)
        assert {} == lcompile("'#py {}")
        assert {kw.keyword("a"): 1, "b": "str"} == lcompile(
            '(quote #py {:a 1 "b" "str"})'
        )

    def test_py_list(self, lcompile: CompileFn):
        assert isinstance(lcompile("'#py []"), list)
        assert [] == lcompile("'#py []")
        assert [1, kw.keyword("a"), "str"] == lcompile('(quote #py [1 :a "str"])')

    def test_py_set(self, lcompile: CompileFn):
        assert isinstance(lcompile("'#py #{}"), set)
        assert set() == lcompile("'#py #{}")
        assert {1, kw.keyword("a"), "str"} == lcompile('(quote #py #{1 :a "str"})')

    def test_py_tuple(self, lcompile: CompileFn):
        assert isinstance(lcompile("'#py ()"), tuple)
        assert tuple() == lcompile("'#py ()")
        assert (1, kw.keyword("a"), "str") == lcompile('(quote #py (1 :a "str"))')


class TestRecur:
    def test_recur(self, lcompile: CompileFn):
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
                             (recur (rest in) (cons (python/str (first in)) out))
                             (cons (python/str (first in)) out)))]
             (.join \"\" (coerce (cons s args) '())))))
         """

        lcompile(code)

        assert "a" == lcompile('(rev-str "a")')
        assert ":ba" == lcompile('(rev-str "a" :b)')
        assert "3:ba" == lcompile('(rev-str "a" :b 3)')

    def test_can_macroexpand_recur(self, lcompile: CompileFn):
        lcompile(
            """
            (defmacro maybe-loop [& body]
              `(loop [x# true]
                (if x#
                  (do ~@body)
                  :done)))

            (macroexpand-1 '(maybe-loop (recur false)))
        """
        )

    def test_recur_arity_must_match_recur_point(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn [s] (recur :a :b))")

        with pytest.raises(compiler.CompilerException):
            lcompile("(fn [a b] (recur a))")

    def test_single_arity_recur(self, lcompile: CompileFn):
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

    def test_multi_arity_recur(self, lcompile: CompileFn):
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

    @pytest.mark.parametrize(
        "code",
        [
            '(fn [a] (def b (recur "a")))',
            '(fn [a] (import* (recur "a")))',
            '(fn [a] (.join "" (recur "a")))',
            '(fn [a] (.-p (recur "a")))',
            '(fn [a] (throw (recur "a"))))',
            '(fn [a] (var (recur "a"))))',
        ],
    )
    def test_disallow_recur_in_special_forms(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            "(recur)",
            "(do (recur))",
            "(if true (recur) :b)",
            '(fn [a] (do (recur "a") :b))',
            '(fn [a] (if (recur "a") :a :b))',
            '(fn [a] (if (recur "a") :a))',
            '(fn [a] (let [a (recur "a")] a))',
            '(fn [a] (let [a (do (recur "a"))] a))',
            '(fn [a] (let [a (do :b (recur "a"))] a))',
            '(fn [a] (let [a (do (recur "a") :c)] a))',
            '(fn [a] (let [a "a"] (recur a) a))',
            '(fn [a] (loop* [a (recur "a")] a))',
            '(fn [a] (loop* [a (do (recur "a"))] a))',
            '(fn [a] (loop* [a (do :b (recur "a"))] a))',
            '(fn [a] (loop* [a (do (recur "a") :c)] a))',
            '(fn [a] (loop* [a "a"] (recur a) a))',
            "(fn [a] (try (do (recur a) :b) (catch AttributeError _ nil)))",
            "(fn [a] (try (recur a) :b (catch AttributeError _ nil)))",
            "(fn [a] (try :b (catch AttributeError _ (do (recur :a) :c))))",
            "(fn [a] (try :b (finally (do (recur :a) :c))))",
        ],
    )
    def test_disallow_recur_outside_tail(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_single_arity_named_anonymous_fn_recursion(self, lcompile: CompileFn):
        code = """
        (let [compute-sum (fn sum [n]
                            (if (operator/eq 0 n)
                              0
                              (operator/add n (sum (operator/sub n 1)))))]
          (compute-sum 5))
        """
        assert 15 == lcompile(code)

    def test_multi_arity_named_anonymous_fn_recursion(self, lcompile: CompileFn):
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


class TestReify:
    @pytest.mark.parametrize("code", ["(reify*)", "(reify* :implements)"])
    def test_reify_number_of_elems(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_reify_has_implements_kw(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
                (reify* [collections.abc/Sized]
                  (__len__ [this] 2))"""
            )

    @pytest.mark.parametrize(
        "code",
        [
            """
            (reify :implements (collections.abc/Sized)
              (__len__ [this] 2))""",
            """
            (reify :implements collections.abc/Sized
              (__len__ [this] 2))""",
        ],
    )
    def test_reify_implements_is_vector(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_reify_must_declare_implements(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
                (reify*
                  (--call-- [this] [1 2 3]))"""
            )

    @pytest.mark.parametrize(
        "code",
        [
            """
            (import* collections.abc)
            (reify :implements [collections.abc/Callable])""",
            """
            (import* collections.abc)
            (reify* :implements [collections.abc/Callable collections.abc/Sized]
              (--call-- [this] [1 2 3]))""",
            """
            (reify*
              (--call-- [this] [1 2 3]))""",
        ],
    )
    def test_reify_impls_must_match_defined_interfaces(
        self, lcompile: CompileFn, code: str
    ):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            """
            (import* collections.abc)
            (reify* :implements [collections.abc/Callable collections.abc/Callable]
              (--call-- [this] [1 2 3])
              (--call-- [this] 1))""",
            """
            (import* collections.abc)
            (reify*
              :implements [collections.abc/Callable collections.abc/Sized collections.abc/Callable]
              (--call-- [this] [1 2 3])
              (--len-- [this] 1)
              (--call-- [this] 1))""",
            """
            (import* collections.abc)
            (reify*
              :implements [collections.abc/Callable collections.abc/Callable collections.abc/Sized]
              (--call-- [this] [1 2 3])
              (--call-- [this] 1)
              (--len-- [this] 1))""",
        ],
    )
    def test_reify_prohibit_duplicate_interface(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            """
            (reify*
              :implements [:collections.abc/Callable]
              (--call-- [this] [1 2 3]))""",
            """
            (import* collections.abc)
            (reify*
              :implements [collections.abc/Callable]
              [--call-- [this] [1 2 3]])""",
            """
            (import* collections.abc)
            (reify*
              :implements [collections.abc/Callable :collections.abc/Sized]
              [--call-- [this] [1 2 3]])""",
        ],
    )
    def test_reify_impls_must_be_sym_or_list(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_reify_interface_must_be_host_form(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
                (let [a :kw]
                  (reify* :implements [a]
                    (--call-- [this] [1 2 3])))"""
            )

    @pytest.mark.parametrize(
        "code,ExceptionType",
        [
            (
                """
                (import* collections)
                (reify* :implements [collections/OrderedDict]
                  (keys [this] [1 2 3]))""",
                compiler.CompilerException,
            ),
            (
                """
                (do
                  (def Shape (python/type "Shape" #py () #py {}))
                  (reify* :implements [Shape]))""",
                compiler.CompilerException,
            ),
            (
                """
                ((fn []
                  (def Shape (python/type "Shape" #py () #py {}))
                  (reify* :implements [Shape])))""",
                runtime.RuntimeException,
            ),
        ],
    )
    def test_reify_interface_must_be_abstract(
        self, lcompile: CompileFn, code: str, ExceptionType
    ):
        with pytest.raises(ExceptionType):
            lcompile(code)

    def test_reify_allows_empty_abstract_interface(self, lcompile: CompileFn):
        reified_obj = lcompile("(reify* :implements [basilisp.lang.interfaces/IType])")
        assert isinstance(reified_obj, IType)

    def test_reify_allows_empty_dynamic_abstract_interface(self, lcompile: CompileFn):
        Shape, make_circle = lcompile(
            """
            (do
              (import* abc)
              (def Shape (python/type "Shape" #py (abc/ABC) #py {}))
              [Shape (fn [] (reify* :implements [Shape]))])"""
        )
        c = make_circle()
        assert isinstance(Shape, type)
        assert isinstance(c, Shape)

    @pytest.mark.parametrize(
        "code,ExceptionType",
        [
            (
                """
                (import* collections.abc)
                (reify* :implements [collections.abc/Collection]
                  (--len-- [this] 3))""",
                compiler.CompilerException,
            ),
            (
                """
                (do
                  (import* abc)
                  (def Shape
                  (python/type "Shape"
                               #py (abc/ABC)
                               #py {"area"
                                     (abc/abstractmethod
                                      (fn []))}))
                  (reify* :implements [Shape]))""",
                compiler.CompilerException,
            ),
            (
                """
                ((fn []
                  (import* abc)
                  (def Shape
                  (python/type "Shape"
                               #py (abc/ABC)
                               #py {"area"
                                     (abc/abstractmethod
                                      (fn []))}))
                  (reify* :implements [Shape])))""",
                runtime.RuntimeException,
            ),
        ],
    )
    def test_reify_interface_must_implement_all_abstract_methods(
        self,
        lcompile: CompileFn,
        code: str,
        ExceptionType,
    ):
        with pytest.raises(ExceptionType):
            lcompile(code)

    @pytest.mark.parametrize(
        "code,ExceptionType",
        [
            (
                """
                (import* collections.abc)
                (reify* :implements [collections.abc/Sized]
                  (--len-- [this] 3)
                  (call [this] :called))""",
                compiler.CompilerException,
            ),
            (
                """
                (import* abc collections.abc)
                (do
                  (def Shape
                  (python/type "Shape"
                               #py (abc/ABC)
                               #py {"area"
                                     (abc/abstractmethod
                                      (fn []))}))
                (reify* :implements [Shape]
                  (area [this] (* 2 1 1))
                  (call [this] :called)))""",
                compiler.CompilerException,
            ),
            (
                """
                (import* abc collections.abc)
                ((fn []
                  (def Shape
                  (python/type "Shape"
                               #py (abc/ABC)
                               #py {"area"
                                     (abc/abstractmethod
                                      (fn []))}))
                (reify* :implements [Shape]
                  (area [this] (* 2 1 1))
                  (call [this] :called))))""",
                runtime.RuntimeException,
            ),
        ],
    )
    def test_reify_may_not_add_extra_methods_to_interface(
        self,
        lcompile: CompileFn,
        code: str,
        ExceptionType,
    ):
        with pytest.raises(ExceptionType):
            lcompile(code)

    @pytest.mark.parametrize(
        "code", ["(reify* :implements [])", "(reify* :implements [python/object])"]
    )
    def test_reify_interface_may_have_no_fields_or_methods(
        self,
        lcompile: CompileFn,
        code: str,
    ):
        lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            """
            (fn [x y z]
              (reify* :implements [python/object]
                (__str__ [this]
                (python/repr #py ("Point" x y z)))))""",
            """
            (do
              (def PyObject python/object)
              (fn [x y z]
                (reify* :implements [PyObject]
                  (__str__ [this]
                  (python/repr #py ("Point" x y z))))))""",
        ],
    )
    def test_reify_interface_may_implement_only_some_object_methods(
        self, lcompile: CompileFn, code: str
    ):
        Point = lcompile(code)
        pt = Point(1, 2, 3)
        assert "('Point', 1, 2, 3)" == str(pt)

    @pytest.mark.parametrize(
        "code",
        [
            """
            (fn* [x y z]
              (reify* :implements [WithProp]
                (^:property prop [this] [x y z])
                (prop [self] self)))""",
            """
            (fn* [x y z]
              (reify* :implements [WithProp]
                (prop [self] self)
                (^:property prop [this] [x y z])))""",
        ],
    )
    def test_reify_property_and_method_names_cannot_overlap(
        self, lcompile: CompileFn, code: str
    ):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                f"""
                (import* abc)
                (def WithProp
                  (python/type "WithProp"
                                 #py (abc/ABC)
                                 #py {{"prop"
                                      (python/property
                                       (abc/abstractmethod
                                        (fn [self])))}}))
                (def WithMember
                  (python/type "WithMember"
                               #py (abc/ABC)
                               #py {{"prop"
                                    (abc/abstractmethod
                                     (fn [cls]))}}))
                {code}"""
            )

    @pytest.mark.parametrize(
        "code",
        [
            """
        (import* abc)
        (def WithCls
          (python/type "WithCls"
                       #py (abc/ABC)
                       #py {"create"
                            (python/classmethod
                             (abc/abstractmethod
                              (fn [self])))}))
        (reify* :implements [WithProp]
          (^:classmethod create [cls] cls))""",
            """
        (import* abc)
        (def WithStatic
          (python/type "WithStatic"
                       #py (abc/ABC)
                       #py {"dostatic"
                            (python/staticmethod
                             (abc/abstractmethod
                              (fn [self])))}))
        (reify* :implements [WithProp]
          (^:staticmethod dostatic [] :staticboi))""",
        ],
    )
    def test_reify_disallows_class_and_static_members(
        self, lcompile: CompileFn, code: str
    ):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_reify_transfers_form_meta_to_obj(self, lcompile: CompileFn):
        make_obj = lcompile(
            """
            (fn [x]
              ^{:passed-through true}
              (reify* :implements [python/object]
                (--call-- [this] x)))"""
        )
        o = make_obj(kw.keyword("x"))
        assert isinstance(o, IWithMeta)
        assert lmap.map({kw.keyword("passed-through"): True}) == o.meta()
        assert kw.keyword("x") == o()

        new_meta = lmap.map({kw.keyword("replaced"): kw.keyword("yes")})
        new_o = o.with_meta(new_meta)
        assert isinstance(o, IWithMeta)
        assert new_meta == new_o.meta()
        assert kw.keyword("x") == new_o()

        assert type(o) is type(new_o)

    class TestReifyBases:
        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* argparse)
                (reify* :implements [^:abstract argparse/Action]
                  (__call__ [this]))""",
                """
                (def AABase
                  (python/type "AABase" #py () #py {"some_method" (fn [this])}))
                (reify* :implements [^:abstract AABase]
                  (some-method [this]))""",
                """
                (do
                  (import* argparse)
                  (reify* :implements [^:abstract argparse/Action]
                    (__call__ [this])))""",
                """
                (do
                  (def AABase
                    (python/type "AABase" #py () #py {"some_method" (fn [this])}))
                  (reify* :implements [^:abstract AABase]
                    (some-method [this])))""",
            ],
        )
        def test_reify_allows_artificially_abstract_super_type(
            self, lcompile: CompileFn, code: str
        ):
            lcompile(code)

        @pytest.mark.parametrize(
            "code,ExceptionType",
            [
                (
                    """
                    (import* argparse)
                    (reify* :implements [^:abstract argparse/Action]
                      (__call__ [this])
                      (do-action [this]))""",
                    compiler.CompilerException,
                ),
                (
                    """
                    (def AABase
                      (python/type "AABase" #py () #py {"some_method" (fn [this])}))
                    (reify* :implements [^:abstract AABase]
                      (some-method [this])
                      (other-method [this]))""",
                    compiler.CompilerException,
                ),
                (
                    """
                    (do
                      (import* argparse)
                      (reify* :implements [^:abstract argparse/Action]
                        (__call__ [this])
                        (do-action [this])))""",
                    compiler.CompilerException,
                ),
                (
                    """
                    (do
                      (def AABase
                        (python/type "AABase" #py () #py {"some_method" (fn [this])}))
                      (reify* :implements [^:abstract AABase]
                        (some-method [this])
                        (other-method [this])))""",
                    compiler.CompilerException,
                ),
                (
                    """
                    ((fn []
                      (def AABase
                        (python/type "AABase" #py () #py {"some_method" (fn [this])}))
                      (reify* :implements [^:abstract AABase]
                        (some-method [this])
                        (other-method [this]))))""",
                    runtime.RuntimeException,
                ),
            ],
        )
        def test_reify_disallows_extra_methods_if_not_in_a_super_type(
            self, lcompile: CompileFn, code: str, ExceptionType
        ):
            with pytest.raises(ExceptionType):
                lcompile(code)

        @pytest.mark.parametrize(
            "code,ExceptionType",
            [
                (
                    """
                    (import* io)
                    (^:mutable reify* :implements [^:abstract ^{:abstract-members '(:read)} io/IOBase]
                      (read [this v]))""",
                    compiler.CompilerException,
                ),
                (
                    """
                    (import* io)
                    (^:mutable reify* :implements [^:abstract ^{:abstract-members [:read]} io/IOBase]
                      (read [this v]))""",
                    compiler.CompilerException,
                ),
                (
                    """
                    (import* io)
                    (^:mutable reify* :implements [^:abstract ^{:abstract-members #py [:read]} io/IOBase]
                      (read [this v]))""",
                    compiler.CompilerException,
                ),
            ],
        )
        def test_reify_abstract_members_must_be_a_set(
            self, lcompile: CompileFn, code: str, ExceptionType
        ):
            with pytest.raises(ExceptionType):
                lcompile(code)

        def test_reify_abstract_members_must_have_elements(self, lcompile: CompileFn):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (import* io)
                    (^:mutable reify* :implements [^:abstract ^{:abstract-members #{}} io/IOBase]
                      (read [this v]))
                    """
                )

        def test_reify_abstract_should_not_have_namespaces(
            self, lcompile: CompileFn, assert_matching_logs
        ):
            lcompile(
                """
                (import* io)
                (^:mutable reify* :implements [^:abstract ^{:abstract-members #{:io/read}} io/IOBase]
                  (read [this v]))
                """
            )
            assert_matching_logs(
                "basilisp.lang.compiler.analyzer",
                logging.WARNING,
                "Unexpected namespace for artificially abstract member to reify*",
            )

        def test_reify_abstract_members_names_must_be_str_keyword_or_symbol(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (import* io)
                    (^:mutable reify* :implements [^:abstract ^{:abstract-members #{1 :read}} io/IOBase]
                      (read [this v]))
                    """,
                )

        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* io)
                (^:mutable reify* :implements [^:abstract ^{:abstract-members #{:read :write}} io/IOBase]
                  (read [this n])
                  (write [this v]))
                """,
                """
                (import* io)
                (^:mutable reify* :implements [^:abstract ^{:abstract-members #{read write}} io/IOBase]
                  (read [this n])
                  (write [this v]))
                """,
                """
                (import* io)
                (^:mutable reify* :implements [^:abstract ^{:abstract-members #{"read" "write"}} io/IOBase]
                  (read [this n])
                  (write [this v]))
                """,
            ],
        )
        def test_reify_artificially_abstract_members(
            self,
            lcompile: CompileFn,
            code: str,
        ):
            instance = lcompile(code)
            assert instance.read is not None
            assert instance.write is not None

    class TestReifyMember:
        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (:--call-- [this] [x y z])))""",
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* collections.abc/Callable
                    (\"--call--\" [this] [x y z])))""",
            ],
        )
        def test_reify_member_is_named_by_sym(self, lcompile: CompileFn, code: str):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

        def test_reify_member_args_are_vec(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (import* collections.abc)
                    (fn [x y z]
                      (reify* :implements [collections.abc/Callable]
                        (--call-- (this) [x y z])))"""
                )

        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (^:property ^:staticmethod __call__ [this]
                      [x y z])))""",
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* collections.abc/Callable
                    (^:classmethod ^:property __call__ [this]
                      [x y z])))""",
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* collections.abc/Callable
                    (^:classmethod ^:staticmethod __call__ [this]
                      [x y z])))""",
            ],
        )
        def test_reify_member_may_not_be_multiple_types(
            self, lcompile: CompileFn, code: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

    class TestReifyMethod:
        def test_reify_fields_and_methods(self, lcompile: CompileFn):
            make_point = lcompile(
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable collections.abc/Sized]
                    (--len-- [this] 1)
                    (--call-- [this] [x y z])))"""
            )
            pt = make_point(1, 2, 3)
            assert 1 == len(pt)
            assert vec.v(1, 2, 3) == pt()

        def test_reify_method_with_args(self, lcompile: CompileFn):
            make_point = lcompile(
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [this i j k] [x i y j z k])))"""
            )
            pt = make_point(1, 2, 3)
            assert vec.v(1, 4, 2, 5, 3, 6) == pt(4, 5, 6)

        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [this &])))""",
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [this & :args])))""",
            ],
        )
        def test_reify_method_with_varargs_malformed(
            self, lcompile: CompileFn, code: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

        def test_reify_method_with_varargs(self, lcompile: CompileFn):
            Mirror = lcompile(
                """
                (import* collections.abc)
                (fn* [x]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [this & args] [x args])))"""
            )
            mirror = Mirror("Beauty is in the eye of the beholder")
            assert vec.v(
                "Beauty is in the eye of the beholder", llist.l(1, 2, 3)
            ) == mirror(1, 2, 3)

        def test_reify_empty_method_body(self, lcompile: CompileFn):
            Point = lcompile(
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [this])))"""
            )
            pt = Point(1, 2, 3)
            assert None is pt()

        def test_reify_method_allows_recur(self, lcompile: CompileFn):
            Point = lcompile(
                """
                (import* collections.abc operator)
                (fn* [x]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [this sum start]
                      (if (operator/gt start 0)
                        (recur (operator/add sum start) (operator/sub start 1))
                        (operator/add sum x)))))"""
            )
            pt = Point(7)
            assert 22 == pt(0, 5)

        def test_reify_method_args_vec_includes_this(self, lcompile: CompileFn):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (import* collections.abc)
                    (fn* [x y z]
                      (reify* :implements [collections.abc/Callable]
                        (--call-- [] [x y z])))"""
                )

        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [\"this\"] [x y z])))""",
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [this :new] [x y z])))""",
            ],
        )
        def test_reify_method_args_are_syms(self, lcompile: CompileFn, code: str):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

        def test_reify_method_returns_value(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
                (import* collections.abc)
                (fn* [x]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [this new-val]
                      (* x new-val))))"""
            )
            pt = Point(3)
            assert 15 == pt(5)

        def test_reify_method_only_support_valid_kwarg_strategies(
            self, lcompile: CompileFn
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (import* collections.abc)
                    (reify* :implements [collections.abc/Callable]
                      (^{:kwargs :kwarg-it} --call-- [this]))"""
                )

        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (^ {:kwargs :apply} --call--
                      [this & args]
                      (merge {:x x :y y :z z} (apply hash-map args)))))""",
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (^{:kwargs :collect} --call--
                      [this kwargs]
                      (merge {:x x :y y :z z} kwargs))))""",
            ],
        )
        def test_reify_method_kwargs(self, lcompile: CompileFn, code: str):
            Point = lcompile(code)

            pt = Point(1, 2, 3)
            assert lmap.map(
                {
                    kw.keyword("w"): 2,
                    kw.keyword("x"): 1,
                    kw.keyword("y"): 4,
                    kw.keyword("z"): 3,
                }
            ) == pt(w=2, y=4)

        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [this]
                      :no-args)
                    (--call-- [this]
                      :also-no-args)))""",
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [this s]
                      :one-arg)
                    (--call-- [this s]
                      :also-one-arg)))""",
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [this]
                      :no-args)
                    (--call-- [this s]
                      :one-arg)
                    (--call-- [this a b]
                      [a b])
                    (--call-- [this s3]
                      :also-one-arg)))""",
            ],
        )
        def test_no_reify_method_arity_has_same_fixed_arity(
            self, lcompile: CompileFn, code: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [this & args]
                      (concat [:no-starter] args))
                    (--call-- [this s & args]
                      (concat [s] args))))""",
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [this s & args]
                      (concat [s] args))
                    (--call-- [this & args]
                      (concat [:no-starter] args))))""",
            ],
        )
        def test_reify_method_cannot_have_two_variadic_arities(
            self, lcompile: CompileFn, code: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

        def test_reify_method_variadic_method_cannot_have_lower_fixed_arity_than_other_methods(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (import* collections.abc)
                    (fn* [x y z]
                      (reify* :implements [collections.abc/Callable]
                        (--call-- [this a b]
                          [a b])
                        (--call-- [this & args]
                          (concat [:no-starter] args))))"""
                )

        @pytest.mark.parametrize(
            "code",
            [
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (--call-- [this s] s)
                    (^{:kwargs :collect} --call-- [this s kwargs]
                       (concat [s] kwargs))))""",
                """
                (import* collections.abc)
                (fn* [x y z]
                  (reify* :implements [collections.abc/Callable]
                    (^{:kwargs :collect} --call-- [this kwargs] kwargs)
                    (^{:kwargs :apply} --call-- [thi shead & kwargs]
                      (apply hash-map :first head kwargs))))""",
            ],
        )
        def test_reify_method_does_not_support_kwargs(
            self, lcompile: CompileFn, code: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(code)

        def test_multi_arity_reify_method_dispatches_properly(
            self,
            lcompile: CompileFn,
            ns: runtime.Namespace,
        ):
            code = """
            (import* abc)
            (def DoubleTrouble
              (python/type "DoubleTrouble"
                           #py (abc/ABC)
                           #py {"_double_up_arity0" (abc/abstractmethod (fn [self]))
                                "_double_up_arity1" (abc/abstractmethod (fn [self arg1]))
                                "double_up"         (abc/abstractmethod (fn [& args]))}))
            (fn* [x y z]
              (reify* :implements [DoubleTrouble]
                (double-up [this] :a)
                (double-up [this s] [:a s])))"""
            Point = lcompile(code)
            assert callable(Point(1, 2, 3).double_up)
            assert kw.keyword("a") == Point(1, 2, 3).double_up()
            assert vec.v(kw.keyword("a"), kw.keyword("c")) == Point(1, 2, 3).double_up(
                kw.keyword("c")
            )

            code = """
            (import* abc)
            (def InTriplicate
              (python/type "InTriplicate"
                           #py (abc/ABC)
                           #py {"_triple_up_arity0"     (abc/abstractmethod (fn [self]))
                                "_triple_up_arity1"     (abc/abstractmethod (fn [self arg1]))
                                "_triple_up_arity_rest" (abc/abstractmethod (fn [self arg1 & args]))
                                "triple_up"             (abc/abstractmethod (fn [& args]))}))
            (fn* [x y z]
              (reify* :implements [InTriplicate]
                (triple-up [this] :no-args)
                (triple-up [this s] s)
                (triple-up [this s & args]
                  (concat [s] args))))"""
            Point = lcompile(code)
            assert callable(Point(1, 2, 3).triple_up)
            assert Point(1, 2, 3).triple_up() == kw.keyword("no-args")
            assert Point(1, 2, 3).triple_up("STRING") == "STRING"
            assert Point(1, 2, 3).triple_up(
                kw.keyword("first-arg"), "second-arg", 3
            ) == llist.l(kw.keyword("first-arg"), "second-arg", 3)

        def test_multi_arity_reify_method_call_fails_if_no_valid_arity(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
                (import* abc)
                (def InTriplicate
                  (python/type "InTriplicate"
                               #py (abc/ABC)
                               #py {"_triple_up_arity0" (abc/abstractmethod (fn [self]))
                                    "_triple_up_arity1" (abc/abstractmethod (fn [self arg1]))
                                    "_triple_up_arity2" (abc/abstractmethod (fn [self arg1 arg2]))
                                    "triple_up"         (abc/abstractmethod (fn [& args]))}))
                (fn* [x y z]
                  (reify* :implements [InTriplicate]
                    (triple-up [this] :send-me-an-arg!)
                    (triple-up [this i] i)
                    (triple-up [this i j] (concat [i] [j]))))"""
            )

            with pytest.raises(runtime.RuntimeException):
                Point(1, 2, 3).triple_up(4, 5, 6)

    class TestReifyProperty:
        @pytest.fixture(autouse=True)
        def property_interface(self, lcompile: CompileFn):
            return lcompile(
                """
                (import* abc)
                (def WithProp
                  (python/type "WithProp"
                                 #py (abc/ABC)
                                 #py {"prop"
                                      (python/property
                                       (abc/abstractmethod
                                        (fn [self])))}))"""
            )

        @pytest.mark.parametrize(
            "code,ExceptionType",
            [
                (
                    """
                    (fn* [x y z]
                      (reify* :implements [WithProp]))""",
                    compiler.CompilerException,
                ),
                (
                    """
                    (do
                      (import* abc)
                      (def WithProperty
                        (python/type "WithProp"
                                     #py (abc/ABC)
                                     #py {"a_property"
                                          (python/property
                                           (abc/abstractmethod
                                            (fn [self])))}))
                      (reify* :implements [WithProperty]))""",
                    compiler.CompilerException,
                ),
                (
                    """
                    ((fn []
                      (import* abc)
                      (def WithProperty
                        (python/type "WithProp"
                                     #py (abc/ABC)
                                     #py {"a_property"
                                          (python/property
                                           (abc/abstractmethod
                                            (fn [self])))}))
                      (reify* :implements [WithProperty])))""",
                    runtime.RuntimeException,
                ),
            ],
        )
        def test_reify_must_implement_interface_property(
            self, lcompile: CompileFn, code: str, ExceptionType
        ):
            with pytest.raises(ExceptionType):
                lcompile(code)

        def test_reify_property_includes_this(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (fn* [x y z]
                      (reify* :implements [WithProp]
                        (^:property prop [] [x y z])))"""
                )

        def test_reify_property_args_are_syms(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (fn* Point [x y z]
                      (reify* :implements [WithProp]
                        (^:property prop [:this] [x y z])))"""
                )

        def test_reify_property_may_not_have_args(
            self,
            lcompile: CompileFn,
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (fn* [x y z]
                      (reify* :implements [WithProp]
                        (^:property prop [this and-that] [x y z])))"""
                )

        def test_reify_property_disallows_recur(self, lcompile: CompileFn):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (fn* [x]
                      (reify* :implements [WithProp]
                        (^:property prop [this]
                          (recur))))"""
                )

        def test_reify_can_have_property(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
                (fn* [x y z]
                  (reify* :implements [WithProp]
                    (^:property prop [this] [x y z])))"""
            )
            assert vec.v(1, 2, 3) == Point(1, 2, 3).prop

        def test_reify_empty_property_body(
            self,
            lcompile: CompileFn,
        ):
            Point = lcompile(
                """
                (fn* [x y z]
                  (reify* :implements [WithProp]
                    (^:property prop [this])))"""
            )
            assert None is Point(1, 2, 3).prop

        @pytest.mark.parametrize("kwarg_support", [":apply", ":collect", ":kwarg-it"])
        def test_reify_property_does_not_support_kwargs(
            self, lcompile: CompileFn, kwarg_support: str
        ):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    f"""
                    (fn* [x y z]
                      (reify* :implements [WithProp]
                        (^:property ^{{:kwargs {kwarg_support}}} prop [this])))"""
                )

        def test_reify_property_may_not_be_multi_arity(self, lcompile: CompileFn):
            with pytest.raises(compiler.CompilerException):
                lcompile(
                    """
                    (fn* [x]
                      (reify* :implements [WithProp]
                        (^:property prop [this] :a)
                        (^:property prop [this] :b)))"""
                )


class TestRequire:
    def test_require_must_have_at_least_one_namespace(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(require*)")

    @pytest.mark.parametrize(
        "code",
        [
            "(require* :edn)",
            "(require* :basilisp/edn)",
            '(require* "basilisp.edn")',
            '(require* basilisp.string "basilisp.edn")',
            "(require* basilisp.string :basilisp/edn)",
        ],
    )
    def test_require_namespace_must_be_symbol(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            "(require* [:basilisp/edn :as edn])",
            "(require* [basilisp.edn edn])",
            "(require* [basilisp.edn :as :edn])",
            "(require* [basilisp.edn :as])",
            "(require* [basilisp.edn :named edn])",
            "(require* [basilisp.edn :as basilisp edn])",
        ],
    )
    def test_require_aliased_namespace_format(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_require_namespace_must_exist(self, lcompile: CompileFn):
        with pytest.raises(ImportError):
            lcompile("(require* real.fake.ns)")

    @pytest.fixture
    def _import_ns(self, ns: runtime.Namespace):
        def _import_ns_module(name: str):
            ns_module = importlib.import_module(name)
            runtime.set_current_ns(ns.name)
            return ns_module

        return _import_ns_module

    @pytest.fixture
    def set_ns(self, _import_ns):
        return _import_ns("basilisp.set")

    @pytest.fixture
    def string_ns(self, _import_ns):
        return _import_ns("basilisp.string")

    def test_require_resolves_within_do_block(self, lcompile: CompileFn, string_ns):
        assert string_ns.join == lcompile(
            "(do (require* basilisp.string)) basilisp.string/join"
        )
        assert string_ns.join == lcompile(
            """
            (do (require* [basilisp.string :as str]))
            str/join
            """
        )

    @pytest.mark.parametrize(
        "code",
        [
            "(require* basilisp.string) basilisp.string/join",
            "(require* [basilisp.string :as str]) str/join",
        ],
    )
    def test_single_require(self, lcompile: CompileFn, string_ns, code: str):
        assert string_ns.join == lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            "(require* [basilisp.string :as str] basilisp.set) [str/join basilisp.set/union]",
            "(require* basilisp.string [basilisp.set :as set]) [basilisp.string/join set/union]",
        ],
    )
    def test_multi_require(self, lcompile: CompileFn, string_ns, set_ns, code: str):
        assert [string_ns.join, set_ns.union] == list(lcompile(code))

    def test_warn_on_duplicated_require_name(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        lcompile("(require* basilisp.string basilisp.string)")
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            r"duplicate name or alias 'basilisp.string' in require",
        )

    def test_warn_on_duplicated_import_name_with_alias(
        self, lcompile: CompileFn, assert_matching_logs
    ):
        lcompile("(require* [basilisp.edn :as serde] [basilisp.json :as serde])")
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            r"duplicate name or alias 'serde' in require",
        )

    def test_warn_on_shadowing_by_existing_import(
        self, ns: runtime.Namespace, lcompile: CompileFn, assert_matching_logs
    ):
        lcompile("(import json) (require* [basilisp.json :as json])")
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"name 'json' may be shadowed by an existing import in '{ns}'",
        )

    def test_warn_on_shadowing_by_existing_import_alias(
        self, ns: runtime.Namespace, lcompile: CompileFn, assert_matching_logs
    ):
        lcompile("(import [collections.abc :as abc]) (require* [basilisp.edn :as abc])")
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"name 'abc' may be shadowed by an existing import alias in '{ns}'",
        )

    def test_warn_on_shadowing_by_existing_namespace_alias(
        self, ns: runtime.Namespace, lcompile: CompileFn, assert_matching_logs
    ):
        lcompile(
            "(require* [basilisp.string :as edn]) (require* [basilisp.json :as edn])"
        )
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"name 'edn' may shadow an existing alias in '{ns}'",
        )


class TestSetBang:
    def test_num_elems(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(set!)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(set! target)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(set! target value arg)")

    def test_set_target_must_be_assignable_type(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile('(set! "a string" "another string")')

        with pytest.raises(compiler.CompilerException):
            lcompile("(set! 3 4)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(set! :kw :new-kw)")

    def test_set_cannot_assign_let_local(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(let [a :b] (set! a :c))")

    def test_set_cannot_assign_loop_local(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(loop [a :b] (set! a :c))")

    def test_set_cannot_assign_fn_arg_local(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn [a b] (set! a :c))")

    def test_set_cannot_assign_non_dynamic_var(self, lcompile: CompileFn):
        with pytest.raises(runtime.RuntimeException):
            lcompile(
                """
            (def static-var :kw)
            (set! static-var \"instead a string\")
            """
            )

    def test_set_cannot_assign_dynamic_var_without_thread_bindings(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        with pytest.raises(runtime.RuntimeException):
            lcompile(
                """
            (def ^:dynamic *dynamic-var* :kw)
            (set! *dynamic-var* \"instead a string\")
            """
            )

    def test_set_can_assign_thread_bound_dynamic_var(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        var = lcompile("(def ^:dynamic *thread-bound-var* :kw)")
        assert not var.is_thread_bound
        var.push_bindings(sym.symbol("a-symbol"))
        assert var.is_thread_bound
        assert sym.symbol("a-symbol") == lcompile("*thread-bound-var*")
        lcompile('(set! *thread-bound-var* "instead a string")')
        assert "instead a string" == lcompile("*thread-bound-var*")
        assert "instead a string" == var.value
        assert kw.keyword("kw") == var.root

    def test_set_can_object_attrs(self, lcompile: CompileFn, ns: runtime.Namespace):
        code = """
        (import* threading)
        (def tl (threading/local))
        (set! (.-some-field tl) :kw)
        """
        assert kw.keyword("kw") == lcompile(code)
        var = Var.find_in_ns(sym.symbol(ns.name), sym.symbol("tl"))
        assert var is not None
        assert kw.keyword("kw") == var.value.some_field

        assert "now a string" == lcompile('(set! (.-some-field tl) "now a string")')
        assert "now a string" == var.value.some_field


class TestThrow:
    def test_throw_not_enough_args(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(throw)")

    def test_throw_too_many_args(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(throw (python/ValueError) nil :a)")

    def test_throw(self, lcompile: CompileFn):
        with pytest.raises(AttributeError):
            lcompile("(throw (python/AttributeError))")

        with pytest.raises(TypeError):
            lcompile("(throw (python/TypeError))")

        with pytest.raises(ValueError):
            lcompile("(throw (python/ValueError))")

    def test_throw_chained(self, lcompile: CompileFn):
        try:
            lcompile(
                """
            (try
              (/ 1.0 0)
              (catch python/ZeroDivisionError e
                (throw (python/ValueError "lol") e)))
            """
            )
        except ValueError as e:
            assert str(e) == "lol"
            assert e.__suppress_context__ is True
            assert isinstance(e.__cause__, ZeroDivisionError)
            assert isinstance(e.__context__, ZeroDivisionError)

    def test_throw_suppress_chain(self, lcompile: CompileFn):
        try:
            lcompile(
                """
            (try
              (/ 1.0 0)
              (catch python/ZeroDivisionError e
                (throw (python/ValueError "lol") nil)))
            """
            )
        except ValueError as e:
            assert str(e) == "lol"
            assert e.__suppress_context__ is True
            assert e.__cause__ is None
            assert isinstance(e.__context__, ZeroDivisionError)

    def test_throw_context_only(self, lcompile: CompileFn):
        try:
            lcompile(
                """
            (try
              (/ 1.0 0)
              (catch python/ZeroDivisionError e
                (throw (python/ValueError "lol"))))
            """
            )
        except ValueError as e:
            assert str(e) == "lol"
            assert e.__suppress_context__ is False
            assert e.__cause__ is None
            assert isinstance(e.__context__, ZeroDivisionError)


class TestTryCatch:
    def test_single_catch_ignoring_binding(
        self,
        lcompile: CompileFn,
        capsys,
    ):
        code = """
          (try
            (.fake-lower "UPPER")
            (catch AttributeError _ "lower"))
        """
        assert "lower" == lcompile(code)

    def test_multiple_expressions_in_try_body(
        self,
        lcompile: CompileFn,
        capsys,
    ):
        code = """
          (try
            (print "hello")
            true
            :keyword
            (let [s "UPPER"]
              (.lower s))
            (catch AttributeError _ "lower"))
        """
        assert "upper" == lcompile(code)

    def test_single_catch_with_binding(self, lcompile: CompileFn, capsys):
        code = """
          (try
            (.fake-lower "UPPER")
            (catch python/AttributeError e (.-args e)))
        """
        assert ("'str' object has no attribute 'fake_lower'",) == lcompile(code)

    def test_multiple_catch(self, lcompile: CompileFn):
        code = """
          (try
            (.fake-lower "UPPER")
            (catch TypeError _ "lower")
            (catch AttributeError _ "mIxEd"))
        """
        assert "mIxEd" == lcompile(code)

    def test_multiple_catch_with_finally(self, lcompile: CompileFn, capsys):
        # If you hit an error here, do yourself a favor
        # and look in the import code first.
        code = """
          (import* builtins)
          (try
            (.fake-lower "UPPER")
            (catch TypeError _ "lower")
            (catch AttributeError _ "mIxEd")
            (finally (python/print "neither")))
        """
        assert "mIxEd" == lcompile(code)
        captured = capsys.readouterr()
        assert "neither\n" == captured.out

    def test_catch_num_elems(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
              (try
                (.lower "UPPER")
                (catch AttributeError _))
            """
            )

        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
              (try
                (.lower "UPPER")
                (catch AttributeError))
            """
            )

        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
              (try
                (.lower "UPPER")
                (catch))
            """
            )

    def test_catch_must_name_exception(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
              (try
                (.lower "UPPER")
                (catch :attribute-error _ "mIxEd"))
            """
            )

    def test_catch_name_must_be_symbol(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
              (try
                (.lower "UPPER")
                (catch AttributeError :e "mIxEd"))
            """
            )

    def test_body_may_not_appear_after_catch(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
              (try
                (.lower "UPPER")
                (catch AttributeError _ "mIxEd")
                "neither")
            """
            )

    def test_body_may_not_appear_after_finally(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
              (try
                (.lower "UPPER")
                (finally (python/print "mIxEd"))
                "neither")
            """
            )

    def test_catch_may_not_appear_after_finally(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
              (try
                (.fake-lower "UPPER")
                (finally (python/print "this is bad!"))
                (catch AttributeError _ "mIxEd"))
            """
            )

    def test_try_may_not_have_multiple_finallys(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile(
                """
              (try
                (.fake-lower "UPPER")
                (catch AttributeError _ "mIxEd")
                (finally (python/print "this is bad!"))
                (finally (python/print "but this is worse")))
            """
            )


class TestSymbolResolution:
    def test_bare_sym_resolves_builtins(self, lcompile: CompileFn):
        assert object is lcompile("object")

    def test_builtin_resolves_builtins(self, lcompile: CompileFn):
        assert object is lcompile("python/object")

    def test_builtins_fails_to_resolve_correctly(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("python/fake")

    def test_namespaced_sym_may_not_contain_period(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("other.ns/with.sep")

    def test_namespaced_sym_cannot_resolve(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("other.ns/name")

    def test_nested_namespaced_sym_will_resolve(self, lcompile: CompileFn):
        assert lmap.MapEntry.of is lcompile("basilisp.lang.map.MapEntry/of")

    def test_nested_bare_sym_will_not_resolve(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("basilisp.lang.map.MapEntry.of")

    @pytest.mark.parametrize(
        "code,v",
        [
            ("python.str/split", str.split),
            ("python.str/.split", str.split),
            ("python.int/from_bytes", int.from_bytes),
            ("python.int/.from_bytes", int.from_bytes),
        ],
    )
    def test_qualified_method_resolves(self, lcompile: CompileFn, code: str, v):
        assert v == lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            "(import* abc) abc",
            "(do (import* abc) abc)",
            "(do (do ((fn [] (import* abc))) abc))",
            "((fn [] (import* abc) abc))",
            "((fn [] (import* abc))) abc",
            """
            (import* collections.abc)
            (deftype* Importer []
              :implements [collections.abc/Callable]
              (--call-- [this] (import* abc) abc))
            ((Importer))""",
            """
            (import* collections.abc)
            (deftype* Importer []
              :implements [collections.abc/Callable]
              (--call-- [this] (import* abc)))
            ((Importer))
            abc""",
            """
            (import* collections.abc)
            (deftype* Importer []
              :implements [collections.abc/Callable]
              (--call-- [this] (import* abc) abc/ABC))
            (do (do ((Importer)) abc))""",
        ],
    )
    def test_imported_module_sym_resolves(self, lcompile: CompileFn, code: str):
        import abc

        imported_abc = lcompile(code)
        assert imported_abc is abc

    @pytest.mark.parametrize(
        "code",
        [
            "(import* abc) abc/ABC",
            "(do (import* abc) abc/ABC)",
            "((fn [] (import* abc) abc/ABC))",
            "((fn [] (import* abc))) abc/ABC",
            "(do ((fn [] (import* abc))) abc/ABC)",
            "(do (do ((fn [] (import* abc))) abc/ABC))"
            """
            (import* collections.abc)
            (deftype* Importer []
              :implements [collections.abc/Callable]
              (--call-- [this] (import* abc) abc/ABC))
            ((Importer))""",
            """
            (import* collections.abc)
            (deftype* Importer []
              :implements [collections.abc/Callable]
              (--call-- [this] (import* abc)))
            ((Importer))
            abc/ABC""",
            """
            (import* collections.abc)
            (deftype* Importer []
              :implements [collections.abc/Callable]
              (--call-- [this] (import* abc) abc/ABC))
            (do ((Importer)) abc/ABC)""",
            """
            (import* collections.abc)
            (deftype* Importer []
              :implements [collections.abc/Callable]
              (--call-- [this] (import* abc) abc/ABC))
            (do (do ((Importer)) abc/ABC))""",
        ],
    )
    def test_sym_from_import_resolves(self, lcompile: CompileFn, code: str):
        from abc import ABC

        imported_ABC = lcompile(code)
        assert imported_ABC is ABC

    @pytest.mark.parametrize(
        "code",
        [
            "(do ((fn [] (import* abc))) abc)",
            """
             (import* collections.abc)
             (deftype* Importer []
               :implements [collections.abc/Callable]
               (--call-- [this] (import* abc) abc/ABC))
             (do ((Importer)) abc)""",
        ],
    )
    def test_module_from_import_resolves(self, lcompile: CompileFn, code: str):
        import abc

        imported_abc = lcompile(code)
        assert imported_abc is abc

    @pytest.mark.parametrize(
        "code,ExceptionType",
        [
            ("(if false (import* abc) nil) abc", NameError),
            ("(do (if false (import* abc) nil) abc)", NameError),
            ("(if false (import* abc) nil) abc/ABC", NameError),
            ("(do (if false (import* abc) nil) abc/ABC)", NameError),
        ],
    )
    def test_unresolvable_imported_symbols(
        self, lcompile: CompileFn, code: str, ExceptionType
    ):
        # Most of these cases are just too dynamic for the compiler to statically
        # resolve these symbols. I suspect these cases are infrequently or never
        # applicable, so I'm not going to spend time making them work right now.
        # If an important use case arises for more complex import resolution,
        # then we can think about reworking the resolver.
        with pytest.raises(ExceptionType):
            lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            """
            (fn []
              (def a :a)
              (var a))""",
            """
            (import* collections.abc)
            (deftype* Definer []
              :implements [collections.abc/Callable]
              (--call-- [this] (def a :a) (var a)))
            (Definer)""",
        ],
    )
    def test_symbol_deffed_in_fn_or_method_will_resolve_in_fn_or_method(
        self,
        ns: runtime.Namespace,
        lcompile: CompileFn,
        code: str,
    ):
        # This behavior is peculiar and perhaps even _wrong_, but it matches how
        # Clojure treats Vars defined in functions. Of course, generally speaking,
        # Vars should not be defined like this, so I suppose it's not a huge deal.
        fn = lcompile(code)

        resolved_var = ns.find(sym.symbol("a"))
        assert not resolved_var.is_bound

        returned_var = fn()
        assert returned_var is resolved_var
        assert returned_var.is_bound
        assert returned_var.value == kw.keyword("a")

    @pytest.mark.parametrize(
        "code",
        [
            """
            (do
              (fn [] (def a :a))
              (var a))""",
            """
            (fn [] (def a :a))
            (var a)""",
            """
            (import* collections.abc)
            (deftype* Definer []
              :implements [collections.abc/Callable]
              (--call-- [this] (def a :a)))
            (var a)""",
        ],
    )
    def test_symbol_deffed_in_fn_or_method_will_resolve_outside_fn_or_method(
        self, ns: runtime.Namespace, lcompile: CompileFn, code: str
    ):
        var = lcompile(code)
        assert not var.is_bound

        resolved_var = ns.find(sym.symbol("a"))
        assert not resolved_var.is_bound

        assert var is resolved_var

    def test_local_deftype_classmethod_resolves(self, lcompile: CompileFn):
        Point = lcompile(
            """
            (import* abc)
            (def WithCls
              (python/type "WithCls"
                             #py (abc/ABC)
                             #py {"create"
                                  (python/classmethod
                                   (abc/abstractmethod
                                    (fn [cls])))}))
            (deftype* Point [x y z]
              :implements [WithCls]
              (^:classmethod create [cls x y z]
                [cls x y z]))
            """
        )

        assert vec.v(Point, 1, 2, 3) == lcompile("(Point/create 1 2 3)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(Point/make 1 2 3)")

    def test_local_deftype_staticmethod_resolves(self, lcompile: CompileFn):
        Point = lcompile(
            """
            (import* abc)
            (def WithStatic
              (python/type "WithStatic"
                             #py (abc/ABC)
                             #py {"dostatic"
                                  (python/staticmethod
                                   (abc/abstractmethod
                                    (fn [])))}))
            (deftype* Point [x y z]
              :implements [WithStatic]
              (^:staticmethod dostatic [arg1 arg2]
                [arg1 arg2]))
            """
        )

        assert Point.dostatic is lcompile("Point/dostatic")
        assert vec.v(kw.keyword("a"), 2) == lcompile("(Point/dostatic :a 2)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(Point/do-non-static 1 2)")

    def test_aliased_namespace_not_hidden_by_python_module(
        self, lcompile: CompileFn, monkeypatch: pytest.MonkeyPatch
    ):
        with TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            monkeypatch.chdir(tmpdir)
            monkeypatch.syspath_prepend(tmpdir)
            monkeypatch.setattr(
                "sys.modules", {name: module for name, module in sys.modules.items()}
            )

            os.makedirs(os.path.join(tmpdir, "project"), exist_ok=True)
            module_file_path = os.path.join(tmpdir, "project", "fileinput.lpy")

            with open(module_file_path, mode="w") as f:
                f.write(
                    """
                (ns project.fileinput
                  (:import fileinput))

                (def some-sym :project.fileinput/test-value)
                """
                )

            try:
                assert kw.keyword("test-value", ns="project.fileinput") == lcompile(
                    """
                (require '[project.fileinput :as fileinput])
                fileinput/some-sym
                """
                )
            finally:
                monkeypatch.chdir(cwd)
                os.unlink(module_file_path)

    def test_import_name_with_underscores_resolves_properly(
        self,
        lcompile: CompileFn,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ):
        package = tmp_path / "a_b"
        package.mkdir(parents=True)
        file = package / "c.py"
        file.write_text("val = 10")
        monkeypatch.syspath_prepend(str(tmp_path))
        assert lcompile("(import a-b.c) [a-b.c/val a_b.c/val]") == vec.v(10, 10)

    @pytest.mark.parametrize(
        "code",
        [
            """
        (import re)

        (defn re [s]
          (re/fullmatch #"[A-Z]+" s))

        (re "YES")
        """,
            """
        (import [re :as regex])

        (defn regex [s]
          (regex/fullmatch #"[A-Z]+" s))

        (regex "YES")
        """,
        ],
    )
    def test_imports_dont_shadow_def_names(self, lcompile: CompileFn, code: str):
        match = lcompile(code)
        assert match is not None

    def test_imports_names_resolve(self, lcompile: CompileFn):
        import re

        assert lcompile("(import re) re") is re
        assert lcompile("(import [re :as regex]) regex") is re

        import urllib.parse

        assert (
            lcompile("(import urllib.parse) urllib.parse/urlparse")
            is urllib.parse.urlparse
        )
        assert (
            lcompile("(import [urllib.parse :as urlparse]) urlparse/urlparse")
            is urllib.parse.urlparse
        )

    def test_aliased_var_does_not_resolve(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        current_ns: runtime.Namespace = ns
        other_ns_name = sym.symbol("other.ns")
        try:
            other_ns = get_or_create_ns(other_ns_name)
            current_ns.add_alias(other_ns, other_ns_name)
            current_ns.add_alias(other_ns, sym.symbol("other"))

            with pytest.raises(compiler.CompilerException):
                lcompile("(other/m :arg)")
        finally:
            runtime.Namespace.remove(other_ns_name)

    def test_private_aliased_var_does_not_resolve(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        current_ns: runtime.Namespace = ns
        other_ns_name = sym.symbol("other.ns")
        private_var_sym = sym.symbol("m")
        try:
            other_ns = get_or_create_ns(other_ns_name)
            current_ns.add_alias(other_ns, sym.symbol("other"))

            private_var = Var(
                other_ns, private_var_sym, meta=lmap.map({SYM_PRIVATE_META_KEY: True})
            )
            private_var.set_value(kw.keyword("private-var"))
            other_ns.intern(private_var_sym, private_var)

            with pytest.raises(compiler.CompilerException):
                lcompile("(other/m :arg)")
        finally:
            runtime.Namespace.remove(other_ns_name)

    def test_private_var_does_not_resolve_during_macroexpansion(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        # In a previous version of the compiler, it was possible to get access
        # to namespace private Vars simply by referring to them from within macros
        # from the same namespace.
        #
        # https://github.com/basilisp-lang/basilisp/issues/646
        with pytest.raises(compiler.CompilerException):
            lcompile("(let [] basilisp.core/*generated-python*)")

    def test_aliased_macro_symbol_resolution(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        current_ns: runtime.Namespace = ns
        other_ns_name = sym.symbol("other.ns")
        try:
            other_ns = get_or_create_ns(other_ns_name)
            current_ns.add_alias(other_ns, other_ns_name)
            current_ns.add_alias(other_ns, sym.symbol("other"))

            with runtime.ns_bindings(other_ns_name.name):
                lcompile("(def ^:macro m (fn* [&env &form v] v))")

            with runtime.ns_bindings(current_ns.name):
                assert kw.keyword("z") == lcompile("(other.ns/m :z)")
                assert kw.keyword("a") == lcompile("(other/m :a)")
        finally:
            runtime.Namespace.remove(other_ns_name)

    def test_fully_namespaced_sym_resolves(self, lcompile: CompileFn):
        """Ensure that references to Vars in other Namespaces by a fully Namespace
        qualified symbol always resolve, regardless of whether the Namespace has
        been aliased within the current Namespace."""
        other_ns_name = sym.symbol("other.ns")
        public_var_sym = sym.symbol("public-var")
        private_var_sym = sym.symbol("private-var")
        third_ns_name = sym.symbol("third.ns")
        try:
            other_ns = runtime.Namespace.get_or_create(other_ns_name)
            runtime.Namespace.get_or_create(third_ns_name)

            # Intern a public symbol in `other.ns`
            public_var = Var(other_ns, public_var_sym)
            public_var.set_value(kw.keyword("public-var"))
            other_ns.intern(public_var_sym, public_var)

            # Intern a private symbol in `other.ns`
            private_var = Var(
                other_ns, private_var_sym, meta=lmap.map({SYM_PRIVATE_META_KEY: True})
            )
            private_var.set_value(kw.keyword("private-var"))
            other_ns.intern(private_var_sym, private_var)

            with runtime.ns_bindings(third_ns_name.name):
                # Verify that we can refer to `other.ns/public-var` with a fully
                # namespace qualified symbol
                assert kw.keyword("public-var") == lcompile(
                    "other.ns/public-var",
                    resolver=runtime.resolve_alias,
                )

                # Verify we cannot refer to `other.ns/private-var` because it is
                # marked private
                with pytest.raises(compiler.CompilerException):
                    lcompile(
                        "other.ns/private-var",
                        resolver=runtime.resolve_alias,
                    )

        finally:
            runtime.Namespace.remove(other_ns_name)
            runtime.Namespace.remove(third_ns_name)

    def test_cross_ns_macro_symbol_resolution(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        """Ensure that a macro symbol, `a`, delegating to another macro, named
        by the symbol `b`, in a namespace directly required by `a`'s namespace
        (and which will not be required by downstream namespaces) is still
        properly resolved when used by the final consumer."""
        current_ns: runtime.Namespace = ns
        other_ns_name = sym.symbol("other.ns")
        third_ns_name = sym.symbol("third.ns")
        try:
            other_ns = get_or_create_ns(other_ns_name)
            current_ns.add_alias(other_ns, other_ns_name)

            third_ns = get_or_create_ns(third_ns_name)
            other_ns.add_alias(third_ns, third_ns_name)

            with runtime.ns_bindings(third_ns_name.name):
                lcompile(
                    "(def ^:macro t (fn* [&env &form v] `(name ~v)))",
                    resolver=runtime.resolve_alias,
                )

            with runtime.ns_bindings(other_ns_name.name):
                lcompile(
                    "(def ^:macro o (fn* [&env &form v] `(third.ns/t ~v)))",
                    resolver=runtime.resolve_alias,
                )

            with runtime.ns_bindings(current_ns.name):
                assert "z" == lcompile(
                    "(other.ns/o :z)", resolver=runtime.resolve_alias
                )
        finally:
            runtime.Namespace.remove(other_ns_name)
            runtime.Namespace.remove(third_ns_name)

    def test_cross_ns_macro_symbol_resolution_with_aliases(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        """Ensure that `a` macro symbol, a, delegating to another macro, named
        by the symbol `b`, which is referenced by `a`'s namespace (and which will
        not be referred by downstream namespaces) is still properly resolved
        when used by the final consumer."""
        current_ns: runtime.Namespace = ns
        other_ns_name = sym.symbol("other.ns")
        third_ns_name = sym.symbol("third.ns")
        try:
            other_ns = get_or_create_ns(other_ns_name)
            current_ns.add_alias(other_ns, other_ns_name)

            third_ns = get_or_create_ns(third_ns_name)
            other_ns.add_alias(third_ns, sym.symbol("third"))

            with runtime.ns_bindings(third_ns_name.name):
                lcompile(
                    "(def ^:macro t (fn* [&env &form v] `(name ~v)))",
                    resolver=runtime.resolve_alias,
                )

            with runtime.ns_bindings(other_ns_name.name):
                lcompile(
                    "(def ^:macro o (fn* [&env &form v] `(third/t ~v)))",
                    resolver=runtime.resolve_alias,
                )

            with runtime.ns_bindings(current_ns.name):
                assert "z" == lcompile(
                    "(other.ns/o :z)", resolver=runtime.resolve_alias
                )
        finally:
            runtime.Namespace.remove(other_ns_name)
            runtime.Namespace.remove(third_ns_name)

    def test_cross_ns_macro_symbol_resolution_with_refers(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        """Ensure that a macro symbol, `a`, delegating to another macro, named
        by the symbol `b`, which is referred by `a`'s namespace (and which will
        not be referred by downstream namespaces) is still properly resolved
        when used by the final consumer."""
        current_ns: runtime.Namespace = ns
        other_ns_name = sym.symbol("other.ns")
        third_ns_name = sym.symbol("third.ns")
        try:
            other_ns = get_or_create_ns(other_ns_name)
            current_ns.add_alias(other_ns, other_ns_name)

            third_ns = get_or_create_ns(third_ns_name)

            with runtime.ns_bindings(third_ns_name.name):
                lcompile(
                    "(def ^:macro t (fn* [&env &form v] `(name ~v)))",
                    resolver=runtime.resolve_alias,
                )

            other_ns.add_refer(sym.symbol("t"), third_ns.find(sym.symbol("t")))

            with runtime.ns_bindings(other_ns_name.name):
                lcompile(
                    "(def ^:macro o (fn* [&env &form v] `(t ~v)))",
                    resolver=runtime.resolve_alias,
                )

            with runtime.ns_bindings(current_ns.name):
                assert "z" == lcompile(
                    "(other.ns/o :z)", resolver=runtime.resolve_alias
                )
        finally:
            runtime.Namespace.remove(other_ns_name)
            runtime.Namespace.remove(third_ns_name)

    def test_cross_ns_macro_deftype_symbol_resolution(
        self,
        lcompile: CompileFn,
        tmp_path: pathlib.Path,
        monkeypatch,
    ):
        """"""
        cross_ns_deftype_test_dir = tmp_path / "cross_ns_deftype_test"
        cross_ns_deftype_test_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.syspath_prepend(tmp_path)

        other_ns_file = cross_ns_deftype_test_dir / "other.lpy"
        other_ns_file.write_text(
            """
            (ns cross-ns-deftype-test.other)

            (deftype TypeOther [])
            """
        )

        main_ns_file = cross_ns_deftype_test_dir / "issue.lpy"
        main_ns_file.write_text(
            """
            (ns cross-ns-deftype-test.issue
              (:require cross-ns-deftype-test.other))

            (defmacro issue []
              `(identity ~(cross-ns-deftype-test.other/TypeOther)))

            (def result [(issue) (cross-ns-deftype-test.other/TypeOther)])
            """
        )

        result = lcompile(
            """
            (require 'cross-ns-deftype-test.issue)

            cross-ns-deftype-test.issue/result
            """
        )

        assert isinstance(result[0], type(result[1]))

        sys.modules.pop("cross_ns_deftype_test.issue")
        sys.modules.pop("cross_ns_deftype_test.other")


class TestWarnOnArityMismatch:
    def test_warning_on_arity_mismatch(
        self,
        lcompile: CompileFn,
        ns: runtime.Namespace,
        compiler_file_path: str,
        assert_matching_logs,
    ):
        var = lcompile("(defn jdkdka [a b c] [a b c])")
        lcompile(
            "(fn* [] (jdkdka :a :b))",
            opts={compiler.WARN_ON_ARITY_MISMATCH: True},
        )
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"calling function {var} \({compiler_file_path}:\d+\) with 2 arguments; expected any of: 3",
        )

    def test_warning_on_arity_mismatch_variadic(
        self,
        lcompile: CompileFn,
        ns: runtime.Namespace,
        compiler_file_path: str,
        assert_matching_logs,
    ):
        var = lcompile(
            "(defn pqkdha ([] :none) ([a b] [a b]) ([a b c d & others] [a b c d others]))"
        )
        lcompile(
            "(fn* [] (pqkdha :a))",
            opts={compiler.WARN_ON_ARITY_MISMATCH: True},
        )
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"calling function {var} \({compiler_file_path}:\d+\) with 1 arguments; expected any of: 0, 2, 4+",
        )

    def test_warning_on_arity_mismatch_variadic_with_partial(
        self,
        lcompile: CompileFn,
        ns: runtime.Namespace,
        compiler_file_path: str,
        assert_matching_logs,
    ):
        var = lcompile(
            """
            (defn dazhqpe ([] :none) ([a b] [a b]) ([a b c d & others] [a b c d others]))
            (def zjpqeee (partial dazhqpe :a :b :c))
            """
        )
        lcompile(
            "(fn* [] (zjpqeee))",
            opts={compiler.WARN_ON_ARITY_MISMATCH: True},
        )
        assert_matching_logs(
            "basilisp.lang.compiler.analyzer",
            logging.WARNING,
            rf"calling function {var} \({compiler_file_path}:\d+\) with 0 arguments; expected any of: 1+",
        )

    def test_no_runtime_warning_on_arity_mismatch_variadic_with_partial(
        self,
        lcompile: CompileFn,
        ns: runtime.Namespace,
        compiler_file_path: str,
        caplog,
    ):
        # This case is tricky to detect at compile time, so we implemented a runtime
        # check instead.
        lcompile(
            """
            (defn klkpqkc [a] a)
            (def yyqeken (partial klkpqkc :a))
            """
        )
        assert not list(
            filter(
                lambda rec: re.match(
                    "invalid partial function application of 'klkpqkc' detected",
                    rec[2],
                ),
                caplog.record_tuples,
            )
        )

    def test_runtime_warning_on_arity_mismatch_variadic_with_partial(
        self,
        lcompile: CompileFn,
        ns: runtime.Namespace,
        compiler_file_path: str,
        assert_matching_logs,
    ):
        # This case is tricky to detect at compile time, so we implemented a runtime
        # check instead.
        lcompile(
            """
            (defn ueqhenn [])
            (def pqencka (partial ueqhenn :a))
            """
        )
        assert_matching_logs(
            "basilisp.lang.runtime",
            logging.WARNING,
            "invalid partial function application of 'ueqhenn' detected: 1 arguments given; expected any of: 0",
        )


class TestWarnOnVarIndirection:
    @pytest.fixture
    def other_ns(self, lcompile: CompileFn, ns: runtime.Namespace):
        current_ns: runtime.Namespace = ns
        other_ns_name = sym.symbol("other.ns")
        try:
            other_ns = get_or_create_ns(other_ns_name)
            Var.intern(other_ns, sym.symbol("m"), lambda x: x)
            current_ns.add_alias(other_ns, other_ns_name)
            current_ns.add_alias(other_ns, sym.symbol("other"))

            with runtime.ns_bindings(current_ns.name):
                yield
        finally:
            runtime.Namespace.remove(other_ns_name)

    def test_warning_for_cross_ns_reference(
        self, lcompile: CompileFn, ns: runtime.Namespace, other_ns, assert_matching_logs
    ):
        lcompile(
            "(fn [] (other.ns/m :z))", opts={compiler.WARN_ON_VAR_INDIRECTION: True}
        )
        assert_matching_logs(
            "basilisp.lang.compiler.generator",
            logging.WARNING,
            rf"could not resolve a direct link to Var 'm' \({ns}:\d+\)",
        )

    def test_no_warning_for_cross_ns_reference_if_warning_disabled(
        self,
        lcompile: CompileFn,
        ns: runtime.Namespace,
        other_ns,
        assert_no_matching_logs,
    ):
        lcompile(
            "(fn [] (other.ns/m :z))", opts={compiler.WARN_ON_VAR_INDIRECTION: False}
        )
        assert_no_matching_logs(
            "basilisp.lang.compiler.generator",
            logging.WARNING,
            rf"could not resolve a direct link to Var 'm' \({ns}:\d+\)",
        )

    def test_warning_for_cross_ns_alias_reference(
        self, lcompile: CompileFn, ns: runtime.Namespace, other_ns, assert_matching_logs
    ):
        lcompile("(fn [] (other/m :z))", opts={compiler.WARN_ON_VAR_INDIRECTION: True})
        assert_matching_logs(
            "basilisp.lang.compiler.generator",
            logging.WARNING,
            rf"could not resolve a direct link to Var 'm' \({ns}:\d+\)",
        )

    def test_no_warning_for_cross_ns_alias_reference_if_warning_enabled_but_suppressed_locally(
        self, lcompile: CompileFn, other_ns, assert_no_matching_logs
    ):
        lcompile(
            "(fn [] (^:no-warn-on-var-indirection other/m :z))",
            opts={compiler.WARN_ON_VAR_INDIRECTION: True},
        )
        assert_no_matching_logs(
            "basilisp.lang.compiler.generator",
            logging.WARNING,
            r"could not resolve a direct link to Var 'm' \(test:\d+\)",
        )

    def test_no_warning_for_cross_ns_alias_reference_if_warning_disabled(
        self, lcompile: CompileFn, other_ns, assert_no_matching_logs
    ):
        lcompile("(fn [] (other/m :z))", opts={compiler.WARN_ON_VAR_INDIRECTION: False})
        assert_no_matching_logs(
            "basilisp.lang.compiler.generator",
            logging.WARNING,
            r"could not resolve a direct link to Var 'm' \(test:\d+\)",
        )

    def test_warning_on_imported_name(
        self, lcompile: CompileFn, ns: runtime.Namespace, assert_no_matching_logs
    ):
        """Basilisp should be able to directly resolve a link to cross-namespace
        imports, so no warning should be raised."""
        ns.add_import(sym.symbol("string"), __import__("string"))

        with runtime.ns_bindings(ns.name):
            lcompile(
                '(fn [] (string/capwords "capitalize this"))',
                opts={compiler.WARN_ON_VAR_INDIRECTION: True},
            )
            assert_no_matching_logs(
                "basilisp.lang.compiler.generator",
                logging.WARNING,
                r"could not resolve a direct link to Python variable 'string/m' \(test:\d+\)",
            )

    def test_exception_raised_for_nonexistent_imported_name(
        self, lcompile: CompileFn, ns: runtime.Namespace, caplog
    ):
        """If a name does not exist, then a CompilerException will be raised."""
        ns.add_import(sym.symbol("string"), __import__("string"))

        with runtime.ns_bindings(ns.name), pytest.raises(compiler.CompilerException):
            lcompile(
                "(fn [] (string/m :z))", opts={compiler.WARN_ON_VAR_INDIRECTION: True}
            )

    def test_exception_raised_for_nonexistent_var_name(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(fn [] m)", opts={compiler.WARN_ON_VAR_INDIRECTION: True})


class TestVar:
    def test_var_num_elems(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(var)")

        with pytest.raises(compiler.CompilerException):
            lcompile("(var test/some-var test/other-var)")

    def test_var_does_not_resolve(self, lcompile: CompileFn):
        with pytest.raises(compiler.CompilerException):
            lcompile("(var test/definitely-not-a-var-in-this-namespace)")

    def test_var(self, lcompile: CompileFn, ns: runtime.Namespace):
        code = f"""
        (def some-var "a value")

        (var {ns}/some-var)"""

        ns_name = ns.name
        v = lcompile(code)
        assert v == Var.find_in_ns(sym.symbol(ns_name), sym.symbol("some-var"))
        assert v.value == "a value"

    def test_var_reader_literal(self, lcompile: CompileFn, ns: runtime.Namespace):
        code = f"""
        (def some-var "a value")

        #'{ns}/some-var"""

        ns_name = ns.name
        v = lcompile(code)
        assert v == Var.find_in_ns(sym.symbol(ns_name), sym.symbol("some-var"))
        assert v.value == "a value"


class TestYield:
    @pytest.mark.parametrize(
        "code",
        [
            "(yield)",
            "(yield :a)",
            "(let* [v :a] (yield v))",
        ],
    )
    def test_yield_must_be_in_fn_context(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    @pytest.mark.parametrize(
        "code",
        [
            "(fn [] (yield :two :items))",
            "(fn [] (yield :three :items :now))",
        ],
    )
    def test_yield_num_elems(self, lcompile: CompileFn, code: str):
        with pytest.raises(compiler.CompilerException):
            lcompile(code)

    def test_yield_control_only(self, lcompile: CompileFn, ns: runtime.Namespace):
        f = lcompile(
            """
            (def state (atom nil))
            (fn []
              (reset! state :started)
              (yield)
              (reset! state :done))
            """
        )
        state = ns.find(sym.symbol("state")).value
        coro = f()
        assert None is state.deref()
        assert None is next(coro)
        assert kw.keyword("started") == state.deref()
        assert None is next(coro, None)
        assert kw.keyword("done") == state.deref()

    def test_yield_value(self, lcompile: CompileFn, ns: runtime.Namespace):
        f = lcompile(
            """
            (def state (atom nil))
            (fn []
              (reset! state :started)
              (yield :yielding)
              (reset! state :done))
            """
        )
        state = ns.find(sym.symbol("state")).value
        coro = f()
        assert None is state.deref()
        assert kw.keyword("yielding") == next(coro)
        assert kw.keyword("started") == state.deref()
        assert None is next(coro, None)
        assert kw.keyword("done") == state.deref()

    def test_yield_as_coroutine(self, lcompile: CompileFn, ns: runtime.Namespace):
        f = lcompile(
            """
            (def state (atom nil))
            (fn []
              (reset! state :started)
              (let [v (yield :yielding)]
                (reset! state v)))
            """
        )
        state = ns.find(sym.symbol("state")).value
        coro = f()
        assert None is state.deref()
        assert kw.keyword("yielding") == next(coro)
        assert kw.keyword("started") == state.deref()
        with pytest.raises(StopIteration):
            coro.send(kw.keyword("coroutine-value"))
        assert kw.keyword("coroutine-value") == state.deref()

    def test_yield_as_coroutine_with_multiple_yields(
        self, lcompile: CompileFn, ns: runtime.Namespace
    ):
        f = lcompile(
            """
            (def state (atom nil))
            (fn []
              (reset! state :started)
              (let [v (yield :yielding)]
                (reset! state v)
                (yield)
                (reset! state :done)))
            """
        )
        state = ns.find(sym.symbol("state")).value
        coro = f()
        assert None is state.deref()
        assert kw.keyword("yielding") == next(coro)
        assert kw.keyword("started") == state.deref()
        assert None is coro.send(kw.keyword("coroutine-value"))
        assert kw.keyword("coroutine-value") == state.deref()
        assert None is next(coro, None)
        assert kw.keyword("done") == state.deref()
