import os
import platform
import shutil
import subprocess
import sys

import pytest

from basilisp.lang import runtime
from basilisp.lang import symbol as sym


class TestTestrunner:
    @pytest.fixture
    def run_result(self, pytester: pytest.Pytester) -> pytest.RunResult:
        runtime.Namespace.remove(sym.symbol("test-testrunner"))
        code = """
        (ns test-testrunner
          (:require
           [basilisp.test :refer [deftest is are testing]]))

        (deftest assertion-test
          (testing "is assertions"
            (is true)
            (is false)
            (is (some #{5} #{6 7}))
            (is (some #{7} #{6 7}))
            (is (= "string" "string"))
            (is (thrown? basilisp.lang.exception/ExceptionInfo (throw (ex-info "Exception" {}))))
            (is (thrown? basilisp.lang.exception/ExceptionInfo (throw (python/Exception))))
            (is (throw (ex-info "Uncaught exception" {})))
            (is (thrown-with-msg?
                  basilisp.lang.exception/ExceptionInfo
                  #"Caught exception"
                  (throw (ex-info "Caught exception message" {}))))
            (is (thrown-with-msg?
                  basilisp.lang.exception/ExceptionInfo
                  #"Known exception"
                  (throw (ex-info "Unexpected exception" {})))))

          (testing "are assertions"
            (are [exp actual] (= exp actual)
              1      1
              :hi    :hi
              "true" false
              4.6    4.6)))

        (deftest passing-test
          (is true))

        (deftest error-test
          (throw
            (ex-info "This test will count as an error." {})))
        
        ;; Test that syntax quoted forms still get expanded correctly into assertions
        (defmacro syntax-quote-test-make []
          `(deftest syntax-quote-seq-test
             (is (= 5 4))))
        (syntax-quote-test-make)
        """
        pytester.makefile(".lpy", test_testrunner=code)
        pytester.syspathinsert()
        yield pytester.runpytest()
        runtime.Namespace.remove(sym.symbol("test-testrunner"))

    def test_outcomes(self, run_result: pytest.RunResult):
        run_result.assert_outcomes(passed=1, failed=3)

    def test_failure_repr(self, run_result: pytest.RunResult):
        run_result.stdout.fnmatch_lines(
            [
                "FAIL in (assertion-test) (test_testrunner.lpy:8)",
                "     is assertions :: Test failure: false",
                "",
                "    expected: (not false)",
                "      actual: false",
            ],
            consecutive=True,
        )

        run_result.stdout.fnmatch_lines(
            [
                "FAIL in (assertion-test) (test_testrunner.lpy:9)",
                "     is assertions :: Test failure: (some #{5} #{6 7})",
                "",
                "    expected: (not nil)",
                "      actual: nil",
            ],
            consecutive=True,
        )

        run_result.stdout.fnmatch_lines(
            [
                "FAIL in (assertion-test) (test_testrunner.lpy:13)",
                "     is assertions :: Expected <class 'basilisp.lang.exception.ExceptionInfo'>; got <class 'Exception'> instead",
                "",
                "    expected: <class 'basilisp.lang.exception.ExceptionInfo'>",
                "      actual: Exception()",
            ],
            consecutive=True,
        )

        run_result.stdout.fnmatch_lines(
            [
                "FAIL in (assertion-test) (test_testrunner.lpy:19)",
                "     is assertions :: Regex pattern did not match",
                "",
                '    expected: #"Known exception"',
                '      actual: "Unexpected exception {}"',
            ],
            consecutive=True,
        )

        run_result.stdout.fnmatch_lines(
            [
                # Note the lack of line number, since `are` assertions generally lose
                # the original line number during templating
                "FAIL in (assertion-test) (test_testrunner.lpy)",
                '     are assertions :: Test failure: (= "true" false)',
                "",
                '    expected: "true"',
                "      actual: false",
            ],
            consecutive=True,
        )

        run_result.stdout.fnmatch_lines(
            [
                "FAIL in (syntax-quote-seq-test) (test_testrunner.lpy)",
                "    Test failure: (basilisp.core/= 5 4)",
                "",
                "    expected: 5",
                "      actual: 4",
            ],
            consecutive=True,
        )

    @pytest.mark.xfail(
        platform.python_implementation() == "PyPy" and sys.version_info < (3, 10),
        reason=(
            "PyPy 3.9 fails this test because it intermittently produces an incorrect"
            "line number (128014) in the exception traceback, which is clearly erroneous."
        ),
    )
    def test_error_repr(self, run_result: pytest.RunResult):
        expected = [
            "ERROR in (assertion-test) (test_testrunner.lpy:14)",
            "",
            "Traceback (most recent call last):",
            '  File "*test_testrunner.lpy", line 14, in assertion_test',
            '    (is (throw (ex-info "Uncaught exception" {})))',
            "basilisp.lang.exception.ExceptionInfo: Uncaught exception {}",
        ]

        run_result.stdout.fnmatch_lines(
            expected,
            consecutive=True,
        )

        run_result.stdout.fnmatch_lines(
            [
                "ERROR in (error-test) (test_testrunner.lpy)",
                "Traceback (most recent call last):",
                '  File "*test_testrunner.lpy", line 35, in error_test',
                "    (throw",
                "basilisp.lang.exception.ExceptionInfo: This test will count as an error. {}",
            ]
        )


def test_fixtures(pytester: pytest.Pytester):
    code = """
    (ns test-fixtures
      (:require
       [basilisp.test :refer [deftest is use-fixtures]]))

    (def once-no-cleanup (volatile! 0))
    (def once-cleanup (volatile! 0))
    (def each-no-cleanup (volatile! 0))
    (def each-cleanup (volatile! 0))

    ;; return here rather than yielding
    (defn once-fixture-no-cleanup []
      (vswap! once-no-cleanup inc))

    (defn once-fixture-w-cleanup []
      (vswap! once-cleanup inc)
      (yield)
      (vswap! once-cleanup dec))

    ;; yield here rather than returning, even w/o cleanup step
    (defn each-fixture-no-cleanup []
      (vswap! each-no-cleanup inc)
      (yield))

    (defn each-fixture-w-cleanup []
      (vswap! each-cleanup inc)
      (yield)
      (vswap! each-cleanup dec))

    (use-fixtures :once once-fixture-no-cleanup once-fixture-w-cleanup)
    (use-fixtures :each each-fixture-no-cleanup each-fixture-w-cleanup)

    (deftest passing-test
      (is true))

    (deftest failing-test
      (is false))
    """
    pytester.makefile(".lpy", test_fixtures=code)
    pytester.syspathinsert()
    result: pytest.RunResult = pytester.runpytest()
    result.assert_outcomes(passed=1, failed=1)

    get_volatile = lambda vname: runtime.Var.find_safe(
        sym.symbol(vname, ns="test-fixtures")
    ).value.deref()
    assert 1 == get_volatile("once-no-cleanup")
    assert 0 == get_volatile("once-cleanup")
    assert 2 == get_volatile("each-no-cleanup")
    assert 0 == get_volatile("each-cleanup")


@pytest.mark.parametrize(
    "fixture,style,errors,passes,failures",
    [
        ("error-during-setup", ":once", 2, 0, 0),
        ("error-during-setup", ":each", 2, 0, 0),
        ("error-during-teardown", ":once", 1, 1, 1),
        ("error-during-teardown", ":each", 2, 1, 1),
    ],
)
def test_fixtures_with_errors(
    pytester: pytest.Pytester,
    fixture: str,
    style: str,
    errors: int,
    passes: int,
    failures: int,
):
    runtime.Namespace.remove(sym.symbol("test-fixtures-with-errors"))
    code = f"""
    (ns test-fixtures-with-errors
      (:require
       [basilisp.test :refer [deftest is use-fixtures]]))

    (defn error-during-setup []
      (throw (ex-info "Setup error" {{}}))
      (yield))

    (defn error-during-teardown []
      (yield)
      (throw (ex-info "Teardown error" {{}})))

    (use-fixtures {style} {fixture})

    (deftest passing-test
      (is true))

    (deftest failing-test
      (is false))
    """
    pytester.makefile(".lpy", test_fixtures_with_errors=code)
    pytester.syspathinsert()
    result: pytest.RunResult = pytester.runpytest()
    result.assert_outcomes(passed=passes, failed=failures, errors=errors)


def test_basilisp_test_noargs(pytester: pytest.Pytester):
    runtime.Namespace.remove(sym.symbol("a.test-path"))

    code = """
    (ns tests.test-path
      (:require
       [basilisp.test :refer [deftest is]]))
    (deftest passing-test
      (is true))
    """
    pytester.makefile(".lpy", **{"./tests/test_path": code})

    # I couldn't find a way to directly manipulate the pytester's
    # `sys.path` with the precise control needed by this test, so we're
    # invoking `basilisp test` directly as a subprocess instead ...
    basilisp = shutil.which("basilisp")
    cmd = [basilisp, "test"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=pytester.path)

    assert "==== 1 passed" in result.stdout.strip()

    assert result.returncode == 0


class TestCollection:
    @pytest.fixture(autouse=True)
    def reset_collection_config(self):
        from basilisp.contrib.pytest import testrunner

        testrunner._get_test_file_path.cache_clear()
        testrunner._get_test_file_pattern.cache_clear()

        yield

        testrunner._get_test_file_path.cache_clear()
        testrunner._get_test_file_pattern.cache_clear()

    def test_test_path_variable(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        runtime.Namespace.remove(sym.symbol("a.test-path-variable"))

        code = """
        (ns a.test-path-variable
          (:require
           [basilisp.test :refer [deftest is]]))
        (deftest passing-test
          (is true))
        (deftest failing-test
          (is false))
        """
        pytester.makefile(".lpy", **{"./test/a/test_path_variable": code})
        pytester.syspathinsert()
        monkeypatch.syspath_prepend(pytester.path / "test")
        monkeypatch.setenv(
            "BASILISP_TEST_PATH", os.pathsep.join([str(pytester.path), str(tmp_path)])
        )
        result: pytest.RunResult = pytester.runpytest("test")
        result.assert_outcomes(passed=1, failed=1)

    def test_test_path_variable_excludes_test(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        runtime.Namespace.remove(sym.symbol("b.test-path-variable"))

        code = """
        (ns b.test-path-variable
          (:require
           [basilisp.test :refer [deftest is]]))
        (deftest passing-test
          (is true))
        (deftest failing-test
          (is false))
        """
        pytester.makefile(".lpy", **{"./test/b/test_path_variable": code})
        pytester.syspathinsert()
        monkeypatch.syspath_prepend(pytester.path / "test")
        monkeypatch.setenv("BASILISP_TEST_PATH", str(tmp_path))
        result: pytest.RunResult = pytester.runpytest("test")
        result.assert_outcomes()

    def test_file_pattern(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        runtime.Namespace.remove(sym.symbol("c.test-file-pattern-variable"))

        code = """
        (ns c.test-file-pattern-variable
          (:require
           [basilisp.test :refer [deftest is]]))
        (deftest passing-test
          (is true))
        (deftest failing-test
          (is false))
        """
        pytester.makefile(".lpy", **{"./test/c/test_file_pattern_variable": code})
        pytester.syspathinsert()
        monkeypatch.syspath_prepend(pytester.path / "test")
        monkeypatch.setenv(
            "BASILISP_TEST_FILE_PATTERN", r"(check_[^.]*|.*_check)\.(lpy|cljc)"
        )
        result: pytest.RunResult = pytester.runpytest("test")
        result.assert_outcomes()


def test_ns_in_syspath(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch):
    runtime.Namespace.remove(sym.symbol("a.test-path"))

    code = """
    (ns a.test-path
      (:require
       [basilisp.test :refer [deftest is]]))
    (deftest passing-test
      (is true))
    (deftest failing-test
      (is false))
    """
    pytester.makefile(".lpy", **{"./test/a/test_path": code})
    pytester.syspathinsert()
    # ensure `a` namespace is in sys.path
    monkeypatch.syspath_prepend(pytester.path / "test")
    result: pytest.RunResult = pytester.runpytest("test")
    result.assert_outcomes(passed=1, failed=1)


def test_ns_in_syspath_w_src(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
):
    runtime.Namespace.remove(sym.symbol("a.src"))
    runtime.Namespace.remove(sym.symbol("a.test-path"))

    code_src = """
    (ns a.src)
    (def abc 5)
    """

    code = """
    (ns a.test-path
      (:require
       a.src
       [basilisp.test :refer [deftest is]]))
    (deftest a-test (is (= a.src/abc 5)))
    (deftest passing-test
      (is true))
    (deftest failing-test
      (is false))
    """
    # a slightly more complicated setup where packages under namespace
    # `a` are both in src and test.
    pytester.makefile(".lpy", **{"./test/a/test_path": code, "./src/a/src": code_src})
    pytester.syspathinsert()
    # ensure src and test is in sys.path
    monkeypatch.syspath_prepend(pytester.path / "test")
    monkeypatch.syspath_prepend(pytester.path / "src")
    result: pytest.RunResult = pytester.runpytest("test")
    result.assert_outcomes(passed=2, failed=1)


def test_ns_not_in_syspath(pytester: pytest.Pytester):
    runtime.Namespace.remove(sym.symbol("a.test-path"))

    code = """
    (ns a.test-path
      (:require
       [basilisp.test :refer [deftest is]]))
    """
    # In this test, we use a `testabc` directory instead of `test`, as
    # the latter can cause issues on macOS.  Specifically, macOS has a
    # `/Library/Frameworks/Python.framework/Versions/3.xx/lib/python3.13/test`
    # directory is picked up, resulting in a slightly different error
    # message.
    pytester.makefile(".lpy", **{"./testabc/a/test_path": code})
    pytester.syspathinsert()
    result: pytest.RunResult = pytester.runpytest("testabc")
    assert result.ret != 0
    result.stdout.fnmatch_lines(
        ["*ModuleNotFoundError: Module named 'a.test-path' is not in sys.path"]
    )


def test_ns_with_underscore(pytester: pytest.Pytester):
    runtime.Namespace.remove(sym.symbol("test_underscore"))

    code = """
    (ns test_underscore
      (:require
       [basilisp.test :refer [deftest is]]))
    (deftest passing-test
      (is true))
    (deftest failing-test
      (is false))
    """
    pytester.makefile(".lpy", test_underscore=code)
    pytester.syspathinsert()
    result: pytest.RunResult = pytester.runpytest()
    result.assert_outcomes(passed=1, failed=1)


def test_no_ns(pytester: pytest.Pytester):
    runtime.Namespace.remove(sym.symbol("abc"))

    code = """
    (in-ns 'abc)
    (require '[basilisp.test :refer [deftest is]]))
    (deftest passing-test
      (is true))
    (deftest failing-test
      (is false))
    """
    pytester.makefile(".lpy", test_under=code)
    pytester.syspathinsert()
    result: pytest.RunResult = pytester.runpytest()
    assert result.ret != 0
    result.stdout.fnmatch_lines(
        [
            "*basilisp.lang.compiler.exception.CompilerException: unable to resolve symbol 'require'*"
        ]
    )
