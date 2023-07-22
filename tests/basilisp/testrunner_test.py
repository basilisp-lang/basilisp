import platform
import sys

import pytest
from _pytest.pytester import Pytester, RunResult

from basilisp.lang import runtime
from basilisp.lang import symbol as sym


class TestTestrunner:
    @pytest.fixture
    def run_result(self, pytester: Pytester) -> RunResult:
        code = """
        (ns test-testrunner
          (:require
           [basilisp.test :refer [deftest is are testing]]))

        (deftest assertion-test
          (testing "is assertions"
            (is true)
            (is false)
            (is (= "string" "string"))
            (is (thrown? basilisp.lang.exception/ExceptionInfo (throw (ex-info "Exception" {}))))
            (is (thrown? basilisp.lang.exception/ExceptionInfo (throw (python/Exception))))
            (is (throw (ex-info "Uncaught exception" {}))))

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
        """
        pytester.makefile(".lpy", test_testrunner=code)
        yield pytester.runpytest()
        runtime.Namespace.remove(sym.symbol("test-testrunner"))

    def test_outcomes(self, run_result: RunResult):
        run_result.assert_outcomes(passed=1, failed=2)

    def test_failure_repr(self, run_result: RunResult):
        run_result.stdout.fnmatch_lines(
            [
                "FAIL in (assertion-test) (test_testrunner.lpy:8)",
                "     is assertions :: Test failure: false",
                "",
                "    expected: false",
                "      actual: false",
            ],
            consecutive=True,
        )

        run_result.stdout.fnmatch_lines(
            [
                "FAIL in (assertion-test) (test_testrunner.lpy:11)",
                "     is assertions :: Expected <class 'basilisp.lang.exception.ExceptionInfo'>; got <class 'Exception'> instead",
                "",
                "    expected: <class 'basilisp.lang.exception.ExceptionInfo'>",
                "      actual: Exception()",
            ]
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

    @pytest.mark.xfail(
        platform.python_implementation() == "PyPy" and sys.version_info < (3, 9),
        reason=(
            "PyPy 3.8 seems to fail this test, but 3.9 doesn't so it doesn't bear "
            "further investigation."
        ),
    )
    @pytest.mark.xfail(
        sys.version_info < (3, 8),
        reason=(
            "This issue seems to stem from this fact that traceback line numbers for "
            "Python 3.8+ point to the beginning of the subexpression, whereas before "
            "they pointed to the end. See https://bugs.python.org/issue12458"
        ),
    )
    def test_error_repr(self, run_result: RunResult):
        if (sys.version_info < (3,11)):
            expected = [
                "ERROR in (assertion-test) (test_testrunner.lpy:12)",
                "",
                "Traceback (most recent call last):",
                '  File "/*/test_testrunner.lpy", line 12, in assertion_test',
                '    (is (throw (ex-info "Uncaught exception" {}))))',
                "basilisp.lang.exception.ExceptionInfo: Uncaught exception {}",
            ]
        else:
            expected = [
                "ERROR in (assertion-test) (test_testrunner.lpy:12)",
                "",
                "Traceback (most recent call last):",
                '  File "*test_testrunner.lpy", line 12, in assertion_test',
                '    (is (throw (ex-info "Uncaught exception" {}))))',
                '    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',
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
                '  File "/*/test_testrunner.lpy", line 25, in error_test',
                "    (throw",
                "basilisp.lang.exception.ExceptionInfo: This test will count as an error. {}",
            ]
        )


def test_fixtures(pytester: Pytester):
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
    result: pytester.RunResult = pytester.runpytest()
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
        ("error-during-teardown", ":once", 3, 0, 0),
        ("error-during-teardown", ":each", 2, 1, 1),
    ],
)
def test_fixtures_with_errors(
    pytester: Pytester,
    fixture: str,
    style: str,
    errors: int,
    passes: int,
    failures: int,
):
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
    result: pytester.RunResult = pytester.runpytest()
    result.assert_outcomes(passed=passes, failed=failures, errors=errors)
