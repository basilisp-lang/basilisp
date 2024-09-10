import platform
import sys

import pytest

from basilisp.lang import runtime
from basilisp.lang import symbol as sym


class TestTestrunner:
    @pytest.fixture
    def run_result(self, pytester: pytest.Pytester) -> pytest.RunResult:
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
            '  File "*test_testrunner.lpy", line 14, in __assertion_test_*',
            '    (is (throw (ex-info "Uncaught exception" {})))',
            "basilisp.lang.exception.ExceptionInfo: Uncaught exception {}",
        ]

        run_result.stdout.fnmatch_lines(
            expected,
            consecutive=True,
        )

        run_result.stdout.fnmatch_lines(
            [
                "ERROR in (error-test) (test_testrunner.lpy:34)",
                "",
                "Traceback (most recent call last):",
                '  File "*basilisp/test.lpy", line *, in execute__STAR__*',
                "    (try",
                '  File "*test_testrunner.lpy", line 35, in __error_test_*',
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
        ("error-during-teardown", ":once", 3, 0, 0),
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
