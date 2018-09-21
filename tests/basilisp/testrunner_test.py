import _pytest.pytester as pytester
import pytest

import basilisp.lang.symbol as sym
import basilisp.main as basilisp
from basilisp.lang.runtime import Namespace


@pytest.fixture
def core_ns() -> Namespace:
    basilisp.init()
    return Namespace.get(sym.symbol('basilisp.core'))


def test_testrunner(core_ns: Namespace, testdir: pytester.Testdir):
    code = """
    (ns fixture
      (:require
       [basilisp.test :refer [deftest is]]))

    (deftest fixture-test
      (is (= 1 1))
      (is (= :hi :hi))
      (is true)
      (is (= "true" false)))
    """
    testdir.makefile('.lpy', fixture=code)

    result: pytester.RunResult = testdir.runpytest_inprocess()
    result.assert_outcomes(failed=1)
