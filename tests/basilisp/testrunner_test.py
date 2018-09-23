import _pytest.pytester as pytester


# TODO: fix namespace after importer namespace issue resolved
# (https://github.com/chrisrink10/basilisp/issues/206)
def test_testrunner(testdir: pytester.Testdir):
    code = """
    (ns test_fixture
      (:require
       [basilisp.test :refer [deftest is]]))

    (deftest fixture-test
      (is (= 1 1))
      (is (= :hi :hi))
      (is true)
      (is (= "true" false)))
    """
    testdir.makefile('.lpy', test_fixture=code)

    result: pytester.RunResult = testdir.runpytest()
    result.assert_outcomes(failed=1)
