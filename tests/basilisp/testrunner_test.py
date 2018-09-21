import _pytest.pytester as pytester


def test_testrunner(testdir: pytester.Testdir):
    code = """
    (ns fixture.test
      (:require
       [basilisp.test :refer [deftest is]]))

    (deftest fixture
      (is (= 1 1))
      (is (= :hi :hi))
      (is true)
      (is (= "true" false)))
    """
    testdir.makefile('lpy', code)

    result: pytester.RunResult = testdir.runpytest()
    result.assert_outcomes(passed=0, failed=1)
