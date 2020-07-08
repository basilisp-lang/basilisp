from _pytest import pytester as pytester


def test_testrunner(testdir: pytester.Testdir, capsys):
    code = """
    (ns test-fixture
      (:require
       [basilisp.test :refer [deftest is]]))

    (deftest fixture-test
      (is (= 1 1))
      (is (= :hi :hi))
      (is true)
      (is false)
      (is (= "true" false))
      (is (thrown? basilisp.lang.exception/ExceptionInfo (throw (ex-info "Exception" {}))))
      (is (thrown? basilisp.lang.exception/ExceptionInfo (throw (python/Exception))))
      (is (= 4.6 4.6))
      (is (throw (ex-info "Uncaught exception" {}))))
    """
    testdir.makefile(".lpy", test_fixture=code)

    result: pytester.RunResult = testdir.runpytest()
    result.assert_outcomes(failed=1)

    captured = capsys.readouterr()

    expected_out = """FAIL in (fixture-test) (test_fixture.lpy:9)
    Test failure: false

    expected: false
      actual: false"""
    assert expected_out in captured.out

    expected_out = """FAIL in (fixture-test) (test_fixture.lpy:10)
    Test failure: (= "true" false)

    expected: "true"
      actual: false"""
    assert expected_out in captured.out

    expected_out = """FAIL in (fixture-test) (test_fixture.lpy:12)
    Expected <class 'basilisp.lang.exception.ExceptionInfo'>; got <class 'Exception'> instead

    expected: <class 'basilisp.lang.exception.ExceptionInfo'>
      actual: Exception()"""
    assert expected_out in captured.out

    expected_out = """ERROR in (fixture-test) (test_fixture.lpy:14)"""
    assert expected_out in captured.out
