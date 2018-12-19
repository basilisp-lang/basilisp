import importlib
import traceback
from typing import Optional, Callable

import pytest

import basilisp.lang.keyword as kw
import basilisp.lang.map as lmap
import basilisp.lang.runtime as runtime
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
import basilisp.main as basilisp
from basilisp.lang.obj import lrepr
from basilisp.util import Maybe

basilisp.init()
importlib.import_module("basilisp.test")

_COLLECTED_TESTS_SYM = sym.symbol("collected-tests", ns="basilisp.test")
_CURRENT_NS_SYM = sym.symbol("current-ns", ns="basilisp.test")


def pytest_collect_file(parent, path):
    """Primary PyTest hook to identify Basilisp test files."""
    if path.ext == ".lpy":
        if path.basename.startswith("test_") or path.purebasename.endswith("_test"):
            return BasilispFile(path, parent)
    return None


def _collected_tests() -> Optional[vec.Vector]:
    """Fetch the collected tests for the namespace ns from
    basilisp.test/collected-tests atom. If no tests are found, return
    None."""
    var = Maybe(runtime.Var.find(_COLLECTED_TESTS_SYM)).or_else_raise(
        lambda: runtime.RuntimeException(
            f"Unable to find test Var {_COLLECTED_TESTS_SYM}."
        )
    )
    return var.value.deref()


def _current_ns() -> str:
    """Fetch the current namespace from basilisp.test/current-ns."""
    var = Maybe(runtime.Var.find(_CURRENT_NS_SYM)).or_else_raise(
        lambda: runtime.RuntimeException(f"Unable to find test Var {_CURRENT_NS_SYM}.")
    )
    ns = var.value.deref()
    return ns.name


def _reset_collected_tests() -> None:
    """Reset the collected tests."""
    var = Maybe(runtime.Var.find(_COLLECTED_TESTS_SYM)).or_else_raise(
        lambda: runtime.RuntimeException(
            f"Unable to find test Var {_COLLECTED_TESTS_SYM}."
        )
    )
    return var.value.reset(vec.Vector.empty())


class TestFailuresInfo(Exception):
    __slots__ = ("_msg", "_data")

    def __init__(self, message: str, data: lmap.Map) -> None:
        super().__init__()
        self._msg = message
        self._data = data

    def __repr__(self):
        return f"basilisp.testrunner.TestFailuresInfo({self._msg}, {lrepr(self._data)})"

    def __str__(self):
        return f"{self._msg} {lrepr(self._data)}"

    @property
    def data(self) -> lmap.Map:
        return self._data

    @property
    def message(self) -> str:
        return self._msg


TestFunction = Callable[[], Optional[vec.Vector]]


class BasilispFile(pytest.File):
    """Files represent a test module in Python or a test namespace in Basilisp."""

    def collect(self):
        """Collect all of the tests in the namespace (module) given.

        Basilisp's test runner imports the namespace which will (as a side
        effect) collect all of the test functions in a namespace (represented
        by `deftest` forms in Basilisp) into an atom in `basilisp.test`.
        BasilispFile.collect fetches those test functions and generates
        BasilispTestItems for PyTest to run the tests."""
        _reset_collected_tests()
        filename = self.fspath.basename
        self.fspath.pyimport()
        ns = _current_ns()
        tests = _collected_tests()
        for test in tests:
            f: TestFunction = test.value
            yield BasilispTestItem(test.name.name, self, f, ns, filename)


_ACTUAL_KW = kw.keyword("actual")
_ERROR_KW = kw.keyword("error")
_EXPECTED_KW = kw.keyword("expected")
_FAILURE_KW = kw.keyword("failure")
_FAILURES_KW = kw.keyword("failures")
_MESSAGE_KW = kw.keyword("message")
_LINE_KW = kw.keyword("line")
_EXPR_KW = kw.keyword("expr")
_TEST_SECTION_KW = kw.keyword("test-section")
_TYPE_KW = kw.keyword("type")


class BasilispTestItem(pytest.Item):
    """Test items correspond to a single `deftest` form in a Basilisp test.

    `deftest` forms run each `is` assertion and collect all failures in an
    atom, reporting their results as a vector of failures when each test
    concludes.

    The BasilispTestItem collects all the failures and returns a report
    to PyTest to show to the end-user."""

    def __init__(
        self,
        name: str,  # pylint: disable=too-many-arguments
        parent: BasilispFile,
        run_test: TestFunction,
        namespace: str,
        filename: str,
    ) -> None:
        super(BasilispTestItem, self).__init__(name, parent)
        self._run_test = run_test
        self._namespace = namespace
        self._filename = filename

    def runtest(self):
        """Run the tests associated with this test item.

        If any tests fail, raise an ExceptionInfo exception with the
        test failures. PyTest will invoke self.repr_failure to display
        the failures to the user."""
        results: lmap.Map = self._run_test()
        failures: Optional[vec.Vector] = results.entry(_FAILURES_KW)
        if runtime.to_seq(failures):
            raise TestFailuresInfo("Test failures", lmap.map(results))

    def repr_failure(self, excinfo):
        """Representation function called when self.runtest() raises an
        exception."""
        if isinstance(excinfo.value, TestFailuresInfo):
            exc = excinfo.value
            failures = exc.data.entry(_FAILURES_KW)
            messages = []

            for details in failures:
                type_ = details.entry(_TYPE_KW)
                if type_ == _FAILURE_KW:
                    messages.append(self._failure_msg(details))
                elif type_ == _ERROR_KW:
                    exc = details.entry(_ACTUAL_KW)
                    line = details.entry(_LINE_KW)
                    messages.append(self._error_msg(exc, line=line))
                else:
                    assert False, "Test failure type must be in #{:error :failure}"

            return "\n\n".join(messages)
        elif isinstance(excinfo.value, Exception):
            exc = excinfo.value
            return self._error_msg(exc)
        else:
            return None

    def reportinfo(self):
        return self.fspath, 0, self.name

    def _error_msg(self, exc: Exception, line: Optional[int] = None) -> str:
        line_msg = Maybe(line).map(lambda l: f":{l}").or_else_get("")
        messages = [f"ERROR in ({self.name}) ({self._filename}{line_msg})", "\n\n"]
        messages.extend(traceback.format_exception(Exception, exc, exc.__traceback__))
        return "".join(messages)

    def _failure_msg(self, details: lmap.Map) -> str:
        assert details.entry(_TYPE_KW) == _FAILURE_KW
        msg: str = details.entry(_MESSAGE_KW)

        actual = details.entry(_ACTUAL_KW)
        expected = details.entry(_EXPECTED_KW)

        test_section = details.entry(_TEST_SECTION_KW)
        line = details.entry(_LINE_KW)
        section_msg = Maybe(test_section).map(lambda s: f" {s} :: ").or_else_get("")

        return "\n".join(
            [
                f"FAIL in ({self.name}) ({self._filename}:{line})",
                f"    {section_msg}{msg}",
                "",
                f"    expected: {lrepr(expected)}",
                f"      actual: {lrepr(actual)}",
            ]
        )
