import importlib
import traceback
from typing import Callable, Iterable, Optional

import pytest

from basilisp import main as basilisp
from basilisp.lang import compiler as compiler
from basilisp.lang import keyword as kw
from basilisp.lang import map as lmap
from basilisp.lang import runtime as runtime
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec
from basilisp.lang.obj import lrepr
from basilisp.util import Maybe

_CURRENT_NS_SYM = sym.symbol("current-ns", ns="basilisp.test")
_TEST_META_KW = kw.keyword("test", "basilisp.test")
_TEST_NUM_META_KW = kw.keyword("order", "basilisp.test")


# pylint: disable=unused-argument
def pytest_configure(config):
    opts = compiler.compiler_opts()
    basilisp.init(opts)
    importlib.import_module("basilisp.test")


def pytest_collect_file(parent, path):
    """Primary PyTest hook to identify Basilisp test files."""
    if path.ext == ".lpy":
        if path.basename.startswith("test_") or path.purebasename.endswith("_test"):
            if hasattr(BasilispFile, "from_parent"):
                return BasilispFile.from_parent(parent, fspath=path)
            else:
                return BasilispFile(path, parent)
    return None


class TestFailuresInfo(Exception):
    __slots__ = ("_msg", "_data")

    def __init__(self, message: str, data: lmap.PersistentMap) -> None:
        super().__init__()
        self._msg = message
        self._data = data

    def __repr__(self):
        return f"basilisp.testrunner.TestFailuresInfo({self._msg}, {lrepr(self._data)})"

    def __str__(self):
        return f"{self._msg} {lrepr(self._data)}"

    @property
    def data(self) -> lmap.PersistentMap:
        return self._data

    @property
    def message(self) -> str:
        return self._msg


TestFunction = Callable[[], lmap.PersistentMap]


class BasilispFile(pytest.File):
    """Files represent a test module in Python or a test namespace in Basilisp."""

    @staticmethod
    def _collected_tests(ns: runtime.Namespace) -> Iterable[runtime.Var]:
        """Return the set of collected tests from the Namespace `ns`.

        Tests defined by `deftest` are annotated with `:basilisp.test/test` metadata
        and `:basilisp.test/order` is a monotonically increasing integer added by
        `deftest` at compile-time to run tests in the order they are defined (which
        matches the default behavior of PyTest)."""

        def _test_num(var: runtime.Var) -> int:
            assert var.meta is not None
            order = var.meta.val_at(_TEST_NUM_META_KW)
            assert isinstance(order, int)
            return order

        return sorted(
            (
                var
                for _, var in ns.interns.items()
                if var.meta is not None and var.meta.val_at(_TEST_META_KW)
            ),
            key=_test_num,
        )

    def collect(self):
        """Collect all of the tests in the namespace (module) given.

        Basilisp's test runner imports the namespace which will (as a side
        effect) collect all of the test functions in a namespace (represented
        by `deftest` forms in Basilisp). BasilispFile.collect fetches those
        test functions and generates BasilispTestItems for PyTest to run the
        tests."""
        filename = self.fspath.basename
        module = self.fspath.pyimport()
        assert isinstance(module, runtime.BasilispModule)
        ns = module.__basilisp_namespace__
        for test in self._collected_tests(ns):
            f: TestFunction = test.value
            yield BasilispTestItem.from_parent(
                self, name=test.name.name, run_test=f, namespace=ns, filename=filename
            )


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

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: str,
        parent: BasilispFile,
        run_test: TestFunction,
        namespace: runtime.Namespace,
        filename: str,
    ) -> None:
        super(BasilispTestItem, self).__init__(name, parent)
        self._run_test = run_test
        self._namespace = namespace
        self._filename = filename

    @classmethod
    def from_parent(  # type: ignore  # pylint: disable=arguments-differ,too-many-arguments
        cls,
        parent: "BasilispFile",
        name: str,
        run_test: TestFunction,
        namespace: runtime.Namespace,
        filename: str,
    ):
        """Create a new BasilispTestItem from the parent Node."""
        # https://github.com/pytest-dev/pytest/pull/6680
        return super().from_parent(
            parent, name=name, run_test=run_test, namespace=namespace, filename=filename
        )

    def runtest(self):
        """Run the tests associated with this test item.

        If any tests fail, raise an ExceptionInfo exception with the
        test failures. PyTest will invoke self.repr_failure to display
        the failures to the user."""
        results: lmap.PersistentMap = self._run_test()
        failures: Optional[vec.PersistentVector] = results.val_at(_FAILURES_KW)
        if runtime.to_seq(failures):
            raise TestFailuresInfo("Test failures", lmap.map(results))

    def repr_failure(self, excinfo, style=None):  # pylint: disable=unused-argument
        """Representation function called when self.runtest() raises an
        exception."""
        if isinstance(excinfo.value, TestFailuresInfo):
            exc = excinfo.value
            failures = exc.data.val_at(_FAILURES_KW)
            messages = []

            for details in failures:
                type_ = details.val_at(_TYPE_KW)
                if type_ == _FAILURE_KW:
                    messages.append(self._failure_msg(details))
                elif type_ == _ERROR_KW:
                    exc = details.val_at(_ACTUAL_KW)
                    line = details.val_at(_LINE_KW)
                    messages.append(self._error_msg(exc, line=line))
                else:
                    assert False, "Test failure type must be in #{:error :failure}"

            return "\n\n".join(messages)
        elif isinstance(excinfo.value, Exception):
            return self._error_msg(excinfo.value)
        else:
            return None

    def reportinfo(self):
        return self.fspath, 0, self.name

    def _error_msg(self, exc: Exception, line: Optional[int] = None) -> str:
        line_msg = Maybe(line).map(lambda l: f":{l}").or_else_get("")
        messages = [f"ERROR in ({self.name}) ({self._filename}{line_msg})", "\n\n"]
        messages.extend(traceback.format_exception(Exception, exc, exc.__traceback__))
        return "".join(messages)

    def _failure_msg(self, details: lmap.PersistentMap) -> str:
        assert details.val_at(_TYPE_KW) == _FAILURE_KW
        msg: str = details.val_at(_MESSAGE_KW)

        actual = details.val_at(_ACTUAL_KW)
        expected = details.val_at(_EXPECTED_KW)

        test_section = details.val_at(_TEST_SECTION_KW)
        line = details.val_at(_LINE_KW)
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
