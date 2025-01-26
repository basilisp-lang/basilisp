import importlib.util
import inspect
import os
import sys
import traceback
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import GeneratorType
from typing import Callable, Optional

import pytest

from basilisp import main as basilisp
from basilisp.lang import keyword as kw
from basilisp.lang import map as lmap
from basilisp.lang import runtime as runtime
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec
from basilisp.lang.obj import lrepr
from basilisp.lang.util import munge
from basilisp.util import Maybe

_EACH_FIXTURES_META_KW = kw.keyword("each-fixtures", "basilisp.test")
_ONCE_FIXTURES_NUM_META_KW = kw.keyword("once-fixtures", "basilisp.test")
_TEST_META_KW = kw.keyword("test", "basilisp.test")

CORE_NS = "basilisp.core"
CORE_NS_SYM = sym.symbol(CORE_NS)
OUT_VAR_NAME = "*out*"
OUT_VAR_SYM = sym.symbol(OUT_VAR_NAME, ns=CORE_NS)
ERR_VAR_NAME = "*err*"
ERR_VAR_SYM = sym.symbol(ERR_VAR_NAME, ns=CORE_NS)


def pytest_configure(config):

    # https://github.com/pytest-dev/pytest/issues/12876
    #
    # Basilisp's standard output streams may be initialized before
    # pytest captures sys streams (sys.stdout and sys.stderr) for
    # testing (e.g., with `basilisp test`). Writing to the original
    # handles during tests on Windows can cause invalid handle
    # errors. To prevent this, we rebind them to pytest's streams
    # during tests and restore them afterward.
    out_var = runtime.Var.find(OUT_VAR_SYM)
    err_var = runtime.Var.find(ERR_VAR_SYM)
    if bindings := {
        k: v for k, v in {out_var: sys.stdout, err_var: sys.stderr}.items() if k
    }:
        runtime.push_thread_bindings(lmap.map(bindings))
        config.basilisp_bindings = bindings

    basilisp.bootstrap("basilisp.test")


def pytest_unconfigure(config):
    if hasattr(config, "basilisp_bindings"):
        runtime.pop_thread_bindings()


def pytest_collect_file(file_path: Path, parent):
    """Primary PyTest hook to identify Basilisp test files."""
    if file_path.suffix == ".lpy":
        if file_path.name.startswith("test_") or file_path.stem.endswith("_test"):
            return BasilispFile.from_parent(parent, path=file_path)
    return None


class TestFailuresInfo(Exception):
    __slots__ = ("_msg", "_data")

    def __init__(self, message: str, data: lmap.PersistentMap) -> None:
        super().__init__()
        self._msg = message
        self._data = data

    def __repr__(self):
        return (
            "basilisp.contrib.pytest.testrunner.TestFailuresInfo"
            f"({self._msg}, {lrepr(self._data)})"
        )

    def __str__(self):
        return f"{self._msg} {lrepr(self._data)}"

    @property
    def data(self) -> lmap.PersistentMap:
        return self._data

    @property
    def message(self) -> str:
        return self._msg


TestFunction = Callable[[], lmap.PersistentMap]
FixtureTeardown = Iterator[None]
FixtureFunction = Callable[[], Optional[FixtureTeardown]]


class FixtureManager:
    """FixtureManager instances manage `basilisp.test` style fixtures on behalf of a
    BasilispFile or BasilispTestItem node."""

    __slots__ = ("_fixtures", "_teardowns")

    def __init__(self, fixtures: Iterable[FixtureFunction]):
        self._fixtures: Iterable[FixtureFunction] = fixtures
        self._teardowns: Iterable[FixtureTeardown] = ()

    @staticmethod
    def _run_fixture(fixture: FixtureFunction) -> Optional[Iterator[None]]:
        """Run a fixture function. If the fixture is a generator function, return the
        generator/coroutine. Otherwise, simply return the value from the function, if
        one."""
        if inspect.isgeneratorfunction(fixture):
            coro = fixture()
            assert isinstance(coro, GeneratorType)
            next(coro)
            return coro
        else:
            fixture()
            return None

    @classmethod
    def _setup_fixtures(
        cls, fixtures: Iterable[FixtureFunction]
    ) -> Iterable[FixtureTeardown]:
        """Set up fixtures by running them as by `_run_fixture`. Collect any fixtures
        teardown steps (e.g. suspended coroutines) and return those so they can be
        resumed later for any cleanup."""
        teardown_fixtures = []
        try:
            for fixture in fixtures:
                teardown = cls._run_fixture(fixture)
                if teardown is not None:
                    teardown_fixtures.append(teardown)
        except Exception as e:
            raise runtime.RuntimeException(
                "Exception occurred during fixture setup"
            ) from e
        else:
            return teardown_fixtures

    @staticmethod
    def _teardown_fixtures(teardowns: Iterable[FixtureTeardown]) -> None:
        """Perform teardown steps returned from a fixture function."""
        for teardown in teardowns:
            try:
                next(teardown)
            except StopIteration:
                pass
            except Exception as e:
                raise runtime.RuntimeException(
                    "Exception occurred during fixture teardown"
                ) from e

    def setup(self) -> None:
        """Setup fixtures and store any teardowns for cleanup later.

        Should have a corresponding call to `FixtureManager.teardown` to clean up
        fixtures."""
        self._teardowns = self._setup_fixtures(self._fixtures)

    def teardown(self) -> None:
        """Teardown fixtures from a previous call to `setup`."""
        self._teardown_fixtures(self._teardowns)
        self._teardowns = ()


def _is_package(path: Path) -> bool:
    """Return `True` if the given path refers to a Python or Basilisp package."""
    _, _, files = next(os.walk(path))
    for file in files:
        if file in {"__init__.lpy", "__init__.py"} or file.endswith(".lpy"):
            return True
    return False


def _get_fully_qualified_module_names(file: Path) -> list[str]:
    """Return the fully qualified module name (from the import root) for a module given
    its location.

    This works by traversing up the filesystem looking for the top-most package. From
    there, we derive a Python module name referring to the given module path."""
    paths = []
    for pth in sys.path:
        root = Path(pth).resolve()
        if file.is_relative_to(root):
            elems = list(file.with_suffix("").relative_to(root).parts)

            if elems[-1] == "__init__":
                elems.pop()
            paths.append(".".join(elems))
    return paths


class BasilispFile(pytest.File):
    """Files represent a test module in Python or a test namespace in Basilisp."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._fixture_manager: Optional[FixtureManager] = None

    @staticmethod
    def _collected_fixtures(
        ns: runtime.Namespace,
    ) -> tuple[Iterable[FixtureFunction], Iterable[FixtureFunction]]:
        """Collect all of the declared fixtures of the namespace."""
        if ns.meta is not None:
            return (
                ns.meta.val_at(_ONCE_FIXTURES_NUM_META_KW) or (),
                ns.meta.val_at(_EACH_FIXTURES_META_KW) or (),
            )
        return (), ()

    @staticmethod
    def _collected_tests(ns: runtime.Namespace) -> Iterable[runtime.Var]:
        """Return the sorted sequence of collected tests from the Namespace `ns`.

        Tests defined by `deftest` are annotated with `:basilisp.test/test` metadata.
        Tests are sorted by their line number, which matches the default behavior of
        PyTest."""

        def _test_num(var: runtime.Var) -> int:
            assert var.meta is not None
            order = var.meta.val_at(_LINE_KW)
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

    def setup(self) -> None:
        assert self._fixture_manager is not None
        self._fixture_manager.setup()

    def teardown(self) -> None:
        assert self._fixture_manager is not None
        self._fixture_manager.teardown()

    def _import_module(self) -> runtime.BasilispModule:
        modnames = _get_fully_qualified_module_names(self.path)
        assert modnames, "Must have at least one module name"

        exc: Optional[ModuleNotFoundError] = None
        for modname in modnames:
            try:
                module = importlib.import_module(modname)
            except ModuleNotFoundError as e:
                exc = e
            else:
                assert isinstance(module, runtime.BasilispModule)
                return module

        assert exc is not None, "Must have an exception or module"
        raise exc

    def collect(self):
        """Collect all tests from the namespace (module) given.

        Basilisp's test runner imports the namespace which will (as a side effect)
        collect the test functions in a namespace (represented by `deftest` forms in
        Basilisp). BasilispFile.collect fetches those test functions and generates
        BasilispTestItems for PyTest to run the tests."""
        filename = self.path.name
        module = self._import_module()
        ns = module.__basilisp_namespace__

        # Ensure the test module was loaded because it was directly
        # relative to an entry in `sys.path`.
        if module.__name__ != munge(str(ns)):
            raise ModuleNotFoundError(f"Module named '{ns}' is not in sys.path")

        once_fixtures, each_fixtures = self._collected_fixtures(ns)
        self._fixture_manager = FixtureManager(once_fixtures)
        for test in self._collected_tests(ns):
            f: TestFunction = test.value
            yield BasilispTestItem.from_parent(
                self,
                name=test.name.name,
                run_test=f,
                namespace=ns,
                filename=filename,
                fixture_manager=FixtureManager(each_fixtures),
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

    `deftest` forms run each `is` assertion and collect all failures in an atom,
    reporting their results as a vector of failures when each test concludes.

    The BasilispTestItem collects all the failures and returns a report to PyTest to
    show to the end-user."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: str,
        parent: BasilispFile,
        run_test: TestFunction,
        namespace: runtime.Namespace,
        filename: str,
        fixture_manager: FixtureManager,
    ) -> None:
        super().__init__(name, parent)
        self._run_test = run_test
        self._namespace = namespace
        self._filename = filename
        self._fixture_manager = fixture_manager

    @classmethod
    def from_parent(  # pylint: disable=arguments-differ,too-many-arguments
        cls,
        parent: "BasilispFile",
        name: str,
        run_test: TestFunction,
        namespace: runtime.Namespace,
        filename: str,
        fixture_manager: FixtureManager,
    ):
        """Create a new BasilispTestItem from the parent Node."""
        # https://github.com/pytest-dev/pytest/pull/6680
        return super().from_parent(
            parent,
            name=name,
            run_test=run_test,
            namespace=namespace,
            filename=filename,
            fixture_manager=fixture_manager,
        )

    def setup(self) -> None:
        self._fixture_manager.setup()

    def teardown(self) -> None:
        self._fixture_manager.teardown()

    def runtest(self):
        """Run the tests associated with this test item.

        If any tests fail, raise an ExceptionInfo exception with the test failures.
        PyTest will invoke self.repr_failure to display the failures to the user."""
        results: lmap.PersistentMap = self._run_test()
        failures: Optional[vec.PersistentVector] = results.val_at(_FAILURES_KW)
        if runtime.to_seq(failures):
            raise TestFailuresInfo("Test failures", lmap.map(results))

    def repr_failure(self, excinfo, style=None):
        """Representation function called when self.runtest() raises an exception."""
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
                else:  # pragma: no cover
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
        line_msg = Maybe(line).map(lambda l: f":{l}").or_else_get("")
        section_msg = Maybe(test_section).map(lambda s: f" {s} :: ").or_else_get("")

        return "\n".join(
            [
                f"FAIL in ({self.name}) ({self._filename}{line_msg})",
                f"    {section_msg}{msg}",
                "",
                f"    expected: {lrepr(expected)}",
                f"      actual: {lrepr(actual)}",
            ]
        )
