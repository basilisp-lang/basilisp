import types

import pytest

import basilisp.compiler as compiler
import basilisp.lang.runtime as runtime
import basilisp.main as basilisp
import basilisp.reader as reader

basilisp.init()


def pytest_collect_file(parent, path):
    if path.ext == ".lpy" and path.basename.startswith("test_"):
        print(path.basename)
        return BasilispFile(path, parent)


def eval_file(filename: str, module: types.ModuleType):
    """Evaluate a file with the given name into a Python module AST node."""
    ctx = compiler.CompilerContext()
    last = None
    for form in reader.read_file(filename, resolver=runtime.resolve_alias):
        last = compiler.compile_and_exec_form(form, ctx, module, filename)
    return last


class BasilispFile(pytest.File):
    def collect(self):
        self.fspath.pyimport()
        # yield BasilispTestItem(name, self, spec)


class BasilispTestItem(pytest.Item):
    def __init__(self, name, parent, spec):
        super(BasilispTestItem, self).__init__(name, parent)
        self.spec = spec

    def runtest(self):
        for name, value in sorted(self.spec.items()):
            # some custom test execution (dumb example follows)
            if name != value:
                raise AssertionError(self, name, value)

    def repr_failure(self, excinfo):
        """ called when self.runtest() raises an exception. """
        if isinstance(excinfo.value, AssertionError):
            return "\n".join(
                [
                    "usecase execution failed",
                    "   spec failed: %r: %r" % excinfo.value.args[1:3],
                    "   no further details known at this point.",
                ]
            )

    def reportinfo(self):
        return self.fspath, 0, "usecase: %s" % self.name
