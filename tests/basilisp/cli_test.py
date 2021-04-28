import io
import os
import platform
import re
import tempfile
from typing import Optional, Sequence
from unittest.mock import patch

import pytest

from basilisp.cli import invoke_cli
from basilisp.prompt import Prompter

pytestmark = pytest.mark.skipif(
    platform.python_implementation() == "PyPy", reason="CLI tests fail on PyPy 3.6"
)


@pytest.fixture
def isolated_filesystem():
    with tempfile.TemporaryDirectory() as d:
        wd = os.getcwd()
        os.chdir(d)
        try:
            yield
        finally:
            os.chdir(wd)


@pytest.fixture
def run_cli(monkeypatch, capsys):
    def _run_cli(args: Sequence[str], input: Optional[str] = None):
        if input is not None:
            monkeypatch.setattr("sys.stdin", io.StringIO(input))
        invoke_cli([*args])
        return capsys.readouterr()

    return _run_cli


class TestREPL:
    @pytest.fixture(scope="class", autouse=True)
    def prompter(self):
        with patch("basilisp.cli.get_prompter", return_value=Prompter()):
            yield

    def test_no_input(self, run_cli):
        result = run_cli(["repl"], input="")
        assert "basilisp.user=> " == result.out

    def test_newline(self, run_cli):
        result = run_cli(["repl"], input="\n")
        assert "basilisp.user=> basilisp.user=> " == result.out

    def test_simple_expression(self, run_cli):
        result = run_cli(["repl"], input="(+ 1 2)")
        assert "basilisp.user=> 3\nbasilisp.user=> " == result.out

    def test_syntax_error(self, run_cli):
        result = run_cli(["repl"], input="(+ 1 2")
        assert (
            "basilisp.lang.reader.UnexpectedEOFError: Unexpected EOF in list "
            "(line: 1, col: 7)\nbasilisp.user=> " in result.out
        )

    def test_compiler_error(self, run_cli):
        result = run_cli(["repl"], input="(fn*)")
        assert (
            "basilisp.lang.compiler.exception.CompilerException: fn form "
            "must match: (fn* name? [arg*] body*) or (fn* name? method*)"
        ) in result.out
        assert result.out.endswith("\nbasilisp.user=> ")

    def test_other_exception(self, run_cli):
        result = run_cli(["repl"], input='(throw (python/Exception "CLI test"))')
        assert "Exception: CLI test\nbasilisp.user=> " in result.out


class TestRun:
    def test_run_code(self, run_cli):
        result = run_cli(["run", "-c", "(+ 1 2)"])
        assert "3\n" == result.out

    def test_run_file(self, isolated_filesystem, run_cli):
        with open("test.lpy", mode="w") as f:
            f.write("(+ 1 2)")
        result = run_cli(["run", "test.lpy"])
        assert "3\n" == result.out

    def test_run_stdin(self, run_cli):
        result = run_cli(["run", "-"], input="(+ 1 2)")
        assert "3\n" == result.out


def test_version(run_cli):
    result = invoke_cli(["version"])
    assert re.compile(r"^Basilisp (\d+)\.(\d+)\.(\w*)(\d+)\n$").match(result.out)
