import os
import platform
import re
import subprocess
import tempfile
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from basilisp.cli import cli
from basilisp.prompt import Prompter

pytestmark = pytest.mark.skipif(
    platform.python_implementation() == "PyPy", reason="CLI tests fail on PyPy 3.6"
)


class TestREPL:
    @pytest.fixture(scope="class", autouse=True)
    def prompter(self):
        with patch("basilisp.cli.get_prompter", return_value=Prompter()):
            yield

    def test_no_input(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["repl"], input="")
        assert "basilisp.user=> " == result.stdout

    def test_newline(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["repl"], input="\n")
        assert "basilisp.user=> basilisp.user=> " == result.stdout

    def test_simple_expression(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["repl"], input="(+ 1 2)")
        assert "basilisp.user=> 3\nbasilisp.user=> " == result.stdout

    def test_syntax_error(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["repl"], input="(+ 1 2")
        assert (
            "basilisp.lang.reader.UnexpectedEOFError: Unexpected EOF in list\nbasilisp.user=> "
            in result.stdout
        )

    def test_compiler_error(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["repl"], input="(fn*)")
        assert (
            "basilisp.lang.compiler.exception.CompilerException: fn form "
            "must match: (fn* name? [arg*] body*) or (fn* name? method*)\nbasilisp.user=> "
        ) in result.stdout

    def test_other_exception(self):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["repl"], input='(throw (python/Exception "CLI test"))'
        )
        assert "Exception: CLI test\nbasilisp.user=> " in result.stdout


class TestRun:
    def test_run_code(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "-c", "(+ 1 2)"])
        assert "3\n" == result.stdout

    def test_run_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("test.lpy", mode="w") as f:
                f.write("(+ 1 2)")
            result = runner.invoke(cli, ["run", "test.lpy"])
            assert "3\n" == result.stdout

    def test_run_stdin(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "-"], input="(+ 1 2)")
        assert "3\n" == result.stdout


class TestRunSubprocess:
    @pytest.fixture
    def run(self):
        def _run(*args, **kwargs):
            return subprocess.run(
                ["basilisp", "run", *args],
                check=True,
                encoding="utf-8",
                env={"BASILISP_DO_NOT_CACHE_NAMESPACES": "true"},
                stdout=subprocess.PIPE,
                **kwargs,
            ).stdout

        return _run

    @pytest.fixture
    def temp_file(self):
        _, filename = tempfile.mkstemp(".lpy", "test")
        try:
            yield filename
        finally:
            os.unlink(filename)

    def test_run_code(self, run):
        assert "3\n" == run("-c", "(+ 1 2)")

    def test_run_file(self, run, temp_file: str):
        with open(temp_file, mode="w") as f:
            f.write("(+ 1 2)")
        assert "3\n" == run(temp_file)

    def test_run_stdin(self, run):
        assert "3\n" == run("-", input="(+ 1 2)")


def test_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["version"])
    assert re.compile(r"^Basilisp (\d+)\.(\d+)\.(\w*)(\d+)\n$").match(result.stdout)
