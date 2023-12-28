import io
import os
import re
import tempfile
import time
from threading import Thread
from typing import Optional, Sequence
from unittest.mock import patch

import pytest

from basilisp.cli import BOOL_FALSE, BOOL_TRUE, invoke_cli
from basilisp.prompt import Prompter


@pytest.fixture(autouse=True)
def env_vars():
    environ = set(os.environ.items())
    try:
        yield
    finally:
        os.environ.clear()
        for var, val in environ:
            os.environ[var] = val


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
            monkeypatch.setattr(
                "sys.stdin", io.TextIOWrapper(io.BytesIO(input.encode("utf-8")))
            )
        invoke_cli([*args])
        return capsys.readouterr()

    return _run_cli


def test_debug_flag(run_cli):
    result = run_cli(["run", "--disable-ns-cache", "true", "-c", "(+ 1 2)"])
    assert "3\n" == result.out
    assert os.environ["BASILISP_DO_NOT_CACHE_NAMESPACES"].lower() == "true"


class TestCompilerFlags:
    def test_no_flag(self, run_cli):
        result = run_cli(["run", "--warn-on-var-indirection", "-c", "(+ 1 2)"])
        assert "3\n" == result.out

    @pytest.mark.parametrize("val", BOOL_TRUE | BOOL_FALSE)
    def test_valid_flag(self, run_cli, val):
        result = run_cli(["run", "--warn-on-var-indirection", val, "-c", "(+ 1 2)"])
        assert "3\n" == result.out

    @pytest.mark.parametrize("val", ["maybe", "not-no", "4"])
    def test_invalid_flag(self, run_cli, val):
        with pytest.raises(SystemExit):
            run_cli(["run", "--warn-on-var-indirection", val, "-c", "(+ 1 2)"])


def run_nrepl(run_cli, tmpfilepath):
    try:
        run_cli(["nrepl-server", "--port-filepath", tmpfilepath])
    except Exception as e:
        print(f":run-nrepl-error {e}")


class TestnREPLServer:
    def test_run_nrepl(self, run_cli):
        with tempfile.TemporaryDirectory() as tmpdirpath:
            tmpfilepath = os.path.join(tmpdirpath, ".nrepl-port-test")
            thread = Thread(target=run_nrepl, args=[run_cli, tmpfilepath], daemon=True)
            thread.start()
            time.sleep(1)  # give server some time to settle down
            with open(tmpfilepath) as tf:
                port = int(tf.readline())
                assert port > 0 and port < 65536


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
        assert "basilisp.user=> basilisp.user=> " == result.out
        assert (
            "basilisp.lang.reader.UnexpectedEOFError: Unexpected EOF in list "
            "(line: 1, col: 7)" in result.err
        )

    def test_compiler_error(self, run_cli):
        result = run_cli(["repl"], input="(fn*)")
        assert "basilisp.user=> basilisp.user=> " == result.out
        assert (
            "basilisp.lang.compiler.exception.CompilerException: fn form "
            "must match: (fn* name? [arg*] body*) or (fn* name? method*)"
        ) in result.err

    def test_other_exception(self, run_cli):
        result = run_cli(["repl"], input='(throw (python/Exception "CLI test"))')
        assert "basilisp.user=> basilisp.user=> " == result.out
        assert "Exception: CLI test" in result.err


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
    result = run_cli(["version"])
    assert re.compile(r"^Basilisp (\d+)\.(\d+)\.(\w*)(\d+)\n$").match(result.out)
