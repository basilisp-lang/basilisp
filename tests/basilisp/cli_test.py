import io
import os
import pathlib
import platform
import re
import stat
import subprocess
import tempfile
import time
from threading import Thread
from typing import Optional, Sequence
from unittest.mock import patch

import attr
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


@attr.frozen
class CapturedIO:
    out: str
    err: str
    lisp_out: str
    lisp_err: str


@pytest.fixture
def run_cli(monkeypatch, capsys, cap_lisp_io):
    def _run_cli(args: Sequence[str], input: Optional[str] = None):
        if input is not None:
            monkeypatch.setattr(
                "sys.stdin", io.TextIOWrapper(io.BytesIO(input.encode("utf-8")))
            )
        invoke_cli([*args])
        python_io = capsys.readouterr()
        lisp_out, lisp_err = cap_lisp_io
        return CapturedIO(
            out=python_io.out,
            err=python_io.err,
            lisp_out=lisp_out.getvalue(),
            lisp_err=lisp_err.getvalue(),
        )

    return _run_cli


def test_debug_flag(run_cli):
    result = run_cli(["run", "--disable-ns-cache", "true", "-c", "(println (+ 1 2))"])
    assert "3\n" == result.lisp_out
    assert os.environ["BASILISP_DO_NOT_CACHE_NAMESPACES"].lower() == "true"


class TestCompilerFlags:
    def test_no_flag(self, run_cli):
        result = run_cli(
            ["run", "--warn-on-var-indirection", "-c", "(println (+ 1 2))"]
        )
        assert "3\n" == result.lisp_out

    @pytest.mark.parametrize("val", BOOL_TRUE | BOOL_FALSE)
    def test_valid_flag(self, run_cli, val):
        result = run_cli(
            ["run", "--warn-on-var-indirection", val, "-c", "(println  (+ 1 2))"]
        )
        assert "3\n" == result.lisp_out

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

            # give the server some time to settle down.
            #
            # the high retries number is to address the slowness when
            # running on pypy.
            retries = 60
            while not os.path.exists(tmpfilepath):
                time.sleep(1)
                retries -= 1
                assert 0 <= retries

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
    cli_args_params = [
        ([], "0\n"),
        (["--"], "0\n"),
        (["1", "2", "3"], "6\n"),
        (["--", "1", "2", "3"], "6\n"),
    ]
    cli_args_code = "(println (apply + (map int *command-line-args*)))"

    def test_run_code(self, run_cli):
        result = run_cli(["run", "-c", "(println (+ 1 2))"])
        assert "3\n" == result.lisp_out

    @pytest.mark.parametrize("args,ret", cli_args_params)
    def test_run_code_with_args(self, run_cli, args: list[str], ret: str):
        result = run_cli(["run", "-c", self.cli_args_code, *args])
        assert ret == result.lisp_out

    def test_run_file(self, tmp_path: pathlib.Path, run_cli):
        script_path = tmp_path / "test.lpy"
        script_path.write_text("(println (+ 1 2))")
        result = run_cli(["run", str(script_path)])
        assert "3\n" == result.lisp_out

    @pytest.mark.parametrize("args,ret", cli_args_params)
    def test_run_file_with_args(
        self, tmp_path: pathlib.Path, run_cli, args: list[str], ret: str
    ):
        script_path = tmp_path / "test.lpy"
        script_path.write_text(self.cli_args_code)
        result = run_cli(["run", str(script_path), *args])
        assert ret == result.lisp_out

    def test_run_stdin(self, run_cli):
        result = run_cli(["run", "-"], input="(println (+ 1 2))")
        assert "3\n" == result.lisp_out

    @pytest.mark.parametrize("args,ret", cli_args_params)
    def test_run_stdin_with_args(self, run_cli, args: list[str], ret: str):
        result = run_cli(
            ["run", "-", *args],
            input=self.cli_args_code,
        )
        assert ret == result.lisp_out


def test_version(run_cli):
    result = run_cli(["version"])
    assert re.compile(r"^Basilisp (\d+)\.(\d+)\.(\w*)(\d+)\n$").match(result.out)


@pytest.mark.skipif(
    platform.system().lower() == "windows",
    reason=(
        "Shebangs are only supported virtually by Windows Python installations, "
        "so this doesn't work natively on Windows"
    ),
)
@pytest.mark.parametrize(
    "args,ret",
    [
        ([], b"nil\n"),
        (["1", "hi", "yes"], b"[1 hi yes]\n"),
    ],
)
def test_run_script(tmp_path: pathlib.Path, args: list[str], ret: bytes):
    script_path = tmp_path / "script.lpy"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env basilisp-run",
                "(ns test-run-script-ns ",
                "  (:import os))",
                "",
                "(println *command-line-args*)",
            ]
        )
    )
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    res = subprocess.run(
        [script_path.resolve(), *args], check=True, capture_output=True
    )
    assert res.stdout == ret
