import importlib
import io
import os
import pathlib
import platform
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from threading import Thread
from typing import List, Optional
from collections.abc import Sequence
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


@pytest.fixture(autouse=True)
def sys_path() -> list[str]:
    sys_path = list(sys.path)
    try:
        yield sys_path
    finally:
        sys.path = sys_path
        importlib.invalidate_caches()


@pytest.fixture()
def temp_paths(tmp_path_factory) -> list[pathlib.Path]:
    paths = []
    for i in range(3):
        path = tmp_path_factory.mktemp(f"dir{i}")
        paths.append(path)
    return paths


@pytest.fixture
def temp_path_args(temp_paths: list[pathlib.Path]) -> list[str]:
    args = []
    for p in temp_paths:
        args.extend(("-p", str(p)))
    return args


@pytest.fixture
def isolated_filesystem():
    with tempfile.TemporaryDirectory() as d:
        wd = os.getcwd()
        os.chdir(d)
        try:
            yield d
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


class TestBootstrap:
    def test_install(self, tmp_path: pathlib.Path, run_cli):
        res = run_cli(["bootstrap", "--site-packages", str(tmp_path)])

        bootstrap_file = tmp_path / "basilispbootstrap.pth"
        assert bootstrap_file.exists()
        assert bootstrap_file.read_text() == "import basilisp.sitecustomize"

        assert res.out == (
            "Your Python installation has been bootstrapped! You can undo this at any "
            "time with with `basilisp bootstrap --uninstall`.\n"
        )

        res = run_cli(["bootstrap", "--uninstall", "--site-packages", str(tmp_path)])

        assert not bootstrap_file.exists()

        assert res.out == f"Removed '{bootstrap_file}'\n"

    def test_install_quiet(self, tmp_path: pathlib.Path, run_cli, capsys):
        run_cli(["bootstrap", "-q", "--site-packages", str(tmp_path)])

        bootstrap_file = tmp_path / "basilispbootstrap.pth"
        assert bootstrap_file.exists()
        assert bootstrap_file.read_text() == "import basilisp.sitecustomize"

        res = capsys.readouterr()
        assert res.out == ""

        run_cli(["bootstrap", "-q", "--uninstall", "--site-packages", str(tmp_path)])

        assert not bootstrap_file.exists()

        res = capsys.readouterr()
        assert res.out == ""

    def test_nothing_to_uninstall(self, tmp_path: pathlib.Path, run_cli, capsys):
        bootstrap_file = tmp_path / "basilispbootstrap.pth"
        assert not bootstrap_file.exists()

        res = run_cli(["bootstrap", "--uninstall", "--site-packages", str(tmp_path)])

        assert not bootstrap_file.exists()
        assert res.out == "No Basilisp bootstrap files were found.\n"


def test_debug_flag(run_cli):
    result = run_cli(["run", "--disable-ns-cache", "true", "-c", "(println (+ 1 2))"])
    assert f"3{os.linesep}" == result.lisp_out
    assert os.environ["BASILISP_DO_NOT_CACHE_NAMESPACES"].lower() == "true"


class TestCompilerFlags:
    def test_no_flag(self, run_cli):
        result = run_cli(
            ["run", "--warn-on-var-indirection", "-c", "(println (+ 1 2))"]
        )
        assert f"3{os.linesep}" == result.lisp_out

    @pytest.mark.parametrize("val", BOOL_TRUE | BOOL_FALSE)
    def test_valid_flag(self, run_cli, val):
        result = run_cli(
            ["run", "--warn-on-var-indirection", val, "-c", "(println  (+ 1 2))"]
        )
        assert f"3{os.linesep}" == result.lisp_out

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
            while not os.path.exists(tmpfilepath) or os.stat(tmpfilepath).st_size == 0:
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
            os.linesep.join(
                (
                    "",
                    "  exception: <class 'basilisp.lang.reader.UnexpectedEOFError'>",
                    "    message: Unexpected EOF in list",
                    "       line: 1:6",
                    "",
                )
            )
            in result.err
        )

    def test_compiler_error(self, run_cli):
        result = run_cli(["repl"], input="(fn*)")
        assert "basilisp.user=> basilisp.user=> " == result.out
        assert (
            os.linesep.join(
                (
                    "",
                    "  exception: <class 'IndexError'> from <class 'basilisp.lang.compiler.exception.CompilerException'>",
                    "      phase: :analyzing",
                    "    message: fn form must match: (fn* name? [arg*] body*) or (fn* name? method*): Index 1 out of bounds",
                    "       form: (fn*)",
                    "   location: <REPL Input>:1",
                    "",
                )
            )
            in result.err
        )

    def test_other_exception(self, run_cli):
        result = run_cli(["repl"], input='(throw (python/Exception "CLI test"))')
        assert "basilisp.user=> basilisp.user=> " == result.out
        assert "Exception: CLI test" in result.err

    class TestPathConfig:
        @pytest.mark.parametrize(
            "args",
            [
                (),
                ("--include-unsafe-path",),
                ("--include-unsafe-path=true",),
                ("--include-unsafe-path", "true"),
            ],
        )
        def test_repl_include_unsafe_path(self, run_cli, args):
            result = run_cli(
                ["repl", *args], input="(import sys) (prn (first sys/path))"
            )
            assert '""' == result.lisp_out.rstrip()
            assert "" == sys.path[0]

        @pytest.mark.parametrize(
            "args",
            [
                ("--include-unsafe-path=false",),
                ("--include-unsafe-path", "false"),
            ],
        )
        def test_repl_do_not_include_unsafe_path(self, run_cli, args, sys_path):
            result = run_cli(
                ["repl", *args], input="(import sys) (prn (first sys/path))"
            )
            assert '""\n' != result.lisp_out
            assert sys_path[0] == sys.path[0]

        def test_repl_include_extra_path(
            self, run_cli, temp_paths: list[pathlib.Path], temp_path_args: list[str]
        ):
            result = run_cli(
                ["repl", *temp_path_args],
                input=" ".join(
                    [
                        f"(import pathlib sys)",
                        "(doseq [path sys/path]",
                        "  (prn (.as-posix (pathlib/Path path))))",
                    ]
                ),
            )
            out_lines = set(result.lisp_out.splitlines())
            assert {'"."', *map(lambda p: f'"{p.as_posix()}"', temp_paths)}.issubset(
                out_lines
            )


cli_run_args_params = [
    ([], f"0{os.linesep}"),
    (["--"], f"0{os.linesep}"),
    (["1", "2", "3"], f"6{os.linesep}"),
    (["--", "1", "2", "3"], f"6{os.linesep}"),
]
cli_run_args_code = "(println (apply + (map int *command-line-args*)))"


class TestRun:
    def test_run_ns_and_code_mutually_exclusive(self, run_cli):
        with pytest.raises(SystemExit):
            run_cli(["run", "-c", "-n"])

    class TestRunCode:
        def test_run_code(self, run_cli):
            result = run_cli(["run", "-c", "(println (+ 1 2))"])
            assert f"3{os.linesep}" == result.lisp_out

        def test_run_code_main_ns(self, run_cli):
            result = run_cli(["run", "-c", "(println *main-ns*)"])
            assert f"nil{os.linesep}" == result.lisp_out

        @pytest.mark.parametrize("args,ret", cli_run_args_params)
        def test_run_code_with_args(self, run_cli, args: list[str], ret: str):
            result = run_cli(["run", "-c", cli_run_args_code, *args])
            assert ret == result.lisp_out

        @pytest.mark.parametrize(
            "args",
            [
                (),
                ("--include-unsafe-path",),
                ("--include-unsafe-path=true",),
                ("--include-unsafe-path", "true"),
            ],
        )
        def test_run_code_include_unsafe_path(self, run_cli, args):
            result = run_cli(
                ["run", *args, "-c", "(import sys) (prn (first sys/path))"]
            )
            assert '""' == result.lisp_out.rstrip()
            assert "" == sys.path[0]

        @pytest.mark.parametrize(
            "args",
            [
                ("--include-unsafe-path=false",),
                ("--include-unsafe-path", "false"),
            ],
        )
        def test_run_code_do_not_include_unsafe_path(self, run_cli, args, sys_path):
            result = run_cli(
                ["run", *args, "-c", "(import sys) (prn (first sys/path))"]
            )
            assert '""\n' != result.lisp_out
            assert sys_path[0] == sys.path[0]

        def test_run_code_include_extra_path(
            self, run_cli, temp_paths: list[pathlib.Path], temp_path_args: list[str]
        ):
            result = run_cli(
                [
                    "run",
                    *temp_path_args,
                    "-c",
                    os.linesep.join(
                        [
                            f"(import pathlib sys)",
                            "(doseq [path sys/path]",
                            "  (prn (.as-posix (pathlib/Path path))))",
                        ]
                    ),
                ]
            )
            out_lines = set(result.lisp_out.splitlines())
            assert {'"."', *map(lambda p: f'"{p.as_posix()}"', temp_paths)}.issubset(
                out_lines
            )

    class TestRunFile:
        def test_run_file_rel(self, isolated_filesystem, run_cli):
            with open("test.lpy", mode="w") as f:
                f.write("(println (+ 1 2))")
            result = run_cli(["run", "test.lpy"])
            assert f"3{os.linesep}" == result.lisp_out

        def test_run_file_abs(self, isolated_filesystem, run_cli):
            with open("test.lpy", mode="w") as f:
                f.write("(println (+ 1 3))")
            full_path = os.path.abspath("test.lpy")
            result = run_cli(["run", full_path])
            assert f"4{os.linesep}" == result.lisp_out

        def test_run_file_not_found(self, isolated_filesystem, run_cli):
            with pytest.raises(FileNotFoundError):
                run_cli(["run", "xyz.lpy"])

        def test_run_file_main_ns(self, isolated_filesystem, run_cli):
            with open("test.lpy", mode="w") as f:
                f.write("(println *main-ns*)")
            result = run_cli(["run", "test.lpy"])
            assert f"nil{os.linesep}" == result.lisp_out

        @pytest.mark.parametrize("args,ret", cli_run_args_params)
        def test_run_file_with_args(
            self, isolated_filesystem, run_cli, args: list[str], ret: str
        ):
            with open("test.lpy", mode="w") as f:
                f.write(cli_run_args_code)
            result = run_cli(["run", "test.lpy", *args])
            assert ret == result.lisp_out

        @pytest.mark.parametrize(
            "args",
            [
                (),
                ("--include-unsafe-path=true",),
                ("--include-unsafe-path", "true"),
            ],
        )
        def test_run_file_include_unsafe_path(self, isolated_filesystem, run_cli, args):
            with open("test.lpy", mode="w") as f:
                f.write(
                    os.linesep.join(
                        [
                            "(import pathlib sys)",
                            "(prn (.as-posix (pathlib/Path (first sys/path))))",
                        ]
                    )
                )
            result = run_cli(["run", *args, "test.lpy"])
            resolved_path = pathlib.Path(isolated_filesystem).resolve()
            assert f'"{resolved_path.as_posix()}"' == result.lisp_out.rstrip()
            assert str(resolved_path) == sys.path[0]

        @pytest.mark.parametrize(
            "args",
            [
                ("--include-unsafe-path=false",),
                ("--include-unsafe-path", "false"),
            ],
        )
        def test_run_file_do_not_include_unsafe_path(
            self, isolated_filesystem, run_cli, args, sys_path
        ):
            with open("test.lpy", mode="w") as f:
                f.write(
                    os.linesep.join(
                        [
                            "(import pathlib sys)",
                            "(prn (.as-posix (pathlib/Path (first sys/path))))",
                        ]
                    )
                )
            result = run_cli(["run", *args, "test.lpy"])
            resolved_path = pathlib.Path(isolated_filesystem).resolve().as_posix()
            assert f'"{resolved_path}"' != result.lisp_out.rstrip()
            assert sys_path[0] == sys.path[0]

        def test_run_file_include_extra_path(
            self,
            isolated_filesystem,
            run_cli,
            temp_paths: list[pathlib.Path],
            temp_path_args: list[str],
        ):
            with open("test.lpy", mode="w") as f:
                f.write(
                    os.linesep.join(
                        [
                            "(import pathlib sys)",
                            "(doseq [path sys/path]",
                            "  (prn (.as-posix (pathlib/Path path))))",
                        ]
                    )
                )
            result = run_cli(["run", *temp_path_args, "test.lpy"])
            resolved_path = pathlib.Path(isolated_filesystem).resolve().as_posix()
            out_lines = set(result.lisp_out.splitlines())
            assert {
                f'"{resolved_path}"',
                *map(lambda p: f'"{p.as_posix()}"', temp_paths),
            }.issubset(out_lines)

    class TestRunNamespace:
        @pytest.fixture
        def namespace_name(self) -> str:
            return f"package.core{secrets.token_hex(4)}"

        @pytest.fixture
        def namespace_file(
            self, monkeypatch, tmp_path: pathlib.Path, namespace_name: str
        ) -> pathlib.Path:
            parent_ns, child_ns = namespace_name.split(".", maxsplit=1)
            parent = tmp_path / parent_ns
            parent.mkdir()
            nsfile = parent / f"{child_ns}.lpy"
            nsfile.touch()
            monkeypatch.syspath_prepend(str(tmp_path))
            yield nsfile

        def test_cannot_run_namespace_with_in_ns_arg(
            self, run_cli, namespace_name: str, namespace_file: pathlib.Path
        ):
            namespace_file.write_text("(println (+ 1 2))")
            with pytest.raises(SystemExit):
                run_cli(["run", "--in-ns", "otherpackage.core", "-n", namespace_name])

        def test_run_namespace(
            self, run_cli, namespace_name: str, namespace_file: pathlib.Path
        ):
            namespace_file.write_text(f"(ns {namespace_name}) (println (+ 1 2))")
            result = run_cli(["run", "-n", namespace_name])
            assert f"3{os.linesep}" == result.lisp_out

        def test_run_namespace_main_ns(
            self, run_cli, namespace_name: str, namespace_file: pathlib.Path
        ):
            namespace_file.write_text(
                f"(ns {namespace_name}) (println (name *ns*)) (println *main-ns*)"
            )
            result = run_cli(["run", "-n", namespace_name])
            assert (
                f"{namespace_name}{os.linesep}{namespace_name}{os.linesep}"
                == result.lisp_out
            )

        @pytest.mark.parametrize("args,ret", cli_run_args_params)
        def test_run_namespace_with_args(
            self,
            run_cli,
            namespace_name: str,
            namespace_file: pathlib.Path,
            args: list[str],
            ret: str,
        ):
            namespace_file.write_text(f"(ns {namespace_name}) {cli_run_args_code}")
            result = run_cli(["run", "-n", namespace_name, *args])
            assert ret == result.lisp_out

        def test_run_namespace_with_subnamespace(
            self, run_cli, monkeypatch, tmp_path: pathlib.Path
        ):
            ns_name = f"parent{secrets.token_hex(4)}"
            child = "child"

            parent_ns_dir = tmp_path / ns_name
            parent_ns_dir.mkdir()

            parent_ns_file = tmp_path / f"{ns_name}.lpy"
            parent_ns_file.touch()
            parent_ns_file.write_text(
                f'(ns {ns_name} (:require [{ns_name}.{child}])) (python/print "loading:" *ns*)'
            )

            child_ns_file = parent_ns_dir / f"{child}.lpy"
            child_ns_file.touch()
            child_ns_file.write_text(
                f'(ns {ns_name}.{child}) (python/print "loading:" *ns*)'
            )

            monkeypatch.syspath_prepend(str(tmp_path))

            result = run_cli(["run", "-n", ns_name])
            assert f"loading: {ns_name}.{child}\nloading: {ns_name}\n" == result.out

        @pytest.mark.parametrize(
            "args",
            [
                (),
                ("--include-unsafe-path",),
                ("--include-unsafe-path=true",),
                ("--include-unsafe-path", "true"),
            ],
        )
        def test_run_namespace_include_unsafe_path(
            self, run_cli, namespace_name: str, namespace_file: pathlib.Path, args
        ):
            namespace_file.write_text(
                f"(ns {namespace_name} (:import sys)) (prn (first sys/path))"
            )
            result = run_cli(["run", *args, "-n", namespace_name])
            assert '""' == result.lisp_out.rstrip()
            assert "" == sys.path[0]

        @pytest.mark.parametrize(
            "args",
            [
                ("--include-unsafe-path=false",),
                ("--include-unsafe-path", "false"),
            ],
        )
        def test_run_namespace_do_not_include_unsafe_path(
            self,
            run_cli,
            namespace_name: str,
            namespace_file: pathlib.Path,
            args,
            sys_path,
        ):
            namespace_file.write_text(
                f"(ns {namespace_name} (:import sys)) (prn (first sys/path))"
            )
            result = run_cli(["run", *args, "-n", namespace_name])
            assert '""\n' != result.lisp_out

        def test_run_namespace_include_extra_path(
            self,
            run_cli,
            namespace_name: str,
            namespace_file: pathlib.Path,
            temp_paths: list[pathlib.Path],
            temp_path_args: list[str],
        ):
            namespace_file.write_text(
                os.linesep.join(
                    [
                        f"(ns {namespace_name} (:import pathlib sys))",
                        "(doseq [path sys/path]",
                        "  (prn (.as-posix (pathlib/Path path))))",
                    ]
                )
            )
            result = run_cli(["run", *temp_path_args, "-n", namespace_name])
            out_lines = set(result.lisp_out.splitlines())
            assert {'"."', *map(lambda p: f'"{p.as_posix()}"', temp_paths)}.issubset(
                out_lines
            )

    class TestRunStdin:
        def test_run_stdin(self, run_cli):
            result = run_cli(["run", "-"], input="(println (+ 1 2))")
            assert f"3{os.linesep}" == result.lisp_out

        def test_run_stdin_main_ns(self, run_cli):
            result = run_cli(["run", "-"], input="(println *main-ns*)")
            assert f"nil{os.linesep}" == result.lisp_out

        @pytest.mark.parametrize("args,ret", cli_run_args_params)
        def test_run_stdin_with_args(self, run_cli, args: list[str], ret: str):
            result = run_cli(
                ["run", "-", *args],
                input=cli_run_args_code,
            )
            assert ret == result.lisp_out

        @pytest.mark.parametrize(
            "args",
            [
                (),
                ("--include-unsafe-path=true",),
                ("--include-unsafe-path", "true"),
            ],
        )
        def test_run_stdin_include_unsafe_path(self, run_cli, args):
            result = run_cli(
                ["run", *args, "-"], input="(import sys) (prn (first sys/path))"
            )
            assert '""' == result.lisp_out.rstrip()
            assert "" == sys.path[0]

        @pytest.mark.parametrize(
            "args",
            [
                ("--include-unsafe-path=false",),
                ("--include-unsafe-path", "false"),
            ],
        )
        def test_run_stdin_do_not_include_unsafe_path(self, run_cli, args, sys_path):
            result = run_cli(
                ["run", *args, "-"], input="(import sys) (prn (first sys/path))"
            )
            assert '""\n' != result.lisp_out
            assert sys_path[0] == sys.path[0]

        def test_run_stdin_include_extra_path(
            self,
            run_cli,
            temp_paths: list[pathlib.Path],
            temp_path_args: list[str],
        ):
            result = run_cli(
                ["run", *temp_path_args, "-"],
                input=os.linesep.join(
                    [
                        f"(import pathlib sys)",
                        "(doseq [path sys/path]",
                        "  (prn (.as-posix (pathlib/Path path))))",
                    ]
                ),
            )
            out_lines = set(result.lisp_out.splitlines())
            assert {
                '"."',
                *map(lambda p: f'"{p.as_posix()}"', temp_paths),
            }.issubset(out_lines)


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
        (["1", "hi", "yes"], b'["1" "hi" "yes"]\n'),
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
