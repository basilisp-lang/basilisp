import importlib
import os.path
import pathlib
import platform
import site
import subprocess
import sys
import tempfile
from multiprocessing import Process, get_start_method
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple
from unittest.mock import patch

import pytest

import basilisp.main
from basilisp import importer as importer
from basilisp.lang import runtime as runtime
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec
from basilisp.lang.util import demunge, munge
from basilisp.main import bootstrap as bootstrap_basilisp


def importer_counter():
    return sum(isinstance(o, importer.BasilispImporter) for o in sys.meta_path)


def test_hook_imports():
    with patch("sys.meta_path", new=[]):
        assert 0 == importer_counter()
        importer.hook_imports()
        assert 1 == importer_counter()
        importer.hook_imports()
        assert 1 == importer_counter()

    with patch("sys.meta_path", new=[importer.BasilispImporter()]):
        assert 1 == importer_counter()
        importer.hook_imports()
        assert 1 == importer_counter()


def test_demunged_import(pytester: pytest.Pytester):
    with TemporaryDirectory() as tmpdir:
        tmp_module = os.path.join(
            tmpdir, "long__AMP__namespace_name__PLUS__with___LT__punctuation__GT__.lpy"
        )
        with open(tmp_module, mode="w") as module:
            code = """
            (ns long&namespace-name+with-<punctuation>)
            """
            module.write(code)

        with runtime.remove_ns_bindings():
            with (
                patch("sys.path", new=[tmpdir]),
                patch("sys.meta_path", new=[importer.BasilispImporter()]),
            ):
                importlib.import_module(
                    "long__AMP__namespace_name__PLUS__with___LT__punctuation__GT__"
                )

            assert (
                runtime.Namespace.get(
                    sym.symbol("long&namespace-name+with-<punctuation>")
                )
                is not None
            )
            assert (
                runtime.Namespace.get(
                    sym.symbol(
                        "long__AMP__namespace_name__PLUS__with___LT__punctuation__GT__"
                    )
                )
                is None
            )


def _ns_and_module(filename: str) -> tuple[str, str]:
    basename = os.path.splitext(os.path.basename(filename))[0]
    return demunge(basename), basename


if get_start_method() != "fork":
    # If `multiprocessing` starts a process using a method other than "fork",
    # the `basilisp.core` namespace will not be loaded in the child process.
    _import_module = bootstrap_basilisp
else:
    _import_module = importlib.import_module


class TestImporter:
    @pytest.fixture
    def do_cache_namespaces(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(importer._NO_CACHE_ENVVAR, "false")

    @pytest.fixture
    def do_not_cache_namespaces(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(importer._NO_CACHE_ENVVAR, "true")

    @pytest.fixture
    def module_cache(self):
        return {name: module for name, module in sys.modules.items()}

    @pytest.fixture
    def module_dir(self, monkeypatch: pytest.MonkeyPatch, module_cache):
        with TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            monkeypatch.chdir(tmpdir)
            monkeypatch.syspath_prepend(tmpdir)
            monkeypatch.setattr("sys.modules", module_cache)
            yield tmpdir
            monkeypatch.chdir(cwd)

    @pytest.fixture
    def make_new_module(self, module_dir):
        """Fixture returning a function which creates a new module and then
        removes it after the test run."""
        filenames = set()

        def _make_new_module(
            *ns_path: str, ns_name: str = "", module_text: Optional[str] = None
        ) -> None:
            """Generate a new module. If ns_name is not the empty string, use that
            name as the name of a Basilisp namespace with a single Var named `val`
            containing the name of the namespace. Otherwise, if module_text is not
            None, emit that module text directly to the generated module.

            You may specify only either the ns_name or module_text.

            This method always cleans up the module and any cached modules that it
            creates."""
            assert ns_name == "" or module_text is None

            if ns_path[:-1]:
                os.makedirs(os.path.join(module_dir, *ns_path[:-1]), exist_ok=True)

            filename = os.path.join(module_dir, *ns_path)
            filenames.add(filename)

            with open(filename, mode="w") as mod:
                if ns_name != "" and module_text is None:
                    mod.write(f"(ns {ns_name}) (def val (name *ns*))")
                else:
                    mod.write(module_text)

        try:
            yield _make_new_module
        finally:
            for filename in filenames:
                os.unlink(filename)

                try:
                    os.unlink(importer._cache_from_source(filename))
                except FileNotFoundError:
                    pass

    @pytest.fixture
    def load_namespace(self):
        """Fixture returning a function which loads a namespace by name and
        then removes it after the test run."""
        namespaces = []

        def _load_namespace(ns_name: str):
            """Load the named Namespace and return it."""
            namespaces.append(ns_name)
            importlib.import_module(munge(ns_name))
            return runtime.Namespace.get(sym.symbol(ns_name))

        try:
            yield _load_namespace
        finally:
            for namespace in namespaces:
                runtime.Namespace.remove(sym.symbol(namespace))

    def test_import_module(self, make_new_module, load_namespace):
        make_new_module(
            "importer",
            "namespace",
            "not_cached.lpy",
            ns_name="importer.namespace.not-cached",
        )

        not_cached = load_namespace("importer.namespace.not-cached")

        assert (
            "importer.namespace.not-cached" == not_cached.find(sym.symbol("val")).value
        )

    def test_import_module_without_cache(
        self, do_not_cache_namespaces, make_new_module, load_namespace
    ):
        make_new_module(
            "importer",
            "namespace",
            "without_cache.lpy",
            ns_name="importer.namespace.without-cache",
        )

        not_cached = load_namespace("importer.namespace.without-cache")

        assert (
            "importer.namespace.without-cache"
            == not_cached.find(sym.symbol("val")).value
        )

    def test_reload_module(
        self, do_not_cache_namespaces, make_new_module, load_namespace
    ):
        ns_name = "importer.namespace.to-reload"

        make_new_module(
            "importer",
            "namespace",
            "to_reload.lpy",
            module_text=f"""
            (ns {ns_name})
            (defn status [] "not-reloaded")
            """,
        )

        ns = load_namespace(ns_name)

        assert "not-reloaded" == ns.module.status()
        assert "not-reloaded" == ns.find(sym.symbol("status")).value()

        make_new_module(
            "importer",
            "namespace",
            "to_reload.lpy",
            module_text=f"""
            (ns {ns_name})
            (defn status [] "reloaded")
            (defn other [] "new function")
            """,
        )

        ns.reload()

        assert "reloaded" == ns.module.status()
        assert "reloaded" == ns.find(sym.symbol("status")).value()

        assert "new function" == ns.module.other()
        assert "new function" == ns.find(sym.symbol("other")).value()

    def test_reload_all_modules(
        self, do_not_cache_namespaces, make_new_module, load_namespace
    ):
        importlib.reload(runtime.Namespace.get(sym.symbol("basilisp.core")).module)

        ns1_name = "importer.namespace.to-reload1"
        ns2_name = "importer.namespace.to-reload2"
        ns3_name = "importer.namespace.to-reload3"

        make_new_module(
            "importer",
            "namespace",
            "to_reload1.lpy",
            module_text=f"""
            (ns {ns1_name}
              (:require [{ns2_name}]))
            (defn statuses []
              (conj ({ns2_name}/statuses) "1 not-reloaded"))
            """,
        )
        make_new_module(
            "importer",
            "namespace",
            "to_reload2.lpy",
            module_text=f"""
            (ns {ns2_name}
              (:require [{ns3_name}]))
            (defn statuses []
              (conj ({ns3_name}/statuses) "2 not-reloaded"))
            """,
        )
        make_new_module(
            "importer",
            "namespace",
            "to_reload3.lpy",
            module_text=f"""
            (ns {ns3_name})
            (defn statuses [] ["3 not-reloaded"])
            """,
        )

        ns1 = load_namespace(ns1_name)

        assert (
            vec.v(
                "3 not-reloaded",
                "2 not-reloaded",
                "1 not-reloaded",
            )
            == ns1.module.statuses()
        )
        assert (
            vec.v(
                "3 not-reloaded",
                "2 not-reloaded",
                "1 not-reloaded",
            )
            == ns1.find(sym.symbol("statuses")).value()
        )

        make_new_module(
            "importer",
            "namespace",
            "to_reload1.lpy",
            module_text=f"""
            (ns {ns1_name}
              (:require [{ns2_name}]))
            (defn statuses []
              (conj ({ns2_name}/statuses) "1 reloaded"))
            """,
        )
        make_new_module(
            "importer",
            "namespace",
            "to_reload3.lpy",
            module_text=f"""
            (ns {ns3_name})
            (defn statuses [] ["3 reloaded"])
            """,
        )

        ns1.reload_all()

        assert (
            vec.v(
                "3 reloaded",
                "2 not-reloaded",
                "1 reloaded",
            )
            == ns1.module.statuses()
        )
        assert (
            vec.v(
                "3 reloaded",
                "2 not-reloaded",
                "1 reloaded",
            )
            == ns1.find(sym.symbol("statuses")).value()
        )

    @pytest.fixture
    def cached_module_ns(self) -> str:
        return "importer.namespace.using-cache"

    @pytest.fixture
    def cached_module_file(
        self, do_cache_namespaces, cached_module_ns, make_new_module
    ):
        file_path = ("importer", "namespace", "using_cache.lpy")
        make_new_module(*file_path, ns_name=cached_module_ns)

        # Import the module out of the current process to avoid having to
        # monkeypatch sys.modules
        p = Process(target=_import_module, args=(munge(cached_module_ns),))
        p.start()
        p.join()
        return os.path.join(*file_path)

    def test_import_module_with_cache(
        self, module_dir, cached_module_ns, cached_module_file, load_namespace
    ):
        using_cache = load_namespace(cached_module_ns)
        assert cached_module_ns == using_cache.find(sym.symbol("val")).value

    def test_import_module_without_writing_cache(
        self,
        monkeypatch,
        module_dir,
        make_new_module,
        load_namespace,
    ):
        monkeypatch.setattr(sys, "dont_write_bytecode", True)
        module_path = ["importer", "namespace", "no_bytecode.lpy"]
        make_new_module(*module_path, ns_name="importer.namespace.no-bytecode")
        load_namespace("importer.namespace.no-bytecode")
        assert not os.path.exists(
            importer._cache_from_source(os.path.join(module_dir, *module_path))
        )

    def test_import_module_with_invalid_cache_magic_number(
        self, module_dir, cached_module_ns, cached_module_file, load_namespace
    ):
        cache_filename = importer._cache_from_source(
            os.path.join(module_dir, cached_module_file)
        )
        with open(cache_filename, mode="r+b") as f:
            f.seek(0)
            f.write(b"1999")

        using_cache = load_namespace(cached_module_ns)
        assert cached_module_ns == using_cache.find(sym.symbol("val")).value

    def test_import_module_with_truncated_timestamp(
        self, module_dir, cached_module_ns, cached_module_file, load_namespace
    ):
        cache_filename = importer._cache_from_source(
            os.path.join(module_dir, cached_module_file)
        )
        with open(cache_filename, mode="w+b") as f:
            f.write(importer.MAGIC_NUMBER)
            f.write(b"abc")

        using_cache = load_namespace(cached_module_ns)
        assert cached_module_ns == using_cache.find(sym.symbol("val")).value

    def test_import_module_with_invalid_timestamp(
        self, module_dir, cached_module_ns, cached_module_file, load_namespace
    ):
        cache_filename = importer._cache_from_source(
            os.path.join(module_dir, cached_module_file)
        )
        with open(cache_filename, mode="r+b") as f:
            f.seek(4)
            f.write(importer._w_long(7_323_337_733))

        using_cache = load_namespace(cached_module_ns)
        assert cached_module_ns == using_cache.find(sym.symbol("val")).value

    def test_import_module_with_truncated_rawsize(
        self, module_dir, cached_module_ns, cached_module_file, load_namespace
    ):
        cache_filename = importer._cache_from_source(
            os.path.join(module_dir, cached_module_file)
        )
        stat = os.stat(cache_filename)
        with open(cache_filename, mode="w+b") as f:
            f.write(importer.MAGIC_NUMBER)
            f.write(importer._w_long(stat.st_mtime))
            f.write(b"abc")

        using_cache = load_namespace(cached_module_ns)
        assert cached_module_ns == using_cache.find(sym.symbol("val")).value

    def test_import_module_with_invalid_rawsize(
        self, module_dir, cached_module_ns, cached_module_file, load_namespace
    ):
        cache_filename = importer._cache_from_source(
            os.path.join(module_dir, cached_module_file)
        )
        with open(cache_filename, mode="r+b") as f:
            f.seek(8)
            f.write(importer._w_long(7733))

        using_cache = load_namespace(cached_module_ns)
        assert cached_module_ns == using_cache.find(sym.symbol("val")).value

    class TestPackageStructure:
        def test_import_module_no_child(self, make_new_module, load_namespace):
            make_new_module("core.lpy", ns_name="core")

            core = load_namespace("core")

            assert "core" == core.find(sym.symbol("val")).value

        def test_import_module_with_non_code_child(
            self, make_new_module, load_namespace
        ):
            make_new_module("core.lpy", ns_name="core")
            make_new_module("core", "resource.txt", module_text="{}")

            core = load_namespace("core")

            assert "core" == core.find(sym.symbol("val")).value

        def test_import_module_without_init_with_python_child(
            self, make_new_module, load_namespace
        ):
            """Load a Basilisp namespace and a Python submodule of that namespace."""
            make_new_module("core.lpy", ns_name="core")
            make_new_module("core", "child.py", module_text="""val = __name__""")

            core = load_namespace("core")
            core_child = importlib.import_module("core.child")

            assert "core" == core.find(sym.symbol("val")).value
            assert "core.child" == core_child.val

        def test_import_basilisp_child_with_python_init(
            self, make_new_module, load_namespace
        ):
            """Load a Python package and a Basilisp submodule namespace of that
            package."""
            make_new_module("core", "__init__.py", module_text="val = __name__")
            make_new_module("core", "child.lpy", ns_name="core.child")

            pycore = importlib.import_module("core")
            core_child = load_namespace("core.child")

            assert "core" == pycore.val
            assert "core.child" == core_child.find(sym.symbol("val")).value

        def test_import_basilisp_and_python_module_siblings(
            self, make_new_module, load_namespace
        ):
            """Load a Python module and Basilisp namespace which are siblings."""
            make_new_module("core.lpy", ns_name="core")
            make_new_module("main.py", module_text="""val = __name__""")

            core = load_namespace("core")
            pymain = importlib.import_module("main")

            assert "core" == core.find(sym.symbol("val")).value
            assert "main" == pymain.val

        def test_import_basilisp_child_with_basilisp_init(
            self, make_new_module, load_namespace
        ):
            """Load a Basilisp package namespace setup as a Python package
            and a child Basilisp namespace of that package."""
            make_new_module("core", "__init__.lpy", ns_name="core")
            make_new_module("core", "sub.lpy", ns_name="core.sub")

            core = load_namespace("core")
            core_sub = load_namespace("core.sub")

            assert "core" == core.find(sym.symbol("val")).value
            assert "core.sub" == core_sub.find(sym.symbol("val")).value

        def test_import_module_without_init(self, make_new_module, load_namespace):
            """Load a Basilisp namespace and a Basilisp child namespace in the
            typical Clojure fashion."""
            make_new_module("core.lpy", ns_name="core")
            make_new_module("core", "child.lpy", ns_name="core.child")

            core = load_namespace("core")
            core_child = load_namespace("core.child")

            assert "core" == core.find(sym.symbol("val")).value
            assert "core.child" == core_child.find(sym.symbol("val")).value

        def test_import_module_with_namespace_only_pkg(
            self, make_new_module, load_namespace
        ):
            """Load a Basilisp namespace and another Basilisp namespace using
            a Python namespace package."""
            make_new_module("core.lpy", ns_name="core")
            make_new_module("core", "nested", "child.lpy", ns_name="core.nested.child")

            core = load_namespace("core")
            core_nested_child = load_namespace("core.nested.child")

            assert "core" == core.find(sym.symbol("val")).value
            assert (
                "core.nested.child" == core_nested_child.find(sym.symbol("val")).value
            )

    class TestExecuteModule:
        def test_no_filename_if_no_module(self):
            with pytest.raises(ImportError):
                importer.BasilispImporter().get_filename("package.module")

        def test_can_get_filename_when_module_exists(self, make_new_module):
            make_new_module("package", "module.lpy", ns_name="package.module")
            filename = importer.BasilispImporter().get_filename("package.module")
            assert filename is not None

            p = pathlib.Path(filename)
            assert p.parts[-2:] == ("package", "module.lpy")

        def test_no_code_if_no_module(self):
            with pytest.raises(ImportError):
                importer.BasilispImporter().get_code("package.module")

        @pytest.mark.parametrize(
            "args,output",
            [
                (["whatever", "1", "2", "3"], "package.module\n[1 2 3]\n"),
                ([], "package.module\nnil\n"),
            ],
        )
        def test_execute_module_correctly(
            self,
            monkeypatch: pytest.MonkeyPatch,
            make_new_module,
            capsys,
            args: list[str],
            output: str,
        ):
            make_new_module(
                "package",
                "module.lpy",
                module_text="""
                (ns package.module)

                (python/print *main-ns*)
                (python/print *command-line-args*)""",
            )

            monkeypatch.setattr("sys.argv", ["whatever", "1", "2", "3"])

            code = importer.BasilispImporter().get_code("package.module")
            exec(code)
            captured = capsys.readouterr()
            assert captured.out == 'package.module\n["1" "2" "3"]\n'


@pytest.fixture
def bootstrap_file() -> pathlib.Path:
    # Generate a bootstrap `.pth` file in the site-packages directory of the current
    # installation.
    #
    # Unfortunately, there is no easy method to copy the current venv (which has
    # Basilisp and it's dependencies installed) to avoid mutating the site-packages
    # of the current environment, so instead we just generate a safe temp `.pth` file
    # in that directory to avoid stomping over any intentionally installed `.pth`
    # files.
    site_package_dir = next(map(pathlib.Path, site.getsitepackages()), None)
    assert site_package_dir is not None

    with tempfile.NamedTemporaryFile(
        dir=site_package_dir, prefix="basilispbootstrap", suffix=".pth", mode="w"
    ) as pth_file:
        pth_file.write("import basilisp.sitecustomize")
        pth_file.flush()
        yield pth_file


@pytest.mark.skipif(
    platform.system().lower() == "windows",
    reason=(
        "Couldn't get this to work and do not have a Windows computer to test on. "
        "Happy to accept a patch!"
    ),
)
@pytest.mark.parametrize(
    "args,ret",
    [
        ([], b"package.test-run-ns-as-pymodule\nnil\n"),
        (["1", "hi", "yes"], b'package.test-run-ns-as-pymodule\n["1" "hi" "yes"]\n'),
    ],
)
def test_run_namespace_as_python_module(
    bootstrap_file: pathlib.Path, tmp_path: pathlib.Path, args: list[str], ret: bytes
):
    parent = tmp_path / "package"
    parent.mkdir()
    nsfile = parent / "test_run_ns_as_pymodule.lpy"

    pythonpath = f"{str(tmp_path)}{os.pathsep}{os.pathsep.join(sys.path)}"

    nsfile.write_text(
        "\n".join(
            [
                "(ns package.test-run-ns-as-pymodule)",
                "",
                "(println *main-ns*)",
                "(println *command-line-args*)",
            ]
        )
    )
    res = subprocess.run(
        [sys.executable, "-m", "package.test_run_ns_as_pymodule", *args],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
    )
    assert res.stdout == ret
