import importlib
import os.path
import sys
from multiprocessing import Process, get_start_method
from tempfile import TemporaryDirectory
from typing import Optional, Tuple
from unittest.mock import patch

import pytest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import Testdir

from basilisp import importer as importer
from basilisp.lang import runtime as runtime
from basilisp.lang import symbol as sym
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


def test_demunged_import(testdir: Testdir):
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
            with patch("sys.path", new=[tmpdir]), patch(
                "sys.meta_path", new=[importer.BasilispImporter()]
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


def _ns_and_module(filename: str) -> Tuple[str, str]:
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
    def do_cache_namespaces(self, monkeypatch):
        monkeypatch.setenv(importer._NO_CACHE_ENVVAR, "false")

    @pytest.fixture
    def do_not_cache_namespaces(self, monkeypatch):
        monkeypatch.setenv(importer._NO_CACHE_ENVVAR, "true")

    @pytest.fixture
    def module_cache(self):
        return {name: module for name, module in sys.modules.items()}

    @pytest.fixture
    def module_dir(self, monkeypatch: MonkeyPatch, module_cache):
        with TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            monkeypatch.chdir(tmpdir)
            monkeypatch.syspath_prepend(tmpdir)
            monkeypatch.setattr("sys.modules", module_cache)
            yield tmpdir
            monkeypatch.chdir(tmpdir)
            monkeypatch.chdir(cwd)

    @pytest.fixture
    def make_new_module(self, module_dir):
        """Fixture returning a function which creates a new module and then
        removes it after the test run."""
        filenames = []

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
            filenames.append(filename)

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
