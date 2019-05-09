import importlib
import os.path
import sys
from tempfile import TemporaryDirectory, mkstemp
from typing import Optional, Tuple
from unittest.mock import patch

import pytest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import Testdir

import basilisp.importer as importer
import basilisp.lang.keyword as kw
import basilisp.lang.runtime as runtime
import basilisp.lang.symbol as sym
from basilisp.lang.util import demunge, munge


def importer_counter():
    return sum([isinstance(o, importer.BasilispImporter) for o in sys.meta_path])


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
            monkeypatch.chdir(tmpdir)
            monkeypatch.syspath_prepend(tmpdir)
            monkeypatch.setattr("sys.modules", module_cache)
            yield tmpdir

    @pytest.fixture
    def new_module_file(self, module_dir):
        files = {}

        def _new_module_file(mode: str = "w"):
            fd, filename = mkstemp(
                suffix=".lpy", prefix="importer", dir=module_dir, text=True
            )
            file = open(filename, mode=mode)
            files[filename] = file
            return file

        try:
            yield _new_module_file
        finally:
            for filename, file in files.items():
                file.close()
                os.unlink(filename)

    def test_import_module(self, new_module_file):
        mod = new_module_file()
        ns_name, mod_name = _ns_and_module(mod.name)
        mod.write(f"(ns {ns_name}) (def val :basilisp.namespace/not-cached)")
        mod.flush()

        importlib.import_module(mod_name)

        assert (
            kw.keyword("not-cached", ns="basilisp.namespace")
            == runtime.Namespace.get(sym.symbol(ns_name)).find(sym.symbol("val")).value
        )

    def test_import_module_without_cache(
        self, new_module_file, do_not_cache_namespaces
    ):
        mod = new_module_file()
        ns_name, mod_name = _ns_and_module(mod.name)
        mod.write(f"(ns {ns_name}) (def val :basilisp.namespace/without-cache)")
        mod.flush()

        importlib.import_module(mod_name)

        assert (
            kw.keyword("without-cache", ns="basilisp.namespace")
            == runtime.Namespace.get(sym.symbol(ns_name)).find(sym.symbol("val")).value
        )

    @pytest.fixture
    def cached_module_file(self, do_cache_namespaces, new_module_file):
        mod = new_module_file()
        ns_name, mod_name = _ns_and_module(mod.name)
        mod.write(f"(ns {ns_name}) (def val :basilisp.namespace/invalid-cache)")
        mod.flush()

        importlib.import_module(mod_name)
        return ns_name, mod_name, importer._cache_from_source(mod.name)

    def test_import_module_with_invalid_cache_magic_number(
        self, monkeypatch: MonkeyPatch, module_cache, cached_module_file
    ):
        with monkeypatch.context() as mctx:
            mctx.setattr("sys.modules", module_cache)
            ns_name, mod_name, cache_filename = cached_module_file

            with open(cache_filename, "r+b") as f:
                f.seek(0)
                f.write(b"1999")

            importlib.import_module(mod_name)
            assert (
                kw.keyword("invalid-cache", ns="basilisp.namespace")
                == runtime.Namespace.get(sym.symbol(ns_name))
                .find(sym.symbol("val"))
                .value
            )

    class TestPackageStructure:
        @pytest.fixture
        def make_new_module(self, module_dir):
            """Fixture returning a function which creates a new module and then
            removes it after the test run."""
            filenames = []

            def _make_new_module(ns_name: str, *ns_path: str, module_text: Optional[str] = None) -> None:
                """Generate a new Namespace with a single Var named 'val' which
                contains the string name of the current namespace."""
                if ns_path[:-1]:
                    os.makedirs(os.path.join(module_dir, *ns_path[:-1]), exist_ok=True)

                filename = os.path.join(module_dir, *ns_path)
                filenames.append(filename)

                with open(filename, mode="w") as mod:
                    if module_text is None:
                        mod.write(f"(ns {ns_name}) (def val (name *ns*))")
                    else:
                        mod.write(module_text)

            try:
                yield _make_new_module
            finally:
                for filename in filenames:
                    os.unlink(filename)

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

        def test_import_module_no_child(self, make_new_module, load_namespace):
            make_new_module("core", "core.lpy")

            core = load_namespace("core")

            assert "core" == core.find(sym.symbol("val")).value

        def test_import_module_with_init(self, make_new_module, load_namespace):
            make_new_module("core", "core", "__init__.lpy")
            make_new_module("core.sub", "core", "sub.lpy")

            core = load_namespace("core")
            core_sub = load_namespace("core.sub")

            assert "core" == core.find(sym.symbol("val")).value
            assert "core.sub" == core_sub.find(sym.symbol("val")).value

        def test_import_module_without_init(self, make_new_module, load_namespace):
            make_new_module("core", "core.lpy")
            make_new_module("core.child", "core", "child.lpy")

            core = load_namespace("core")
            core_child = load_namespace("core.child")

            assert "core" == core.find(sym.symbol("val")).value
            assert "core.child" == core_child.find(sym.symbol("val")).value

        def test_import_module_with_namespace_only_pkg(
            self, make_new_module, load_namespace
        ):
            make_new_module("core", "core.lpy")
            make_new_module("core.nested.child", "core", "nested", "child.lpy")

            core = load_namespace("core")
            core_nested_child = load_namespace("core.nested.child")

            assert "core" == core.find(sym.symbol("val")).value
            assert (
                "core.nested.child" == core_nested_child.find(sym.symbol("val")).value
            )
