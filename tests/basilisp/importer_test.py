import importlib
import os.path
import sys
from tempfile import TemporaryDirectory, mkstemp
from typing import Tuple
from unittest.mock import patch

import pytest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import Testdir

import basilisp.importer as importer
import basilisp.lang.keyword as kw
import basilisp.lang.runtime as runtime
import basilisp.lang.symbol as sym
from basilisp.lang.util import demunge


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

    def test_import_module_with_init(self, module_dir):
        os.mkdir(os.path.join(module_dir, "core"))

        with open(os.path.join(module_dir, "core", "__init__.lpy"), mode="w") as mod:
            mod.write(f"(ns core) (def val :basilisp.core/init-namespace)")
            mod.flush()

        with open(os.path.join(module_dir, "core", "sub.lpy"), mode="w") as mod:
            mod.write(f"(ns core.sub) (def val :basilisp.core.sub/namespace)")
            mod.flush()

        importlib.import_module("core")
        importlib.import_module("core.sub")

        assert (
            kw.keyword("init-namespace", ns="basilisp.core")
            == runtime.Namespace.get(sym.symbol("core")).find(sym.symbol("val")).value
        )

        assert (
            kw.keyword("namespace", ns="basilisp.core.sub")
            == runtime.Namespace.get(sym.symbol("core.sub"))
            .find(sym.symbol("val"))
            .value
        )

        runtime.Namespace.remove(sym.symbol("core"))
        runtime.Namespace.remove(sym.symbol("core.sub"))

    def test_import_module_without_init(self, module_dir):
        os.mkdir(os.path.join(module_dir, "core"))

        with open(os.path.join(module_dir, "core.lpy"), mode="w") as mod:
            mod.write(f"(ns core) (def val :basilisp.core/standard-namespace)")
            mod.flush()

        with open(os.path.join(module_dir, "core", "child.lpy"), mode="w") as mod:
            mod.write(
                f"(ns core.child) (def val :basilisp.core.child/standard-namespace)"
            )
            mod.flush()

        importlib.import_module("core")
        importlib.import_module("core.child")

        assert (
            kw.keyword("standard-namespace", ns="basilisp.core")
            == runtime.Namespace.get(sym.symbol("core")).find(sym.symbol("val")).value
        )

        assert (
            kw.keyword("standard-namespace", ns="basilisp.core.child")
            == runtime.Namespace.get(sym.symbol("core.child"))
            .find(sym.symbol("val"))
            .value
        )

        runtime.Namespace.remove(sym.symbol("core"))
        runtime.Namespace.remove(sym.symbol("core.child"))

    def test_import_module_with_namespace_only_pkg(self, module_dir):
        os.mkdir(os.path.join(module_dir, "core"))
        os.mkdir(os.path.join(module_dir, "core", "nested"))

        with open(os.path.join(module_dir, "core.lpy"), mode="w") as mod:
            mod.write(f"(ns core) (def val :basilisp.core/grandparent-namespace)")
            mod.flush()

        with open(os.path.join(module_dir, "core", "nested", "child.lpy"), mode="w") as mod:
            mod.write(
                f"(ns core.nested.child) (def val :basilisp.core.nested.child/standard-namespace)"
            )
            mod.flush()

        importlib.import_module("core")
        importlib.import_module("core.nested.child")

        assert (
            kw.keyword("grandparent-namespace", ns="basilisp.core")
            == runtime.Namespace.get(sym.symbol("core")).find(sym.symbol("val")).value
        )

        assert (
            kw.keyword("standard-namespace", ns="basilisp.core.nested.child")
            == runtime.Namespace.get(sym.symbol("core.nested.child"))
            .find(sym.symbol("val"))
            .value
        )

        runtime.Namespace.remove(sym.symbol("core"))
        runtime.Namespace.remove(sym.symbol("core.nested.child"))

    # @pytest.fixture
    # def cached_module_file(self, do_cache_namespaces, new_module_file):
    #     mod = new_module_file()
    #     ns_name, mod_name = _ns_and_module(mod.name)
    #     mod.write(f"(ns {ns_name}) (def val :basilisp.namespace/invalid-cache)")
    #     mod.flush()
    #
    #     importlib.import_module(mod_name)
    #     return ns_name, mod_name, importer._cache_from_source(mod.name)
    #
    # def test_import_module_with_invalid_cache_magic_number(
    #     self, monkeypatch: MonkeyPatch, module_cache, cached_module_file
    # ):
    #     with monkeypatch.context() as mctx:
    #         mctx.setattr("sys.modules", module_cache)
    #         ns_name, mod_name, cache_filename = cached_module_file
    #
    #         with open(cache_filename, "r+b") as f:
    #             f.seek(0)
    #             f.write(b"1999")
    #
    #         importlib.import_module(mod_name)
    #         assert (
    #             kw.keyword("invalid-cache", ns="basilisp.namespace")
    #             == runtime.Namespace.get(sym.symbol(ns_name))
    #             .find(sym.symbol("val"))
    #             .value
    #         )
