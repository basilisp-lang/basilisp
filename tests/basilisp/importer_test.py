import importlib
import os.path
import sys
import tempfile
from unittest.mock import patch

import _pytest.pytester as pytester

import basilisp.importer as importer
import basilisp.lang.runtime as runtime
import basilisp.lang.symbol as sym


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


def test_demunged_import(testdir: pytester.Testdir):
    with tempfile.TemporaryDirectory() as tmpdir:
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
