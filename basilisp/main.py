import importlib

import basilisp.importer as importer
import basilisp.lang.runtime as runtime


def init():
    """Initialize the runtime environment for evaluation."""
    runtime.init_ns_var()
    runtime.bootstrap()
    importer.hook_imports()
    importlib.import_module('basilisp.core')
