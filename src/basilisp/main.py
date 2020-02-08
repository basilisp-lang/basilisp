import importlib
import logging

import basilisp.importer as importer
import basilisp.lang.runtime as runtime
from basilisp.logconfig import DEFAULT_HANDLER, DEFAULT_LEVEL

logger = logging.getLogger("basilisp")
logger.setLevel(DEFAULT_LEVEL)
logger.addHandler(DEFAULT_HANDLER)


def init():
    """Initialize the runtime environment for evaluation."""
    runtime.init_ns_var()
    runtime.bootstrap_core()
    importer.hook_imports()
    importlib.import_module("basilisp.core")
