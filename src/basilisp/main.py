import importlib
import logging

from basilisp import importer as importer
from basilisp.lang import runtime as runtime
from basilisp.lang.typing import CompilerOpts
from basilisp.logconfig import DEFAULT_HANDLER, DEFAULT_LEVEL

logger = logging.getLogger("basilisp")
logger.setLevel(DEFAULT_LEVEL)
logger.addHandler(DEFAULT_HANDLER)


def init(compiler_opts: CompilerOpts) -> None:
    """Initialize the runtime environment for evaluation."""
    runtime.init_ns_var()
    runtime.bootstrap_core(compiler_opts)
    importer.hook_imports()
    importlib.import_module("basilisp.core")
