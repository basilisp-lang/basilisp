import importlib
import logging
from typing import Optional

from basilisp import importer as importer
from basilisp.lang import runtime as runtime
from basilisp.lang.compiler import compiler_opts
from basilisp.lang.typing import CompilerOpts
from basilisp.lang.util import munge
from basilisp.logconfig import DEFAULT_HANDLER, DEFAULT_LEVEL

logger = logging.getLogger("basilisp")
logger.setLevel(DEFAULT_LEVEL)
logger.addHandler(DEFAULT_HANDLER)


def init(opts: Optional[CompilerOpts] = None) -> None:
    """
    Initialize the runtime environment for Basilisp code evaluation.

    If you want to execute a Basilisp file which is stored in a well-formed package
    or module structure, you probably want to use `bootstrap()` below.
    """
    runtime.init_ns_var()
    runtime.bootstrap_core(opts if opts is not None else compiler_opts())
    importer.hook_imports()
    importlib.import_module("basilisp.core")


def bootstrap(
    target: str, opts: Optional[CompilerOpts] = None
) -> None:  # pragma: no cover
    """
    Import a Basilisp namespace or function identified by `target`.

    Basilisp only needs to be bootstrapped once per Python VM invocation. Subsequent
    imports of Basilisp namespaces will work using Python's standard `import` statement
    and `importlib.import_module` function.

    `target` must be a string naming a Basilisp namespace. Namespace references may
    be given exactly as they are found in Basilisp code. `target` may optionally
    include a trailing function reference, delimited by ":", which will be executed
    after the target namespace is imported. If a function reference is given, the
    function will be called with no arguments.

    `opts` is a mapping of compiler options that may be supplied for bootstrapping.
    This setting should be left alone unless you know what you are doing.
    """
    init(opts=opts)
    pkg_name, *rest = target.split(":", maxsplit=1)
    mod = importlib.import_module(munge(pkg_name))
    if rest:
        fn_name = munge(rest[0])
        getattr(mod, fn_name)()
