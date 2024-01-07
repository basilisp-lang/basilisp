import importlib
import logging
import site
from pathlib import Path
from typing import List, Optional

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

    Basilisp only needs to be initialized once per Python VM invocation. Subsequent
    imports of Basilisp namespaces will work using Python's standard ``import``
    statement and ``importlib.import_module`` function.

    If you want to execute a Basilisp file which is stored in a well-formed package
    or module structure, you probably want to use :py:func:`bootstrap`.
    """
    runtime.init_ns_var()
    runtime.bootstrap_core(opts if opts is not None else compiler_opts())
    importer.hook_imports()
    importlib.import_module("basilisp.core")


def bootstrap(
    target: str, opts: Optional[CompilerOpts] = None
) -> None:  # pragma: no cover
    """
    Import a Basilisp namespace or function identified by ``target``. If a function
    reference is given, the function will be called with no arguments.

    Basilisp only needs to be initialized once per Python VM invocation. Subsequent
    imports of Basilisp namespaces will work using Python's standard ``import``
    statement and ``importlib.import_module`` function.

    ``target`` must be a string naming a Basilisp namespace. Namespace references may
    be given exactly as they are found in Basilisp code. ``target`` may optionally
    include a trailing function reference, delimited by ":", which will be executed
    after the target namespace is imported.

    ``opts`` is a mapping of compiler options that may be supplied for bootstrapping.
    This setting should be left alone unless you know what you are doing.
    """
    init(opts=opts)
    pkg_name, *rest = target.split(":", maxsplit=1)
    mod = importlib.import_module(munge(pkg_name))
    if rest:
        fn_name = munge(rest[0])
        getattr(mod, fn_name)()


def bootstrap_python(site_packages: Optional[List[str]] = None) -> None:
    """Bootstrap a Python installation by installing a ``.pth`` file in the first
    available ``site-packages`` directory (as by ``site.getsitepackages()``).

    Subsequent startups of the Python interpreter will have Basilisp already
    bootstrapped and available to run."""
    if site_packages is None:  # pragma: no cover
        site_packages = site.getsitepackages()

    assert site_packages, "Expected at least one site-package directory"

    for d in site_packages:
        p = Path(d)
        with open(p / "basilispbootstrap.pth", mode="w") as f:
            f.write("import basilisp.sitecustomize")
        break


def unbootstrap_python(site_packages: Optional[List[str]] = None) -> List[str]:
    """Remove any `basilispbootstrap.pth` files found in any Python site-packages
    directory (as by ``site.getsitepackages()``). Return a list of removed
    filenames."""
    if site_packages is None:  # pragma: no cover
        site_packages = site.getsitepackages()

    assert site_packages, "Expected at least one site-package directory"

    removed = []
    for d in site_packages:
        p = Path(d)
        for file in p.glob("basilispbootstrap.pth"):
            try:
                file.unlink()
            except FileNotFoundError:  # pragma: no cover
                pass
            else:
                removed.append(str(file))
    return removed
