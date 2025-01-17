import importlib.util
import logging
import marshal
import os
import os.path
import sys
import types
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from functools import lru_cache
from importlib.abc import MetaPathFinder, SourceLoader
from importlib.machinery import ModuleSpec
from typing import Any, Optional, cast

from typing_extensions import TypedDict

from basilisp.lang import compiler as compiler
from basilisp.lang import reader as reader
from basilisp.lang import runtime as runtime
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec
from basilisp.lang.runtime import BasilispModule
from basilisp.lang.typing import ReaderForm
from basilisp.lang.util import demunge
from basilisp.util import timed

_NO_CACHE_ENVVAR = "BASILISP_DO_NOT_CACHE_NAMESPACES"

MAGIC_NUMBER = (1149).to_bytes(2, "little") + b"\r\n"

logger = logging.getLogger(__name__)


def _r_long(int_bytes: bytes) -> int:
    """Convert 4 bytes in little-endian to an integer."""
    return int.from_bytes(int_bytes, "little")


def _w_long(x: int) -> bytes:
    """Convert a 32-bit integer to little-endian."""
    return (int(x) & 0xFFFFFFFF).to_bytes(4, "little")


def _basilisp_bytecode(
    mtime: int, source_size: int, code: list[types.CodeType]
) -> bytes:
    """Return the bytes for a Basilisp bytecode cache file."""
    data = bytearray(MAGIC_NUMBER)
    data.extend(_w_long(mtime))
    data.extend(_w_long(source_size))
    data.extend(marshal.dumps(code))
    return data


def _get_basilisp_bytecode(
    fullname: str, mtime: int, source_size: int, cache_data: bytes
) -> list[types.CodeType]:
    """Unmarshal the bytes from a Basilisp bytecode cache file, validating the
    file header prior to returning. If the file header does not match, throw
    an exception."""
    exc_details = {"name": fullname}
    magic = cache_data[:4]
    raw_timestamp = cache_data[4:8]
    raw_size = cache_data[8:12]
    if magic != MAGIC_NUMBER:
        message = (
            f"Incorrect magic number ({magic!r}) in {fullname}; "
            f"expected {MAGIC_NUMBER!r}"
        )
        logger.debug(message)
        raise ImportError(message, **exc_details)
    elif len(raw_timestamp) != 4:
        message = f"Reached EOF while reading timestamp in {fullname}"
        logger.debug(message)
        raise EOFError(message)
    elif _r_long(raw_timestamp) != mtime:
        message = f"Non-matching timestamp ({_r_long(raw_timestamp)}) in {fullname} bytecode cache; expected {mtime}"
        logger.debug(message)
        raise ImportError(message, **exc_details)
    elif len(raw_size) != 4:
        message = f"Reached EOF while reading size of source in {fullname}"
        logger.debug(message)
        raise EOFError(message)
    elif _r_long(raw_size) != source_size:
        message = f"Non-matching filesize ({_r_long(raw_size)}) in {fullname} bytecode cache; expected {source_size}"
        logger.debug(message)
        raise ImportError(message, **exc_details)

    return marshal.loads(cache_data[12:])  # nosec 6302


def _cache_from_source(path: str) -> str:
    """Return the path to the cached file for the given path. The original path
    does not have to exist."""
    cache_path, cache_file = os.path.split(importlib.util.cache_from_source(path))
    filename, _ = os.path.splitext(cache_file)
    return os.path.join(cache_path, filename + ".lpyc")


@lru_cache
def _is_package(path: str) -> bool:
    """Return True if path should be considered a Basilisp (and consequently
    a Python) package.

    A path would be considered a package if it contains at least one Basilisp
    or Python code file."""
    for _, _, files in os.walk(path):
        for file in files:
            if file.endswith(".lpy") or file.endswith(".py") or file.endswith(".cljc"):
                return True
    return False


@lru_cache
def _is_namespace_package(path: str) -> bool:
    """Return True if the current directory is a namespace Basilisp package.

    Basilisp namespace packages are directories containing no __init__.py or
    __init__.lpy files and at least one other Basilisp code file."""
    no_inits = True
    has_basilisp_files = False
    _, _, files = next(os.walk(path))
    for file in files:
        if file in {"__init__.lpy", "__init__.py"}:
            no_inits = False
        elif file.endswith(".lpy") or file.endswith(".cljc"):
            has_basilisp_files = True
    return no_inits and has_basilisp_files


class ImporterCacheEntry(TypedDict, total=False):
    spec: ModuleSpec
    module: BasilispModule


class BasilispImporter(  # type: ignore[misc]  # pylint: disable=abstract-method
    MetaPathFinder, SourceLoader
):
    """Python import hook to allow directly loading Basilisp code within
    Python."""

    def __init__(self):
        self._cache: MutableMapping[str, ImporterCacheEntry] = {}

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[types.ModuleType] = None,
    ) -> Optional[ModuleSpec]:
        """Find the ModuleSpec for the specified Basilisp module.

        Returns None if the module is not a Basilisp module to allow import processing to continue.
        """
        package_components = fullname.split(".")
        if not path:
            path = sys.path
            module_name = package_components
        else:
            module_name = [package_components[-1]]

        for entry in path:
            root_path = os.path.join(entry, *module_name)
            filenames = [
                f"{os.path.join(root_path, '__init__')}.lpy",
                f"{root_path}.lpy",
                f"{os.path.join(root_path, '__init__')}.cljc",
                f"{root_path}.cljc",
            ]
            for filename in filenames:
                if os.path.isfile(filename):
                    state = {
                        "fullname": fullname,
                        "filename": filename,
                        "path": entry,
                        "target": target,
                        "cache_filename": _cache_from_source(filename),
                    }
                    logger.debug(
                        f"Found potential Basilisp module '{fullname}' in file '{filename}'"
                    )
                    is_package = filename.endswith("__init__.lpy") or _is_package(
                        root_path
                    )
                    spec = ModuleSpec(
                        fullname,
                        self,
                        origin=filename,
                        loader_state=state,
                        is_package=is_package,
                    )
                    # The Basilisp loader can find packages regardless of
                    # submodule_search_locations, but the Python loader cannot.
                    # Set this to the root path to allow the Python loader to
                    # load submodules of Basilisp "packages".
                    if is_package:
                        assert (
                            spec.submodule_search_locations is not None
                        ), "Package module spec must have submodule_search_locations list"
                        spec.submodule_search_locations.append(root_path)
                    return spec
            if os.path.isdir(root_path):
                if _is_namespace_package(root_path):
                    return ModuleSpec(fullname, None, is_package=True)
        return None

    def invalidate_caches(self) -> None:
        super().invalidate_caches()
        self._cache = {}

    def _cache_bytecode(self, source_path: str, cache_path: str, data: bytes) -> None:
        self.set_data(cache_path, data)

    def path_stats(self, path: str) -> Mapping[str, Any]:
        stat = os.stat(path)
        return {"mtime": int(stat.st_mtime), "size": stat.st_size}

    def get_data(self, path: str) -> bytes:
        with open(path, mode="r+b") as f:
            return f.read()

    def set_data(self, path: str, data: bytes) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode="w+b") as f:
            f.write(data)

    def get_filename(self, fullname: str) -> str:
        try:
            cached = self._cache[fullname]
        except KeyError as e:
            if (spec := self.find_spec(fullname, None)) is None:
                raise ImportError(f"Could not import module '{fullname}'") from e
        else:
            spec = cached["spec"]
            assert spec is not None, "spec must be defined here"
        return spec.loader_state["filename"]

    def get_code(self, fullname: str) -> Optional[types.CodeType]:
        """Return code to load a Basilisp module.

        This function is part of the ABC for `importlib.abc.ExecutionLoader` which is
        what Python uses to execute modules at the command line as `python -m module`.
        """
        core_ns = runtime.Namespace.get(runtime.CORE_NS_SYM)
        assert core_ns is not None

        with runtime.ns_bindings("basilisp.namespace-executor") as ns:
            ns.refer_all(core_ns)

            # Set the *main-ns* variable to the current namespace.
            main_ns_var = core_ns.find(sym.symbol(runtime.MAIN_NS_VAR_NAME))
            assert main_ns_var is not None
            main_ns_var.bind_root(sym.symbol(demunge(fullname)))

            # Set command line args passed to the module
            if pyargs := sys.argv[1:]:
                cli_args_var = core_ns.find(
                    sym.symbol(runtime.COMMAND_LINE_ARGS_VAR_NAME)
                )
                assert cli_args_var is not None
                cli_args_var.bind_root(vec.vector(pyargs))

            # Basilisp can only ever product multiple `types.CodeType` objects for any
            # given module because it compiles each form as a separate unit, but
            # `ExecutionLoader.get_code` expects a single `types.CodeType` object. To
            # simulate this requirement, we generate a single `(load "...")` to execute
            # in a synthetic namespace.
            #
            # The target namespace is free to interpret
            code: list[types.CodeType] = []
            path = "/" + "/".join(fullname.split("."))
            try:
                compiler.load(
                    path,
                    compiler.CompilerContext(
                        filename="<Basilisp Namespace Executor>",
                        opts=runtime.get_compiler_opts(),
                    ),
                    ns,
                    collect_bytecode=code.append,
                )
            except Exception as e:
                raise ImportError(f"Could not import module '{fullname}'") from e
            else:
                assert len(code) == 1
                return code[0]

    def create_module(self, spec: ModuleSpec) -> BasilispModule:
        logger.debug(f"Creating Basilisp module '{spec.name}'")
        mod = BasilispModule(spec.name)
        mod.__file__ = spec.loader_state["filename"]
        mod.__loader__ = spec.loader
        mod.__package__ = spec.parent
        mod.__spec__ = spec
        self._cache[spec.name] = {"spec": spec}
        return mod

    def _exec_cached_module(
        self,
        fullname: str,
        loader_state: Mapping[str, str],
        path_stats: Mapping[str, int],
        module: BasilispModule,
    ) -> None:
        """Load and execute a cached Basilisp module."""
        filename = loader_state["filename"]
        cache_filename = loader_state["cache_filename"]

        with timed(
            lambda duration: logger.debug(
                f"Loaded cached Basilisp module '{fullname}' in {duration / 1000000}ms"
            )
        ):
            logger.debug(f"Checking for cached Basilisp module '{fullname}''")
            cache_data = self.get_data(cache_filename)
            cached_code = _get_basilisp_bytecode(
                fullname, path_stats["mtime"], path_stats["size"], cache_data
            )
            compiler.compile_bytecode(
                cached_code,
                compiler.GeneratorContext(
                    filename=filename, opts=runtime.get_compiler_opts()
                ),
                compiler.PythonASTOptimizer(),
                module,
            )

    def _exec_module(
        self,
        fullname: str,
        loader_state: Mapping[str, str],
        path_stats: Mapping[str, int],
        module: BasilispModule,
    ) -> None:
        """Load and execute a non-cached Basilisp module."""
        filename = loader_state["filename"]
        cache_filename = loader_state["cache_filename"]

        with timed(
            lambda duration: logger.debug(
                f"Loaded Basilisp module '{fullname}' in {duration / 1000000}ms"
            )
        ):
            # During compilation, bytecode objects are added to the list which is
            # passed to the compiler. The collected bytecodes will be used to generate
            # an .lpyc file for caching the compiled file.
            all_bytecode: list[types.CodeType] = []

            logger.debug(f"Reading and compiling Basilisp module '{fullname}'")
            # Cast to basic ReaderForm since the reader can never return a reader conditional
            # form unprocessed in internal usage. There are reader settings which permit
            # callers to leave unprocessed reader conditionals in the stream, however.
            forms = cast(
                Iterable[ReaderForm],
                reader.read_file(filename, resolver=runtime.resolve_alias),
            )
            compiler.compile_module(
                forms,
                compiler.CompilerContext(
                    filename=filename, opts=runtime.get_compiler_opts()
                ),
                module,
                collect_bytecode=all_bytecode.append,
            )

        if sys.dont_write_bytecode:
            logger.debug(f"Skipping bytecode generation for '{fullname}'")
            return

        # Cache the bytecode that was collected through the compilation run.
        cache_file_bytes = _basilisp_bytecode(
            path_stats["mtime"], path_stats["size"], all_bytecode
        )
        self._cache_bytecode(filename, cache_filename, cache_file_bytes)

    def exec_module(self, module: types.ModuleType) -> None:
        """Compile the Basilisp module into Python code.

        Basilisp is fundamentally a form-at-a-time compilation, meaning that
        each form in a module may require code compiled from an earlier form, so
        we incrementally compile a Python module by evaluating a single top-level
        form at a time and inserting the resulting AST nodes into the Pyton module."""
        assert isinstance(module, BasilispModule)

        fullname = module.__name__
        if (cached := self._cache.get(fullname)) is None:
            spec = module.__spec__
            assert spec is not None, "Module must have a spec"
            cached = {"spec": spec}
            self._cache[spec.name] = cached
        cached["module"] = module
        spec = cached["spec"]
        filename = spec.loader_state["filename"]
        path_stats = self.path_stats(filename)

        # During the bootstrapping process, the 'basilisp.core namespace is created with
        # a blank module. If we do not replace the module here with the module we are
        # generating, then we will not be able to use advanced compilation features such
        # as direct Python variable access to functions and other def'ed values.
        if fullname == runtime.CORE_NS:
            ns: runtime.Namespace = runtime.Namespace.get_or_create(runtime.CORE_NS_SYM)
            ns.module = module
            module.__basilisp_namespace__ = ns

        # Set the currently importing module so it can be attached to the namespace when
        # it is created.
        with runtime.bindings(
            {runtime.Var.find_safe(runtime.IMPORT_MODULE_VAR_SYM): module}
        ):

            # Check if a valid, cached version of this Basilisp namespace exists and, if so,
            # load it and bypass the expensive compilation process below.
            if os.getenv(_NO_CACHE_ENVVAR, "").lower() == "true":
                self._exec_module(fullname, spec.loader_state, path_stats, module)
            else:
                try:
                    self._exec_cached_module(
                        fullname, spec.loader_state, path_stats, module
                    )
                except (EOFError, ImportError, OSError) as e:
                    logger.debug(f"Failed to load cached Basilisp module: {e}")
                    self._exec_module(fullname, spec.loader_state, path_stats, module)


def hook_imports() -> None:
    """Hook into Python's import machinery with a custom Basilisp code
    importer.

    Once this is called, Basilisp code may be called from within Python code
    using standard `import module.submodule` syntax."""
    if any(isinstance(o, BasilispImporter) for o in sys.meta_path):
        return
    sys.meta_path.insert(0, BasilispImporter())
