import importlib
import logging
import marshal
import os
import os.path
import sys
import types
from functools import lru_cache
from importlib.abc import MetaPathFinder, SourceLoader
from importlib.machinery import ModuleSpec
from typing import (
    Any,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    cast,
)

from basilisp.lang import compiler as compiler
from basilisp.lang import reader as reader
from basilisp.lang import runtime as runtime
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec
from basilisp.lang.runtime import BasilispModule, ModuleLoadState
from basilisp.lang.typing import ReaderForm
from basilisp.lang.util import demunge
from basilisp.util import timed

_EAGER_IMPORT_ENVVAR = "BASILISP_USE_EAGER_IMPORT"
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
    mtime: int, source_size: int, code: List[types.CodeType]
) -> bytes:
    """Return the bytes for a Basilisp bytecode cache file."""
    data = bytearray(MAGIC_NUMBER)
    data.extend(_w_long(mtime))
    data.extend(_w_long(source_size))
    data.extend(marshal.dumps(code))
    return data


def _get_basilisp_bytecode(
    fullname: str, mtime: int, source_size: int, cache_data: bytes
) -> List[types.CodeType]:
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

    return marshal.loads(cache_data[12:])


def _cache_from_source(path: str) -> str:
    """Return the path to the cached file for the given path. The original path
    does not have to exist."""
    cache_path, cache_file = os.path.split(importlib.util.cache_from_source(path))
    filename, _ = os.path.splitext(cache_file)
    return os.path.join(cache_path, filename + ".lpyc")


@lru_cache()
def _is_package(path: str) -> bool:
    """Return True if path should be considered a Basilisp (and consequently
    a Python) package.

    A path would be considered a package if it contains at least one Basilisp
    or Python code file."""
    for _, _, files in os.walk(path):
        for file in files:
            if file.endswith(".lpy") or file.endswith(".py"):
                return True
    return False


@lru_cache()
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
        elif file.endswith(".lpy"):
            has_basilisp_files = True
    return no_inits and has_basilisp_files


class BasilispImporter(MetaPathFinder, SourceLoader):  # pylint: disable=abstract-method
    """Python import hook to allow directly loading Basilisp code within
    Python."""

    def __init__(self):
        self._cache: MutableMapping[str, dict] = {}

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
            code: List[types.CodeType] = []
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
        mod.__basilisp_loaded__ = ModuleLoadState()
        mod.__basilisp_bootstrapped__ = False
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
        ns: runtime.Namespace,
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
                ns,
            )

    def _exec_module(
        self,
        fullname: str,
        loader_state: Mapping[str, str],
        path_stats: Mapping[str, int],
        ns: runtime.Namespace,
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
            all_bytecode: List[types.CodeType] = []

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
                ns,
                collect_bytecode=all_bytecode.append,
            )

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
        load_state = module.__basilisp_loaded__
        try:
            load_state.loading()
            fullname = module.__name__
            cached = self._cache[fullname]
            cached["module"] = module
            spec = cached["spec"]
            filename = spec.loader_state["filename"]
            path_stats = self.path_stats(filename)

            # During the bootstrapping process, the 'basilisp.core namespace is created with
            # a blank module. If we do not replace the module here with the module we are
            # generating, then we will not be able to use advanced compilation features such
            # as direct Python variable access to functions and other def'ed values.
            ns_name = demunge(fullname)
            ns: runtime.Namespace = runtime.Namespace.get_or_create(
                sym.symbol(ns_name), module
            )
            ns.module = module
            module.__basilisp_namespace__ = ns

            # Check if a valid, cached version of this Basilisp namespace exists and, if so,
            # load it and bypass the expensive compilation process below.
            if os.getenv(_NO_CACHE_ENVVAR, "").lower() == "true":
                self._exec_module(fullname, spec.loader_state, path_stats, ns)
            else:
                try:
                    self._exec_cached_module(
                        fullname, spec.loader_state, path_stats, ns
                    )
                except (EOFError, ImportError, IOError, OSError) as e:
                    logger.debug(f"Failed to load cached Basilisp module: {e}")
                    self._exec_module(fullname, spec.loader_state, path_stats, ns)
            load_state.complete()
        except BaseException as e:
            load_state.complete(e)
            raise e

    def module_repr(self, module: types.ModuleType) -> str:
        return module.__name__


class BasilispLazyModule(BasilispModule):
    """Represents a BasilispModule that will be loaded as soon as one of it's members is accessed.

    Adpated from `importlib.util._LazyModule`."""

    def __getattribute__(self, attr):
        """Trigger the load of the module and return the attribute."""

        # We don't need to trigger a load to retrieve this information
        if attr in {
            "__name__",
            "__path__",
            "__spec__",
            "__loader__",
            "__dict__",
            "__class__",
            "__file__",
            "__package__",
        }:
            return object.__getattribute__(self, attr)

        spec = self.__spec__
        if spec is None:
            raise ImportError(f"No spec for module {self.__name__}")

        load_state = spec.loader_state["basilisp_load_state"]

        with load_state.lock():
            loader = spec.loader
            assert isinstance(loader, BasilispLazyImporter)
            # if the loader cache was invalidated then we need to refresh the module
            if not loader.cache_has(self.__name__):
                loader.recreate_module(self)
                return self.__getattribute__(attr)

            # Only the first thread to get the lock should trigger the load
            # and reset the module's class. The rest can now getattr().
            if object.__getattribute__(self, "__class__") is BasilispLazyModule:
                if not load_state.is_waiting:
                    # If the load was a failure then raise that failure to the
                    # thread trying to access it's members
                    failure = load_state.failure
                    if failure:
                        raise failure
                    # Reentrant calls from the same thread must be allowed to
                    # proceed without triggering the load again.  exec_module()
                    # and self-referential imports are the primary ways this can
                    # happen, but in any case we must return something to avoid
                    # deadlock.
                    return object.__getattribute__(self, attr)

                load_state.loading()

                module_dict = self.__dict__

                logger.debug(f"Load '{spec.name}' triggered for '{attr}'")

                # All module metadata must be gathered from __spec__ in order to avoid
                # using mutated values.
                # Get the original name to make sure no object substitution occurred
                # in sys.modules.
                original_name = spec.name
                # Figure out exactly what attributes were mutated between the creation
                # of the module and now.
                attrs_then = spec.loader_state["__dict__"]
                attrs_now = module_dict
                attrs_updated = {}
                for key, value in attrs_now.items():
                    # Code that set an attribute may have kept a reference to the
                    # assigned object, making identity more important than equality.
                    if key not in attrs_then:
                        attrs_updated[key] = value
                    elif id(attrs_now[key]) != id(attrs_then[key]):
                        attrs_updated[key] = value

                try:
                    loader.complete_exec_module(self)
                except BaseException as e:
                    load_state.complete(e)
                    raise e

                # If exec_module() was used directly there is no guarantee the module
                # object was put into sys.modules.
                if original_name in sys.modules:
                    if id(self) != id(sys.modules[original_name]):
                        raise ValueError(
                            f"module object for {original_name!r} "
                            "substituted in sys.modules during a lazy "
                            "load"
                        )
                # Update after loading since that's what would happen in an eager
                # loading situation.
                module_dict.update(attrs_updated)
                # Stop triggering this method
                self.__class__ = BasilispModule  # type: ignore[assignment]
                self.__getattribute__(attr)
            return self.__getattribute__(attr)

    def __delattr__(self, attr):
        """Trigger the load and then perform the deletion."""
        # To trigger the load and raise an exception if the attribute
        # doesn't exist.
        self.__getattribute__(attr)
        delattr(self, attr)


class BasilispLazyImporter(BasilispImporter):
    """Python import hook to allow directly loading Basilisp code within
    Python. Modules are compiled and loaded lazily.

    Adpapted from `importlib.util.LazyLoader`
    """

    def cache_has(self, module_name: str) -> bool:
        """Return true if the module spec is cached"""
        return module_name in self._cache

    def recreate_module(self, module: BasilispModule, is_loading: bool = False):
        """Recreate a module that has gone stale. lazy modules need to be
        recreated when they are dropped from the loader cache."""
        module_name = module.__name__
        if module.__spec__ is None:
            raise ImportError(
                f"Missing loader state for stale Basilisp module {module_name}"
            )
        stale_spec = module.__spec__
        load_state = stale_spec.loader_state.get("basilisp_load_state")
        if load_state is None:
            # XXX: We lost the lock, recreate it, cross fingers, and pray.
            load_state = ModuleLoadState()
            module.__basilisp_loaded__ = load_state
        module_dict = stale_spec.loader_state.get("__dict__")
        if module_dict is None:
            module_dict = module.__dict__.copy()

        spec = self.find_spec(module_name, None, module)
        if spec is None:
            raise ModuleNotFoundError(
                f"Stale Basilisp module cannot be loaded. {module_name}"
            )
        spec.loader_state.update(
            {"__dict__": module_dict, "basilisp_load_state": load_state}
        )
        module.__file__ = spec.loader_state["filename"]
        module.__loader__ = spec.loader
        module.__package__ = spec.parent
        module.__spec__ = spec
        self._cache[spec.name] = {"spec": spec}
        if is_loading:
            load_state.loading()
        else:
            load_state.waiting()
        module.__class__ = BasilispLazyModule
        logger.debug(f"Recreated stale Basilisp module '{spec.name}'")

    def complete_exec_module(self, module: BasilispModule):
        """Load and compile the Basilisp module."""
        if not self.cache_has(module.__name__):
            self.recreate_module(module, True)
        BasilispImporter.exec_module(self, module)

    def create_module(self, spec: ModuleSpec) -> BasilispModule:
        module = BasilispImporter.create_module(self, spec)
        spec.loader_state["basilisp_load_state"] = module.__basilisp_loaded__
        module.__class__ = BasilispLazyModule
        return module

    def exec_module(self, module: types.ModuleType):
        """Apply lazy wrapper to the module then delegate to
        `BasilispImporter.exec_module`."""
        assert isinstance(module, BasilispModule)

        if os.getenv(_EAGER_IMPORT_ENVVAR, "").lower() == "true":
            module.__class__ = BasilispModule
            self.complete_exec_module(module)
            return

        logger.debug(f"Preparing module '{module.__name__}' for loading")
        if module.__spec__ is None:
            raise ImportError(f"No spec for module {module.__name__}")
        module.__spec__.loader_state.update({"__dict__": module.__dict__.copy()})
        module.__basilisp_loaded__.waiting()
        if not isinstance(module, BasilispLazyModule):
            module.__class__ = BasilispLazyModule


def hook_imports() -> None:
    """Hook into Python's import machinery with a custom Basilisp code
    importer.

    Once this is called, Basilisp code may be called from within Python code
    using standard `import module.submodule` syntax."""
    if any(isinstance(o, BasilispImporter) for o in sys.meta_path):
        return
    sys.meta_path.insert(0, BasilispLazyImporter())
