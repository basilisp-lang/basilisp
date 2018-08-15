import importlib.machinery
import importlib.util
import os.path
import sys
import types
from importlib.abc import MetaPathFinder, SourceLoader
from typing import Optional

import basilisp.compiler as compiler
import basilisp.lang.runtime as runtime
import basilisp.reader as reader


class BasilispImporter(MetaPathFinder, SourceLoader):
    """Python import hook to allow directly loading Basilisp code within
    Python."""

    def __init__(self):
        self._cache = {}

    def find_spec(self,
                  fullname: str,
                  path,  # Optional[List[str]] # MyPy complains this is incompatible with supertype
                  target: types.ModuleType = None) -> Optional[importlib.machinery.ModuleSpec]:
        """Find the ModuleSpec for the specified Basilisp module.

        Returns None if the module is not a Basilisp module to allow import processing to continue."""
        package_components = fullname.split('.')
        if path is None:
            path = sys.path
            module_name = package_components
        else:
            module_name = [package_components[-1]]

        for entry in path:
            filenames = [f"{os.path.join(entry, *module_name, '__init__')}.lpy",
                         f"{os.path.join(entry, *module_name)}.lpy"]
            for filename in filenames:
                if os.path.exists(filename):
                    state = {'fullname': fullname, "filename": filename, 'path': entry, 'target': target}
                    return importlib.machinery.ModuleSpec(fullname, self, origin=filename, loader_state=state)
        return None

    def invalidate_caches(self):
        super().invalidate_caches()
        self._cache = {}

    def get_data(self, path) -> bytes:
        with open(path, mode='r+b') as f:
            return f.read()

    def get_filename(self, fullname: str) -> str:
        try:
            cached = self._cache[fullname]
        except KeyError:
            raise ImportError(f"Could not import module '{fullname}'")
        spec = cached["spec"]
        return spec.loader_state.filename

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        mod = types.ModuleType(spec.name)
        mod.__loader__ = spec.loader
        mod.__package__ = spec.parent
        mod.__spec__ = spec
        self._cache[spec.name] = {"spec": spec}
        return mod

    def exec_module(self, module):
        """Compile the Basilisp module into Python code.

        Basilisp is fundamentally a form-at-a-time compilation, meaning that
        each form in a module may require code compiled from an earlier form, so
        we incrementally compile a Python module by evaluating a single top-level
        form at a time and inserting the resulting AST nodes into the Pyton module."""
        fullname = module.__name__
        cached = self._cache[fullname]
        cached["module"] = module
        spec = cached["spec"]
        filename = spec.loader_state["filename"]

        # During the bootstrapping process, the 'basilisp.core namespace is created with
        # a blank module. If we do not replace the module here with the module we are
        # generating, then we will not be able to use advanced compilation features such
        # as direct Python variable access to functions and other def'ed values.
        ns: runtime.Namespace = runtime.set_current_ns(fullname).value
        ns.module = module

        forms = reader.read_file(filename)
        compiler.compile_module(forms, compiler.CompilerContext(), module, filename)

        # Because we want to (by default) add 'basilisp.core into every namespace by default,
        # we want to make sure we don't try to add 'basilisp.core into itself, causing a
        # circular import error.
        #
        # Later on, we can probably remove this and just use the 'ns macro to auto-refer
        # all 'basilisp.core values into the current namespace.
        runtime.Namespace.add_default_import(fullname)


def hook_imports():
    """Hook into Python's import machinery with a custom Basilisp code
    importer.

    Once this is called, Basilisp code may be called from within Python code
    using standard `import module.submodule` syntax."""
    sys.meta_path.insert(0, BasilispImporter())  # pylint:disable=abstract-class-instantiated
