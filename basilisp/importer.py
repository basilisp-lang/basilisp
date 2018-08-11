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
        mod = super().create_module(spec)
        self._cache[spec.name] = {"module": mod, "spec": spec}
        return mod

    def get_code(self, fullname: str) -> types.CodeType:
        runtime.set_current_ns(fullname)
        cached = self._cache[fullname]
        spec = cached["spec"]
        filename = spec.loader_state["filename"]
        forms = reader.read_file(filename)
        bytecode = compiler.compile_module_bytecode(forms, compiler.CompilerContext(), filename)
        runtime.Namespace.add_default_import(fullname)
        return bytecode


def hook_imports():
    """Hook into Python's import machinery with a custom Basilisp code
    importer.

    Once this is called, Basilisp code may be called from within Python code
    using standard `import module.submodule` syntax."""
    sys.meta_path.insert(0, BasilispImporter())
