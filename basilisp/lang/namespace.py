import threading
from typing import Optional

from pyrsistent import PMap, pmap, PSet, pset

import basilisp.lang.atom as atom
import basilisp.lang.symbol as sym

_CORE_NS = 'basilisp.core'
_NAMESPACES = atom.Atom(pmap())


class Namespace:
    __slots__ = ('_name', '_mappings', '_aliases', '_imports', '_lock')

    def __init__(self, name: sym.Symbol) -> None:
        self._name = name
        self._mappings: PMap = pmap()
        self._aliases: PMap = pmap()
        self._imports: PSet = pset([sym.symbol('builtins')])
        self._lock: threading.Lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name.name

    @property
    def mappings(self) -> PMap:
        with self._lock:
            return self._mappings

    @property
    def imports(self) -> PSet:
        with self._lock:
            return self._imports

    def __repr__(self):
        return f"{self._name}"

    def __hash__(self):
        return hash(self._name)

    def add_alias(self, alias: sym.Symbol, namespace: "Namespace") -> None:
        with self._lock:
            self._aliases = self._aliases.set(alias, namespace)

    def get_alias(self, alias: sym.Symbol) -> "Namespace":
        with self._lock:
            return self._aliases[alias]

    def intern(self, sym: sym.Symbol, make_var):
        return self._intern(sym, make_var(self, sym))

    def _intern(self, sym: sym.Symbol, new_var):
        with self._lock:
            var = self._mappings.get(sym, None)
            if var is None:
                var = new_var
                self._mappings = self._mappings.set(sym, var)
            return var

    def find(self, sym: sym.Symbol):
        with self._lock:
            return self._mappings.get(sym, None)

    def add_import(self, sym: sym.Symbol) -> None:
        with self._lock:
            if sym not in self._imports:
                self._imports = self._imports.add(sym)

    def get_import(self, sym: sym.Symbol) -> Optional[sym.Symbol]:
        with self._lock:
            if sym in self._imports:
                return sym
            return None


def __import_core_mappings(ns_cache: PMap,
                           new_ns: Namespace,
                           core_ns_name=_CORE_NS) -> None:
    """Import the Core namespace mappings into the Namespace `new_ns`."""
    core_ns = ns_cache.get(sym.symbol(core_ns_name), None)
    if core_ns is None:
        raise KeyError(f"Namespace {core_ns_name} not found")
    for s, var in core_ns.mappings.items():
        new_ns._intern(s, var)


def __get_or_create(ns_cache: PMap, name: sym.Symbol,
                    core_ns_name=_CORE_NS) -> PMap:
    """Private swap function used by `get_or_create` to atomically swap
    the new namespace map into the global cache."""
    ns = ns_cache.get(name, None)
    if ns is not None:
        return ns_cache
    new_ns = Namespace(name)
    if name.name != core_ns_name:
        __import_core_mappings(ns_cache, new_ns, core_ns_name=core_ns_name)
    return ns_cache.set(name, new_ns)


def get_or_create(name: sym.Symbol,
                  ns_cache: atom.Atom = _NAMESPACES) -> Namespace:
    """Get the namespace bound to the symbol `name` in the global namespace
    cache, creating it if it does not exist.

    Return the namespace."""
    return ns_cache.swap(__get_or_create, name)[name]


def remove(name: sym.Symbol,
           ns_cache: atom.Atom = _NAMESPACES) -> Optional[Namespace]:
    """Remove the namespace bound to the symbol `name` in the global
    namespace cache and return that namespace.

    Return None if the namespace did not exist in the cache."""
    while True:
        oldval: PMap = ns_cache.deref()
        ns: Optional[Namespace] = oldval.get(name, None)
        newval = oldval
        if ns is not None:
            newval = oldval.discard(name)
        if ns_cache.compare_and_set(oldval, newval):
            return ns
