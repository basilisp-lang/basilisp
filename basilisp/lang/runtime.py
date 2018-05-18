import threading
from typing import Optional

import pkg_resources
from pyrsistent import pmap, PMap, PSet, pset

import basilisp.lang.symbol as sym
from basilisp.lang import atom
from basilisp.lang.maybe import Maybe
from basilisp.util import Maybe

_CORE_NS = 'basilisp.core'
_CORE_NS_FILE = 'core.lpy'
_REPL_DEFAULT_NS = 'user'
_NS_VAR_NAME = '*ns*'
_NS_VAR_NS = _CORE_NS
_PYTHON_PACKAGE_NAME = 'basilisp'
_PRINT_GENERATED_PY_VAR_NAME = '*print-generated-python*'


class Var:
    __slots__ = ('_name', '_ns', '_root', '_dynamic', '_tl', '_meta')

    def __init__(self,
                 ns: "Namespace",
                 name: sym.Symbol,
                 dynamic: bool = False,
                 meta=None) -> None:
        self._ns = ns
        self._name = name
        self._root = None
        self._dynamic = dynamic
        self._tl = threading.local()
        self._meta = meta

    def __repr__(self):
        return f"#'{self.ns.name}/{self.name}"

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, meta):
        self._meta = meta

    @property
    def ns(self) -> "Namespace":
        return self._ns

    @property
    def name(self) -> sym.Symbol:
        return self._name

    @property
    def dynamic(self) -> bool:
        return self._dynamic

    @dynamic.setter
    def dynamic(self, is_dynamic: bool):
        self._dynamic = is_dynamic

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, val):
        self._root = val

    def push_bindings(self, val):
        if not hasattr(self._tl, 'bindings'):
            self._tl.bindings = []
        self._tl.bindings.append(val)

    def pop_bindings(self):
        return self._tl.bindings.pop()

    @property
    def value(self):
        if self._dynamic and hasattr(
                self._tl, 'bindings') and len(self._tl.bindings) > 0:
            return self._tl.bindings[-1]
        return self._root

    @value.setter
    def value(self, v):
        if self._dynamic and hasattr(
                self._tl, 'bindings') and len(self._tl.bindings) > 0:
            self._tl.bindings[-1] = v
        self._root = v

    @staticmethod
    def intern(ns: sym.Symbol,
               name: sym.Symbol,
               val,
               dynamic: bool = False,
               meta=None) -> "Var":
        """Intern the value bound to the symbol `name` in namespace `ns`.

        This function uses a dirty hack of hiding the `Var` constructor in
        a lambda to avoid circular dependencies between the var and namespace
        modules. Shameful."""
        var_ns = Namespace.get_or_create(ns)
        var = var_ns.intern(name,
                            lambda ns, name: Var(ns, name, dynamic=dynamic))
        var.root = val
        var.meta = meta
        return var

    @staticmethod
    def find_in_ns(ns_sym: sym.Symbol, name_sym: sym.Symbol) -> "Var":
        """Return the value current bound to the name `name_sym` in the namespace
        specified by `ns_sym`."""
        ns = Namespace.get_or_create(ns_sym)
        return ns.find(name_sym)

    @staticmethod
    def find(ns_qualified_sym: sym.Symbol) -> "Var":
        """Return the value currently bound to the name in the namespace specified
        by `ns_qualified_sym`."""
        ns = Maybe(ns_qualified_sym.ns).or_else_raise(
            lambda: ValueError(f"Namespace must be specified in Symbol {ns_qualified_sym}")
        )
        ns_sym = sym.symbol(ns)
        name_sym = sym.symbol(ns_qualified_sym.name)
        return Var.find_in_ns(ns_sym, name_sym)


class Namespace:
    _NAMESPACES = atom.Atom(pmap())

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

    @classmethod
    def ns_cache(cls) -> PMap:
        """Return a snapshot of the Namespace cache."""
        return cls._NAMESPACES.deref()

    @staticmethod
    def __import_core_mappings(ns_cache: PMap,
                               new_ns: "Namespace",
                               core_ns_name=_CORE_NS) -> None:
        """Import the Core namespace mappings into the Namespace `new_ns`."""
        core_ns = ns_cache.get(sym.symbol(core_ns_name), None)
        if core_ns is None:
            raise KeyError(f"Namespace {core_ns_name} not found")
        for s, var in core_ns.mappings.items():
            new_ns._intern(s, var)

    @staticmethod
    def __get_or_create(ns_cache: PMap,
                        name: sym.Symbol,
                        core_ns_name=_CORE_NS) -> PMap:
        """Private swap function used by `get_or_create` to atomically swap
        the new namespace map into the global cache."""
        ns = ns_cache.get(name, None)
        if ns is not None:
            return ns_cache
        new_ns = Namespace(name)
        if name.name != core_ns_name:
            Namespace.__import_core_mappings(
                ns_cache, new_ns, core_ns_name=core_ns_name)
        return ns_cache.set(name, new_ns)

    @classmethod
    def get_or_create(cls, name: sym.Symbol) -> "Namespace":
        """Get the namespace bound to the symbol `name` in the global namespace
        cache, creating it if it does not exist.

        Return the namespace."""
        return cls._NAMESPACES.swap(Namespace.__get_or_create, name)[name]

    @classmethod
    def remove(cls, name: sym.Symbol) -> Optional["Namespace"]:
        """Remove the namespace bound to the symbol `name` in the global
        namespace cache and return that namespace.

        Return None if the namespace did not exist in the cache."""
        while True:
            oldval: PMap = cls._NAMESPACES.deref()
            ns: Optional[Namespace] = oldval.get(name, None)
            newval = oldval
            if ns is not None:
                newval = oldval.discard(name)
            if cls._NAMESPACES.compare_and_set(oldval, newval):
                return ns


def init_ns_var(which_ns: str = _CORE_NS,
                ns_var_name: str = _NS_VAR_NAME) -> Var:
    """Initialize the dynamic `*ns*` variable in the Namespace `which_ns`."""
    core_sym = sym.Symbol(which_ns)
    core_ns = Namespace.get_or_create(core_sym)
    ns_var = Var.intern(
        core_sym, sym.Symbol(ns_var_name), core_ns, dynamic=True)
    return ns_var


def set_current_ns(ns_name: str,
                   ns_var_name: str = _NS_VAR_NAME,
                   ns_var_ns: str = _NS_VAR_NS) -> Var:
    """Set the value of the dynamic variable `*ns*` in the current thread."""
    symbol = sym.Symbol(ns_name)
    ns = Namespace.get_or_create(symbol)
    ns_var = Var.find(sym.Symbol(ns_var_name, ns=ns_var_ns))
    ns_var.push_bindings(ns)
    return ns_var


def get_current_ns(ns_var_name: str = _NS_VAR_NAME,
                   ns_var_ns: str = _NS_VAR_NS) -> Var:
    """Set the value of the dynamic variable `*ns*` in the current thread."""
    ns_sym = sym.Symbol(ns_var_name, ns=ns_var_ns)
    return Var.find(ns_sym)


def print_generated_python(var_name: str = _PRINT_GENERATED_PY_VAR_NAME,
                           core_ns_name: str = _CORE_NS) -> bool:
    """Return the value of the `*print-generated-python*` dynamic variable."""
    ns_sym = sym.Symbol(var_name, ns=core_ns_name)
    return Var.find(ns_sym).value


def core_resource(package: str = _PYTHON_PACKAGE_NAME,
                  resource: str = _CORE_NS_FILE) -> str:
    return pkg_resources.resource_filename(package, resource)


def bootstrap(ns_var_name: str = _NS_VAR_NAME,
              core_ns_name: str = _CORE_NS) -> None:
    """Bootstrap the environment with functions that are are difficult to
    express with the very minimal lisp environment."""
    core_ns_sym = sym.symbol(core_ns_name)
    ns_var_sym = sym.symbol(ns_var_name, ns=core_ns_name)
    __NS = Var.find(ns_var_sym)

    def set_BANG_(var_sym: sym.Symbol, expr):
        ns = Maybe(var_sym.ns).or_else(lambda: __NS.value.name)
        name = var_sym.name

        v = Var.find(sym.symbol(name, ns=ns))
        v.value = expr
        return expr

    def in_ns(s: sym.Symbol):
        ns = Namespace.get_or_create(s)
        set_BANG_(ns_var_sym, ns)
        return ns

    Var.intern(core_ns_sym, sym.symbol('set!'), set_BANG_)
    Var.intern(core_ns_sym, sym.symbol('in-ns'), in_ns)
    Var.intern(
        core_ns_sym,
        sym.symbol(_PRINT_GENERATED_PY_VAR_NAME),
        True,
        dynamic=True)
