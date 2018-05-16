import threading

import basilisp.lang.atom as atom
import basilisp.lang.namespace as namespace
import basilisp.lang.symbol as sym
from basilisp.lang.maybe import Maybe


class Var:
    __slots__ = ('_name', '_ns', '_root', '_dynamic', '_tl', '_meta')

    def __init__(self,
                 ns: namespace.Namespace,
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
    def ns(self) -> namespace.Namespace:
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


def intern(ns: sym.Symbol,
           name: sym.Symbol,
           val,
           ns_cache: atom.Atom = namespace._NAMESPACES,
           dynamic: bool = False,
           meta=None) -> Var:
    """Intern the value bound to the symbol `name` in namespace `ns`.

    This function uses a dirty hack of hiding the `Var` constructor in
    a lambda to avoid circular dependencies between the var and namespace
    modules. Shameful."""
    var_ns = namespace.get_or_create(ns, ns_cache=ns_cache)
    var = var_ns.intern(name, lambda ns, name: Var(ns, name, dynamic=dynamic))
    var.root = val
    var.meta = meta
    return var


def find_in_ns(ns_sym: sym.Symbol,
               name_sym: sym.Symbol,
               ns_cache: atom.Atom = namespace._NAMESPACES) -> Var:
    """Return the value current bound to the name `name_sym` in the namespace
    specified by `ns_sym`."""
    ns = namespace.get_or_create(ns_sym, ns_cache=ns_cache)
    return ns.find(name_sym)


def find(ns_qualified_sym: sym.Symbol,
         ns_cache: atom.Atom = namespace._NAMESPACES) -> Var:
    """Return the value currently bound to the name in the namespace specified
    by `ns_qualified_sym`."""
    ns = Maybe(ns_qualified_sym.ns).or_else_raise(
        lambda: ValueError(f"Namespace must be specified in Symbol {ns_qualified_sym}")
    )
    ns_sym = sym.symbol(ns)
    name_sym = sym.symbol(ns_qualified_sym.name)
    return find_in_ns(ns_sym, name_sym, ns_cache=ns_cache)
