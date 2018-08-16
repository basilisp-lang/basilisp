import itertools
import threading
import types
from typing import Optional

from functional import seq
from pyrsistent import pmap, PMap, PSet, pset

import basilisp.lang.list as llist
import basilisp.lang.seq as lseq
import basilisp.lang.symbol as sym
from basilisp.lang import atom
from basilisp.util import Maybe

_CORE_NS = 'basilisp.core'
_REPL_DEFAULT_NS = 'user'
_NS_VAR_NAME = '*ns*'
_NS_VAR_NS = _CORE_NS
_PYTHON_PACKAGE_NAME = 'basilisp'
_PRINT_GENERATED_PY_VAR_NAME = '*print-generated-python*'


def _new_module(name: str, doc=None) -> types.ModuleType:
    """Create a new empty Basilisp Python module.

    Modules are created for each Namespace when it is created."""
    mod = types.ModuleType(name, doc=doc)
    mod.__loader__ = None
    mod.__package__ = None
    mod.__spec__ = None
    mod.__basilisp_bootstrapped__ = False  # type: ignore
    return mod


class RuntimeException(Exception):
    pass


class Var:
    __slots__ = ('_name', '_ns', '_root', '_dynamic', '_is_bound', '_tl', '_meta')

    def __init__(self,
                 ns: "Namespace",
                 name: sym.Symbol,
                 dynamic: bool = False,
                 meta=None) -> None:
        self._ns = ns
        self._name = name
        self._root = None
        self._dynamic = dynamic
        self._is_bound = False
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
        self._is_bound = True
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
        """Intern the value bound to the symbol `name` in namespace `ns`."""
        var_ns = Namespace.get_or_create(ns)
        var = var_ns.intern(name, Var(var_ns, name, dynamic=dynamic))
        var.root = val
        var.meta = meta
        return var

    @staticmethod
    def intern_unbound(ns: sym.Symbol,
                       name: sym.Symbol,
                       dynamic: bool = False,
                       meta=None) -> "Var":
        """Create a new unbound `Var` instance to the symbol `name` in namespace `ns`."""
        var_ns = Namespace.get_or_create(ns)
        var = var_ns.intern(name, Var(var_ns, name, dynamic=dynamic))
        var.meta = meta
        return var

    @staticmethod
    def find_in_ns(ns_sym: sym.Symbol, name_sym: sym.Symbol) -> "Optional[Var]":
        """Return the value current bound to the name `name_sym` in the namespace
        specified by `ns_sym`."""
        ns = Namespace.get_or_create(ns_sym)
        return ns.find(name_sym)

    @staticmethod
    def find(ns_qualified_sym: sym.Symbol) -> "Optional[Var]":
        """Return the value currently bound to the name in the namespace specified
        by `ns_qualified_sym`."""
        ns = Maybe(ns_qualified_sym.ns).or_else_raise(
            lambda: ValueError(f"Namespace must be specified in Symbol {ns_qualified_sym}")
        )
        ns_sym = sym.symbol(ns)
        name_sym = sym.symbol(ns_qualified_sym.name)
        return Var.find_in_ns(ns_sym, name_sym)


class Namespace:
    """Namespaces serve as organizational units in Basilisp code, just as
    they do in Clojure code. Vars are mutable containers for functions and
    data which may be interned in a namespace and referred to by a Symbol.

    Namespaces additionally may have aliases to other namespaces, so code
    organized in one namespace may conveniently refer to code or data in
    other namespaces using that alias as the Symbol's namespace.

    Namespaces are constructed def-by-def as Basilisp reads in each form
    in a file (which will typically declare a namespace at the top).

    Namespaces have the following fields of interest:

    - `mappings` is a mapping between a symbolic name and a Var. The
      Var may point to code, data, or nothing, if it is unbound.

    - `aliases` is a mapping between a symbolic alias and another
      Namespace. The fully qualified name of a namespace is also
      an alias for itself.

    - `imports` is a set of Python modules imported into the current
      namespace"""
    DEFAULT_IMPORTS = atom.Atom(pset(seq(['builtins',
                                          'basilisp.lang.exception',
                                          'basilisp.lang.keyword',
                                          'basilisp.lang.list',
                                          'basilisp.lang.map',
                                          'basilisp.lang.runtime',
                                          'basilisp.lang.seq',
                                          'basilisp.lang.set',
                                          'basilisp.lang.symbol',
                                          'basilisp.lang.vector',
                                          'basilisp.lang.util'])
                                     .map(sym.symbol)))
    GATED_IMPORTS = pset(['basilisp.core'])
    _NAMESPACES = atom.Atom(pmap())

    __slots__ = ('_name', '_module', '_mappings', '_refers', '_aliases', '_imports')

    def __init__(self, name: sym.Symbol, module: types.ModuleType = None) -> None:
        self._name = name
        self._module = Maybe(module).or_else(lambda: _new_module(name.as_python_sym()))
        self._mappings: atom.Atom = atom.Atom(pmap())
        self._aliases: atom.Atom = atom.Atom(pmap())
        self._imports: atom.Atom = atom.Atom(pset(Namespace.DEFAULT_IMPORTS.deref()))

    @classmethod
    def add_default_import(cls, module: str):
        """Add a gated default import to the default imports.

        In particular, we need to avoid importing 'basilisp.core' before we have
        finished macro-expanding."""
        if module in cls.GATED_IMPORTS:
            cls.DEFAULT_IMPORTS.swap(lambda s: s.add(sym.symbol(module)))

    @property
    def name(self) -> str:
        return self._name.name

    @property
    def module(self):
        return self._module

    @module.setter
    def module(self, m: types.ModuleType):
        """Override the Python module for this Namespace.

        This should only be done by basilisp.importer code to make sure the
        correct module is generated for `basilisp.core`."""
        self._module = m

    @property
    def aliases(self) -> PMap:
        return self._aliases.deref()

    @property
    def mappings(self) -> PMap:
        return self._mappings.deref()

    @property
    def imports(self) -> PSet:
        return self._imports.deref()

    def __repr__(self):
        return f"{self._name}"

    def __hash__(self):
        return hash(self._name)

    def add_alias(self, alias: sym.Symbol, namespace: "Namespace") -> None:
        """Add a Symbol alias for the given Namespace."""
        self._aliases.swap(lambda m: m.set(alias, namespace))

    def get_alias(self, alias: sym.Symbol) -> "Optional[Namespace]":
        """Get the Namespace aliased by Symbol or None if it does not exist."""
        return self.aliases.get(alias, None)

    def intern(self, sym: sym.Symbol, var: Var, force: bool = False) -> Var:
        """Intern the Var given in this namespace mapped by the given Symbol.

        If the Symbol already maps to a Var, this method _will not overwrite_
        the existing Var mapping unless the force keyword argument is given
        and is True."""
        m: PMap = self._mappings.swap(Namespace._intern, sym, var, force=force)
        return m.get(sym)

    @staticmethod
    def _intern(m: PMap, sym: sym.Symbol, new_var: Var,
                force: bool = False) -> PMap:
        """Swap function used by intern to atomically intern a new variable in
        the symbol mapping for this Namespace."""
        var = m.get(sym, None)
        if var is None or force:
            return m.set(sym, new_var)
        return m

    def find(self, sym: sym.Symbol) -> Optional[Var]:
        """Find Vars mapped by the given Symbol input or None if no Vars are
        mapped by that Symbol."""
        return self.mappings.get(sym, None)

    def add_import(self, sym: sym.Symbol) -> None:
        """Add the Symbol as an imported Symbol in this Namespace."""
        self._imports.swap(lambda s: s.add(sym))

    def get_import(self, sym: sym.Symbol) -> Optional[sym.Symbol]:
        """Return the Symbol if it is imported into this Namespace, None otherwise."""
        if sym in self.imports:
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
            new_ns.intern(s, var)

    @staticmethod
    def __get_or_create(ns_cache: PMap,
                        name: sym.Symbol,
                        module: types.ModuleType = None,
                        core_ns_name=_CORE_NS) -> PMap:
        """Private swap function used by `get_or_create` to atomically swap
        the new namespace map into the global cache."""
        ns = ns_cache.get(name, None)
        if ns is not None:
            return ns_cache
        new_ns = Namespace(name, module=module)
        if name.name != core_ns_name:
            Namespace.__import_core_mappings(
                ns_cache, new_ns, core_ns_name=core_ns_name)
        return ns_cache.set(name, new_ns)

    @classmethod
    def get_or_create(cls, name: sym.Symbol, module: types.ModuleType = None) -> "Namespace":
        """Get the namespace bound to the symbol `name` in the global namespace
        cache, creating it if it does not exist.

        Return the namespace."""
        return cls._NAMESPACES.swap(Namespace.__get_or_create, name, module=module)[name]

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


###################
# Runtime Support #
###################


def first(o):
    """If o is a Seq, return the first element from o. If o is None, return
    None. Otherwise, coerces o to a Seq and returns the first."""
    if o is None:
        return None
    if isinstance(o, lseq.Seq):
        return o.first
    return to_seq(o).first


def rest(o):
    """If o is a Seq, return the elements after the first in o. If o is None,
    returns an empty seq. Otherwise, coerces o to a seq and returns the rest."""
    if o is None:
        return None
    if isinstance(o, lseq.Seq):
        s = o.rest
        if s is None:
            return llist.List.empty()
        return s
    s = to_seq(o).rest
    if s is None:
        return llist.List.empty()
    return s


def next(o):  # pylint:disable=redefined-builtin
    """Calls rest on o. If o returns an empty sequence or None, returns None.
    Otherwise, returns the elements after the first in o."""
    s = rest(o)
    if not s:
        return None
    return s


def cons(o, seq) -> lseq.Seq:
    """Creates a new sequence where o is the first element and seq is the rest.
    If seq is None, return a list containing o. If seq is not a Seq, attempt
    to coerce it to a Seq and then cons o onto the resulting sequence."""
    if seq is None:
        return llist.l(o)
    if isinstance(seq, lseq.Seq):
        return seq.cons(o)
    return to_seq(seq).cons(o)  # type: ignore


def _seq_or_nil(s: lseq.Seq) -> Optional[lseq.Seq]:
    """Return None if a Seq is empty, the Seq otherwise."""
    if s.first is None and s.rest == lseq.empty():
        return None
    return s


def to_seq(o) -> Optional[lseq.Seq]:
    """Coerce the argument o to a Seq. If o is None, return None."""
    if o is None:
        return None
    if isinstance(o, lseq.Seq):
        return _seq_or_nil(o)
    if isinstance(o, lseq.Seqable):
        return _seq_or_nil(o.seq())
    return _seq_or_nil(lseq.sequence(o))


def concat(*seqs) -> lseq.Seq:
    """Concatenate the sequences given by seqs into a single Seq."""
    allseqs = lseq.sequence(itertools.chain(*filter(None, map(to_seq, seqs))))
    if allseqs is None:
        return lseq.empty()
    return allseqs


def apply(f, args):
    """Apply function f to the arguments provided.

    The last argument must always be coercible to a Seq. Intermediate
    arguments are not modified.

    For example:
        (apply max [1 2 3])   ;=> 3
        (apply max 4 [1 2 3]) ;=> 4"""
    final = list(args[:-1])
    try:
        final.extend(to_seq(args[-1]))
    except TypeError:
        pass
    return f(*final)


def _collect_args(args) -> lseq.Seq:
    """Collect Python starred arguments into a Basilisp list."""
    if isinstance(args, tuple):
        return llist.list(args)
    raise TypeError("Python variadic arguments should always be a tuple")


#########################
# Bootstrap the Runtime #
#########################

def init_ns_var(which_ns: str = _CORE_NS,
                ns_var_name: str = _NS_VAR_NAME) -> Var:
    """Initialize the dynamic `*ns*` variable in the Namespace `which_ns`."""
    core_sym = sym.Symbol(which_ns)
    core_ns = Namespace.get_or_create(core_sym)
    ns_var = Var.intern(
        core_sym, sym.Symbol(ns_var_name), core_ns, dynamic=True)
    return ns_var


def set_current_ns(ns_name: str,
                   module: types.ModuleType = None,
                   ns_var_name: str = _NS_VAR_NAME,
                   ns_var_ns: str = _NS_VAR_NS) -> Var:
    """Set the value of the dynamic variable `*ns*` in the current thread."""
    symbol = sym.Symbol(ns_name)
    ns = Namespace.get_or_create(symbol, module=module)
    ns_var = Maybe(Var.find(sym.Symbol(ns_var_name, ns=ns_var_ns))) \
        .or_else_raise(lambda: RuntimeException(f"Dynamic Var {sym.Symbol(ns_var_name, ns=ns_var_ns)} not bound!"))
    ns_var.push_bindings(ns)
    return ns_var


def get_current_ns(ns_var_name: str = _NS_VAR_NAME,
                   ns_var_ns: str = _NS_VAR_NS) -> Namespace:
    """Get the value of the dynamic variable `*ns*` in the current thread."""
    ns_sym = sym.Symbol(ns_var_name, ns=ns_var_ns)
    ns: Namespace = Maybe(Var.find(ns_sym)) \
        .map(lambda v: v.value) \
        .or_else_raise(lambda: RuntimeException(f"Dynamic Var {ns_sym} not bound!"))
    return ns


def resolve_alias(s: sym.Symbol) -> sym.Symbol:
    """Resolve the aliased symbol in the current namespace."""
    ns = get_current_ns()
    if s.ns is not None:
        aliased_ns = ns.get_alias(sym.symbol(s.ns))
        if aliased_ns is not None:
            return sym.symbol(s.name, aliased_ns.name)
        else:
            return s
    else:
        which_var = ns.find(sym.symbol(s.name))
        if which_var is not None:
            return sym.symbol(which_var.name.name, which_var.ns.name)
        else:
            return sym.symbol(s.name, ns=ns.name)


def print_generated_python(var_name: str = _PRINT_GENERATED_PY_VAR_NAME,
                           core_ns_name: str = _CORE_NS) -> bool:
    """Return the value of the `*print-generated-python*` dynamic variable."""
    ns_sym = sym.Symbol(var_name, ns=core_ns_name)
    return Maybe(Var.find(ns_sym)) \
        .map(lambda v: v.value) \
        .or_else_raise(lambda: RuntimeException(f"Dynamic Var {ns_sym} not bound!"))


def bootstrap(ns_var_name: str = _NS_VAR_NAME,
              core_ns_name: str = _CORE_NS) -> None:
    """Bootstrap the environment with functions that are are difficult to
    express with the very minimal lisp environment."""
    core_ns_sym = sym.symbol(core_ns_name)
    ns_var_sym = sym.symbol(ns_var_name, ns=core_ns_name)
    __NS = Maybe(Var.find(ns_var_sym)) \
        .or_else_raise(lambda: RuntimeException(f"Dynamic Var {ns_var_sym} not bound!"))

    def set_BANG_(var_sym: sym.Symbol, expr):
        ns = Maybe(var_sym.ns).or_else(lambda: __NS.value.name)
        name = var_sym.name

        v = Maybe(Var.find(sym.symbol(name, ns=ns))) \
            .or_else_raise(lambda: RuntimeException(f"Var {ns_var_sym} not bound!"))
        v.value = expr
        return expr

    def in_ns(s: sym.Symbol):
        ns = Namespace.get_or_create(s)
        set_BANG_(ns_var_sym, ns)
        return ns

    Var.intern_unbound(core_ns_sym, sym.symbol('unquote'))
    Var.intern_unbound(core_ns_sym, sym.symbol('unquote-splicing'))
    Var.intern(core_ns_sym, sym.symbol('set!'), set_BANG_)
    Var.intern(core_ns_sym, sym.symbol('in-ns'), in_ns)
    Var.intern(
        core_ns_sym,
        sym.symbol(_PRINT_GENERATED_PY_VAR_NAME),
        True,
        dynamic=True)
