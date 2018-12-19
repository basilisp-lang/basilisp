import contextlib
import functools
import importlib
import itertools
import logging
import math
import threading
import types
from fractions import Fraction
from typing import Optional, Dict, Tuple

from functional import seq

import basilisp.lang.associative as lassoc
import basilisp.lang.collection as lcoll
import basilisp.lang.deref as lderef
import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.obj as lobj
import basilisp.lang.seq as lseq
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
from basilisp.lang import atom
from basilisp.lang.typing import LispNumber
from basilisp.util import Maybe

logger = logging.getLogger(__name__)

_CORE_NS = "basilisp.core"
_REPL_DEFAULT_NS = "user"
_NS_VAR_NAME = "*ns*"
_NS_VAR_NS = _CORE_NS
_PYTHON_PACKAGE_NAME = "basilisp"
_GENERATED_PYTHON_VAR_NAME = "*generated-python*"
_PRINT_GENERATED_PY_VAR_NAME = "*print-generated-python*"
_PRINT_DUP_VAR_NAME = "*print-dup*"
_PRINT_LENGTH_VAR_NAME = "*print-length*"
_PRINT_LEVEL_VAR_NAME = "*print-level*"
_PRINT_META_VAR_NAME = "*print-meta*"
_PRINT_READABLY_VAR_NAME = "*print-readably*"


_DYNAMIC_META_KEY = kw.keyword("dynamic")
_PRIVATE_META_KEY = kw.keyword("private")
_REDEF_META_KEY = kw.keyword("redef")

_CATCH = sym.symbol("catch")
_DEF = sym.symbol("def")
_DO = sym.symbol("do")
_FINALLY = sym.symbol("finally")
_FN = sym.symbol("fn*")
_IF = sym.symbol("if")
_IMPORT = sym.symbol("import*")
_INTEROP_CALL = sym.symbol(".")
_INTEROP_PROP = sym.symbol(".-")
_LET = sym.symbol("let*")
_LOOP = sym.symbol("loop*")
_QUOTE = sym.symbol("quote")
_RECUR = sym.symbol("recur")
_THROW = sym.symbol("throw")
_TRY = sym.symbol("try")
_VAR = sym.symbol("var")
_SPECIAL_FORMS = lset.s(
    _CATCH,
    _DEF,
    _DO,
    _FINALLY,
    _FN,
    _IF,
    _IMPORT,
    _INTEROP_CALL,
    _INTEROP_PROP,
    _LET,
    _LOOP,
    _QUOTE,
    _RECUR,
    _THROW,
    _TRY,
    _VAR,
)


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
    __slots__ = ("_name", "_ns", "_root", "_dynamic", "_is_bound", "_tl", "_meta")

    def __init__(
        self, ns: "Namespace", name: sym.Symbol, dynamic: bool = False, meta=None
    ) -> None:
        self._ns = ns
        self._name = name
        self._root = None
        self._dynamic = dynamic
        self._is_bound = False
        self._tl = threading.local()
        self._meta = meta

        if dynamic:
            # If this var was created with the dynamic keyword argument, then the
            # Var metadata should also specify that the Var is dynamic.
            if isinstance(self._meta, lmap.Map):
                if not self._meta.entry(_DYNAMIC_META_KEY):
                    self._meta = self._meta.assoc(_DYNAMIC_META_KEY, True)
            else:
                self._meta = lmap.map({_DYNAMIC_META_KEY: True})

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
    def is_private(self) -> Optional[bool]:
        try:
            return self.meta.entry(_PRIVATE_META_KEY)
        except AttributeError:
            return False

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, val):
        self._is_bound = True
        self._root = val

    def push_bindings(self, val):
        if not hasattr(self._tl, "bindings"):
            self._tl.bindings = []
        self._tl.bindings.append(val)

    def pop_bindings(self):
        return self._tl.bindings.pop()

    @property
    def value(self):
        if (
            self._dynamic
            and hasattr(self._tl, "bindings")
            and len(self._tl.bindings) > 0
        ):
            return self._tl.bindings[-1]
        return self._root

    @value.setter
    def value(self, v):
        if (
            self._dynamic
            and hasattr(self._tl, "bindings")
            and len(self._tl.bindings) > 0
        ):
            self._tl.bindings[-1] = v
            return
        self._root = v

    @staticmethod
    def intern(
        ns: sym.Symbol, name: sym.Symbol, val, dynamic: bool = False, meta=None
    ) -> "Var":
        """Intern the value bound to the symbol `name` in namespace `ns`."""
        var_ns = Namespace.get_or_create(ns)
        var = var_ns.intern(name, Var(var_ns, name, dynamic=dynamic, meta=meta))
        var.root = val
        return var

    @staticmethod
    def intern_unbound(
        ns: sym.Symbol, name: sym.Symbol, dynamic: bool = False, meta=None
    ) -> "Var":
        """Create a new unbound `Var` instance to the symbol `name` in namespace `ns`."""
        var_ns = Namespace.get_or_create(ns)
        return var_ns.intern(name, Var(var_ns, name, dynamic=dynamic, meta=meta))

    @staticmethod
    def find_in_ns(ns_sym: sym.Symbol, name_sym: sym.Symbol) -> "Optional[Var]":
        """Return the value current bound to the name `name_sym` in the namespace
        specified by `ns_sym`."""
        ns = Namespace.get(ns_sym)
        if ns:
            return ns.find(name_sym)
        return None

    @staticmethod
    def find(ns_qualified_sym: sym.Symbol) -> "Optional[Var]":
        """Return the value currently bound to the name in the namespace specified
        by `ns_qualified_sym`."""
        ns = Maybe(ns_qualified_sym.ns).or_else_raise(
            lambda: ValueError(
                f"Namespace must be specified in Symbol {ns_qualified_sym}"
            )
        )
        ns_sym = sym.symbol(ns)
        name_sym = sym.symbol(ns_qualified_sym.name)
        return Var.find_in_ns(ns_sym, name_sym)

    @staticmethod
    def find_safe(ns_qualified_sym: sym.Symbol) -> "Var":
        """Return the Var currently bound to the name in the namespace specified
        by `ns_qualified_sym`. If no Var is bound to that name, raise an exception.

        This is a utility method to return useful debugging information when code
        refers to an invalid symbol at runtime."""
        v = Var.find(ns_qualified_sym)
        if v is None:
            raise RuntimeException(
                f"Unable to resolve symbol {ns_qualified_sym} in this context"
            )
        return v


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
    - `aliases` is a mapping between a symbolic alias and another
      Namespace. The fully qualified name of a namespace is also
      an alias for itself.

    - `imports` is a mapping of names to Python modules imported
      into the current namespace.

    - `interns` is a mapping between a symbolic name and a Var. The
      Var may point to code, data, or nothing, if it is unbound. Vars
      in `interns` are interned in _this_ namespace.

    - `refers` is a mapping between a symbolic name and a Var. Vars in
      `refers` are interned in another namespace and are only referred
      to without an alias in this namespace.
    """

    DEFAULT_IMPORTS = atom.Atom(
        lset.set(
            seq(
                [
                    "builtins",
                    "io",
                    "operator",
                    "sys",
                    "basilisp.lang.atom",
                    "basilisp.lang.compiler",
                    "basilisp.lang.delay",
                    "basilisp.lang.exception",
                    "basilisp.lang.keyword",
                    "basilisp.lang.list",
                    "basilisp.lang.map",
                    "basilisp.lang.multifn",
                    "basilisp.lang.reader",
                    "basilisp.lang.runtime",
                    "basilisp.lang.seq",
                    "basilisp.lang.set",
                    "basilisp.lang.symbol",
                    "basilisp.lang.vector",
                    "basilisp.lang.util",
                ]
            )
            .map(sym.symbol)
            .to_list()
        )
    )
    GATED_IMPORTS = lset.set(["basilisp.core"])

    _NAMESPACES = atom.Atom(lmap.Map.empty())

    __slots__ = (
        "_name",
        "_module",
        "_interns",
        "_refers",
        "_aliases",
        "_imports",
        "_import_aliases",
    )

    def __init__(self, name: sym.Symbol, module: types.ModuleType = None) -> None:
        self._name = name
        self._module = Maybe(module).or_else(lambda: _new_module(name.as_python_sym()))

        self._aliases: atom.Atom = atom.Atom(lmap.Map.empty())
        self._imports: atom.Atom = atom.Atom(
            lmap.map(
                seq(Namespace.DEFAULT_IMPORTS.deref())
                .map(lambda s: (s, importlib.import_module(s.name)))
                .to_dict()
            )
        )
        self._import_aliases: atom.Atom = atom.Atom(lmap.Map.empty())
        self._interns: atom.Atom = atom.Atom(lmap.Map.empty())
        self._refers: atom.Atom = atom.Atom(lmap.Map.empty())

    @classmethod
    def add_default_import(cls, module: str):
        """Add a gated default import to the default imports.

        In particular, we need to avoid importing 'basilisp.core' before we have
        finished macro-expanding."""
        if module in cls.GATED_IMPORTS:
            cls.DEFAULT_IMPORTS.swap(lambda s: s.cons(sym.symbol(module)))

    @property
    def name(self) -> str:
        return self._name.name

    @property
    def module(self):
        return self._module

    @module.setter
    def module(self, m: types.ModuleType):
        """Override the Python module for this Namespace.

        ***WARNING**
        This should only be done by basilisp.importer code to make sure the
        correct module is generated for `basilisp.core`."""
        self._module = m

    @property
    def aliases(self) -> lmap.Map:
        """A mapping between a symbolic alias and another Namespace. The
        fully qualified name of a namespace is also an alias for itself."""
        return self._aliases.deref()

    @property
    def imports(self) -> lmap.Map:
        """A mapping of names to Python modules imported into the current
        namespace."""
        return self._imports.deref()

    @property
    def import_aliases(self) -> lmap.Map:
        """A mapping of a symbolic alias and a Python module name."""
        return self._import_aliases.deref()

    @property
    def interns(self) -> lmap.Map:
        """A mapping between a symbolic name and a Var. The Var may point to
        code, data, or nothing, if it is unbound. Vars in `interns` are
        interned in _this_ namespace."""
        return self._interns.deref()

    @property
    def refers(self) -> lmap.Map:
        """A mapping between a symbolic name and a Var. Vars in refers are
        interned in another namespace and are only referred to without an
        alias in this namespace."""
        return self._refers.deref()

    def __repr__(self):
        return f"{self._name}"

    def __hash__(self):
        return hash(self._name)

    def add_alias(self, alias: sym.Symbol, namespace: "Namespace") -> None:
        """Add a Symbol alias for the given Namespace."""
        self._aliases.swap(lambda m: m.assoc(alias, namespace))

    def get_alias(self, alias: sym.Symbol) -> "Optional[Namespace]":
        """Get the Namespace aliased by Symbol or None if it does not exist."""
        return self.aliases.entry(alias, None)

    def intern(self, sym: sym.Symbol, var: Var, force: bool = False) -> Var:
        """Intern the Var given in this namespace mapped by the given Symbol.
        If the Symbol already maps to a Var, this method _will not overwrite_
        the existing Var mapping unless the force keyword argument is given
        and is True."""
        m: lmap.Map = self._interns.swap(Namespace._intern, sym, var, force=force)
        return m.entry(sym)

    @staticmethod
    def _intern(
        m: lmap.Map, sym: sym.Symbol, new_var: Var, force: bool = False
    ) -> lmap.Map:
        """Swap function used by intern to atomically intern a new variable in
        the symbol mapping for this Namespace."""
        var = m.entry(sym, None)
        if var is None or force:
            return m.assoc(sym, new_var)
        return m

    def find(self, sym: sym.Symbol) -> Optional[Var]:
        """Find Vars mapped by the given Symbol input or None if no Vars are
        mapped by that Symbol."""
        v = self.interns.entry(sym, None)
        if v is None:
            return self.refers.entry(sym, None)
        return v

    def add_import(
        self, sym: sym.Symbol, module: types.ModuleType, *aliases: sym.Symbol
    ) -> None:
        """Add the Symbol as an imported Symbol in this Namespace. If aliases are given,
        the aliases will be applied to the """
        self._imports.swap(lambda m: m.assoc(sym, module))
        if aliases:
            self._import_aliases.swap(
                lambda m: m.assoc(
                    *itertools.chain.from_iterable([(alias, sym) for alias in aliases])
                )
            )

    def get_import(self, sym: sym.Symbol) -> Optional[types.ModuleType]:
        """Return the module if a moduled named by sym has been imported into
        this Namespace, None otherwise.

        First try to resolve a module directly with the given name. If no module
        can be resolved, attempt to resolve the module using import aliases."""
        mod = self.imports.entry(sym, None)
        if mod is None:
            alias = self.import_aliases.get(sym, None)
            if alias is None:
                return None
            return self.imports.entry(alias, None)
        return mod

    def add_refer(self, sym: sym.Symbol, var: Var) -> None:
        """Refer var in this namespace under the name sym."""
        if not var.is_private:
            self._refers.swap(lambda s: s.assoc(sym, var))

    def get_refer(self, sym: sym.Symbol) -> Optional[Var]:
        """Get the Var referred by Symbol or None if it does not exist."""
        return self.refers.entry(sym, None)

    @classmethod
    def __refer_all(cls, refers: lmap.Map, other_ns_interns: lmap.Map) -> lmap.Map:
        """Refer all _public_ interns from another namespace."""
        final_refers = refers
        for entry in other_ns_interns:
            s: sym.Symbol = entry.key
            var: Var = entry.value
            if not var.is_private:
                final_refers = final_refers.assoc(s, var)
        return final_refers

    def refer_all(self, other_ns: "Namespace"):
        """Refer all the Vars in the other namespace."""
        self._refers.swap(Namespace.__refer_all, other_ns.interns)

    @classmethod
    def ns_cache(cls) -> lmap.Map:
        """Return a snapshot of the Namespace cache."""
        return cls._NAMESPACES.deref()

    @staticmethod
    def __get_or_create(
        ns_cache: lmap.Map,
        name: sym.Symbol,
        module: types.ModuleType = None,
        core_ns_name=_CORE_NS,
    ) -> lmap.Map:
        """Private swap function used by `get_or_create` to atomically swap
        the new namespace map into the global cache."""
        ns = ns_cache.entry(name, None)
        if ns is not None:
            return ns_cache
        new_ns = Namespace(name, module=module)
        if name.name != core_ns_name:
            core_ns = ns_cache.entry(sym.symbol(core_ns_name), None)
            assert core_ns is not None, "Core namespace not loaded yet!"
            new_ns.refer_all(core_ns)
        return ns_cache.assoc(name, new_ns)

    @classmethod
    def get_or_create(
        cls, name: sym.Symbol, module: types.ModuleType = None
    ) -> "Namespace":
        """Get the namespace bound to the symbol `name` in the global namespace
        cache, creating it if it does not exist.
        Return the namespace."""
        return cls._NAMESPACES.swap(Namespace.__get_or_create, name, module=module)[
            name
        ]

    @classmethod
    def get(cls, name: sym.Symbol) -> "Optional[Namespace]":
        """Get the namespace bound to the symbol `name` in the global namespace
        cache. Return the namespace if it exists or None otherwise.."""
        return cls._NAMESPACES.deref().entry(name, None)

    @classmethod
    def remove(cls, name: sym.Symbol) -> Optional["Namespace"]:
        """Remove the namespace bound to the symbol `name` in the global
        namespace cache and return that namespace.
        Return None if the namespace did not exist in the cache."""
        while True:
            oldval: lmap.Map = cls._NAMESPACES.deref()
            ns: Optional[Namespace] = oldval.entry(name, None)
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
    s = to_seq(o)
    if s is None:
        return None
    return s.first


def rest(o) -> Optional[lseq.Seq]:
    """If o is a Seq, return the elements after the first in o. If o is None,
    returns an empty seq. Otherwise, coerces o to a seq and returns the rest."""
    if o is None:
        return None
    if isinstance(o, lseq.Seq):
        s = o.rest
        if s is None:
            return lseq.EMPTY
        return s
    n = to_seq(o)
    if n is None:
        return lseq.EMPTY
    return n.rest


def nthrest(coll, i: int):
    """Returns the nth rest sequence of coll, or coll if i is 0."""
    while True:
        if coll is None:
            return None
        if i == 0:
            return coll
        i -= 1
        coll = rest(coll)


def next_(o) -> Optional[lseq.Seq]:
    """Calls rest on o. If o returns an empty sequence or None, returns None.
    Otherwise, returns the elements after the first in o."""
    s = rest(o)
    if not s:
        return None
    return s


def nthnext(coll, i: int) -> Optional[lseq.Seq]:
    """Returns the nth next sequence of coll."""
    while True:
        if coll is None:
            return None
        if i == 0:
            return to_seq(coll)
        i -= 1
        coll = next_(coll)


def cons(o, seq) -> lseq.Seq:
    """Creates a new sequence where o is the first element and seq is the rest.
    If seq is None, return a list containing o. If seq is not a Seq, attempt
    to coerce it to a Seq and then cons o onto the resulting sequence."""
    if seq is None:
        return llist.l(o)
    if isinstance(seq, lseq.Seq):
        return seq.cons(o)
    return Maybe(to_seq(seq)).map(lambda s: s.cons(o)).or_else(lambda: llist.l(o))


def _seq_or_nil(s: lseq.Seq) -> Optional[lseq.Seq]:
    """Return None if a Seq is empty, the Seq otherwise."""
    if s.is_empty:
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
        return lseq.EMPTY
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
        last = args[-1]
    except TypeError as e:
        logger.debug("Ignored %s: %s", type(e).__name__, e)

    s = to_seq(last)
    if s is not None:
        final.extend(s)

    return f(*final)


__nth_sentinel = object()


def nth(coll, i, notfound=__nth_sentinel):
    """Returns the ith element of coll (0-indexed), if it exists.
    None otherwise. If i is out of bounds, throws an IndexError unless
    notfound is specified."""
    if coll is None:
        return None

    try:
        return coll[i]
    except IndexError as ex:
        if notfound is not __nth_sentinel:
            return notfound
        raise ex
    except TypeError as ex:
        logger.debug("Ignored %s: %s", type(ex).__name__, ex)

    try:
        for j, e in enumerate(coll):
            if i == j:
                return e
        if notfound is not __nth_sentinel:
            return notfound
        raise IndexError(f"Index {i} out of bounds")
    except TypeError:
        pass

    raise TypeError(f"nth not supported on object of type {type(coll)}")


def assoc(m, *kvs):
    """Associate keys to values in associative data structure m. If m is None,
    returns a new Map with key-values kvs."""
    if m is None:
        return lmap.Map.empty().assoc(*kvs)
    if isinstance(m, lassoc.Associative):
        return m.assoc(*kvs)
    raise TypeError(
        f"Object of type {type(m)} does not implement Associative interface"
    )


def update(m, k, f, *args):
    """Updates the value for key k in associative data structure m with the return value from
    calling f(old_v, *args). If m is None, use an empty map. If k is not in m, old_v will be
    None."""
    if m is None:
        return lmap.Map.empty().assoc(k, f(None, *args))
    if isinstance(m, lassoc.Associative):
        old_v = m.entry(k)
        new_v = f(old_v, *args)
        return m.assoc(k, new_v)
    raise TypeError(
        f"Object of type {type(m)} does not implement Associative interface"
    )


def conj(coll, *xs):
    """Conjoin xs to collection. New elements may be added in different positions
    depending on the type of coll. conj returns the same type as coll. If coll
    is None, return a list with xs conjoined."""
    if coll is None:
        l = llist.List.empty()
        return l.cons(*xs)
    if isinstance(coll, lcoll.Collection):
        return coll.cons(*xs)
    raise TypeError(
        f"Object of type {type(coll)} does not implement Collection interface"
    )


def partial(f, *args):
    """Return a function which is the partial application of f with args."""

    @functools.wraps(f)
    def partial_f(*inner_args):
        return f(*itertools.chain(args, inner_args))

    return partial_f


def deref(o):
    """Dereference a Deref object and return its contents."""
    if isinstance(o, lderef.Deref):
        return o.deref()
    raise TypeError(f"Object of type {type(o)} cannot be dereferenced")


def swap(a: atom.Atom, f, *args):
    """Atomically swap the value of an atom to the return value of (apply f
    current-value args). The function f may be called multiple times while
    swapping, so should be free of side effects. Return the new value."""
    return a.swap(f, *args)


def equals(v1, v2) -> bool:
    """Compare two objects by value. Unlike the standard Python equality operator,
    this function does not consider 1 == True or 0 == False. All other equality
    operations are the same and performed using Python's equality operator."""
    if isinstance(v1, (bool, type(None))) or isinstance(v2, (bool, type(None))):
        return v1 is v2
    return v1 == v2


def divide(x: LispNumber, y: LispNumber) -> LispNumber:
    """Division reducer. If both arguments are integers, return a Fraction.
    Otherwise, return the true division of x and y."""
    if isinstance(x, int) and isinstance(y, int):
        return Fraction(x, y)
    return x / y  # type: ignore


def quotient(num, div) -> LispNumber:
    """Return the integral quotient resulting from the division of num by div."""
    return math.trunc(num / div)


def sort(coll, f=None) -> Optional[lseq.Seq]:
    """Return a sorted sequence of the elements in coll. If a comparator
    function f is provided, compare elements in coll using f."""
    return to_seq(sorted(coll, key=Maybe(f).map(functools.cmp_to_key).value))


def contains(coll, k):
    """Return true if o contains the key k."""
    if isinstance(coll, lassoc.Associative):
        return coll.contains(k)
    return k in coll


def get(m, k, default=None):
    """Return the value of k in m. Return default if k not found in m."""
    if isinstance(m, lassoc.Associative):
        return m.entry(k, default=default)

    try:
        return m[k]
    except (KeyError, IndexError, TypeError) as e:
        logger.debug("Ignored %s: %s", type(e).__name__, e)
        return default


def lrepr(o, human_readable: bool = False) -> str:
    """Produce a string representation of an object. If human_readable is False,
    the string representation of Lisp objects is something that can be read back
    in by the reader as the same object."""
    core_ns = Namespace.get(sym.symbol(_CORE_NS))
    assert core_ns is not None
    return lobj.lrepr(
        o,
        human_readable=human_readable,
        print_dup=core_ns.find(sym.symbol(_PRINT_DUP_VAR_NAME)).value,  # type: ignore
        print_length=core_ns.find(  # type: ignore
            sym.symbol(_PRINT_LENGTH_VAR_NAME)
        ).value,
        print_level=core_ns.find(  # type: ignore
            sym.symbol(_PRINT_LEVEL_VAR_NAME)
        ).value,
        print_meta=core_ns.find(sym.symbol(_PRINT_META_VAR_NAME)).value,  # type: ignore
        print_readably=core_ns.find(  # type: ignore
            sym.symbol(_PRINT_READABLY_VAR_NAME)
        ).value,
    )


def lstr(o) -> str:
    """Produce a human readable string representation of an object."""
    return lrepr(o, human_readable=True)


def _collect_args(args) -> lseq.Seq:
    """Collect Python starred arguments into a Basilisp list."""
    if isinstance(args, tuple):
        return llist.list(args)
    raise TypeError("Python variadic arguments should always be a tuple")


class _TrampolineArgs:
    __slots__ = ("_has_varargs", "_args", "_kwargs")

    def __init__(self, has_varargs: bool, *args, **kwargs) -> None:
        self._has_varargs = has_varargs
        self._args = args
        self._kwargs = kwargs

    @property
    def args(self) -> Tuple:
        """Return the arguments for a trampolined function. If the function
        that is being trampolined has varargs, unroll the final argument if
        it is a sequence."""
        if not self._has_varargs:
            return self._args

        try:
            final = self._args[-1]
            if isinstance(final, lseq.Seq):
                inits = self._args[:-1]
                return tuple(itertools.chain(inits, final))
            return self._args
        except IndexError:
            return ()

    @property
    def kwargs(self) -> Dict:
        return self._kwargs


def _trampoline(f):
    """Trampoline a function repeatedly until it is finished recurring to help
    avoid stack growth."""

    @functools.wraps(f)
    def trampoline(*args, **kwargs):
        while True:
            ret = f(*args, **kwargs)
            if isinstance(ret, _TrampolineArgs):
                args = ret.args
                kwargs = ret.kwargs
                continue
            return ret

    return trampoline


#########################
# Bootstrap the Runtime #
#########################


def init_ns_var(which_ns: str = _CORE_NS, ns_var_name: str = _NS_VAR_NAME) -> Var:
    """Initialize the dynamic `*ns*` variable in the Namespace `which_ns`."""
    core_sym = sym.Symbol(which_ns)
    core_ns = Namespace.get_or_create(core_sym)
    ns_var = Var.intern(core_sym, sym.Symbol(ns_var_name), core_ns, dynamic=True)
    logger.debug(f"Created namespace variable {sym.symbol(ns_var_name, ns=which_ns)}")
    return ns_var


def set_current_ns(
    ns_name: str,
    module: types.ModuleType = None,
    ns_var_name: str = _NS_VAR_NAME,
    ns_var_ns: str = _NS_VAR_NS,
) -> Var:
    """Set the value of the dynamic variable `*ns*` in the current thread."""
    symbol = sym.Symbol(ns_name)
    ns = Namespace.get_or_create(symbol, module=module)
    ns_var_sym = sym.Symbol(ns_var_name, ns=ns_var_ns)
    ns_var = Maybe(Var.find(ns_var_sym)).or_else_raise(
        lambda: RuntimeException(
            f"Dynamic Var {sym.Symbol(ns_var_name, ns=ns_var_ns)} not bound!"
        )
    )
    ns_var.push_bindings(ns)
    logger.debug(f"Setting {ns_var_sym} to {ns}")
    return ns_var


@contextlib.contextmanager
def ns_bindings(
    ns_name: str,
    module: types.ModuleType = None,
    ns_var_name: str = _NS_VAR_NAME,
    ns_var_ns: str = _NS_VAR_NS,
):
    """Context manager for temporarily changing the value of basilisp.core/*ns*."""
    symbol = sym.Symbol(ns_name)
    ns = Namespace.get_or_create(symbol, module=module)
    ns_var_sym = sym.Symbol(ns_var_name, ns=ns_var_ns)
    ns_var = Maybe(Var.find(ns_var_sym)).or_else_raise(
        lambda: RuntimeException(
            f"Dynamic Var {sym.Symbol(ns_var_name, ns=ns_var_ns)} not bound!"
        )
    )

    try:
        logger.debug(f"Binding {ns_var_sym} to {ns}")
        ns_var.push_bindings(ns)
        yield ns_var.value
    finally:
        ns_var.pop_bindings()
        logger.debug(f"Reset bindings for {ns_var_sym} to {ns_var.value}")


@contextlib.contextmanager
def remove_ns_bindings(ns_var_name: str = _NS_VAR_NAME, ns_var_ns: str = _NS_VAR_NS):
    """Context manager to pop the most recent bindings for basilisp.core/*ns* after
    completion of the code under management."""
    ns_var_sym = sym.Symbol(ns_var_name, ns=ns_var_ns)
    ns_var = Maybe(Var.find(ns_var_sym)).or_else_raise(
        lambda: RuntimeException(
            f"Dynamic Var {sym.Symbol(ns_var_name, ns=ns_var_ns)} not bound!"
        )
    )
    try:
        yield
    finally:
        ns_var.pop_bindings()
        logger.debug(f"Reset bindings for {ns_var_sym} to {ns_var.value}")


def get_current_ns(
    ns_var_name: str = _NS_VAR_NAME, ns_var_ns: str = _NS_VAR_NS
) -> Namespace:
    """Get the value of the dynamic variable `*ns*` in the current thread."""
    ns_sym = sym.Symbol(ns_var_name, ns=ns_var_ns)
    ns: Namespace = Maybe(Var.find(ns_sym)).map(lambda v: v.value).or_else_raise(
        lambda: RuntimeException(f"Dynamic Var {ns_sym} not bound!")
    )
    return ns


def resolve_alias(s: sym.Symbol, ns: Optional[Namespace] = None) -> sym.Symbol:
    """Resolve the aliased symbol in the current namespace."""
    if s in _SPECIAL_FORMS:
        return s

    ns = Maybe(ns).or_else(get_current_ns)
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


def resolve_var(s: sym.Symbol, ns: Optional[Namespace] = None) -> Optional[Var]:
    """Resolve the aliased symbol to a Var from the specified
    namespace, or the current namespace if none is specified."""
    return Var.find(resolve_alias(s, ns))


def add_generated_python(
    generated_python: str,
    var_name: str = _GENERATED_PYTHON_VAR_NAME,
    which_ns: Optional[str] = None,
) -> None:
    """Add generated Python code to a dynamic variable in which_ns."""
    if which_ns is None:
        which_ns = get_current_ns().name
    ns_sym = sym.Symbol(var_name, ns=which_ns)
    v = Maybe(Var.find(ns_sym)).or_else(
        lambda: Var.intern(
            sym.symbol(which_ns),  # type: ignore
            sym.symbol(var_name),
            "",
            dynamic=True,
            meta=lmap.map({_PRIVATE_META_KEY: True}),
        )
    )
    v.value = v.value + generated_python


def print_generated_python(
    var_name: str = _PRINT_GENERATED_PY_VAR_NAME, core_ns_name: str = _CORE_NS
) -> bool:
    """Return the value of the `*print-generated-python*` dynamic variable."""
    ns_sym = sym.Symbol(var_name, ns=core_ns_name)
    return (
        Maybe(Var.find(ns_sym))
        .map(lambda v: v.value)
        .or_else_raise(lambda: RuntimeException(f"Dynamic Var {ns_sym} not bound!"))
    )


def bootstrap(ns_var_name: str = _NS_VAR_NAME, core_ns_name: str = _CORE_NS) -> None:
    """Bootstrap the environment with functions that are are difficult to
    express with the very minimal lisp environment."""
    core_ns_sym = sym.symbol(core_ns_name)
    ns_var_sym = sym.symbol(ns_var_name, ns=core_ns_name)
    __NS = Maybe(Var.find(ns_var_sym)).or_else_raise(
        lambda: RuntimeException(f"Dynamic Var {ns_var_sym} not bound!")
    )

    def set_BANG_(var_sym: sym.Symbol, expr):
        ns = Maybe(var_sym.ns).or_else(lambda: __NS.value.name)
        name = var_sym.name

        v = Maybe(Var.find(sym.symbol(name, ns=ns))).or_else_raise(
            lambda: RuntimeException(f"Var {ns_var_sym} not bound!")
        )
        v.value = expr
        return expr

    def in_ns(s: sym.Symbol):
        ns = Namespace.get_or_create(s)
        set_BANG_(ns_var_sym, ns)
        return ns

    Var.intern_unbound(core_ns_sym, sym.symbol("unquote"))
    Var.intern_unbound(core_ns_sym, sym.symbol("unquote-splicing"))
    Var.intern(
        core_ns_sym,
        sym.symbol("set!"),
        set_BANG_,
        meta=lmap.map({_REDEF_META_KEY: True}),
    )
    Var.intern(
        core_ns_sym, sym.symbol("in-ns"), in_ns, meta=lmap.map({_REDEF_META_KEY: True})
    )
    Var.intern(
        core_ns_sym,
        sym.symbol(_PRINT_GENERATED_PY_VAR_NAME),
        False,
        dynamic=True,
        meta=lmap.map({_PRIVATE_META_KEY: True}),
    )
    Var.intern(
        core_ns_sym,
        sym.symbol(_GENERATED_PYTHON_VAR_NAME),
        "",
        dynamic=True,
        meta=lmap.map({_PRIVATE_META_KEY: True}),
    )

    # Dynamic Vars for controlling printing
    Var.intern(
        core_ns_sym, sym.symbol(_PRINT_DUP_VAR_NAME), lobj.PRINT_DUP, dynamic=True
    )
    Var.intern(
        core_ns_sym, sym.symbol(_PRINT_LENGTH_VAR_NAME), lobj.PRINT_LENGTH, dynamic=True
    )
    Var.intern(
        core_ns_sym, sym.symbol(_PRINT_LEVEL_VAR_NAME), lobj.PRINT_LEVEL, dynamic=True
    )
    Var.intern(
        core_ns_sym, sym.symbol(_PRINT_META_VAR_NAME), lobj.PRINT_META, dynamic=True
    )
    Var.intern(
        core_ns_sym,
        sym.symbol(_PRINT_READABLY_VAR_NAME),
        lobj.PRINT_READABLY,
        dynamic=True,
    )
