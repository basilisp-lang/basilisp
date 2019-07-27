import contextlib
import functools
import importlib
import inspect
import itertools
import logging
import math
import re
import threading
import types
from fractions import Fraction
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.obj as lobj
import basilisp.lang.seq as lseq
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.vector as vec
from basilisp.lang.atom import Atom
from basilisp.lang.interfaces import (
    IAssociative,
    IBlockingDeref,
    IDeref,
    ILookup,
    IPersistentCollection,
    IPersistentList,
    IPersistentMap,
    IPersistentSet,
    IPersistentVector,
    ISeq,
    ISeqable,
)
from basilisp.lang.typing import LispNumber
from basilisp.lang.util import munge
from basilisp.logconfig import TRACE
from basilisp.util import Maybe

logger = logging.getLogger(__name__)

# Public constants
CORE_NS = "basilisp.core"
NS_VAR_NAME = "*ns*"
NS_VAR_NS = CORE_NS
REPL_DEFAULT_NS = "user"

# Private string constants
_GENERATED_PYTHON_VAR_NAME = "*generated-python*"
_PRINT_GENERATED_PY_VAR_NAME = "*print-generated-python*"
_PRINT_DUP_VAR_NAME = "*print-dup*"
_PRINT_LENGTH_VAR_NAME = "*print-length*"
_PRINT_LEVEL_VAR_NAME = "*print-level*"
_PRINT_META_VAR_NAME = "*print-meta*"
_PRINT_READABLY_VAR_NAME = "*print-readably*"

# Common meta keys
_DYNAMIC_META_KEY = kw.keyword("dynamic")
_PRIVATE_META_KEY = kw.keyword("private")
_REDEF_META_KEY = kw.keyword("redef")

# Special form values, used for resolving Vars
_AWAIT = sym.symbol("await")
_CATCH = sym.symbol("catch")
_DEF = sym.symbol("def")
_DEFTYPE = sym.symbol("deftype*")
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
_SET_BANG = sym.symbol("set!")
_THROW = sym.symbol("throw")
_TRY = sym.symbol("try")
_VAR = sym.symbol("var")
_SPECIAL_FORMS = lset.s(
    _AWAIT,
    _CATCH,
    _DEF,
    _DEFTYPE,
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
    _SET_BANG,
    _THROW,
    _TRY,
    _VAR,
)

CompletionMatcher = Callable[[Tuple[sym.Symbol, Any]], bool]
CompletionTrimmer = Callable[[Tuple[sym.Symbol, Any]], str]


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


class _VarBindings(threading.local):
    __slots__ = ("bindings",)

    def __init__(self):
        self.bindings: List = []


class Var(IDeref):
    __slots__ = (
        "_name",
        "_ns",
        "_root",
        "_dynamic",
        "_is_bound",
        "_tl",
        "_meta",
        "_lock",
    )

    def __init__(
        self, ns: "Namespace", name: sym.Symbol, dynamic: bool = False, meta=None
    ) -> None:
        self._ns = ns
        self._name = name
        self._root = None
        self._dynamic = dynamic
        self._is_bound = False
        self._tl = None
        self._meta = meta
        self._lock = threading.Lock()

        if dynamic:
            self._tl = _VarBindings()

            # If this var was created with the dynamic keyword argument, then the
            # Var metadata should also specify that the Var is dynamic.
            if isinstance(self._meta, lmap.Map):
                if not self._meta.val_at(_DYNAMIC_META_KEY):
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
            return self.meta.val_at(_PRIVATE_META_KEY)
        except AttributeError:
            return False

    @property
    def is_bound(self) -> bool:
        return self._is_bound or self.is_thread_bound

    @property
    def root(self):
        with self._lock:
            return self._root

    @root.setter
    def root(self, val):
        with self._lock:
            self._is_bound = True
            self._root = val

    def alter_root(self, f, *args):
        with self._lock:
            self._root = f(self._root, *args)

    def push_bindings(self, val):
        if self._tl is None:
            raise RuntimeException("Can only push bindings to dynamic Vars")
        self._tl.bindings.append(val)

    def pop_bindings(self):
        if self._tl is None:
            raise RuntimeException("Can only pop bindings from dynamic Vars")
        return self._tl.bindings.pop()

    def deref(self):
        return self.value

    @property
    def is_thread_bound(self):
        return bool(self._dynamic and self._tl.bindings)

    @property
    def value(self):
        if self._dynamic:
            assert self._tl is not None
            if len(self._tl.bindings) > 0:
                return self._tl.bindings[-1]
        return self.root

    @value.setter
    def value(self, v):
        if self._dynamic:
            assert self._tl is not None
            if len(self._tl.bindings) > 0:
                self._tl.bindings[-1] = v
            else:
                self.push_bindings(v)
            return
        self.root = v

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


Frame = FrozenSet[Var]
FrameStack = List[Frame]


class _ThreadBindings(threading.local):
    __slots__ = ("_bindings",)

    def __init__(self):
        self._bindings: FrameStack = []

    def push_bindings(self, frame: Frame):
        self._bindings.append(frame)

    def pop_bindings(self):
        return self._bindings.pop()


_THREAD_BINDINGS = _ThreadBindings()


AliasMap = lmap.Map[sym.Symbol, sym.Symbol]
ModuleMap = lmap.Map[sym.Symbol, types.ModuleType]
NamespaceMap = lmap.Map[sym.Symbol, "Namespace"]
VarMap = lmap.Map[sym.Symbol, Var]


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

    DEFAULT_IMPORTS = Atom(
        lset.set(
            map(
                sym.symbol,
                [
                    "attr",
                    "builtins",
                    "io",
                    "operator",
                    "sys",
                    "basilisp.lang.atom",
                    "basilisp.lang.compiler",
                    "basilisp.lang.delay",
                    "basilisp.lang.exception",
                    "basilisp.lang.interfaces",
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
                ],
            )
        )
    )
    GATED_IMPORTS = lset.set(["basilisp.core"])

    _NAMESPACES: Atom[NamespaceMap] = Atom(lmap.Map.empty())

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

        self._aliases: Atom[NamespaceMap] = Atom(lmap.Map.empty())
        self._imports: Atom[ModuleMap] = Atom(
            lmap.map(
                dict(
                    map(
                        lambda s: (s, importlib.import_module(s.name)),
                        Namespace.DEFAULT_IMPORTS.deref(),
                    )
                )
            )
        )
        self._import_aliases: Atom[AliasMap] = Atom(lmap.Map.empty())
        self._interns: Atom[VarMap] = Atom(lmap.Map.empty())
        self._refers: Atom[VarMap] = Atom(lmap.Map.empty())

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
    def module(self) -> types.ModuleType:
        return self._module

    @module.setter
    def module(self, m: types.ModuleType):
        """Override the Python module for this Namespace.

        ***WARNING**
        This should only be done by basilisp.importer code to make sure the
        correct module is generated for `basilisp.core`."""
        self._module = m

    @property
    def aliases(self) -> NamespaceMap:
        """A mapping between a symbolic alias and another Namespace. The
        fully qualified name of a namespace is also an alias for itself."""
        return self._aliases.deref()

    @property
    def imports(self) -> ModuleMap:
        """A mapping of names to Python modules imported into the current
        namespace."""
        return self._imports.deref()

    @property
    def import_aliases(self) -> AliasMap:
        """A mapping of a symbolic alias and a Python module name."""
        return self._import_aliases.deref()

    @property
    def interns(self) -> VarMap:
        """A mapping between a symbolic name and a Var. The Var may point to
        code, data, or nothing, if it is unbound. Vars in `interns` are
        interned in _this_ namespace."""
        return self._interns.deref()

    @property
    def refers(self) -> VarMap:
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
        return self.aliases.val_at(alias, None)

    def remove_alias(self, alias: sym.Symbol) -> None:
        """Remove the Namespace aliased by Symbol. Return None."""
        self._aliases.swap(lambda m: m.dissoc(alias))

    def intern(self, sym: sym.Symbol, var: Var, force: bool = False) -> Var:
        """Intern the Var given in this namespace mapped by the given Symbol.
        If the Symbol already maps to a Var, this method _will not overwrite_
        the existing Var mapping unless the force keyword argument is given
        and is True."""
        m: lmap.Map = self._interns.swap(Namespace._intern, sym, var, force=force)
        return m.val_at(sym)

    @staticmethod
    def _intern(
        m: lmap.Map, sym: sym.Symbol, new_var: Var, force: bool = False
    ) -> lmap.Map:
        """Swap function used by intern to atomically intern a new variable in
        the symbol mapping for this Namespace."""
        var = m.val_at(sym, None)
        if var is None or force:
            return m.assoc(sym, new_var)
        return m

    def unmap(self, sym: sym.Symbol) -> None:
        self._interns.swap(lambda m: m.dissoc(sym))

    def find(self, sym: sym.Symbol) -> Optional[Var]:
        """Find Vars mapped by the given Symbol input or None if no Vars are
        mapped by that Symbol."""
        v = self.interns.val_at(sym, None)
        if v is None:
            return self.refers.val_at(sym, None)
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
        mod = self.imports.val_at(sym, None)
        if mod is None:
            alias = self.import_aliases.get(sym, None)
            if alias is None:
                return None
            return self.imports.val_at(alias, None)
        return mod

    def add_refer(self, sym: sym.Symbol, var: Var) -> None:
        """Refer var in this namespace under the name sym."""
        if not var.is_private:
            self._refers.swap(lambda s: s.assoc(sym, var))

    def get_refer(self, sym: sym.Symbol) -> Optional[Var]:
        """Get the Var referred by Symbol or None if it does not exist."""
        return self.refers.val_at(sym, None)

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
        ns_cache: NamespaceMap,
        name: sym.Symbol,
        module: types.ModuleType = None,
        core_ns_name=CORE_NS,
    ) -> lmap.Map:
        """Private swap function used by `get_or_create` to atomically swap
        the new namespace map into the global cache."""
        ns = ns_cache.val_at(name, None)
        if ns is not None:
            return ns_cache
        new_ns = Namespace(name, module=module)
        if name.name != core_ns_name:
            core_ns = ns_cache.val_at(sym.symbol(core_ns_name), None)
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
        return cls._NAMESPACES.deref().val_at(name, None)

    @classmethod
    def remove(cls, name: sym.Symbol) -> Optional["Namespace"]:
        """Remove the namespace bound to the symbol `name` in the global
        namespace cache and return that namespace.
        Return None if the namespace did not exist in the cache."""
        if name == sym.symbol(CORE_NS):
            raise ValueError("Cannot remove the Basilisp core namespace")
        while True:
            oldval: lmap.Map = cls._NAMESPACES.deref()
            ns: Optional[Namespace] = oldval.val_at(name, None)
            newval = oldval
            if ns is not None:
                newval = oldval.dissoc(name)
            if cls._NAMESPACES.compare_and_set(oldval, newval):
                return ns

    # REPL Completion support

    @staticmethod
    def __completion_matcher(text: str) -> CompletionMatcher:
        """Return a function which matches any symbol keys from map entries
        against the given text."""

        def is_match(entry: Tuple[sym.Symbol, Any]) -> bool:
            return entry[0].name.startswith(text)

        return is_match

    def __complete_alias(
        self, prefix: str, name_in_ns: Optional[str] = None
    ) -> Iterable[str]:
        """Return an iterable of possible completions matching the given
        prefix from the list of aliased namespaces. If name_in_ns is given,
        further attempt to refine the list to matching names in that namespace."""
        candidates = filter(
            Namespace.__completion_matcher(prefix), [(s, n) for s, n in self.aliases]
        )
        if name_in_ns is not None:
            for _, candidate_ns in candidates:
                for match in candidate_ns.__complete_interns(
                    name_in_ns, include_private_vars=False
                ):
                    yield f"{prefix}/{match}"
        else:
            for alias, _ in candidates:
                yield f"{alias}/"

    def __complete_imports_and_aliases(
        self, prefix: str, name_in_module: Optional[str] = None
    ) -> Iterable[str]:
        """Return an iterable of possible completions matching the given
        prefix from the list of imports and aliased imports. If name_in_module
        is given, further attempt to refine the list to matching names in that
        namespace."""
        imports = self.imports
        aliases = lmap.map(
            {
                alias: imports.val_at(import_name)
                for alias, import_name in self.import_aliases
            }
        )

        candidates = filter(
            Namespace.__completion_matcher(prefix), itertools.chain(aliases, imports)
        )
        if name_in_module is not None:
            for _, module in candidates:
                for name in module.__dict__:
                    if name.startswith(name_in_module):
                        yield f"{prefix}/{name}"
        else:
            for candidate_name, _ in candidates:
                yield f"{candidate_name}/"

    def __complete_interns(
        self, value: str, include_private_vars: bool = True
    ) -> Iterable[str]:
        """Return an iterable of possible completions matching the given
        prefix from the list of interned Vars."""
        if include_private_vars:
            is_match = Namespace.__completion_matcher(value)
        else:
            _is_match = Namespace.__completion_matcher(value)

            def is_match(entry: Tuple[sym.Symbol, Var]) -> bool:
                return _is_match(entry) and not entry[1].is_private

        return map(
            lambda entry: f"{entry[0].name}",
            filter(is_match, [(s, v) for s, v in self.interns]),
        )

    def __complete_refers(self, value: str) -> Iterable[str]:
        """Return an iterable of possible completions matching the given
        prefix from the list of referred Vars."""
        return map(
            lambda entry: f"{entry[0].name}",
            filter(
                Namespace.__completion_matcher(value), [(s, v) for s, v in self.refers]
            ),
        )

    def complete(self, text: str) -> Iterable[str]:
        """Return an iterable of possible completions for the given text in
        this namespace."""
        assert not text.startswith(":")

        if "/" in text:
            prefix, suffix = text.split("/", maxsplit=1)
            results = itertools.chain(
                self.__complete_alias(prefix, name_in_ns=suffix),
                self.__complete_imports_and_aliases(prefix, name_in_module=suffix),
            )
        else:
            results = itertools.chain(
                self.__complete_alias(text),
                self.__complete_imports_and_aliases(text),
                self.__complete_interns(text),
                self.__complete_refers(text),
            )

        return results


def push_thread_bindings(m: IAssociative[Var, Any]) -> None:
    """Push thread local bindings for the Var keys in m using the values."""
    bindings = set()

    for entry in m:
        var: Var = entry.key  # type: ignore
        val = entry.value
        if not var.dynamic:
            raise RuntimeException(
                "cannot set thread-local bindings for non-dynamic Var"
            )
        var.push_bindings(val)
        bindings.add(var)

    _THREAD_BINDINGS.push_bindings(frozenset(bindings))


def pop_thread_bindings() -> None:
    """Pop the thread local bindings set by push_thread_bindings above."""
    try:
        bindings = _THREAD_BINDINGS.pop_bindings()
    except IndexError:
        raise RuntimeException("cannot pop thread-local bindings without prior push")

    for var in bindings:
        var.pop_bindings()


###################
# Runtime Support #
###################


def first(o):
    """If o is a ISeq, return the first element from o. If o is None, return
    None. Otherwise, coerces o to a Seq and returns the first."""
    if o is None:
        return None
    if isinstance(o, ISeq):
        return o.first
    s = to_seq(o)
    if s is None:
        return None
    return s.first


def rest(o) -> Optional[ISeq]:
    """If o is a ISeq, return the elements after the first in o. If o is None,
    returns an empty seq. Otherwise, coerces o to a seq and returns the rest."""
    if o is None:
        return None
    if isinstance(o, ISeq):
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


def next_(o) -> Optional[ISeq]:
    """Calls rest on o. If o returns an empty sequence or None, returns None.
    Otherwise, returns the elements after the first in o."""
    s = rest(o)
    if not s:
        return None
    return s


def nthnext(coll, i: int) -> Optional[ISeq]:
    """Returns the nth next sequence of coll."""
    while True:
        if coll is None:
            return None
        if i == 0:
            return to_seq(coll)
        i -= 1
        coll = next_(coll)


def cons(o, seq) -> ISeq:
    """Creates a new sequence where o is the first element and seq is the rest.
    If seq is None, return a list containing o. If seq is not a ISeq, attempt
    to coerce it to a ISeq and then cons o onto the resulting sequence."""
    if seq is None:
        return llist.l(o)
    if isinstance(seq, ISeq):
        return seq.cons(o)
    return Maybe(to_seq(seq)).map(lambda s: s.cons(o)).or_else(lambda: llist.l(o))


def _seq_or_nil(s: ISeq) -> Optional[ISeq]:
    """Return None if a ISeq is empty, the ISeq otherwise."""
    if s.is_empty:
        return None
    return s


def to_seq(o) -> Optional[ISeq]:
    """Coerce the argument o to a ISeq. If o is None, return None."""
    if o is None:
        return None
    if isinstance(o, ISeq):
        return _seq_or_nil(o)
    if isinstance(o, ISeqable):
        return _seq_or_nil(o.seq())
    return _seq_or_nil(lseq.sequence(o))


def concat(*seqs) -> ISeq:
    """Concatenate the sequences given by seqs into a single ISeq."""
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


def apply_kw(f, args):
    """Apply function f to the arguments provided.
    The last argument must always be coercible to a Mapping. Intermediate
    arguments are not modified.
    For example:
        (apply python/dict {:a 1} {:b 2})   ;=> #py {:a 1 :b 2}
        (apply python/dict {:a 1} {:a 2})   ;=> #py {:a 2}"""
    final = list(args[:-1])

    try:
        last = args[-1]
    except TypeError as e:
        logger.debug("Ignored %s: %s", type(e).__name__, e)

    kwargs = {
        to_py(k, lambda kw: munge(kw.name, allow_builtins=True)): v
        for k, v in last.items()
    }
    return f(*final, **kwargs)


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
        # Log these at TRACE so they don't gum up the DEBUG logs since most
        # cases where this exception occurs are not bugs.
        logger.log(TRACE, "Ignored %s: %s", type(ex).__name__, ex)

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
    if isinstance(m, IAssociative):
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
    if isinstance(m, IAssociative):
        old_v = m.val_at(k)
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
    if isinstance(coll, IPersistentCollection):
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


def deref(o, timeout_s=None, timeout_val=None):
    """Dereference a Deref object and return its contents.

    If o is an object implementing IBlockingDeref and timeout_s and
    timeout_val are supplied, deref will wait at most timeout_s seconds,
    returning timeout_val if timeout_s seconds elapse and o has not
    returned."""
    if isinstance(o, IDeref):
        return o.deref()
    elif isinstance(o, IBlockingDeref):
        return o.deref(timeout_s, timeout_val)
    raise TypeError(f"Object of type {type(o)} cannot be dereferenced")


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
    return x / y


def quotient(num, div) -> LispNumber:
    """Return the integral quotient resulting from the division of num by div."""
    return math.trunc(num / div)


def sort(coll, f=None) -> Optional[ISeq]:
    """Return a sorted sequence of the elements in coll. If a comparator
    function f is provided, compare elements in coll using f."""
    return to_seq(sorted(coll, key=Maybe(f).map(functools.cmp_to_key).value))


def contains(coll, k):
    """Return true if o contains the key k."""
    if isinstance(coll, IAssociative):
        return coll.contains(k)
    return k in coll


def get(m, k, default=None):
    """Return the value of k in m. Return default if k not found in m."""
    if isinstance(m, ILookup):
        return m.val_at(k, default=default)

    try:
        return m[k]
    except (KeyError, IndexError, TypeError) as e:
        logger.log(TRACE, "Ignored %s: %s", type(e).__name__, e)
        return default


def is_special_form(s: sym.Symbol) -> bool:
    """Return True if s names a special form."""
    return s in _SPECIAL_FORMS


@functools.singledispatch
def to_lisp(o, keywordize_keys: bool = True):
    """Recursively convert Python collections into Lisp collections."""
    if not isinstance(o, (dict, frozenset, list, set, tuple)):
        return o
    else:  # pragma: no cover
        return _to_lisp_backup(o, keywordize_keys=keywordize_keys)


def _to_lisp_backup(o, keywordize_keys: bool = True):  # pragma: no cover
    if isinstance(o, Mapping):
        return _to_lisp_map(o, keywordize_keys=keywordize_keys)
    elif isinstance(o, AbstractSet):
        return _to_lisp_set(o, keywordize_keys=keywordize_keys)
    elif isinstance(o, Iterable):
        return _to_lisp_vec(o, keywordize_keys=keywordize_keys)
    else:
        return o


@to_lisp.register(list)
@to_lisp.register(tuple)
def _to_lisp_vec(o: Iterable, keywordize_keys: bool = True) -> vec.Vector:
    return vec.vector(
        map(functools.partial(to_lisp, keywordize_keys=keywordize_keys), o)
    )


@to_lisp.register(dict)
def _to_lisp_map(o: Mapping, keywordize_keys: bool = True) -> lmap.Map:
    kvs = {}
    for k, v in o.items():
        if isinstance(k, str) and keywordize_keys:
            processed_key = kw.keyword(k)
        else:
            processed_key = to_lisp(k, keywordize_keys=keywordize_keys)

        kvs[processed_key] = to_lisp(v, keywordize_keys=keywordize_keys)
    return lmap.map(kvs)


@to_lisp.register(frozenset)
@to_lisp.register(set)
def _to_lisp_set(o: AbstractSet, keywordize_keys: bool = True) -> lset.Set:
    return lset.set(map(functools.partial(to_lisp, keywordize_keys=keywordize_keys), o))


def _kw_name(kw: kw.Keyword) -> str:
    return kw.name


@functools.singledispatch
def to_py(o, keyword_fn: Callable[[kw.Keyword], Any] = _kw_name):
    """Recursively convert Lisp collections into Python collections."""
    if isinstance(o, ISeq):
        return _to_py_list(o, keyword_fn=keyword_fn)
    elif not isinstance(
        o, (IPersistentList, IPersistentMap, IPersistentSet, IPersistentVector)
    ):
        return o
    else:  # pragma: no cover
        return _to_py_backup(o, keyword_fn=keyword_fn)


def _to_py_backup(
    o, keyword_fn: Callable[[kw.Keyword], Any] = _kw_name
):  # pragma: no cover
    if isinstance(o, (IPersistentList, IPersistentVector)):
        return _to_py_list(o, keyword_fn=keyword_fn)
    elif isinstance(o, IPersistentMap):
        return _to_py_map(o, keyword_fn=keyword_fn)
    elif isinstance(o, IPersistentSet):
        return _to_py_set(o, keyword_fn=keyword_fn)
    else:
        return o


@to_py.register(kw.Keyword)
def _to_py_kw(o: kw.Keyword, keyword_fn: Callable[[kw.Keyword], Any] = _kw_name) -> Any:
    return keyword_fn(o)


@to_py.register(llist.List)
@to_py.register(ISeq)
@to_py.register(vec.Vector)
def _to_py_list(
    o: Union[IPersistentList, ISeq, IPersistentVector],
    keyword_fn: Callable[[kw.Keyword], Any] = _kw_name,
) -> list:
    return list(map(functools.partial(to_py, keyword_fn=keyword_fn), o))


@to_py.register(lmap.Map)
def _to_py_map(
    o: IPersistentMap, keyword_fn: Callable[[kw.Keyword], Any] = _kw_name
) -> dict:
    return {
        to_py(key, keyword_fn=keyword_fn): to_py(value, keyword_fn=keyword_fn)
        for key, value in o
    }


@to_py.register(lset.Set)
def _to_py_set(
    o: IPersistentSet, keyword_fn: Callable[[kw.Keyword], Any] = _kw_name
) -> set:
    return set(to_py(e, keyword_fn=keyword_fn) for e in o)


def lrepr(o, human_readable: bool = False) -> str:
    """Produce a string representation of an object. If human_readable is False,
    the string representation of Lisp objects is something that can be read back
    in by the reader as the same object."""
    core_ns = Namespace.get(sym.symbol(CORE_NS))
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


__NOT_COMPLETEABLE = re.compile(r"^[0-9].*")


def repl_complete(text: str, state: int) -> Optional[str]:
    """Completer function for Python's readline/libedit implementation."""
    # Can't complete Keywords, Numerals
    if __NOT_COMPLETEABLE.match(text):
        return None
    elif text.startswith(":"):
        completions = kw.complete(text)
    else:
        ns = get_current_ns()
        completions = ns.complete(text)

    return list(completions)[state] if completions is not None else None


####################
# Compiler Support #
####################


def _collect_args(args) -> ISeq:
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
            if isinstance(final, ISeq):
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


def _with_attrs(**kwargs):
    """Decorator to set attributes on a function. Returns the original
    function after setting the attributes named by the keyword arguments."""

    def decorator(f):
        for k, v in kwargs.items():
            setattr(f, k, v)
        return f

    return decorator


def _fn_with_meta(f, meta: Optional[lmap.Map]):
    """Return a new function with the given meta. If the function f already
    has a meta map, then merge the """

    if not isinstance(meta, lmap.Map):
        raise TypeError("meta must be a map")

    if inspect.iscoroutinefunction(f):

        @functools.wraps(f)
        async def wrapped_f(*args, **kwargs):
            return await f(*args, **kwargs)

    else:

        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            return f(*args, **kwargs)

    wrapped_f.meta = (  # type: ignore
        f.meta.update(meta)
        if hasattr(f, "meta") and isinstance(f.meta, lmap.Map)
        else meta
    )
    wrapped_f.with_meta = partial(_fn_with_meta, wrapped_f)  # type: ignore
    return wrapped_f


def _basilisp_fn(f):
    """Create a Basilisp function, setting meta and supplying a with_meta
    method implementation."""
    assert not hasattr(f, "meta")
    f._basilisp_fn = True
    f.meta = None
    f.with_meta = partial(_fn_with_meta, f)
    return f


#########################
# Bootstrap the Runtime #
#########################


def init_ns_var(which_ns: str = CORE_NS, ns_var_name: str = NS_VAR_NAME) -> Var:
    """Initialize the dynamic `*ns*` variable in the Namespace `which_ns`."""
    core_sym = sym.Symbol(which_ns)
    core_ns = Namespace.get_or_create(core_sym)
    ns_var = Var.intern(core_sym, sym.Symbol(ns_var_name), core_ns, dynamic=True)
    logger.debug(f"Created namespace variable {sym.symbol(ns_var_name, ns=which_ns)}")
    return ns_var


def set_current_ns(
    ns_name: str,
    module: types.ModuleType = None,
    ns_var_name: str = NS_VAR_NAME,
    ns_var_ns: str = NS_VAR_NS,
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
    ns_var_name: str = NS_VAR_NAME,
    ns_var_ns: str = NS_VAR_NS,
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
def remove_ns_bindings(ns_var_name: str = NS_VAR_NAME, ns_var_ns: str = NS_VAR_NS):
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
    ns_var_name: str = NS_VAR_NAME, ns_var_ns: str = NS_VAR_NS
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
    # Accessing the Var root via the property uses a lock, which is the
    # desired behavior for Basilisp code, but it introduces additional
    # startup time when there will not realistically be any contention.
    v._root = v._root + generated_python  # type: ignore


def print_generated_python(
    var_name: str = _PRINT_GENERATED_PY_VAR_NAME, core_ns_name: str = CORE_NS
) -> bool:
    """Return the value of the `*print-generated-python*` dynamic variable."""
    ns_sym = sym.Symbol(var_name, ns=core_ns_name)
    return (
        Maybe(Var.find(ns_sym))
        .map(lambda v: v.value)
        .or_else_raise(lambda: RuntimeException(f"Dynamic Var {ns_sym} not bound!"))
    )


def bootstrap(ns_var_name: str = NS_VAR_NAME, core_ns_name: str = CORE_NS) -> None:
    """Bootstrap the environment with functions that are are difficult to
    express with the very minimal lisp environment."""
    core_ns_sym = sym.symbol(core_ns_name)
    ns_var_sym = sym.symbol(ns_var_name, ns=core_ns_name)
    __NS = Maybe(Var.find(ns_var_sym)).or_else_raise(
        lambda: RuntimeException(f"Dynamic Var {ns_var_sym} not bound!")
    )

    def in_ns(s: sym.Symbol):
        ns = Namespace.get_or_create(s)
        __NS.value = ns
        return ns

    Var.intern_unbound(core_ns_sym, sym.symbol("unquote"))
    Var.intern_unbound(core_ns_sym, sym.symbol("unquote-splicing"))
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
