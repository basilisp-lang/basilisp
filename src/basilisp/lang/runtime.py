import contextlib
import decimal
import functools
import importlib
import inspect
import itertools
import logging
import math
import re
import sys
import threading
import types
from collections.abc import Sequence
from fractions import Fraction
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from basilisp.lang import keyword as kw
from basilisp.lang import list as llist
from basilisp.lang import map as lmap
from basilisp.lang import obj as lobj
from basilisp.lang import seq as lseq
from basilisp.lang import set as lset
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec
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
    ITransientSet,
)
from basilisp.lang.reference import ReferenceBase
from basilisp.lang.typing import CompilerOpts, LispNumber
from basilisp.lang.util import OBJECT_DUNDER_METHODS, demunge, is_abstract, munge
from basilisp.util import Maybe

logger = logging.getLogger(__name__)

# Public constants
CORE_NS = "basilisp.core"
CORE_NS_SYM = sym.symbol(CORE_NS)
NS_VAR_NAME = "*ns*"
NS_VAR_SYM = sym.symbol(NS_VAR_NAME, ns=CORE_NS)
NS_VAR_NS = CORE_NS
REPL_DEFAULT_NS = "basilisp.user"
SUPPORTED_PYTHON_VERSIONS = frozenset({(3, 6), (3, 7), (3, 8), (3, 9)})

# Private string constants
_COMPILER_OPTIONS_VAR_NAME = "*compiler-options*"
_DEFAULT_READER_FEATURES_VAR_NAME = "*default-reader-features*"
_GENERATED_PYTHON_VAR_NAME = "*generated-python*"
_PRINT_GENERATED_PY_VAR_NAME = "*print-generated-python*"
_PRINT_DUP_VAR_NAME = "*print-dup*"
_PRINT_LENGTH_VAR_NAME = "*print-length*"
_PRINT_LEVEL_VAR_NAME = "*print-level*"
_PRINT_META_VAR_NAME = "*print-meta*"
_PRINT_READABLY_VAR_NAME = "*print-readably*"
_PYTHON_VERSION = "*python-version*"
_BASILISP_VERSION = "*basilisp-version*"

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
_LETFN = sym.symbol("letfn*")
_LOOP = sym.symbol("loop*")
_QUOTE = sym.symbol("quote")
_REIFY = sym.symbol("reify*")
_RECUR = sym.symbol("recur")
_REQUIRE = sym.symbol("require*")
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
    _LETFN,
    _LOOP,
    _QUOTE,
    _RECUR,
    _REIFY,
    _REQUIRE,
    _SET_BANG,
    _THROW,
    _TRY,
    _VAR,
)


# Reader Conditional default features
def _supported_python_versions_features() -> Iterable[kw.Keyword]:
    """Yield successive reader features corresponding to the various Python
    `major`.`minor` versions the current Python VM corresponds to amongst the
    set of supported Python versions.

    For example, for Python 3.6, we emit:
     - :lpy36  - to exactly match Basilisp running on Python 3.6
     - :lpy36+ - to match Basilisp running on Python 3.6 and later versions
     - :lpy36- - to match Basilisp running on Python 3.6 and earlier versions
     - :lpy37- - to match Basilisp running on Python 3.7 and earlier versions
     - :lpy38- - to match Basilisp running on Python 3.8 and earlier versions"""
    feature_kw = lambda major, minor, suffix="": kw.keyword(
        f"lpy{major}{minor}{suffix}"
    )

    yield feature_kw(sys.version_info.major, sys.version_info.minor)

    current = (sys.version_info.major, sys.version_info.minor)
    for version in SUPPORTED_PYTHON_VERSIONS:
        if current <= version:
            yield feature_kw(version[0], version[1], suffix="-")
        if current >= version:
            yield feature_kw(version[0], version[1], suffix="+")


READER_COND_BASILISP_FEATURE_KW = kw.keyword("lpy")
READER_COND_DEFAULT_FEATURE_KW = kw.keyword("default")
READER_COND_DEFAULT_FEATURE_SET = lset.s(
    READER_COND_BASILISP_FEATURE_KW,
    READER_COND_DEFAULT_FEATURE_KW,
    *_supported_python_versions_features(),
)

CompletionMatcher = Callable[[Tuple[sym.Symbol, Any]], bool]
CompletionTrimmer = Callable[[Tuple[sym.Symbol, Any]], str]


class BasilispModule(types.ModuleType):
    __basilisp_namespace__: "Namespace"
    __basilisp_bootstrapped__: bool = False


def _new_module(name: str, doc=None) -> BasilispModule:
    """Create a new empty Basilisp Python module.
    Modules are created for each Namespace when it is created."""
    mod = BasilispModule(name, doc=doc)
    mod.__loader__ = None
    mod.__package__ = None
    mod.__spec__ = None
    mod.__basilisp_bootstrapped__ = False
    return mod


class RuntimeException(Exception):
    pass


class _VarBindings(threading.local):
    __slots__ = ("bindings",)

    def __init__(self):
        self.bindings: List = []


class Unbound:
    __slots__ = ("var",)

    def __init__(self, v: "Var"):
        self.var = v

    def __repr__(self):  # pragma: no cover
        return f"Unbound(var={self.var})"

    def __eq__(self, other):
        return self is other or (isinstance(other, Unbound) and self.var == other.var)


class Var(IDeref, ReferenceBase):
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
        self,
        ns: "Namespace",
        name: sym.Symbol,
        dynamic: bool = False,
        meta: Optional[IPersistentMap] = None,
    ) -> None:
        self._ns = ns
        self._name = name
        self._root = Unbound(self)
        self._dynamic = dynamic
        self._is_bound = False
        self._tl = None
        self._meta = meta
        self._lock = threading.Lock()

        if dynamic:
            self._tl = _VarBindings()

            # If this var was created with the dynamic keyword argument, then the
            # Var metadata should also specify that the Var is dynamic.
            if isinstance(self._meta, lmap.PersistentMap):
                if not self._meta.val_at(_DYNAMIC_META_KEY):
                    self._meta = self._meta.assoc(_DYNAMIC_META_KEY, True)
            else:
                self._meta = lmap.map({_DYNAMIC_META_KEY: True})

    def __repr__(self):
        return f"#'{self.ns.name}/{self.name}"

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
        if self._meta is not None:
            return self._meta.val_at(_PRIVATE_META_KEY)
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
        return bool(self._dynamic and self._tl and self._tl.bindings)

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
        ns: Union["Namespace", sym.Symbol],
        name: sym.Symbol,
        val,
        dynamic: bool = False,
        meta=None,
    ) -> "Var":
        """Intern the value bound to the symbol `name` in namespace `ns`."""
        if isinstance(ns, sym.Symbol):
            ns = Namespace.get_or_create(ns)
        var = ns.intern(name, Var(ns, name, dynamic=dynamic, meta=meta))
        var.root = val
        return var

    @staticmethod
    def intern_unbound(
        ns: Union["Namespace", sym.Symbol],
        name: sym.Symbol,
        dynamic: bool = False,
        meta=None,
    ) -> "Var":
        """Create a new unbound `Var` instance to the symbol `name` in namespace `ns`."""
        if isinstance(ns, sym.Symbol):
            ns = Namespace.get_or_create(ns)
        return ns.intern(name, Var(ns, name, dynamic=dynamic, meta=meta))

    @staticmethod
    def find_in_ns(
        ns_or_sym: Union["Namespace", sym.Symbol], name_sym: sym.Symbol
    ) -> "Optional[Var]":
        """Return the value current bound to the name `name_sym` in the namespace
        specified by `ns_sym`."""
        ns = (
            Namespace.get(ns_or_sym) if isinstance(ns_or_sym, sym.Symbol) else ns_or_sym
        )
        if ns is not None:
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


AliasMap = lmap.PersistentMap[sym.Symbol, sym.Symbol]
Module = Union[BasilispModule, types.ModuleType]
ModuleMap = lmap.PersistentMap[sym.Symbol, Module]
NamespaceMap = lmap.PersistentMap[sym.Symbol, "Namespace"]
VarMap = lmap.PersistentMap[sym.Symbol, Var]


class Namespace(ReferenceBase):
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

    DEFAULT_IMPORTS = lset.set(
        map(
            sym.symbol,
            [
                "attr",
                "builtins",
                "functools",
                "io",
                "importlib",
                "operator",
                "sys",
                "basilisp.lang.atom",
                "basilisp.lang.compiler",
                "basilisp.lang.delay",
                "basilisp.lang.exception",
                "basilisp.lang.futures",
                "basilisp.lang.interfaces",
                "basilisp.lang.keyword",
                "basilisp.lang.list",
                "basilisp.lang.map",
                "basilisp.lang.multifn",
                "basilisp.lang.promise",
                "basilisp.lang.queue",
                "basilisp.lang.reader",
                "basilisp.lang.reduced",
                "basilisp.lang.runtime",
                "basilisp.lang.seq",
                "basilisp.lang.set",
                "basilisp.lang.symbol",
                "basilisp.lang.vector",
                "basilisp.lang.volatile",
                "basilisp.lang.util",
            ],
        )
    )

    _NAMESPACES: Atom[NamespaceMap] = Atom(lmap.PersistentMap.empty())

    __slots__ = (
        "_name",
        "_module",
        "_meta",
        "_lock",
        "_interns",
        "_refers",
        "_aliases",
        "_imports",
        "_import_aliases",
    )

    def __init__(self, name: sym.Symbol, module: BasilispModule = None) -> None:
        self._name = name
        self._module = Maybe(module).or_else(lambda: _new_module(name.as_python_sym()))

        self._meta: Optional[IPersistentMap] = None
        self._lock = threading.Lock()

        self._aliases: NamespaceMap = lmap.PersistentMap.empty()
        self._imports: ModuleMap = lmap.map(
            dict(
                map(
                    lambda s: (s, importlib.import_module(s.name)),
                    Namespace.DEFAULT_IMPORTS,
                )
            )
        )
        self._import_aliases: AliasMap = lmap.PersistentMap.empty()
        self._interns: VarMap = lmap.PersistentMap.empty()
        self._refers: VarMap = lmap.PersistentMap.empty()

    @property
    def name(self) -> str:
        return self._name.name

    @property
    def module(self) -> BasilispModule:
        return self._module

    @module.setter
    def module(self, m: BasilispModule):
        """Override the Python module for this Namespace.

        ***WARNING**
        This should only be done by basilisp.importer code to make sure the
        correct module is generated for `basilisp.core`."""
        self._module = m

    @property
    def aliases(self) -> NamespaceMap:
        """A mapping between a symbolic alias and another Namespace. The
        fully qualified name of a namespace is also an alias for itself."""
        with self._lock:
            return self._aliases

    @property
    def imports(self) -> ModuleMap:
        """A mapping of names to Python modules imported into the current
        namespace."""
        with self._lock:
            return self._imports

    @property
    def import_aliases(self) -> AliasMap:
        """A mapping of a symbolic alias and a Python module name."""
        with self._lock:
            return self._import_aliases

    @property
    def interns(self) -> VarMap:
        """A mapping between a symbolic name and a Var. The Var may point to
        code, data, or nothing, if it is unbound. Vars in `interns` are
        interned in _this_ namespace."""
        with self._lock:
            return self._interns

    @property
    def refers(self) -> VarMap:
        """A mapping between a symbolic name and a Var. Vars in refers are
        interned in another namespace and are only referred to without an
        alias in this namespace."""
        with self._lock:
            return self._refers

    def __repr__(self):
        return f"{self._name}"

    def __hash__(self):
        return hash(self._name)

    def require(self, ns_name: str, *aliases: sym.Symbol) -> BasilispModule:
        """Require the Basilisp Namespace named by `ns_name` and add any aliases given
        to this Namespace.

        This method is called in code generated for the `require*` special form."""
        try:
            ns_module = importlib.import_module(munge(ns_name))
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Basilisp namespace '{ns_name}' not found",
            ) from e
        else:
            assert isinstance(ns_module, BasilispModule)
            ns_sym = sym.symbol(ns_name)
            ns = self.get(ns_sym)
            assert ns is not None, "Namespace must exist after being required"
            if aliases:
                self.add_alias(ns, *aliases)
            return ns_module

    def add_alias(self, namespace: "Namespace", *aliases: sym.Symbol) -> None:
        """Add Symbol aliases for the given Namespace."""
        with self._lock:
            new_m = self._aliases
            for alias in aliases:
                new_m = new_m.assoc(alias, namespace)
            self._aliases = new_m

    def get_alias(self, alias: sym.Symbol) -> "Optional[Namespace]":
        """Get the Namespace aliased by Symbol or None if it does not exist."""
        with self._lock:
            return self._aliases.val_at(alias, None)

    def remove_alias(self, alias: sym.Symbol) -> None:
        """Remove the Namespace aliased by Symbol. Return None."""
        with self._lock:
            self._aliases = self._aliases.dissoc(alias)

    def intern(self, sym: sym.Symbol, var: Var, force: bool = False) -> Var:
        """Intern the Var given in this namespace mapped by the given Symbol.
        If the Symbol already maps to a Var, this method _will not overwrite_
        the existing Var mapping unless the force keyword argument is given
        and is True."""
        with self._lock:
            old_var = self._interns.val_at(sym, None)
            if old_var is None or force:
                self._interns = self._interns.assoc(sym, var)
            return self._interns.val_at(sym)

    def unmap(self, sym: sym.Symbol) -> None:
        with self._lock:
            self._interns = self._interns.dissoc(sym)

    def find(self, sym: sym.Symbol) -> Optional[Var]:
        """Find Vars mapped by the given Symbol input or None if no Vars are
        mapped by that Symbol."""
        with self._lock:
            v = self._interns.val_at(sym, None)
            if v is None:
                return self._refers.val_at(sym, None)
            return v

    def add_import(self, sym: sym.Symbol, module: Module, *aliases: sym.Symbol) -> None:
        """Add the Symbol as an imported Symbol in this Namespace. If aliases are given,
        the aliases will be applied to the"""
        with self._lock:
            self._imports = self._imports.assoc(sym, module)
            if aliases:
                m = self._import_aliases
                for alias in aliases:
                    m = m.assoc(alias, sym)
                self._import_aliases = m

    def get_import(self, sym: sym.Symbol) -> Optional[BasilispModule]:
        """Return the module if a moduled named by sym has been imported into
        this Namespace, None otherwise.

        First try to resolve a module directly with the given name. If no module
        can be resolved, attempt to resolve the module using import aliases."""
        with self._lock:
            mod = self._imports.val_at(sym, None)
            if mod is None:
                alias = self._import_aliases.get(sym, None)
                if alias is None:
                    return None
                return self._imports.val_at(alias, None)
            return mod

    def add_refer(self, sym: sym.Symbol, var: Var) -> None:
        """Refer var in this namespace under the name sym."""
        if not var.is_private:
            with self._lock:
                self._refers = self._refers.assoc(sym, var)

    def get_refer(self, sym: sym.Symbol) -> Optional[Var]:
        """Get the Var referred by Symbol or None if it does not exist."""
        with self._lock:
            return self._refers.val_at(sym, None)

    def refer_all(self, other_ns: "Namespace") -> None:
        """Refer all the Vars in the other namespace."""
        with self._lock:
            final_refers = self._refers
            for s, var in other_ns.interns.items():
                if not var.is_private:
                    final_refers = final_refers.assoc(s, var)
            self._refers = final_refers

    @classmethod
    def ns_cache(cls) -> lmap.PersistentMap:
        """Return a snapshot of the Namespace cache."""
        return cls._NAMESPACES.deref()

    @staticmethod
    def __get_or_create(
        ns_cache: NamespaceMap,
        name: sym.Symbol,
        module: BasilispModule = None,
    ) -> lmap.PersistentMap:
        """Private swap function used by `get_or_create` to atomically swap
        the new namespace map into the global cache."""
        ns = ns_cache.val_at(name, None)
        if ns is not None:
            return ns_cache
        new_ns = Namespace(name, module=module)
        return ns_cache.assoc(name, new_ns)

    @classmethod
    def get_or_create(
        cls, name: sym.Symbol, module: BasilispModule = None
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
        if name == CORE_NS_SYM:
            raise ValueError("Cannot remove the Basilisp core namespace")
        while True:
            oldval: lmap.PersistentMap = cls._NAMESPACES.deref()
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

    # pylint: disable=unnecessary-comprehension
    def __complete_alias(
        self, prefix: str, name_in_ns: Optional[str] = None
    ) -> Iterable[str]:
        """Return an iterable of possible completions matching the given
        prefix from the list of aliased namespaces. If name_in_ns is given,
        further attempt to refine the list to matching names in that namespace."""
        candidates = filter(
            Namespace.__completion_matcher(prefix),
            ((s, n) for s, n in self.aliases.items()),
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
                for alias, import_name in self.import_aliases.items()
            }
        )

        candidates = filter(
            Namespace.__completion_matcher(prefix),
            itertools.chain(aliases.items(), imports.items()),
        )
        if name_in_module is not None:
            for _, module in candidates:
                for name in module.__dict__:
                    if name.startswith(name_in_module):
                        yield f"{prefix}/{name}"
        else:
            for candidate_name, _ in candidates:
                yield f"{candidate_name}/"

    # pylint: disable=unnecessary-comprehension
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
            filter(is_match, ((s, v) for s, v in self.interns.items())),
        )

    # pylint: disable=unnecessary-comprehension
    def __complete_refers(self, value: str) -> Iterable[str]:
        """Return an iterable of possible completions matching the given
        prefix from the list of referred Vars."""
        return map(
            lambda entry: f"{entry[0].name}",
            filter(
                Namespace.__completion_matcher(value),
                ((s, v) for s, v in self.refers.items()),
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


def push_thread_bindings(m: IPersistentMap[Var, Any]) -> None:
    """Push thread local bindings for the Var keys in m using the values."""
    bindings = set()

    for var, val in m.items():
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

T = TypeVar("T")


@functools.singledispatch
def first(o):
    """If o is a ISeq, return the first element from o. If o is None, return
    None. Otherwise, coerces o to a Seq and returns the first."""
    s = to_seq(o)
    if s is None:
        return None
    return s.first


@first.register(type(None))
def _first_none(_: None) -> None:
    return None


@first.register(ISeq)
def _first_iseq(o: ISeq[T]) -> Optional[T]:
    return o.first


@functools.singledispatch
def rest(o) -> ISeq:
    """If o is a ISeq, return the elements after the first in o. If o is None,
    returns an empty seq. Otherwise, coerces o to a seq and returns the rest."""
    n = to_seq(o)
    if n is None:
        return lseq.EMPTY
    return n.rest


@rest.register(type(None))
def _rest_none(_: None) -> ISeq:
    return lseq.EMPTY


@rest.register(type(ISeq))
def _rest_iseq(o: ISeq[T]) -> ISeq:
    s = o.rest
    if s is None:
        return lseq.EMPTY
    return s


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
    return to_seq(rest(o))


def nthnext(coll, i: int) -> Optional[ISeq]:
    """Returns the nth next sequence of coll."""
    while True:
        if coll is None:
            return None
        if i == 0:
            return to_seq(coll)
        i -= 1
        coll = next_(coll)


@functools.singledispatch
def _cons(seq, o) -> ISeq:
    return Maybe(to_seq(seq)).map(lambda s: s.cons(o)).or_else(lambda: llist.l(o))


@_cons.register(type(None))
def _cons_none(_: None, o) -> ISeq:
    return llist.l(o)


@_cons.register(ISeq)
def _cons_iseq(seq: ISeq, o) -> ISeq:
    return seq.cons(o)


def cons(o, seq) -> ISeq:
    """Creates a new sequence where o is the first element and seq is the rest.
    If seq is None, return a list containing o. If seq is not a ISeq, attempt
    to coerce it to a ISeq and then cons o onto the resulting sequence."""
    return _cons(seq, o)


to_seq = lseq.to_seq


def concat(*seqs) -> ISeq:
    """Concatenate the sequences given by seqs into a single ISeq."""
    allseqs = lseq.sequence(itertools.chain.from_iterable(filter(None, seqs)))
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


def count(coll) -> int:
    try:
        return len(coll)
    except (AttributeError, TypeError):
        try:
            return sum(1 for _ in coll)
        except TypeError:
            raise TypeError(f"count not supported on object of type {type(coll)}")


__nth_sentinel = object()


@functools.singledispatch
def nth(coll, i: int, notfound=__nth_sentinel):
    """Returns the ith element of coll (0-indexed), if it exists.
    None otherwise. If i is out of bounds, throws an IndexError unless
    notfound is specified."""
    raise TypeError(f"nth not supported on object of type {type(coll)}")


@nth.register(type(None))
def _nth_none(_: None, i: int, notfound=__nth_sentinel) -> None:
    return notfound if notfound is not __nth_sentinel else None


@nth.register(Sequence)
def _nth_sequence(coll: Sequence, i: int, notfound=__nth_sentinel):
    try:
        return coll[i]
    except IndexError as ex:
        if notfound is not __nth_sentinel:
            return notfound
        raise ex


@nth.register(ISeq)
def _nth_iseq(coll: ISeq, i: int, notfound=__nth_sentinel):
    for j, e in enumerate(coll):
        if i == j:
            return e

    if notfound is not __nth_sentinel:
        return notfound

    raise IndexError(f"Index {i} out of bounds")


@functools.singledispatch
def contains(coll, k):
    """Return true if o contains the key k."""
    return k in coll


@contains.register(IAssociative)
def _contains_iassociative(coll, k):
    return coll.contains(k)


@functools.singledispatch
def get(m, k, default=None):  # pylint: disable=unused-argument
    """Return the value of k in m. Return default if k not found in m."""
    return default


@get.register(dict)
@get.register(list)
@get.register(str)
def _get_others(m, k, default=None):
    try:
        return m[k]
    except (KeyError, IndexError):
        return default


@get.register(IPersistentSet)
@get.register(ITransientSet)
@get.register(frozenset)
@get.register(set)
def _get_settypes(m, k, default=None):
    if k in m:
        return k
    return default


@get.register(ILookup)
def _get_ilookup(m, k, default=None):
    return m.val_at(k, default)


@functools.singledispatch
def assoc(m, *kvs):
    """Associate keys to values in associative data structure m. If m is None,
    returns a new Map with key-values kvs."""
    raise TypeError(
        f"Object of type {type(m)} does not implement IAssociative interface"
    )


@assoc.register(type(None))
def _assoc_none(_: None, *kvs) -> lmap.PersistentMap:
    return lmap.PersistentMap.empty().assoc(*kvs)


@assoc.register(IAssociative)
def _assoc_iassociative(m: IAssociative, *kvs):
    return m.assoc(*kvs)


@functools.singledispatch
def update(m, k, f, *args):
    """Updates the value for key k in associative data structure m with the return value from
    calling f(old_v, *args). If m is None, use an empty map. If k is not in m, old_v will be
    None."""
    raise TypeError(
        f"Object of type {type(m)} does not implement IAssociative interface"
    )


@update.register(type(None))
def _update_none(_: None, k, f, *args) -> lmap.PersistentMap:
    return lmap.PersistentMap.empty().assoc(k, f(None, *args))


@update.register(IAssociative)
def _update_iassociative(m: IAssociative, k, f, *args):
    old_v = m.val_at(k)
    new_v = f(old_v, *args)
    return m.assoc(k, new_v)


@functools.singledispatch
def conj(coll, *xs):
    """Conjoin xs to collection. New elements may be added in different positions
    depending on the type of coll. conj returns the same type as coll. If coll
    is None, return a list with xs conjoined."""
    raise TypeError(
        f"Object of type {type(coll)} does not implement "
        "IPersistentCollection interface"
    )


@conj.register(type(None))
def _conj_none(_: None, *xs):
    l = llist.PersistentList.empty()
    return l.cons(*xs)


@conj.register(IPersistentCollection)
def _conj_ipersistentcollection(coll: IPersistentCollection, *xs):
    return coll.cons(*xs)


def partial(f, *args, **kwargs):
    """Return a function which is the partial application of f with args and kwargs."""

    @functools.wraps(f)
    def partial_f(*inner_args, **inner_kwargs):
        return f(*itertools.chain(args, inner_args), **{**kwargs, **inner_kwargs})

    return partial_f


@functools.singledispatch
def deref(o, timeout_s=None, timeout_val=None):
    """Dereference a Deref object and return its contents.

    If o is an object implementing IBlockingDeref and timeout_s and
    timeout_val are supplied, deref will wait at most timeout_s seconds,
    returning timeout_val if timeout_s seconds elapse and o has not
    returned."""
    raise TypeError(f"Object of type {type(o)} cannot be dereferenced")


@deref.register(IBlockingDeref)
def _deref_blocking(
    o: IBlockingDeref, timeout_s: Optional[float] = None, timeout_val=None
):
    return o.deref(timeout_s, timeout_val)


@deref.register(IDeref)
def _deref(o: IDeref):
    return o.deref()


def equals(v1, v2) -> bool:
    """Compare two objects by value. Unlike the standard Python equality operator,
    this function does not consider 1 == True or 0 == False. All other equality
    operations are the same and performed using Python's equality operator."""
    if isinstance(v1, (bool, type(None))) or isinstance(v2, (bool, type(None))):
        return v1 is v2
    return v1 == v2


@functools.singledispatch
def divide(x: LispNumber, y: LispNumber) -> LispNumber:
    """Division reducer. If both arguments are integers, return a Fraction.
    Otherwise, return the true division of x and y."""
    return x / y


@divide.register(int)
def _divide_ints(x: int, y: LispNumber) -> LispNumber:
    if isinstance(y, int):
        return Fraction(x, y)
    return x / y


def quotient(num, div) -> LispNumber:
    """Return the integral quotient resulting from the division of num by div."""
    return math.trunc(num / div)


@functools.singledispatch
def compare(x, y) -> int:
    """Return either -1, 0, or 1 to indicate the relationship between x and y.

    This is a 3-way comparator commonly used in Java-derived systems. Python does not
    typically use 3-way comparators, so this function convert's Python's `__lt__` and
    `__gt__` method returns into one of the 3-way comparator return values."""
    if y is None:
        assert x is not None, "x cannot be nil"
        return 1
    return (x > y) - (x < y)


@compare.register(type(None))
def _compare_nil(_: None, y) -> int:
    # nil is less than all values, except itself.
    return 0 if y is None else -1


@compare.register(decimal.Decimal)
def _compare_decimal(x: decimal.Decimal, y) -> int:
    # Decimal instances will not compare with float("nan"), so we need a special case
    if isinstance(y, float):
        return -compare(y, x)  # pylint: disable=arguments-out-of-order
    return (x > y) - (x < y)


@compare.register(float)
def _compare_float(x, y) -> int:
    if y is None:
        return 1
    if math.isnan(x):
        return 0
    return (x > y) - (x < y)


@compare.register(IPersistentSet)
def _compare_sets(x: IPersistentSet, y) -> int:
    # Sets are not comparable (because there is no total ordering between sets).
    # However, in Python comparison is done using __lt__ and __gt__, which AbstractSet
    # inconveniently also uses as part of it's API for comparing sets with subset and
    # superset relationships. To "break" that, we just override the comparison method.
    # One consequence of this is that it may be possible to sort a collection of sets,
    # since `compare` isn't actually consulted in sorting.
    raise TypeError(
        f"cannot compare instances of '{type(x).__name__}' and '{type(y).__name__}'"
    )


def sort(coll, f=None) -> Optional[ISeq]:
    """Return a sorted sequence of the elements in coll. If a comparator
    function f is provided, compare elements in coll using f."""
    return lseq.sequence(sorted(coll, key=Maybe(f).map(functools.cmp_to_key).value))


def sort_by(keyfn, coll, cmp=None) -> Optional[ISeq]:
    """Return a sorted sequence of the elements in coll. If a comparator
    function f is provided, compare elements in coll using f."""
    if cmp is not None:

        class key:
            __slots__ = ("obj",)

            def __init__(self, obj):
                self.obj = obj

            def __lt__(self, other):
                return cmp(keyfn(self.obj), keyfn(other.obj)) < 0

            def __gt__(self, other):
                return cmp(keyfn(self.obj), keyfn(other.obj)) > 0

            def __eq__(self, other):
                return cmp(keyfn(self.obj), keyfn(other.obj)) == 0

            def __le__(self, other):
                return cmp(keyfn(self.obj), keyfn(other.obj)) <= 0

            def __ge__(self, other):
                return cmp(keyfn(self.obj), keyfn(other.obj)) >= 0

            __hash__ = None  # type: ignore

    else:
        key = keyfn  # type: ignore

    return lseq.sequence(sorted(coll, key=key))


def is_special_form(s: sym.Symbol) -> bool:
    """Return True if s names a special form."""
    return s in _SPECIAL_FORMS


@functools.singledispatch
def to_lisp(o, keywordize_keys: bool = True):  # pylint: disable=unused-argument
    """Recursively convert Python collections into Lisp collections."""
    return o


@to_lisp.register(list)
@to_lisp.register(tuple)
def _to_lisp_vec(o: Iterable, keywordize_keys: bool = True) -> vec.PersistentVector:
    return vec.vector(
        map(functools.partial(to_lisp, keywordize_keys=keywordize_keys), o)
    )


@functools.singledispatch
def _keywordize_keys(k):
    return k


@_keywordize_keys.register(str)
def _keywordize_keys_str(k):
    return kw.keyword(k)


@to_lisp.register(dict)
def _to_lisp_map(o: Mapping, keywordize_keys: bool = True) -> lmap.PersistentMap:
    process_key = _keywordize_keys if keywordize_keys else lambda x: x
    return lmap.map({process_key(k): v for k, v in o.items()})  # type: ignore[operator]


@to_lisp.register(frozenset)
@to_lisp.register(set)
def _to_lisp_set(o: AbstractSet, keywordize_keys: bool = True) -> lset.PersistentSet:
    return lset.set(map(functools.partial(to_lisp, keywordize_keys=keywordize_keys), o))


def _kw_name(kw: kw.Keyword) -> str:
    return kw.name


@functools.singledispatch
def to_py(
    o, keyword_fn: Callable[[kw.Keyword], Any] = _kw_name
):  # pylint: disable=unused-argument
    """Recursively convert Lisp collections into Python collections."""
    return o


@to_py.register(kw.Keyword)
def _to_py_kw(o: kw.Keyword, keyword_fn: Callable[[kw.Keyword], Any] = _kw_name) -> Any:
    return keyword_fn(o)


@to_py.register(IPersistentList)
@to_py.register(ISeq)
@to_py.register(IPersistentVector)
def _to_py_list(
    o: Union[IPersistentList, ISeq, IPersistentVector],
    keyword_fn: Callable[[kw.Keyword], Any] = _kw_name,
) -> list:
    return list(map(functools.partial(to_py, keyword_fn=keyword_fn), o))


@to_py.register(IPersistentMap)
def _to_py_map(
    o: IPersistentMap, keyword_fn: Callable[[kw.Keyword], Any] = _kw_name
) -> dict:
    return {
        to_py(key, keyword_fn=keyword_fn): to_py(value, keyword_fn=keyword_fn)
        for key, value in o.items()
    }


@to_py.register(IPersistentSet)
def _to_py_set(
    o: IPersistentSet, keyword_fn: Callable[[kw.Keyword], Any] = _kw_name
) -> set:
    return set(to_py(e, keyword_fn=keyword_fn) for e in o)


def lrepr(o, human_readable: bool = False) -> str:
    """Produce a string representation of an object. If human_readable is False,
    the string representation of Lisp objects is something that can be read back
    in by the reader as the same object."""
    core_ns = Namespace.get(CORE_NS_SYM)
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


def repl_completions(text: str) -> Iterable[str]:
    """Return an optional iterable of REPL completions."""
    # Can't complete Keywords, Numerals
    if __NOT_COMPLETEABLE.match(text):
        return ()
    elif text.startswith(":"):
        return kw.complete(text)
    else:
        ns = get_current_ns()
        return ns.complete(text)


####################
# Compiler Support #
####################


@functools.singledispatch
def _collect_args(args) -> ISeq:
    """Collect Python starred arguments into a Basilisp list."""
    raise TypeError("Python variadic arguments should always be a tuple")


@_collect_args.register(tuple)
def _collect_args_tuple(args: tuple) -> ISeq:
    return llist.list(args)


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


def _lisp_fn_apply_kwargs(f):
    """Convert a Python function into a Lisp function.

    Python keyword arguments will be converted into Lisp keyword/argument pairs
    that can be easily understood by Basilisp.

    Lisp functions annotated with the `:apply` value for the `:kwargs` metadata key
    will be wrapped with this decorator by the compiler."""

    @functools.wraps(f)
    def wrapped_f(*args, **kwargs):
        return f(
            *args,
            *itertools.chain.from_iterable(
                (kw.keyword(demunge(k)), v) for k, v in kwargs.items()
            ),
        )

    return wrapped_f


def _lisp_fn_collect_kwargs(f):
    """Convert a Python function into a Lisp function.

    Python keyword arguments will be collected into a single map, which is supplied
    as the final positional argument.

    Lisp functions annotated with the `:collect` value for the `:kwargs` metadata key
    will be wrapped with this decorator by the compiler."""

    @functools.wraps(f)
    def wrapped_f(*args, **kwargs):
        return f(
            *args,
            lmap.map({kw.keyword(demunge(k)): v for k, v in kwargs.items()}),
        )

    return wrapped_f


def _with_attrs(**kwargs):
    """Decorator to set attributes on a function. Returns the original
    function after setting the attributes named by the keyword arguments."""

    def decorator(f):
        for k, v in kwargs.items():
            setattr(f, k, v)
        return f

    return decorator


def _fn_with_meta(f, meta: Optional[lmap.PersistentMap]):
    """Return a new function with the given meta. If the function f already
    has a meta map, then merge the new meta with the existing meta."""

    if not isinstance(meta, lmap.PersistentMap):
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
        if hasattr(f, "meta") and isinstance(f.meta, lmap.PersistentMap)
        else meta
    )
    wrapped_f.with_meta = partial(_fn_with_meta, wrapped_f)  # type: ignore
    return wrapped_f


def _basilisp_fn(arities: Tuple[Union[int, kw.Keyword]]):
    """Create a Basilisp function, setting meta and supplying a with_meta
    method implementation."""

    def wrap_fn(f):
        assert not hasattr(f, "meta")
        f._basilisp_fn = True
        f.arities = lset.set(arities)
        f.meta = None
        f.with_meta = partial(_fn_with_meta, f)
        return f

    return wrap_fn


def _basilisp_type(
    fields: Iterable[str],
    interfaces: Iterable[Type],
    artificially_abstract_bases: AbstractSet[Type],
    members: Iterable[str],
):
    """Check that a Basilisp type (defined by `deftype*`) only declares abstract
    super-types and that all abstract methods are implemented."""

    def wrap_class(cls: Type):
        field_names = frozenset(fields)
        member_names = frozenset(members)
        artificially_abstract_base_members: Set[str] = set()
        all_member_names = field_names.union(member_names)
        all_interface_methods: Set[str] = set()
        for interface in interfaces:
            if interface is object:
                continue

            if is_abstract(interface):
                interface_names: FrozenSet[str] = interface.__abstractmethods__
                interface_property_names: FrozenSet[str] = frozenset(
                    method
                    for method in interface_names
                    if isinstance(getattr(interface, method), property)
                )
                interface_method_names = interface_names - interface_property_names
                if not interface_method_names.issubset(member_names):
                    missing_methods = ", ".join(interface_method_names - member_names)
                    raise RuntimeException(
                        "deftype* definition missing interface members for interface "
                        f"{interface}: {missing_methods}",
                    )
                elif not interface_property_names.issubset(all_member_names):
                    missing_fields = ", ".join(interface_property_names - field_names)
                    raise RuntimeException(
                        "deftype* definition missing interface properties for interface "
                        f"{interface}: {missing_fields}",
                    )

                all_interface_methods.update(interface_names)
            elif interface in artificially_abstract_bases:
                artificially_abstract_base_members.update(
                    map(
                        lambda v: v[0],
                        inspect.getmembers(
                            interface,
                            predicate=lambda v: inspect.isfunction(v)
                            or isinstance(v, (property, staticmethod))
                            or inspect.ismethod(v),
                        ),
                    )
                )
            else:
                raise RuntimeException(
                    "deftype* interface must be Python abstract class or object",
                )

        extra_methods = member_names - all_interface_methods - OBJECT_DUNDER_METHODS
        if extra_methods and not extra_methods.issubset(
            artificially_abstract_base_members
        ):
            extra_method_str = ", ".join(extra_methods)
            raise RuntimeException(
                "deftype* definition for interface includes members not part of "
                f"defined interfaces: {extra_method_str}"
            )

        return cls

    return wrap_class


###############################
# Symbol and Alias Resolution #
###############################


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


#######################
# Namespace Utilities #
#######################


@contextlib.contextmanager
def ns_bindings(ns_name: str, module: BasilispModule = None) -> Iterator[Namespace]:
    """Context manager for temporarily changing the value of basilisp.core/*ns*."""
    symbol = sym.symbol(ns_name)
    ns = Namespace.get_or_create(symbol, module=module)
    ns_var = Maybe(Var.find(NS_VAR_SYM)).or_else_raise(
        lambda: RuntimeException(f"Dynamic Var {NS_VAR_SYM} not bound!")
    )

    try:
        logger.debug(f"Binding {NS_VAR_SYM} to {ns}")
        ns_var.push_bindings(ns)
        yield ns_var.value
    finally:
        ns_var.pop_bindings()
        logger.debug(f"Reset bindings for {NS_VAR_SYM} to {ns_var.value}")


@contextlib.contextmanager
def remove_ns_bindings():
    """Context manager to pop the most recent bindings for basilisp.core/*ns* after
    completion of the code under management."""
    ns_var = Maybe(Var.find(NS_VAR_SYM)).or_else_raise(
        lambda: RuntimeException(f"Dynamic Var {NS_VAR_SYM} not bound!")
    )
    try:
        yield
    finally:
        ns_var.pop_bindings()
        logger.debug(f"Reset bindings for {NS_VAR_SYM} to {ns_var.value}")


def get_current_ns() -> Namespace:
    """Get the value of the dynamic variable `*ns*` in the current thread."""
    ns: Namespace = (
        Maybe(Var.find(NS_VAR_SYM))
        .map(lambda v: v.value)
        .or_else_raise(lambda: RuntimeException(f"Dynamic Var {NS_VAR_SYM} not bound!"))
    )
    return ns


def set_current_ns(
    ns_name: str,
    module: BasilispModule = None,
) -> Var:
    """Set the value of the dynamic variable `*ns*` in the current thread."""
    symbol = sym.symbol(ns_name)
    ns = Namespace.get_or_create(symbol, module=module)
    ns_var = Maybe(Var.find(NS_VAR_SYM)).or_else_raise(
        lambda: RuntimeException(f"Dynamic Var {NS_VAR_SYM} not bound!")
    )
    ns_var.push_bindings(ns)
    logger.debug(f"Setting {NS_VAR_SYM} to {ns}")
    return ns_var


##############################
# Emit Generated Python Code #
##############################


def add_generated_python(
    generated_python: str,
    which_ns: Optional[Namespace] = None,
) -> None:
    """Add generated Python code to a dynamic variable in which_ns."""
    if which_ns is None:
        which_ns = get_current_ns()
    v = Maybe(which_ns.find(sym.symbol(_GENERATED_PYTHON_VAR_NAME))).or_else(
        lambda: Var.intern(
            which_ns,  # type: ignore
            sym.symbol(_GENERATED_PYTHON_VAR_NAME),
            "",
            dynamic=True,
            meta=lmap.map({_PRIVATE_META_KEY: True}),
        )
    )
    # Accessing the Var root via the property uses a lock, which is the
    # desired behavior for Basilisp code, but it introduces additional
    # startup time when there will not realistically be any contention.
    v._root = v._root + generated_python  # type: ignore


def print_generated_python() -> bool:
    """Return the value of the `*print-generated-python*` dynamic variable."""
    ns_sym = sym.symbol(_PRINT_GENERATED_PY_VAR_NAME, ns=CORE_NS)
    return (
        Maybe(Var.find(ns_sym))
        .map(lambda v: v.value)
        .or_else_raise(lambda: RuntimeException(f"Dynamic Var {ns_sym} not bound!"))
    )


#########################
# Bootstrap the Runtime #
#########################


def init_ns_var() -> Var:
    """Initialize the dynamic `*ns*` variable in the `basilisp.core` Namespace."""
    core_ns = Namespace.get_or_create(CORE_NS_SYM)
    ns_var = Var.intern(core_ns, sym.symbol(NS_VAR_NAME), core_ns, dynamic=True)
    logger.debug(f"Created namespace variable {NS_VAR_SYM}")
    return ns_var


def bootstrap_core(compiler_opts: CompilerOpts) -> None:
    """Bootstrap the environment with functions that are either difficult to express
    with the very minimal Lisp environment or which are expected by the compiler."""
    _NS = Maybe(Var.find(NS_VAR_SYM)).or_else_raise(
        lambda: RuntimeException(f"Dynamic Var {NS_VAR_SYM} not bound!")
    )

    def in_ns(s: sym.Symbol):
        ns = Namespace.get_or_create(s)
        _NS.value = ns
        return ns

    # Vars used in bootstrapping the runtime
    Var.intern_unbound(CORE_NS_SYM, sym.symbol("unquote"))
    Var.intern_unbound(CORE_NS_SYM, sym.symbol("unquote-splicing"))
    Var.intern(
        CORE_NS_SYM, sym.symbol("in-ns"), in_ns, meta=lmap.map({_REDEF_META_KEY: True})
    )

    # Dynamic Var examined by the compiler when importing new Namespaces
    Var.intern(
        CORE_NS_SYM,
        sym.symbol(_COMPILER_OPTIONS_VAR_NAME),
        compiler_opts,
        dynamic=True,
    )

    # Dynamic Var for introspecting the default reader featureset
    Var.intern(
        CORE_NS_SYM,
        sym.symbol(_DEFAULT_READER_FEATURES_VAR_NAME),
        READER_COND_DEFAULT_FEATURE_SET,
        dynamic=True,
    )

    # Dynamic Vars examined by the compiler for generating Python code for debugging
    Var.intern(
        CORE_NS_SYM,
        sym.symbol(_PRINT_GENERATED_PY_VAR_NAME),
        False,
        dynamic=True,
        meta=lmap.map({_PRIVATE_META_KEY: True}),
    )
    Var.intern(
        CORE_NS_SYM,
        sym.symbol(_GENERATED_PYTHON_VAR_NAME),
        "",
        dynamic=True,
        meta=lmap.map({_PRIVATE_META_KEY: True}),
    )

    # Dynamic Vars for controlling printing
    Var.intern(
        CORE_NS_SYM, sym.symbol(_PRINT_DUP_VAR_NAME), lobj.PRINT_DUP, dynamic=True
    )
    Var.intern(
        CORE_NS_SYM, sym.symbol(_PRINT_LENGTH_VAR_NAME), lobj.PRINT_LENGTH, dynamic=True
    )
    Var.intern(
        CORE_NS_SYM, sym.symbol(_PRINT_LEVEL_VAR_NAME), lobj.PRINT_LEVEL, dynamic=True
    )
    Var.intern(
        CORE_NS_SYM, sym.symbol(_PRINT_META_VAR_NAME), lobj.PRINT_META, dynamic=True
    )
    Var.intern(
        CORE_NS_SYM,
        sym.symbol(_PRINT_READABLY_VAR_NAME),
        lobj.PRINT_READABLY,
        dynamic=True,
    )

    # Version info
    from basilisp.__version__ import VERSION

    Var.intern(
        CORE_NS_SYM,
        sym.symbol(_PYTHON_VERSION),
        vec.vector(sys.version_info),
        dynamic=True,
    )
    Var.intern(
        CORE_NS_SYM, sym.symbol(_BASILISP_VERSION), vec.vector(VERSION), dynamic=True
    )


def get_compiler_opts() -> CompilerOpts:
    """Return the current compiler options map."""
    v = Var.find_in_ns(CORE_NS_SYM, sym.symbol(_COMPILER_OPTIONS_VAR_NAME))
    assert v is not None, "*compiler-options* Var not defined"
    return cast(CompilerOpts, v.value)
