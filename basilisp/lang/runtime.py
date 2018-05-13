import pkg_resources
import basilisp.lang.namespace as namespace
import basilisp.lang.symbol as sym
import basilisp.lang.var as var

_CORE_NS = namespace._CORE_NS
_CORE_NS_FILE = 'core.lpy'
_REPL_DEFAULT_NS = 'user'
_NS_VAR_NAME = '*ns*'
_NS_VAR_NS = _CORE_NS
_PRINT_GENERATED_PY_VAR_NAME = '*print-generated-python*'


def init_ns_var(which_ns=_CORE_NS, ns_var_name=_NS_VAR_NAME) -> var.Var:
    """Initialize the dynamic `*ns*` variable in the Namespace `which_ns`."""
    core_sym = sym.Symbol(which_ns)
    core_ns = namespace.get_or_create(core_sym)
    ns_var = var.intern(
        core_sym, sym.Symbol(ns_var_name), core_ns, dynamic=True)
    return ns_var


def set_current_ns(ns_name, ns_var_name=_NS_VAR_NAME,
                   ns_var_ns=_NS_VAR_NS) -> var.Var:
    """Set the value of the dynamic variable `*ns*` in the current thread."""
    symbol = sym.Symbol(ns_name)
    ns = namespace.get_or_create(symbol)
    ns_var = var.find(sym.Symbol(ns_var_name, ns=ns_var_ns))
    ns_var.push_bindings(ns)
    return ns_var


def get_current_ns(ns_var_name=_NS_VAR_NAME, ns_var_ns=_NS_VAR_NS) -> var.Var:
    """Set the value of the dynamic variable `*ns*` in the current thread."""
    ns_sym = sym.Symbol(ns_var_name, ns=ns_var_ns)
    return var.find(ns_sym)


def print_generated_python(var_name=_PRINT_GENERATED_PY_VAR_NAME,
                           core_ns_name=_CORE_NS) -> bool:
    """Return the value of the `*print-generated-python*` dynamic variable."""
    ns_sym = sym.Symbol(var_name, ns=core_ns_name)
    return var.find(ns_sym).value


def core_resource(resource=_CORE_NS_FILE):
    return pkg_resources.resource_filename('basilisp', resource)


def bootstrap(ns_var_name=_NS_VAR_NAME, core_ns_name=_CORE_NS) -> None:
    """Bootstrap the environment with functions that are are difficult to
    express with the very minimal lisp environment."""
    core_ns_sym = sym.symbol(core_ns_name)
    ns_var_sym = sym.symbol(ns_var_name, ns=core_ns_name)
    __NS = var.find(ns_var_sym)

    def set_BANG_(var_sym: sym.Symbol, expr):
        ns, name = None, var_sym.name
        if var_sym.ns is None:
            ns = __NS.value.name

        v = var.find(sym.symbol(name, ns=ns))
        v.value = expr
        return expr

    def in_ns(s: sym.Symbol):
        ns = namespace.get_or_create(s)
        set_BANG_(ns_var_sym, ns)
        return ns

    var.intern(core_ns_sym, sym.symbol('set!'), set_BANG_)
    var.intern(core_ns_sym, sym.symbol('in-ns'), in_ns)
    var.intern(
        core_ns_sym,
        sym.symbol(_PRINT_GENERATED_PY_VAR_NAME),
        True,
        dynamic=True)
