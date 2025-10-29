from collections.abc import Callable
from typing import Any, Tuple

from basilisp.lang import symbol as sym
from basilisp.lang.runtime import CORE_NS_SYM, Namespace

CompileFn = Callable[[str], Any]


def get_or_create_ns(
    name: sym.Symbol, refer: tuple[sym.Symbol] = (CORE_NS_SYM,)
) -> Namespace:
    """Get or create the namespace named by `name`, referring in all of the symbols
    of the namespaced named by `refer`."""
    ns = Namespace.get_or_create(name)
    for refer_name in filter(lambda s: s != name, refer):
        refer_ns = Namespace.get_or_create(refer_name)
        ns.refer_all(refer_ns)
    return ns
