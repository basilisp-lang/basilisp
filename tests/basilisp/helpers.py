from typing import Tuple

import basilisp.lang.symbol as sym
from basilisp.lang.runtime import CORE_NS, Namespace


def get_or_create_ns(
    name: sym.Symbol, refer: Tuple[sym.Symbol] = (sym.symbol(CORE_NS),)
) -> Namespace:
    """Get or create the namespace named by `name`, referring in all of the symbols
    of the namespaced named by `refer`."""
    ns = get_or_create_ns(name)
    for refer_name in filter(lambda s: s != name, refer):
        refer_ns = get_or_create_ns(refer_name)
        ns.refer_all(refer_ns)
    return ns
