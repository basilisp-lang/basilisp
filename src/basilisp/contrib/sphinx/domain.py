from typing import Any, Dict, List, Tuple

from docutils.nodes import Element
from docutils.parsers.rst import directives
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.domains import Domain, ObjType
from sphinx.domains.python import (
    PyClasslike,
    PyClassMethod,
    PyFunction,
    PyMethod,
    PyModule,
    PyObject,
    PyProperty,
    PyStaticMethod,
    PyVariable,
)
from sphinx.roles import XRefRole
from sphinx.util.typing import OptionSpec


class BasilispFunction(PyFunction):

    option_spec: OptionSpec = PyFunction.option_spec.copy()
    option_spec.update(
        {
            "macro": directives.flag,
        }
    )

    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        print(f"{sig=} {signode=}")
        return super().handle_signature(sig, signode)

    def get_signature_prefix(self, sig: str) -> str:
        prefix = ""
        if "macro" in self.options:
            prefix = "macro "
        return prefix + super().get_signature_prefix(sig)


class BasilispClassLike(PyClasslike):
    def get_signature_prefix(self, sig: str) -> str:
        return self.objtype


class BasilispDomain(Domain):
    name = "lpy"
    label = "basilisp"
    object_types: Dict[str, ObjType] = {
        "namespace": ObjType("namespace", "ns", "obj"),
        "var": ObjType("var", "var", "obj"),
        "function": ObjType("lispfunction", "func", "obj"),
        "protocol": ObjType("protocol", "rec", "obj"),
        "record": ObjType("record", "rec", "obj"),
        "type": ObjType("type", "type", "obj"),
        "method": ObjType("method", "meth", "obj"),
        "classmethod": ObjType("class method", "meth", "obj"),
        "staticmethod": ObjType("static method", "meth", "obj"),
        "property": ObjType("property", "prop", "obj"),
    }

    directives = {
        "namespace": PyModule,
        "var": PyVariable,
        "lispfunction": BasilispFunction,
        "protocol": BasilispClassLike,
        "record": BasilispClassLike,
        "type": BasilispClassLike,
        "method": PyMethod,
        "classmethod": PyClassMethod,
        "staticmethod": PyStaticMethod,
        "property": PyProperty,
    }
    roles = {
        "ns": XRefRole(),
        "var": XRefRole(),
        "proto": XRefRole(),
        "rec": XRefRole(),
        "type": XRefRole(),
        "meth": XRefRole(),
        "prop": XRefRole(),
    }
    initial_data: Dict[str, Dict[str, Tuple[Any]]] = {
        "objects": {},  # fullname -> docname, objtype
        "modules": {},  # modname -> docname, synopsis, platform, deprecated
    }
    indices = []

    def merge_domaindata(self, docnames: List[str], otherdata: Dict) -> None:
        pass

    def resolve_any_xref(
        self,
        env: "BuildEnvironment",
        fromdocname: str,
        builder: "Builder",
        target: str,
        node: pending_xref,
        contnode: Element,
    ) -> List[Tuple[str, Element]]:
        pass
