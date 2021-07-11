from typing import Any, Dict, List, Tuple

from docutils import nodes
from docutils.nodes import Element
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.domains import Domain, ObjType
from sphinx.domains.python import (
    PyClassMethod,
    PyMethod,
    PyModule,
    PyObject,
    PyProperty,
    PyStaticMethod,
    PythonModuleIndex,
)
from sphinx.roles import XRefRole
from sphinx.util.typing import OptionSpec

from basilisp.lang import reader, runtime
from basilisp.lang import symbol as sym
from basilisp.lang.interfaces import IPersistentList


class BasilispObject(PyObject):
    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        return sig, ""

    def get_index_text(self, modname: str, name: Tuple[str, str]) -> str:
        sig, prefix = name
        sig_sexp = next(reader.read_str(sig), None)
        if isinstance(sig_sexp, sym.Symbol):
            return str(sym.symbol(sig, ns=modname))
        elif isinstance(sig_sexp, IPersistentList):
            return str(sym.symbol(runtime.first(sig_sexp), ns=modname))
        else:
            return name[0]


class BasilispVar(BasilispObject):
    option_spec: OptionSpec = PyObject.option_spec.copy()
    option_spec.update(
        {
            "dynamic": directives.unchanged,
            "type": directives.unchanged,
            "value": directives.unchanged,
        }
    )

    def get_signature_prefix(self, sig: str) -> str:
        return "" if "dynamic" not in self.options else "dynamic "

    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        prefix = self.get_signature_prefix(sig)
        if prefix:
            signode += addnodes.desc_annotation(prefix, prefix)

        signode += addnodes.desc_annotation("Var", "Var")
        signode += addnodes.desc_name(sig, sig)

        type_ = self.options.get("type")
        if type_:
            signode += addnodes.desc_annotation(type_, "", nodes.Text(": "), type_)

        value = self.options.get("value")
        if value:
            signode += addnodes.desc_annotation(value, " = " + value)

        return sig, prefix


class BasilispFunction(BasilispObject):
    option_spec: OptionSpec = BasilispObject.option_spec.copy()
    option_spec.update(
        {
            "async": directives.flag,
            "macro": directives.flag,
        }
    )

    def needs_arglist(self) -> bool:
        return True

    def get_signature_prefix(self, sig: str) -> str:
        prefix = "fn " if "macro" not in self.options else "macro "
        if "async" in self.options:
            prefix = f"async {prefix}"
        return prefix

    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        prefix = self.get_signature_prefix(sig)
        if prefix:
            signode += addnodes.desc_annotation(prefix, prefix)

        signode += addnodes.desc_name(sig, sig)
        return sig, prefix


class BasilispClassLike(BasilispObject):
    def get_signature_prefix(self, sig: str) -> str:
        return self.objtype

    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        prefix = self.get_signature_prefix(sig)
        if prefix:
            signode += addnodes.desc_annotation(prefix, prefix)

        signode += addnodes.desc_name(sig, sig)
        return sig, prefix


class BasilispNamespaceIndex(PythonModuleIndex):
    name = "nsindex"
    localname = "Basilisp Namespace Index"
    shortname = "namespaces"


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
        "var": BasilispVar,
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
    indices = [BasilispNamespaceIndex]

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
