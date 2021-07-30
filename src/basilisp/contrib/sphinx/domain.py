from collections import defaultdict
from typing import Any, Dict, Iterable, List, NamedTuple, Tuple, cast

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives  # type: ignore[attr-defined]
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.builders import Builder
from sphinx.domains import Domain, Index, IndexEntry, ObjType
from sphinx.domains.python import (
    PyClassMethod,
    PyMethod,
    PyObject,
    PyProperty,
    PyStaticMethod,
)
from sphinx.environment import BuildEnvironment
from sphinx.roles import XRefRole
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_id
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator

from basilisp.lang import reader, runtime
from basilisp.lang import symbol as sym
from basilisp.lang.interfaces import IPersistentList, IPersistentMap, IPersistentVector


class desc_lispparameterlist(addnodes.desc_parameterlist):
    """
    Node for a Lisp function parameter list.

    This is required to be distinct from `sphinx.addnodes.desc_parameterlist` to avoid
    Sphinx's HTML emitter to include surrounding parens for the argument list.
    """

    child_text_separator = " "


class desc_lispparameter(addnodes.desc_parameter):
    """
    Node for Lisp function parameter.
    """

    pass


# Visitor functions for the Sphinx HTML translator which are required to not have
# Sphinx emit Python-focused HTML for function signatures.


def visit_lispparameterlist(self: HTMLTranslator, node: Element):
    self.first_param = True
    if len(node):
        self.body.append(" ")
    self.param_separator = node.child_text_separator


def depart_lispparameterlist(_: HTMLTranslator, __: Element):
    pass


def visit_lispparameter(self: HTMLTranslator, _: Element):
    self.body.append(self.param_separator)


def depart_lispparameter(_: HTMLTranslator, __: Element):
    pass


class BasilispCurrentNamespace(SphinxDirective):
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False

    option_spec: OptionSpec = {}

    def run(self) -> List[Node]:
        modname = self.arguments[0].strip()
        if modname == "None":
            self.env.ref_context.pop("lpy:namespace", None)
        else:
            self.env.ref_context["lpy:namespace"] = modname
        return []


class BasilispNamespace(SphinxDirective):
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False

    option_spec: OptionSpec = {
        "platform": lambda x: x,
        "synopsis": lambda x: x,
        "noindex": directives.flag,
        "deprecated": directives.flag,
    }

    def run(self) -> List[Node]:
        domain = cast(BasilispDomain, self.env.get_domain("lpy"))

        modname = self.arguments[0].strip()
        noindex = "noindex" in self.options
        self.env.ref_context["lpy:namespace"] = modname
        ret: List[Node] = []
        if not noindex:
            # note module to the domain
            node_id = make_id(self.env, self.state.document, "namespace", modname)
            target = nodes.target("", "", ids=[node_id], ismod=True)
            self.set_source_info(target)

            self.state.document.note_explicit_target(target)

            domain.note_namespace(
                modname,
                node_id,
                self.options.get("synopsis", ""),
                self.options.get("platform", ""),
                "deprecated" in self.options,
            )
            domain.note_var(modname, "namespace", node_id, location=target)

            ret.append(target)
            indextext = f"namespace; {modname}"
            inode = addnodes.index(entries=[("pair", indextext, node_id, "", None)])
            ret.append(inode)
        return ret


class BasilispObject(PyObject):
    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        """Subclasses should implement this themselves."""
        return NotImplemented


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
        prefix = "Var "
        if "dynamic" in self.options:
            prefix = f"dynamic {prefix}"
        return prefix

    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        prefix = self.get_signature_prefix(sig)
        if prefix:
            signode += addnodes.desc_annotation(prefix, prefix)

        signode += addnodes.desc_name(sig, sig)

        type_ = self.options.get("type")
        if type_:
            signode += addnodes.desc_annotation(type_, "", nodes.Text(": "), type_)

        value = self.options.get("value")
        if value:
            signode += addnodes.desc_annotation(value, " = " + value)

        return sig, prefix

    def get_index_text(self, modname: str, name: Tuple[str, str]) -> str:
        sig, prefix = name
        return f"{sig} ({prefix} Var in {modname})"


class BasilispFunction(BasilispObject):
    option_spec: OptionSpec = BasilispObject.option_spec.copy()
    option_spec.update(
        {
            "async": directives.flag,
            "macro": directives.flag,
            "multi": directives.flag,
        }
    )

    def get_signature_prefix(self, sig: str) -> str:
        prefix = "fn " if "macro" not in self.options else "macro "
        if "multi" in self.options:
            prefix = f"multi {prefix}"
        if "async" in self.options:
            prefix = f"async {prefix}"
        return prefix

    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        prefix = self.get_signature_prefix(sig)
        if prefix:
            signode += addnodes.desc_annotation(prefix, prefix)

        sig_sexp = runtime.first(reader.read_str(sig))
        assert isinstance(sig_sexp, IPersistentList)
        fn_sym = runtime.first(sig_sexp)
        assert isinstance(fn_sym, sym.Symbol)

        signode += addnodes.desc_addname("(", "(")
        signode += addnodes.desc_name(fn_sym.name, fn_sym.name)

        param_list = desc_lispparameterlist()
        for param in runtime.rest(sig_sexp):
            param_node = desc_lispparameter()
            assert isinstance(param, (sym.Symbol, IPersistentVector, IPersistentMap))
            param_node += addnodes.desc_sig_name("", runtime.lrepr(param))
            param_list += param_node
        signode += param_list

        signode += addnodes.desc_addname(")", ")")
        return fn_sym.name, prefix

    def get_index_text(self, modname: str, name: Tuple[str, str]) -> str:
        sig, prefix = name
        sig_sexp = runtime.first(reader.read_str(sig))
        if isinstance(sig_sexp, IPersistentList):
            sig = runtime.first(sig_sexp)
        return f"{sig} ({prefix} in {modname})"


class BasilispClassLike(BasilispObject):
    def get_signature_prefix(self, sig: str) -> str:
        return self.objtype

    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        prefix = self.get_signature_prefix(sig)
        if prefix:
            signode += addnodes.desc_annotation(prefix, prefix)

        signode += addnodes.desc_name(sig, sig)
        return sig, prefix

    def get_index_text(self, modname: str, name: Tuple[str, str]) -> str:
        sig, prefix = name
        return f"{sig} ({prefix} in {modname})"


class BasilispNamespaceIndex(Index):
    name = "nsindex"
    localname = "Basilisp Namespace Index"
    shortname = "namespaces"

    def generate(
        self, docnames: Iterable[str] = None
    ) -> Tuple[List[Tuple[str, List[IndexEntry]]], bool]:
        content: Dict[str, List[IndexEntry]] = defaultdict(list)

        ignores: List[str] = self.domain.env.config["modindex_common_prefix"]
        ignores = sorted(ignores, key=len, reverse=True)
        namespaces = sorted(
            self.domain.data["namespaces"].items(), key=lambda x: x[0].lower()
        )

        prev_ns = ""
        num_toplevels = 0
        for nsname, (docname, node_id, synopsis, platforms, deprecated) in namespaces:
            if docnames and docname not in docnames:
                continue

            for ignore in ignores:
                if nsname.startswith(ignore):
                    nsname = nsname[len(ignore) :]
                    stripped = ignore
                    break
            else:
                stripped = ""

            if not nsname:
                nsname, stripped = stripped, ""

            entries = content[nsname[0].lower()]

            package = nsname.split(".")[0]
            if package != nsname:
                # it's a child namespace
                if prev_ns == package:
                    # first child namespace - make parent a group head
                    if entries:
                        last = entries[-1]
                        entries[-1] = IndexEntry(
                            last[0], 1, last[2], last[3], last[4], last[5], last[6]
                        )
                elif not prev_ns.startswith(package):
                    # child namespace without parent in list, add dummy entry
                    entries.append(
                        IndexEntry(stripped + package, 1, "", "", "", "", "")
                    )
                subtype = 2
            else:
                num_toplevels += 1
                subtype = 0

            qualifier = "Deprecated" if deprecated else ""
            entries.append(
                IndexEntry(
                    stripped + nsname,
                    subtype,
                    docname,
                    node_id,
                    platforms,
                    qualifier,
                    synopsis,
                )
            )
            prev_ns = nsname

        # apply heuristics when to collapse modindex at page load:
        # only collapse if number of toplevel modules is larger than
        # number of submodules
        collapse = len(namespaces) - num_toplevels < num_toplevels

        # sort by first letter
        sorted_content = sorted(content.items())

        return sorted_content, collapse


class VarEntry(NamedTuple):
    docname: str
    node_id: str
    objtype: str
    aliased: bool


class NamespaceEntry(NamedTuple):
    docname: str
    node_id: str
    synopsis: str
    platform: str
    deprecated: bool


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
        "currentns": BasilispCurrentNamespace,
        "namespace": BasilispNamespace,
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
        "fn": XRefRole(),
        "proto": XRefRole(),
        "rec": XRefRole(),
        "type": XRefRole(),
        "meth": XRefRole(),
        "prop": XRefRole(),
    }
    initial_data: Dict[str, Dict[str, Tuple[Any]]] = {
        "vars": {},  # fullname -> docname, objtype
        "namespaces": {},  # nsname -> docname, synopsis, platform, deprecated
    }
    indices = [BasilispNamespaceIndex]

    @property
    def vars(self) -> Dict[str, VarEntry]:
        return self.data.setdefault("vars", {})

    def note_var(
        self,
        name: str,
        objtype: str,
        node_id: str,
        aliased: bool = False,
        location: Any = None,
    ) -> None:
        """Note a Basilisp var for cross reference."""
        # if name in self.vars:
        #     other = self.vars[name]
        #     if other.aliased and aliased is False:
        #         # The original definition found. Override it!
        #         pass
        #     elif other.aliased is False and aliased:
        #         # The original definition is already registered.
        #         return
        #     else:
        #         # duplicated
        #         pass
        self.vars[name] = VarEntry(self.env.docname, node_id, objtype, aliased)

    @property
    def namespaces(self) -> Dict[str, NamespaceEntry]:
        return self.data.setdefault("namespaces", {})

    def note_namespace(
        self, name: str, node_id: str, synopsis: str, platform: str, deprecated: bool
    ) -> None:
        """Note a Basilisp Namespace module for cross reference."""
        self.namespaces[name] = NamespaceEntry(
            self.env.docname,
            node_id,
            synopsis,
            platform,
            deprecated,
        )

    def merge_domaindata(self, docnames: List[str], otherdata: Dict) -> None:
        pass

    def resolve_any_xref(
        self,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Builder,
        target: str,
        node: pending_xref,
        contnode: Element,
    ) -> List[Tuple[str, Element]]:
        pass
