import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, NamedTuple, Optional, Union, cast

from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
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
from sphinx.util.nodes import make_id, make_refnode
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator

from basilisp.lang import reader, runtime
from basilisp.lang import symbol as sym
from basilisp.lang.interfaces import IPersistentList

logger = logging.getLogger(__name__)


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

    def run(self) -> list[Node]:
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

    def run(self) -> list[Node]:
        domain = cast(BasilispDomain, self.env.get_domain("lpy"))

        modname = self.arguments[0].strip()
        noindex = "noindex" in self.options
        self.env.ref_context["lpy:namespace"] = modname
        ret: list[Node] = []
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

            ret.append(target)
            indextext = f"namespace; {modname}"
            inode = addnodes.index(entries=[("pair", indextext, node_id, "", None)])
            ret.append(inode)
        return ret


class BasilispObject(PyObject):  # pylint: disable=abstract-method
    option_spec: OptionSpec = PyObject.option_spec.copy()
    option_spec.update(
        {
            "filename": directives.unchanged,
            "lines": directives.unchanged,
        }
    )

    def handle_signature(self, sig: str, signode: desc_signature) -> tuple[str, str]:
        """Subclasses should implement this themselves."""
        return NotImplemented

    def add_target_and_index(
        self, name_cls: tuple[str, str], sig: str, signode: desc_signature
    ) -> None:
        modname = self.options.get("module", self.env.ref_context.get("lpy:namespace"))
        fullname = (modname + "/" if modname else "") + name_cls[0]
        node_id = fullname
        signode["ids"].append(node_id)

        self.state.document.note_explicit_target(signode)

        domain = cast(BasilispDomain, self.env.get_domain("lpy"))
        domain.note_var(fullname, self.objtype, node_id)

        if "noindexentry" not in self.options:
            indextext = self.get_index_text(  # pylint: disable=assignment-from-no-return, useless-suppression
                modname, name_cls
            )
            if indextext:
                self.indexnode["entries"].append(
                    ("single", indextext, node_id, "", None)
                )

    def _add_source_annotations(self, signode: desc_signature) -> None:
        for metadata in ("filename", "lines"):
            if val := self.options.get(metadata):
                signode[metadata] = val


class BasilispVar(BasilispObject):
    option_spec: OptionSpec = BasilispObject.option_spec.copy()
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

    def handle_signature(self, sig: str, signode: desc_signature) -> tuple[str, str]:
        prefix = self.get_signature_prefix(sig)
        if prefix:
            signode += addnodes.desc_annotation(prefix, prefix)

        signode += addnodes.desc_name(sig, sig)

        self._add_source_annotations(signode)

        type_ = self.options.get("type")
        if type_:
            signode += addnodes.desc_annotation(type_, "", nodes.Text(": "), type_)

        value = self.options.get("value")
        if value:
            signode += addnodes.desc_annotation(value, " = " + value)

        return sig, prefix

    def get_index_text(self, modname: str, name: tuple[str, str]) -> str:
        sig, prefix = name
        return f"{sig} ({prefix} in {modname})"


class BasilispFunctionLike(BasilispObject):  # pylint: disable=abstract-method
    option_spec: OptionSpec = BasilispObject.option_spec.copy()

    def handle_signature(self, sig: str, signode: desc_signature) -> tuple[str, str]:
        prefix = self.get_signature_prefix(sig)
        if prefix:
            signode += addnodes.desc_annotation(prefix, prefix)

        self._add_source_annotations(signode)

        sig_sexp = runtime.first(reader.read_str(sig))
        assert isinstance(sig_sexp, IPersistentList)
        fn_sym = runtime.first(sig_sexp)
        assert isinstance(fn_sym, sym.Symbol)

        signode += addnodes.desc_addname("(", "(")
        signode += addnodes.desc_name(fn_sym.name, fn_sym.name)

        param_list = desc_lispparameterlist()
        for param in runtime.rest(sig_sexp):
            param_node = desc_lispparameter()
            param_node += addnodes.desc_sig_name("", runtime.lrepr(param))
            param_list += param_node
        signode += param_list

        signode += addnodes.desc_addname(")", ")")
        return fn_sym.name, prefix


class BasilispSpecialForm(BasilispFunctionLike):
    def get_signature_prefix(self, sig: str) -> str:
        return "special form"

    def add_target_and_index(
        self, name_cls: tuple[str, str], sig: str, signode: desc_signature
    ) -> None:
        fullname = name_cls[0]
        node_id = fullname
        signode["ids"].append(node_id)

        self.state.document.note_explicit_target(signode)

        domain = cast(BasilispDomain, self.env.get_domain("lpy"))
        domain.note_form(fullname, node_id)

    def get_index_text(self, modname: str, name: tuple[str, str]) -> str:
        sig, prefix = name
        sig_sexp = runtime.first(reader.read_str(sig))
        if isinstance(sig_sexp, IPersistentList):
            sig = runtime.first(sig_sexp)
        return f"{sig} ({prefix})"


class BasilispFunction(BasilispFunctionLike):
    option_spec: OptionSpec = BasilispFunctionLike.option_spec.copy()
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

    def get_index_text(self, modname: str, name: tuple[str, str]) -> str:
        sig, prefix = name
        sig_sexp = runtime.first(reader.read_str(sig))
        if isinstance(sig_sexp, IPersistentList):
            sig = runtime.first(sig_sexp)
        return f"{sig} ({prefix} in {modname})"


class BasilispClassLike(BasilispObject):
    def get_signature_prefix(self, sig: str) -> str:
        return self.objtype

    def handle_signature(self, sig: str, signode: desc_signature) -> tuple[str, str]:
        prefix = self.get_signature_prefix(sig)
        if prefix:
            signode += addnodes.desc_annotation(prefix, prefix)

        self._add_source_annotations(signode)

        signode += addnodes.desc_name(sig, sig)
        return sig, prefix

    def get_index_text(self, modname: str, name: tuple[str, str]) -> str:
        sig, prefix = name
        return f"{sig} ({prefix} in {modname})"


class BasilispNamespaceIndex(Index):
    name = "nsindex"
    localname = "Namespace Index"
    shortname = "namespaces"

    def generate(  # pylint: disable=too-many-locals
        self, docnames: Optional[Iterable[str]] = None
    ) -> tuple[list[tuple[str, list[IndexEntry]]], bool]:
        content: dict[str, list[IndexEntry]] = defaultdict(list)

        ignores: list[str] = self.domain.env.config["modindex_common_prefix"]
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
                    nsname = nsname[  # pylint: disable=redefined-loop-name
                        len(ignore) :
                    ]
                    stripped = ignore
                    break
            else:
                stripped = ""

            if not nsname:
                nsname, stripped = stripped, ""  # pylint: disable=redefined-loop-name

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


class BasilispXRefRole(XRefRole):
    def process_link(  # pylint: disable=too-many-arguments
        self,
        env: BuildEnvironment,
        refnode: Element,
        has_explicit_title: bool,
        title: str,
        target: str,
    ) -> tuple[str, str]:
        if refnode.attributes.get("reftype") != "form":
            refnode["lpy:namespace"] = env.ref_context.get("lpy:namespace")
        return title, target


class FormEntry(NamedTuple):
    docname: str
    node_id: str


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
    object_types: dict[str, ObjType] = {
        "form": ObjType("special form", "form", "specialform", "obj"),
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
        "specialform": BasilispSpecialForm,
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
        "form": BasilispXRefRole(),
        "ns": BasilispXRefRole(),
        "var": BasilispXRefRole(),
        "fn": BasilispXRefRole(),
        "proto": BasilispXRefRole(),
        "rec": BasilispXRefRole(),
        "type": BasilispXRefRole(),
        "meth": BasilispXRefRole(),
        "prop": BasilispXRefRole(),
    }
    initial_data: dict[str, dict[str, tuple[Any]]] = {
        "forms": {},  # name -> docname
        "vars": {},  # fullname -> docname, objtype
        "namespaces": {},  # nsname -> docname, synopsis, platform, deprecated
    }
    indices = [BasilispNamespaceIndex]

    @property
    def forms(self) -> dict[str, FormEntry]:
        return self.data.setdefault("forms", {})

    def note_form(
        self,
        name: str,
        node_id: str,
    ) -> None:
        """Note a Basilisp var for cross reference."""
        self.forms[name] = FormEntry(self.env.docname, node_id)

    @property
    def vars(self) -> dict[str, VarEntry]:
        return self.data.setdefault("vars", {})

    def note_var(
        self,
        name: str,
        objtype: str,
        node_id: str,
        aliased: bool = False,
    ) -> None:
        """Note a Basilisp var for cross reference."""
        self.vars[name] = VarEntry(self.env.docname, node_id, objtype, aliased)

    @property
    def namespaces(self) -> dict[str, NamespaceEntry]:
        return self.data.setdefault("namespaces", {})

    def note_namespace(  # pylint: disable=too-many-arguments
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

    def clear_doc(self, docname: str) -> None:
        for frm_name, form_entry in list(self.forms.items()):
            if form_entry.docname == docname:
                del self.forms[frm_name]
        for var_name, var_entry in list(self.vars.items()):
            if var_entry.docname == docname:
                del self.vars[var_name]
        for ns_name, ns_entry in list(self.namespaces.items()):
            if ns_entry.docname == docname:
                del self.namespaces[ns_name]

    def merge_domaindata(self, docnames: list[str], otherdata: dict) -> None:
        for frm_name, form_entry in otherdata["forms"].items():
            if form_entry.docname in docnames:
                self.forms[frm_name] = form_entry
        for var_name, var_entry in otherdata["vars"].items():
            if var_entry.docname in docnames:
                self.vars[var_name] = var_entry
        for ns_name, ns_entry in otherdata["namespaces"].items():
            if ns_entry.docname in docnames:
                self.namespaces[ns_name] = ns_entry

    def resolve_xref(  # pylint: disable=too-many-arguments
        self,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Builder,
        typ: str,
        target: str,
        node: pending_xref,
        contnode: Element,
    ) -> Optional[Element]:
        nsname = node.get("lpy:namespace")

        maybe_obj: Union[FormEntry, NamespaceEntry, VarEntry, None]
        reftype = node.get("reftype")
        if reftype == "ns":
            maybe_obj = self.namespaces.get(target)
            title = target
        elif reftype == "form":
            maybe_obj = self.forms.get(target)
            title = target
        else:
            obj_sym = runtime.first(reader.read_str(target))
            assert isinstance(
                obj_sym, sym.Symbol
            ), f"Symbol expected; not {obj_sym.__class__}"
            maybe_obj = self.vars.get(
                f"{nsname}/{obj_sym}" if obj_sym.ns is None else str(obj_sym)
            )
            title = (
                obj_sym.name
                if obj_sym.ns is None or obj_sym.ns == nsname
                else str(obj_sym)
            )

        if not maybe_obj:
            logger.warning(
                f"Unable to resolve xref for type={type} from {node.source} "
                f"with target={target}"
            )
            return None

        docname, node_id, *_ = maybe_obj
        return make_refnode(
            builder, fromdocname, docname, node_id, contnode, title=title
        )

    def resolve_any_xref(  # pylint: disable=too-many-arguments
        self,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Builder,
        target: str,
        node: pending_xref,
        contnode: Element,
    ) -> list[tuple[str, Element]]:
        raise NotImplementedError
