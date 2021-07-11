import importlib
import inspect
import logging
import types
from typing import Any, List, Optional, Tuple

from sphinx.ext.autodoc import (
    Documenter,
    ObjectMember,
    ObjectMembers,
    bool_option,
    exclude_members_option,
    identity,
    members_option,
)
from sphinx.util.docstrings import prepare_docstring
from sphinx.util.typing import OptionSpec

from basilisp.lang import keyword as kw
from basilisp.lang import reader, runtime
from basilisp.lang import symbol as sym
from basilisp.lang.interfaces import (
    IPersistentMap,
    IPersistentVector,
    IRecord,
    IReference,
    IType,
)

logger = logging.getLogger(__name__)

_DOC_KW = kw.keyword("doc")
_PRIVATE_KW = kw.keyword("private")
_DYNAMIC_KW = kw.keyword("dynamic")
_FILE_KW = kw.keyword("file")
_ARGLISTS_KW = kw.keyword("arglists")
_MACRO_KW = kw.keyword("macro")
_PROTOCOL_KW = kw.keyword("protocol", ns=runtime.CORE_NS)
_SOURCE_PROTOCOL_KW = kw.keyword("source-protocol", ns=runtime.CORE_NS)
_METHODS_KW = kw.keyword("methods")


def _get_doc(reference: IReference) -> Optional[List[List[str]]]:
    """Return the docstring of an IReference type (e.g. Namespace or Var)."""
    docstring: Optional[str] = reference.meta and reference.meta.val_at(_DOC_KW)
    if docstring is None:
        return None

    return [prepare_docstring(docstring)]


class NamespaceDocumenter(Documenter):
    domain = "lpy"
    objtype = "namespace"
    content_indent = ""
    titles_allowed = True

    option_spec: OptionSpec = {
        "members": members_option,
        "undoc-members": bool_option,
        "noindex": bool_option,
        "synopsis": identity,
        "platform": identity,
        "deprecated": bool_option,
        "exclude-members": exclude_members_option,
        "private-members": members_option,
    }

    object: Optional[runtime.Namespace]

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return False

    def parse_name(self) -> bool:
        v = next(reader.read_str(self.name), None)
        if isinstance(v, sym.Symbol) and v.ns is None:
            self.modname = v.name
            self.objpath = []
            self.args = ""
            self.retann = ""
            self.fullname = v.name
            return True
        return False

    def resolve_name(
        self, modname: str, parents: Any, path: str, base: Any
    ) -> Tuple[str, List[str]]:
        return NotImplemented

    def import_object(self, raiseerror: bool = False) -> bool:
        try:
            importlib.import_module(self.modname)
        except (ImportError, ModuleNotFoundError):
            if raiseerror:
                raise
            return False

        ns = runtime.Namespace.get(sym.symbol(self.modname))
        if ns is None:
            return False

        self.object = ns
        self.object_name = ns.name
        self.module = ns.module
        return True

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
        assert self.object is not None
        return _get_doc(self.object)

    def get_object_members(self, want_all: bool) -> Tuple[bool, ObjectMembers]:
        interns = self.object.interns

        if want_all:
            return False, list(
                map(lambda s: ObjectMember(s[0].name, s[1]), interns.items())
            )

        selected = []
        for m in self.options.members:
            val = interns.val_at(sym.symbol(m))
            if val is not None:
                selected.append(ObjectMember(m, val))
            else:
                logger.warning(f"Member {m} not found in namespace {self.object}")
        return False, selected

    def filter_members(
        self, members: ObjectMembers, want_all: bool
    ) -> List[Tuple[str, Any, bool]]:
        filtered = []
        for name, val in members:
            assert isinstance(val, runtime.Var)
            if val.meta is not None:
                # Private members will be excluded
                #
                # Namespace members with :basilisp.core/source-protocol meta will be
                # documented and grouped under the protocol they are defined by
                if (
                    val.meta.val_at(_PRIVATE_KW)
                    or val.meta.val_at(_SOURCE_PROTOCOL_KW) is not None
                ):
                    continue
            filtered.append((name, val, False))
        return filtered


class VarDocumenter(Documenter):
    domain = "lpy"
    objtype = "var"
    priority = 10

    option_spec: OptionSpec = {
        "noindex": bool_option,
        "synopsis": identity,
        "platform": identity,
        "deprecated": bool_option,
    }

    object: Optional[runtime.Var]

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return isinstance(member, runtime.Var)

    def get_sourcename(self) -> str:
        if self.object.meta is not None:
            file = self.object.meta.val_at(_FILE_KW)
            return f"{file}:docstring of {self.object}"
        return f"docstring of {self.object}"

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()
        super().add_directive_header(sig)

        if self.object.meta is not None and self.object.meta.val_at(_DYNAMIC_KW):
            self.add_line("   :dynamic:", sourcename)

    def parse_name(self) -> bool:
        ns, name = self.name.split("::")
        self.modname = ns
        self.objpath = name.split(".")
        self.args = ""
        self.retann = ""
        self.fullname = str(sym.symbol(name, ns=ns))
        return True

    def resolve_name(
        self, modname: str, parents: Any, path: str, base: Any
    ) -> Tuple[str, List[str]]:
        """Unused method since parse_name is overridden."""
        return NotImplemented

    def import_object(self, raiseerror: bool = False) -> bool:
        try:
            importlib.import_module(self.modname)
        except (ImportError, ModuleNotFoundError):
            if raiseerror:
                raise
            return False

        # Protocol methods are passed as `some.ns::Protocol.method` which is annoying
        # but probably the lowest friction way of getting Sphinx to work for us.
        name = self.objpath[-1]
        var = runtime.Var.find(sym.symbol(name, ns=self.modname))
        print(f"Found var {var=} {name=} {self.modname=}")
        if var is None:
            return False

        self.object = var
        self.object_name = name
        self.parent = self.object.ns
        return True

    def get_doc(self, ignore: int = None) -> Optional[List[List[str]]]:
        assert self.object is not None
        return _get_doc(self.object)

    def format_name(self) -> str:
        return self.object_name

    def format_signature(self, **kwargs: Any) -> str:
        return ""


class VarFnDocumenter(VarDocumenter):
    domain = "lpy"
    objtype = "lispfunction"
    priority = 15

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return isinstance(member, runtime.Var) and isinstance(
            member.value, types.FunctionType
        )

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()
        super().add_directive_header(sig)

        if inspect.iscoroutinefunction(self.object.value):
            self.add_line("   :async:", sourcename)
        if self.object.meta is not None and self.object.meta.val_at(_MACRO_KW):
            self.add_line("   :macro:", sourcename)

    def format_name(self) -> str:
        return f"({self.object_name}"

    def format_signature(self, **kwargs: Any) -> str:
        is_macro = (
            False if self.object.meta is None else self.object.meta.val_at(_MACRO_KW)
        )

        def _format_sig(arglist: IPersistentVector) -> str:
            if is_macro:
                arglist = runtime.nthrest(arglist, 2)
            call = " ".join(map(str, arglist))
            if call:
                return f" {call})"
            return ")"

        arglists = self.object.meta.val_at(_ARGLISTS_KW)
        assert arglists is not None
        return "\n".join(_format_sig(arglist) for arglist in arglists)


class ProtocolDocumenter(VarDocumenter):
    domain = "lpy"
    objtype = "protocol"
    priority = 15

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return (
            isinstance(member, runtime.Var)
            and member.meta is not None
            and member.meta.val_at(_PROTOCOL_KW) is True
        )

    def get_object_members(self, want_all: bool) -> Tuple[bool, ObjectMembers]:
        assert want_all
        ns = self.object.ns
        proto: IPersistentMap = self.object.value
        proto_methods: IPersistentMap[kw.Keyword, Any] = proto.val_at(_METHODS_KW, ())
        return False, list(
            map(
                lambda k: ObjectMember(k.name, ns.find(sym.symbol(k.name))),
                proto_methods.keys(),
            )
        )

    def filter_members(
        self, members: ObjectMembers, want_all: bool
    ) -> List[Tuple[str, Any, bool]]:
        filtered = []
        for name, val in members:
            assert isinstance(val, runtime.Var)
            if val.meta is not None:
                if val.meta.val_at(_PRIVATE_KW):
                    continue
            filtered.append((name, val, False))
        return filtered


class TypeDocumenter(VarDocumenter):
    domain = "lpy"
    objtype = "type"
    priority = 15

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return (
            isinstance(member, runtime.Var)
            and isinstance(member.value, type)
            and issubclass(member.value, IType)
        )


class RecordDocumenter(VarDocumenter):
    domain = "lpy"
    objtype = "record"
    priority = 15

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return (
            isinstance(member, runtime.Var)
            and isinstance(member.value, type)
            and issubclass(member.value, IRecord)
        )
