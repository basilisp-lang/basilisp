import importlib
import inspect
import logging
import sys
import types
from typing import Any, Optional, cast

from sphinx.ext.autodoc import (
    ClassDocumenter,
    Documenter,
    ObjectMember,
    bool_option,
    exclude_members_option,
    identity,
    inherited_members_option,
    member_order_option,
    members_option,
)
from sphinx.util.docstrings import prepare_docstring
from sphinx.util.typing import OptionSpec

from basilisp.lang import keyword as kw
from basilisp.lang import map as lmap
from basilisp.lang import reader, runtime
from basilisp.lang import symbol as sym
from basilisp.lang.interfaces import (
    IPersistentMap,
    IPersistentVector,
    IRecord,
    IReference,
    IType,
)
from basilisp.lang.multifn import MultiFunction

logger = logging.getLogger(__name__)

# Var metadata used for annotations
_DOC_KW = kw.keyword("doc")
_LINE_KW = kw.keyword("line")
_END_LINE_KW = kw.keyword("end-line")
_PRIVATE_KW = kw.keyword("private")
_DYNAMIC_KW = kw.keyword("dynamic")
_DEPRECATED_KW = kw.keyword("deprecated")
_FILE_KW = kw.keyword("file")
_ARGLISTS_KW = kw.keyword("arglists")
_MACRO_KW = kw.keyword("macro")

# Protocol support
_PROTOCOL_KW = kw.keyword("protocol", ns=runtime.CORE_NS)
_SOURCE_PROTOCOL_KW = kw.keyword("source-protocol", ns=runtime.CORE_NS)
_METHODS_KW = kw.keyword("methods")


def _get_doc(reference: IReference) -> Optional[list[list[str]]]:
    """Return the docstring of an IReference type (e.g. Namespace or Var)."""
    docstring = reference.meta and reference.meta.val_at(_DOC_KW)
    if docstring is None:
        return None

    assert isinstance(docstring, str)
    return [prepare_docstring(docstring)]


class NamespaceDocumenter(Documenter):
    domain = "lpy"
    objtype = "namespace"
    content_indent = ""
    titles_allowed = True

    option_spec: OptionSpec = {
        "members": members_option,
        "undoc-members": bool_option,
        "exclude-members": exclude_members_option,
        "private-members": members_option,
        "member-order": member_order_option,
        "noindex": bool_option,
        "synopsis": identity,
        "platform": identity,
        "deprecated": bool_option,
    }

    object: Optional[runtime.Namespace]

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return False

    def parse_name(self) -> bool:
        v = runtime.first(reader.read_str(self.name))
        if isinstance(v, sym.Symbol) and v.ns is None:
            self.modname = v.name
            self.objpath: list[str] = []
            self.args = ""
            self.retann = ""
            self.fullname = v.name
            return True
        return False

    def resolve_name(
        self, modname: str, parents: Any, path: str, base: Any
    ) -> tuple[str, list[str]]:
        """Unused method since parse_name is overridden."""
        return NotImplemented

    def import_object(self, raiseerror: bool = False) -> bool:
        try:
            importlib.import_module(self.modname.replace("-", "_"))
        except (ImportError, ModuleNotFoundError):
            logger.exception(f"Error: can't import namespace {self.modname}")
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

    def get_doc(self) -> Optional[list[list[str]]]:
        assert self.object is not None
        return _get_doc(self.object)

    def get_object_members(self, want_all: bool) -> tuple[bool, list[ObjectMember]]:
        assert self.object is not None
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
        self, members: list[ObjectMember], want_all: bool
    ) -> list[tuple[str, Any, bool]]:
        filtered = []
        for member in members:
            name, val = member.__name__, member.object
            assert isinstance(val, runtime.Var)
            if self.options.exclude_members and name in self.options.exclude_members:
                continue
            if val.meta is not None:
                # Ignore undocumented members unless undoc_members is set
                docstring: Optional[str] = val.meta.val_at(_DOC_KW)
                if docstring is None and not self.options.undoc_members:
                    continue
                # Private members will be excluded unless they are requested
                is_private = cast(bool, val.meta.val_at(_PRIVATE_KW, False))
                if is_private and (
                    self.options.private_members is None
                    or name not in self.options.private_members
                ):
                    continue
                # Namespace members with :basilisp.core/source-protocol meta will be
                # documented and grouped under the protocol they are defined by
                if val.meta.val_at(_SOURCE_PROTOCOL_KW) is not None:
                    continue
            filtered.append((name, val, False))
        return filtered

    def sort_members(
        self, documenters: list[tuple["Documenter", bool]], order: str
    ) -> list[tuple["Documenter", bool]]:
        assert self.object is not None
        if order == "bysource":
            # By the time this method is called, the object isn't hydrated in the
            # Documenter wrapper, so we cannot rely on the existence of that to get
            # line numbers. Instead, we have to build an index manually.
            line_numbers: dict[str, int] = {
                s.name: (
                    cast(int, v.meta.val_at(_LINE_KW, sys.maxsize))
                    if v.meta is not None
                    else sys.maxsize
                )
                for s, v in self.object.interns.items()
            }

            def _line_num(e: tuple["Documenter", bool]) -> int:
                documenter = e[0]
                _, name = documenter.name.split("::", maxsplit=1)
                return line_numbers.get(name, sys.maxsize)

            documenters.sort(key=_line_num)
            return documenters
        return super().sort_members(documenters, order)

    def format_name(self) -> str:
        return self.object_name

    def format_signature(self, **kwargs: Any) -> str:
        return ""


class VarDocumenter(Documenter):
    domain = "lpy"
    objtype = "var"
    priority = 10

    option_spec: OptionSpec = {
        "noindex": bool_option,
        "deprecated": bool_option,
    }

    object: Optional[runtime.Var]

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return isinstance(member, runtime.Var)

    def parse_name(self) -> bool:
        # Names may be given in either of these formats depending on the object:
        #
        #   basilisp.core::map
        #   basilisp.walk::IWalkable.walk*
        ns, name = self.name.split("::")
        self.modname = ns
        self.objpath = name.split(".")
        self.args = ""
        self.retann = ""
        self.fullname = str(sym.symbol(name, ns=ns))
        return True

    def resolve_name(
        self, modname: str, parents: Any, path: str, base: Any
    ) -> tuple[str, list[str]]:
        """Unused method since parse_name is overridden."""
        return NotImplemented

    def import_object(self, raiseerror: bool = False) -> bool:
        try:
            importlib.import_module(self.modname.replace("-", "_"))
        except (ImportError, ModuleNotFoundError):
            if raiseerror:
                raise
            return False

        # Protocol methods are passed as `some.ns::Protocol.method` which is annoying
        # but probably the lowest friction way of getting Sphinx to work for us.
        name = self.objpath[-1] if self.fullname != "basilisp.core/.." else ".."
        var = runtime.Var.find(sym.symbol(name, ns=self.modname))
        if var is None:
            logger.warning(f"Could not find Var {sym.symbol(name, ns=self.modname)}")
            return False

        self.object = var
        self.object_name = name
        self.parent = self.object.ns
        return True

    def get_sourcename(self) -> str:
        assert self.object is not None
        if self.object.meta is not None:
            file = self.object.meta.val_at(_FILE_KW)
            return f"{file}:docstring of {self.object}"
        return f"docstring of {self.object}"

    def get_object_members(self, want_all: bool) -> tuple[bool, list[ObjectMember]]:
        assert self.object is not None
        return False, []

    def add_directive_header(self, sig: str) -> None:
        assert self.object is not None
        sourcename = self.get_sourcename()
        super().add_directive_header(sig)

        if self.object.meta is not None:
            if (file := self.object.meta.val_at(_FILE_KW)) is not None:
                self.add_line(f"   :filename: {file}", sourcename)
            if isinstance(line := self.object.meta.val_at(_LINE_KW), int):
                if isinstance(end_line := self.object.meta.val_at(_END_LINE_KW), int):
                    self.add_line(f"   :lines: {line}:{end_line}", sourcename)
                else:
                    self.add_line(f"   :lines: {line}", sourcename)
            if self.object.meta.val_at(_DYNAMIC_KW):
                self.add_line("   :dynamic:", sourcename)
            if self.object.meta.val_at(_DEPRECATED_KW):
                self.add_line("   :deprecated:", sourcename)

    def get_doc(self) -> Optional[list[list[str]]]:
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
        return (
            isinstance(member, runtime.Var)
            and not member.dynamic
            and isinstance(member.value, (types.FunctionType, MultiFunction))
        )

    def add_directive_header(self, sig: str) -> None:
        assert self.object is not None
        sourcename = self.get_sourcename()
        super().add_directive_header(sig)

        if inspect.iscoroutinefunction(self.object.value):
            self.add_line("   :async:", sourcename)
        if self.object.meta is not None and self.object.meta.val_at(_MACRO_KW):
            self.add_line("   :macro:", sourcename)
        if isinstance(self.object.value, MultiFunction):
            self.add_line("   :multi:", sourcename)

    def format_name(self) -> str:
        return f"({self.object_name}"

    def format_signature(self, **kwargs: Any) -> str:
        assert self.object is not None
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

        assert self.object.meta is not None
        arglists = self.object.meta.val_at(_ARGLISTS_KW)
        if arglists is not None:
            return "\n".join(_format_sig(arglist) for arglist in arglists)

        # MultiFunctions don't automatically have :arglists meta set, so we provide
        # a default signature for such cases to avoid Sphinx errors.
        return " & args)"


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

    def get_object_members(self, want_all: bool) -> tuple[bool, list[ObjectMember]]:
        assert self.object is not None
        assert want_all
        ns = self.object.ns
        proto = self.object.value
        assert isinstance(proto, IPersistentMap)
        proto_methods = cast(
            IPersistentMap[kw.Keyword, Any],
            proto.val_at(_METHODS_KW, lmap.EMPTY),
        )
        return False, list(
            map(
                lambda k: ObjectMember(k.name, ns.find(sym.symbol(k.name))),
                proto_methods.keys(),
            )
        )

    def filter_members(
        self, members: list[ObjectMember], want_all: bool
    ) -> list[tuple[str, Any, bool]]:
        filtered = []
        for member in members:
            name, val = member.__name__, member.object
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

    option_spec: OptionSpec = {
        "members": members_option,
        "undoc-members": bool_option,
        "inherited-members": inherited_members_option,
        "exclude-members": exclude_members_option,
        "private-members": members_option,
        "special-members": members_option,
        "noindex": bool_option,
    }

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return (
            isinstance(member, runtime.Var)
            and isinstance(member.value, type)
            and issubclass(member.value, IType)
        )

    def get_object_members(self, want_all: bool) -> tuple[bool, list[ObjectMember]]:
        return ClassDocumenter.get_object_members(self, want_all)


class RecordDocumenter(TypeDocumenter):
    objtype = "record"

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return (
            isinstance(member, runtime.Var)
            and isinstance(member.value, type)
            and issubclass(member.value, IRecord)
        )
