from sphinx.application import Sphinx

from basilisp.contrib.sphinx.autodoc import (
    NamespaceDocumenter,
    ProtocolDocumenter,
    RecordDocumenter,
    TypeDocumenter,
    VarDocumenter,
    VarFnDocumenter,
)
from basilisp.contrib.sphinx.domain import BasilispDomain
from basilisp.main import init as init_basilisp


def setup(app: Sphinx) -> None:
    init_basilisp()
    app.setup_extension("sphinx.ext.autodoc")
    app.add_domain(BasilispDomain)
    app.add_autodocumenter(NamespaceDocumenter)
    app.add_autodocumenter(VarDocumenter)
    app.add_autodocumenter(VarFnDocumenter)
    app.add_autodocumenter(ProtocolDocumenter)
    app.add_autodocumenter(TypeDocumenter)
    app.add_autodocumenter(RecordDocumenter)


__all__ = ("setup",)
