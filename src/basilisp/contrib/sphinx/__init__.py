from sphinx.application import Sphinx

from basilisp.contrib.sphinx import linkcode
from basilisp.contrib.sphinx.autodoc import (
    NamespaceDocumenter,
    ProtocolDocumenter,
    RecordDocumenter,
    TypeDocumenter,
    VarDocumenter,
    VarFnDocumenter,
)
from basilisp.contrib.sphinx.domain import (
    BasilispDomain,
    depart_lispparameter,
    depart_lispparameterlist,
    desc_lispparameter,
    desc_lispparameterlist,
    visit_lispparameter,
    visit_lispparameterlist,
)
from basilisp.main import init as init_basilisp


def setup(app: Sphinx) -> None:
    init_basilisp()
    app.setup_extension("sphinx.ext.autodoc")

    app.add_node(
        desc_lispparameterlist, html=(visit_lispparameterlist, depart_lispparameterlist)
    )
    app.add_node(desc_lispparameter, html=(visit_lispparameter, depart_lispparameter))

    app.add_domain(BasilispDomain)

    app.add_autodocumenter(NamespaceDocumenter)
    app.add_autodocumenter(VarDocumenter)
    app.add_autodocumenter(VarFnDocumenter)
    app.add_autodocumenter(ProtocolDocumenter)
    app.add_autodocumenter(TypeDocumenter)
    app.add_autodocumenter(RecordDocumenter)

    app.connect("doctree-read", linkcode.doctree_read)
    app.add_config_value("lpy_linkcode_resolve", None, "")


__all__ = ("setup",)
