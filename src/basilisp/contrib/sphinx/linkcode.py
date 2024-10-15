from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def doctree_read(app: Sphinx, doctree: Node) -> None:
    env = app.builder.env

    resolve_link = getattr(env.config, "lpy_linkcode_resolve", None)
    if not callable(resolve_link):
        logger.warning("No Basilisp linkcode resolver!")
        return

    for objnode in list(doctree.findall(addnodes.desc)):
        domain = objnode.get("domain")
        if domain != "lpy":
            continue

        uris: set[str] = set()
        for signode in objnode:
            if not isinstance(signode, addnodes.desc_signature):
                continue

            info = {}
            for key in ("filename", "lines"):
                value = signode.get(key)
                if not value:
                    value = ""
                info[key] = value

            if not info or all(not v for v in info.values()):
                continue

            if not (uri := resolve_link(info.get("filename"), info.get("lines"))):
                continue

            if uri in uris:
                continue

            uris.add(uri)

            inline = nodes.inline("", "[source]", classes=["viewcode-link"])
            onlynode = addnodes.only(expr="html")
            onlynode += nodes.reference("", "", inline, internal=False, refuri=uri)
            signode += onlynode  # pylint: disable=redefined-loop-name
