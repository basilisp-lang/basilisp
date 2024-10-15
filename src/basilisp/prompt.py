# pylint: disable=ungrouped-imports

import os
import re
from collections.abc import Iterable, Mapping
from functools import partial
from types import MappingProxyType
from typing import Any, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout.processors import HighlightMatchingBracketProcessor

from basilisp.lang import reader as reader
from basilisp.lang import runtime as runtime
from basilisp.lang.exception import print_exception

_USER_DATA_HOME = os.getenv("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
BASILISP_USER_DATA = os.path.abspath(os.path.join(_USER_DATA_HOME, "basilisp"))
os.makedirs(BASILISP_USER_DATA, exist_ok=True)

BASILISP_REPL_HISTORY_FILE_PATH = os.getenv(
    "BASILISP_REPL_HISTORY_FILE_PATH",
    os.path.join(BASILISP_USER_DATA, ".basilisp_history"),
)
BASILISP_NO_COLOR = os.environ.get("BASILISP_NO_COLOR", "false").lower() in {
    "1",
    "true",
}


class Prompter:
    __slots__ = ()

    def prompt(self, msg: str) -> str:
        """Prompt the user for input with the input string `msg`."""
        return input(msg)

    def print(self, msg: str) -> None:
        """Print the message to standard out."""
        print(msg)


_DELIMITED_WORD_PATTERN = re.compile(r"([^\[\](){\}\s]+)")


class REPLCompleter(Completer):
    __slots__ = ()

    def get_completions(
        self, document: Document, _: CompleteEvent
    ) -> Iterable[Completion]:
        """Yield successive REPL completions for Prompt Toolkit."""
        word_before_cursor = document.get_word_before_cursor(
            pattern=_DELIMITED_WORD_PATTERN
        )
        completions = runtime.repl_completions(word_before_cursor) or ()
        for completion in completions:
            yield Completion(completion, start_position=-len(word_before_cursor))


class PromptToolkitPrompter(Prompter):
    """Prompter class which wraps Prompt Toolkit utilities to provide advanced
    line editing functionality."""

    __slots__ = ("_session",)

    def __init__(self):
        self._session: PromptSession = PromptSession(
            auto_suggest=AutoSuggestFromHistory(),
            completer=REPLCompleter(),
            history=FileHistory(BASILISP_REPL_HISTORY_FILE_PATH),
            key_bindings=self._get_key_bindings(),
            lexer=self._prompt_toolkit_lexer,
            multiline=True,
            input_processors=[HighlightMatchingBracketProcessor(chars="[](){}")],
            **self._style_settings,
        )

    @staticmethod
    def _get_key_bindings() -> KeyBindings:
        """Return `KeyBindings` which override the builtin `enter` handler to
        allow multi-line input.

        Inputs are read by the reader to determine if they represent valid
        Basilisp syntax. If an `UnexpectedEOFError` is raised, then allow multiline
        input. If a more general `SyntaxError` is raised, then the exception will
        be printed to the terminal. In all other cases, handle the input normally."""
        kb = KeyBindings()
        _eof = object()

        @kb.add("enter")
        def _(event: KeyPressEvent) -> None:
            try:
                list(
                    reader.read_str(
                        event.current_buffer.text,
                        resolver=runtime.resolve_alias,
                        eof=_eof,
                    )
                )
            except reader.UnexpectedEOFError:
                event.current_buffer.insert_text("\n")
            except reader.SyntaxError as e:
                run_in_terminal(
                    partial(
                        print_exception,
                        e,
                        reader.SyntaxError,
                        e.__traceback__,
                    )
                )
            else:
                event.current_buffer.validate_and_handle()

        return kb

    _prompt_toolkit_lexer: Optional["PygmentsLexer"] = None
    _style_settings: Mapping[str, Any] = MappingProxyType({})

    def prompt(self, msg: str) -> str:
        return self._session.prompt(msg)


_DEFAULT_PROMPTER: type[Prompter] = PromptToolkitPrompter


try:
    import pygments
    from pygments.lexers.jvm import ClojureLexer
    from pygments.styles import get_style_by_name
except ImportError:  # pragma: no cover
    pass
else:
    from prompt_toolkit import print_formatted_text
    from prompt_toolkit.formatted_text import PygmentsTokens
    from prompt_toolkit.lexers import PygmentsLexer
    from prompt_toolkit.styles import style_from_pygments_cls

    BASILISP_REPL_PYGMENTS_STYLE_NAME = os.getenv(
        "BASILISP_REPL_PYGMENTS_STYLE_NAME", "emacs"
    )

    class StyledPromptToolkitPrompter(PromptToolkitPrompter):
        """Prompter class which adds Pygments based terminal styling to the
        PromptToolKit prompt."""

        _prompt_toolkit_lexer = PygmentsLexer(ClojureLexer)
        _pygments_lexer = ClojureLexer()
        _style_settings = MappingProxyType(
            {
                "style": style_from_pygments_cls(
                    get_style_by_name(BASILISP_REPL_PYGMENTS_STYLE_NAME)
                ),
                "include_default_pygments_style": False,
            }
        )

        def print(self, msg: str) -> None:
            tokens = list(pygments.lex(msg, lexer=self._pygments_lexer))
            print_formatted_text(PygmentsTokens(tokens), **self._style_settings)

    if not BASILISP_NO_COLOR:
        _DEFAULT_PROMPTER = StyledPromptToolkitPrompter


def get_prompter() -> Prompter:
    """Return a Prompter instance for reading user input from the REPL.

    Prompter instances may be stateful, so the Prompter instance returned by
    this function can be reused within a single REPL session."""
    return _DEFAULT_PROMPTER()


__all__ = ["Prompter", "get_prompter"]
