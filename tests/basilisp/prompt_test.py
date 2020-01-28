from typing import Callable, Iterable, Tuple
from unittest.mock import MagicMock, patch

import pytest
from prompt_toolkit.completion import CompleteEvent, Completer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding.key_bindings import Binding
from prompt_toolkit.keys import Keys

from basilisp.prompt import PromptToolkitPrompter, REPLCompleter, get_prompter


class TestCompleter:
    @pytest.fixture(scope="class")
    def completer(self) -> Completer:
        return REPLCompleter()

    @pytest.fixture
    def completions(self) -> Iterable[str]:
        return (
            "macroexpand",
            "macroexpand-1",
            "map",
            "map-entry",
            "map-entry?",
            "map-indexed",
            "map?",
            "mapcat",
            "mapv",
            "max",
            "max-key",
            "merge",
            "meta",
            "methods",
            "min",
            "min-key",
            "mod",
            "munge",
        )

    @pytest.fixture(scope="class")
    def patch_completions(self, completions: Iterable[str]):
        with patch(
            "basilisp.prompt.runtime.repl_completions", return_value=completions
        ):
            yield

    @pytest.mark.parametrize(
        "val,expected",
        [
            (
                "m",
                (
                    "macroexpand",
                    "macroexpand-1",
                    "map",
                    "map-entry",
                    "map-entry?",
                    "map-indexed",
                    "map?",
                    "mapcat",
                    "mapv",
                    "max",
                    "max-key",
                    "merge",
                    "meta",
                    "methods",
                    "min",
                    "min-key",
                    "mod",
                    "munge",
                ),
            ),
            (
                "ma",
                (
                    "macroexpand",
                    "macroexpand-1",
                    "map",
                    "map-entry",
                    "map-entry?",
                    "map-indexed",
                    "map?",
                    "mapcat",
                    "mapv",
                    "max",
                    "max-key",
                ),
            ),
            (
                "map",
                (
                    "map",
                    "map-entry",
                    "map-entry?",
                    "map-indexed",
                    "map?",
                    "mapcat",
                    "mapv",
                ),
            ),
            ("mak", (),),
            ("(map-", ("map-entry", "map-entry?", "map-indexed",),),
        ],
    )
    def test_completer(self, completer: Completer, val: str, expected: Tuple[str]):
        doc = Document(val, len(val))
        completions = list(completer.get_completions(doc, CompleteEvent()))
        assert len(completions) == len(expected)
        assert set(c.text for c in completions) == set(expected)


def test_prompter():
    session = MagicMock()
    with patch("basilisp.prompt.PromptSession", return_value=session) as session_cls:
        prompter = get_prompter()
        session_cls.assert_called_once()

        prompter.prompt("=>")
        session.prompt.assert_called_once_with("=>")

        with patch("basilisp.prompt.print_formatted_text") as prn:
            prompter.print(":a")
            prn.assert_called_once()


class TestKeyBindings:
    def make_key_press_event(self, text: str):
        e = MagicMock()
        e.current_buffer = MagicMock()
        e.current_buffer.text = text
        return e

    @pytest.fixture(scope="class")
    def handler(self) -> Binding:
        kb = PromptToolkitPrompter._get_key_bindings()
        handler, *_ = kb.get_bindings_for_keys((Keys.ControlM,))
        return handler

    @pytest.fixture(autouse=True)
    def assert_syntax_error(self):
        with patch("basilisp.prompt.run_in_terminal") as run_in_terminal, patch(
            "basilisp.prompt.partial"
        ) as partial:
            marker = object()
            partial.return_value = marker

            def _assert_syntax_error():
                partial.assert_called_once()
                run_in_terminal.assert_called_once_with(marker)

            yield _assert_syntax_error

    @pytest.mark.parametrize(
        "line",
        [
            "#{:a :b :c}",
            "{:a 3}",
            "\\newline",
            ":a",
            "(map odd? [1 2 3])",
            '"just a string"',
        ],
    )
    def test_valid_single_line_syntax(
        self, handler: Binding, line: str,
    ):
        e = self.make_key_press_event(line)
        handler.call(e)
        e.current_buffer.validate_and_handle.assert_called_once()

    @pytest.mark.parametrize(
        "lines",
        [
            ("(defn f [] :a)",),
            ("(", "map odd? [1 2 3])"),
            ("(defn f", "[]", ":a)"),
            ("[", "1", ":b", '"c"', "]"),
            ("[", "          ", "          ", ":a :b :c", "", "]"),
        ],
    )
    def test_multiline_input(self, handler: Binding, lines: Tuple[str]):
        *begin, last = lines

        line_buffer = []
        for l in begin:
            line_buffer.append(l)
            e = self.make_key_press_event("\n".join(line_buffer))
            handler.call(e)

            e.current_buffer.insert_text.assert_called_with("\n")

        line_buffer.append(last)
        e = self.make_key_press_event("\n".join(line_buffer))
        handler.call(e)
        e.current_buffer.validate_and_handle.assert_called_once()

    @pytest.mark.parametrize(
        "line",
        [
            "#{:a :a}",
            "{:a}",
            "\\fakeescape",
            "#?@(:clj [:a :b :c])",
            "ns.of..a/badsymbol",
            ":bad.keyword",
        ],
    )
    def test_syntax_error(
        self, handler: Binding, line: str, assert_syntax_error: Callable[[], None]
    ):
        e = self.make_key_press_event(line)
        handler.call(e)
        assert_syntax_error()
