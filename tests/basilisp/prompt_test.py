from typing import Iterable, Tuple
from unittest.mock import MagicMock, patch

import pytest
from prompt_toolkit.completion import CompleteEvent, Completer
from prompt_toolkit.document import Document

from basilisp.prompt import REPLCompleter, get_prompter


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


def test_prompter(capsys):
    session = MagicMock()
    with patch("basilisp.prompt.PromptSession", return_value=session) as session_cls:
        prompter = get_prompter()
        session_cls.assert_called_once()

        prompter.prompt("=>")
        session.prompt.assert_called_once_with("=>")

        with patch("basilisp.prompt.print_formatted_text") as prn:
            prompter.print(":a")
            prn.assert_called_once()
