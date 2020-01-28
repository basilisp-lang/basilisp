from typing import Iterable, Tuple
from unittest.mock import patch

import pytest
from prompt_toolkit.completion import CompleteEvent, Completer
from prompt_toolkit.document import Document

from basilisp.prompt import REPLCompleter


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


class TestPromptToolkitPrompt:
    pass
