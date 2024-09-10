import os
import textwrap
from pathlib import Path

import pytest

from basilisp.lang import compiler as compiler
from basilisp.lang.source import format_source_context


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    return tmp_path / "source_test.lpy"


@pytest.fixture
def source_file_path(source_file: Path) -> str:
    return str(source_file)


def test_format_source_context(monkeypatch, source_file, source_file_path):
    source_file.write_text(
        textwrap.dedent(
            """
            (ns source-test)
            
            (a)
            (let [a 5]
              (b))
            """
        )
    )
    format_c = format_source_context(source_file_path, 2, end_line=4)
    assert [
        " 1   | \x1b[37m\x1b[39;49;00m\n",
        " 2 > | (\x1b[34mns \x1b[39;49;00m\x1b[31msource-test\x1b[39;49;00m)\x1b[37m\x1b[39;49;00m\n",
        " 3 > | \x1b[37m\x1b[39;49;00m\n",
        " 4 > | (\x1b[32ma\x1b[39;49;00m)\x1b[37m\x1b[39;49;00m\n",
        " 5   | (\x1b[34mlet \x1b[39;49;00m[\x1b[31ma\x1b[39;49;00m\x1b[37m \x1b[39;49;00m\x1b[34m5\x1b[39;49;00m]\x1b[37m\x1b[39;49;00m\n",
        " 6   | \x1b[37m  \x1b[39;49;00m(\x1b[32mb\x1b[39;49;00m))\x1b[37m\x1b[39;49;00m\n",
    ] == format_c

    format_nc = format_source_context(
        source_file_path, 2, end_line=4, disable_color=True
    )
    assert [
        " 1   | " + os.linesep,
        " 2 > | (ns source-test)" + os.linesep,
        " 3 > | " + os.linesep,
        " 4 > | (a)" + os.linesep,
        " 5   | (let [a 5]" + os.linesep,
        " 6   |   (b))" + os.linesep,
    ] == format_nc

    monkeypatch.setenv("BASILISP_NO_COLOR", "true")
    format_bnc = format_source_context(source_file_path, 2, end_line=4)
    assert [
        " 1   | " + os.linesep,
        " 2 > | (ns source-test)" + os.linesep,
        " 3 > | " + os.linesep,
        " 4 > | (a)" + os.linesep,
        " 5   | (let [a 5]" + os.linesep,
        " 6   |   (b))" + os.linesep,
    ] == format_bnc


def test_format_source_context_file_change(monkeypatch, source_file, source_file_path):
    source_file.write_text(
        textwrap.dedent(
            """
            (ns source-test)

            (a)
            (let [a 5]
              (b))
            """
        )
    )
    format_nc1 = format_source_context(
        source_file_path, 2, end_line=4, disable_color=True
    )
    assert [
        " 1   | " + os.linesep,
        " 2 > | (ns source-test)" + os.linesep,
        " 3 > | " + os.linesep,
        " 4 > | (a)" + os.linesep,
        " 5   | (let [a 5]" + os.linesep,
        " 6   |   (b))" + os.linesep,
    ] == format_nc1

    source_file.write_text(
        textwrap.dedent(
            """
            (ns source-test)
            (a)
            (abcd)
            """
        )
    )
    format_nc2 = format_source_context(
        source_file_path, 2, end_line=4, disable_color=True
    )
    assert [
        " 1   | " + os.linesep,
        " 2 > | (ns source-test)" + os.linesep,
        " 3 > | (a)" + os.linesep,
        " 4 > | (abcd)" + os.linesep,
    ] == format_nc2
