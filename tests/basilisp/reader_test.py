import datetime
import io
import math
import os
import re
import textwrap
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Optional

import pytest

from basilisp.lang import keyword as kw
from basilisp.lang import list as llist
from basilisp.lang import map as lmap
from basilisp.lang import queue as lqueue
from basilisp.lang import reader as reader
from basilisp.lang import runtime as runtime
from basilisp.lang import set as lset
from basilisp.lang import symbol as sym
from basilisp.lang import util as langutil
from basilisp.lang import vector as vec
from basilisp.lang.exception import format_exception
from basilisp.lang.interfaces import IPersistentSet
from basilisp.lang.reader import Resolver
from basilisp.lang.tagged import tagged_literal


@pytest.fixture
def test_ns() -> str:
    return "basilisp.reader-test"


def read_str_first(
    s: str,
    resolver: reader.Resolver = None,
    data_readers=None,
    is_eof_error: bool = False,
    features: Optional[IPersistentSet[kw.Keyword]] = None,
    process_reader_cond: bool = True,
    default_data_reader_fn=None,
):
    """Read the first form from the input string. If no form
    is found, return None."""
    try:
        return next(
            reader.read_str(
                s,
                resolver=resolver,
                data_readers=data_readers,
                is_eof_error=is_eof_error,
                features=features,
                process_reader_cond=process_reader_cond,
                default_data_reader_fn=default_data_reader_fn,
            )
        )
    except StopIteration:
        return None


def test_stream_reader():
    sreader = reader.StreamReader(io.StringIO("12345"))

    assert "1" == sreader.peek()
    assert (1, 0) == sreader.loc

    assert "2" == sreader.next_char()
    assert (1, 1) == sreader.loc

    assert "2" == sreader.peek()
    assert (1, 1) == sreader.loc

    sreader.pushback()
    assert "1" == sreader.peek()
    assert (1, 0) == sreader.loc

    assert "2" == sreader.next_char()
    assert (1, 1) == sreader.loc

    assert "3" == sreader.next_char()
    assert (1, 2) == sreader.loc

    assert "4" == sreader.next_char()
    assert (1, 3) == sreader.loc

    assert "5" == sreader.next_char()
    assert (1, 4) == sreader.loc

    assert "" == sreader.next_char()
    assert (1, 5) == sreader.loc


def test_stream_reader_loc():
    s = "i=1\n" "b=2\n" "i"
    sreader = reader.StreamReader(io.StringIO(s))
    assert (1, 0) == sreader.loc

    assert "i" == sreader.peek()
    assert (1, 0) == sreader.loc

    assert "=" == sreader.next_char()
    assert (1, 1) == sreader.loc

    assert "=" == sreader.peek()
    assert (1, 1) == sreader.loc

    sreader.pushback()
    assert "i" == sreader.peek()
    assert (1, 0) == sreader.loc

    assert "=" == sreader.next_char()
    assert (1, 1) == sreader.loc

    assert "1" == sreader.next_char()
    assert (1, 2) == sreader.loc

    assert "\n" == sreader.next_char()
    assert (1, 3) == sreader.loc

    assert "b" == sreader.next_char()
    assert (2, 0) == sreader.loc

    assert "=" == sreader.next_char()
    assert (2, 1) == sreader.loc

    assert "2" == sreader.next_char()
    assert (2, 2) == sreader.loc

    assert "\n" == sreader.next_char()
    assert (2, 3) == sreader.loc

    assert "i" == sreader.next_char()
    assert (3, 0) == sreader.loc

    assert "" == sreader.next_char()
    assert (3, 1) == sreader.loc


def test_stream_reader_loc_other():
    s = "i=1\n" "b=2\n" "i"
    sreader = reader.StreamReader(io.StringIO(s), init_line=5, init_column=3)
    assert (5, 3) == sreader.loc

    assert "i" == sreader.peek()
    assert (5, 3) == sreader.loc

    assert "=" == sreader.next_char()
    assert (5, 4) == sreader.loc

    assert "=" == sreader.peek()
    assert (5, 4) == sreader.loc

    sreader.pushback()
    assert "i" == sreader.peek()
    assert (5, 3) == sreader.loc

    assert "=" == sreader.next_char()
    assert (5, 4) == sreader.loc

    assert "1" == sreader.next_char()
    assert (5, 5) == sreader.loc

    assert "\n" == sreader.next_char()
    assert (5, 6) == sreader.loc

    assert "b" == sreader.next_char()
    assert (6, 0) == sreader.loc

    assert "=" == sreader.next_char()
    assert (6, 1) == sreader.loc


class TestReaderLines:
    def test_reader_lines_from_str(self, tmp_path):
        _, _, l = list(reader.read_str("1\n2\n(/ 5 0)"))

        assert (3, 3, 0, 7) == (
            l.meta.get(reader.READER_LINE_KW),
            l.meta.get(reader.READER_END_LINE_KW),
            l.meta.get(reader.READER_COL_KW),
            l.meta.get(reader.READER_END_COL_KW),
        )

    def test_reader_lines_from_str_other_loc(self, tmp_path):
        l1, _, l3 = list(
            reader.read_str("(+ 1 2)\n2\n(/ 5 0)", init_line=6, init_column=7)
        )

        assert (6, 6, 7, 14) == (
            l1.meta.get(reader.READER_LINE_KW),
            l1.meta.get(reader.READER_END_LINE_KW),
            l1.meta.get(reader.READER_COL_KW),
            l1.meta.get(reader.READER_END_COL_KW),
        )

        assert (8, 8, 0, 7) == (
            l3.meta.get(reader.READER_LINE_KW),
            l3.meta.get(reader.READER_END_LINE_KW),
            l3.meta.get(reader.READER_COL_KW),
            l3.meta.get(reader.READER_END_COL_KW),
        )

    @pytest.mark.parametrize(
        "evalstr,first,second",
        [
            ("[5]\n(def n 123)", (1, 1, 0, 3), (2, 2, 0, 11)),
            ("[5]\r(def n 123)", (1, 1, 0, 3), (2, 2, 0, 11)),
            ("[5]\r\n(def n 123)", (1, 1, 0, 3), (2, 2, 0, 11)),
            ("[5]\n\n(def n 123)", (1, 1, 0, 3), (3, 3, 0, 11)),
            ("[5]\r\r(def n 123)", (1, 1, 0, 3), (3, 3, 0, 11)),
            ("[5]\r\n\r\n(def n 123)", (1, 1, 0, 3), (3, 3, 0, 11)),
            ("\n[5]\n(def n 123)", (2, 2, 0, 3), (3, 3, 0, 11)),
            ("\r[5]\r(def n 123)", (2, 2, 0, 3), (3, 3, 0, 11)),
            ("\r\n[5]\r\n(def n 123)", (2, 2, 0, 3), (3, 3, 0, 11)),
        ],
    )
    def test_reader_newlines_from_str(self, evalstr, first, second):
        l0, l1 = list(reader.read_str(evalstr))
        assert first == (
            l0.meta.get(reader.READER_LINE_KW),
            l0.meta.get(reader.READER_END_LINE_KW),
            l0.meta.get(reader.READER_COL_KW),
            l0.meta.get(reader.READER_END_COL_KW),
        )
        assert second == (
            l1.meta.get(reader.READER_LINE_KW),
            l1.meta.get(reader.READER_END_LINE_KW),
            l1.meta.get(reader.READER_COL_KW),
            l1.meta.get(reader.READER_END_COL_KW),
        )

    def test_reader_lines_from_file(self, tmp_path):
        filename = tmp_path / "test.lpy"

        with open(filename, mode="w", encoding="utf-8") as f:
            f.write("1\n2\n(/ 5 0)")

        with open(filename, encoding="utf-8") as f:
            _, _, l = list(reader.read(f))

        assert (3, 3, 0, 7) == (
            l.meta.get(reader.READER_LINE_KW),
            l.meta.get(reader.READER_END_LINE_KW),
            l.meta.get(reader.READER_COL_KW),
            l.meta.get(reader.READER_END_COL_KW),
        )


class TestSyntaxErrorFormat:
    def test_no_cause_exception(self):
        with pytest.raises(reader.SyntaxError) as e:
            read_str_first("[:a :b :c")

        assert [
            f"{os.linesep}",
            f"  exception: <class 'basilisp.lang.reader.UnexpectedEOFError'>{os.linesep}",
            f"    message: Unexpected EOF in vector{os.linesep}",
            f"       line: 1:9{os.linesep}",
        ] == format_exception(e.value)

    def test_exception_with_cause(self):
        with pytest.raises(reader.SyntaxError) as e:
            read_str_first("{:a 1 :a}")

        assert [
            f"{os.linesep}",
            f"  exception: <class 'ValueError'> from <class 'basilisp.lang.reader.SyntaxError'>{os.linesep}",
            f"    message: Unexpected char '}}'; expected map value: not enough values to unpack (expected 2, got 1){os.linesep}",
            f"       line: 1:9{os.linesep}",
        ] == format_exception(e.value)

    class TestExceptionsWithSourceContext:
        @pytest.fixture
        def source_file(self, tmp_path: Path) -> Path:
            return tmp_path / "reader_test.lpy"

        def test_shows_source_context(self, monkeypatch, source_file: Path):
            source_file.write_text(
                textwrap.dedent(
                    """
                    (ns reader-test)

                    (let [a :b]
                      a
                    """
                ).strip()
            )
            monkeypatch.syspath_prepend(source_file.parent)

            with pytest.raises(reader.SyntaxError) as e:
                list(reader.read_file(source_file))

            v = format_exception(e.value, disable_color=True)
            assert re.match(
                (
                    rf"{os.linesep}"
                    rf"  exception: <class 'basilisp\.lang\.reader\.UnexpectedEOFError'>{os.linesep}"
                    rf"    message: Unexpected EOF in list{os.linesep}"
                    rf"   location: (?:\w:)?[^:]*:4:3{os.linesep}"
                    rf"    context:{os.linesep}"
                    rf"{os.linesep}"
                    rf" 1   \| \(ns reader-test\){os.linesep}"
                    rf" 2   \| {os.linesep}"
                    rf" 3   \| \(let \[a :b\]{os.linesep}"
                    rf" 4 > \|   a{os.linesep}"
                ),
                "".join(v),
            )

        def test_shows_source_context_(self, monkeypatch, source_file: Path):
            source_file.write_text(
                textwrap.dedent(
                    """
                    (ns reader-test)

                    (let [a :b]
                      a
                    """
                ).strip()
                + "\n"
            )
            monkeypatch.setenv("BASILISP_NO_COLOR", "true")
            monkeypatch.syspath_prepend(source_file.parent)

            with pytest.raises(reader.SyntaxError) as e:
                list(reader.read_file(source_file))

            v = format_exception(e.value)
            assert re.match(
                (
                    rf"{os.linesep}"
                    rf"  exception: <class 'basilisp\.lang\.reader\.UnexpectedEOFError'>{os.linesep}"
                    rf"    message: Unexpected EOF in list{os.linesep}"
                    rf"   location: (?:\w:)?[^:]*:5:0{os.linesep}"
                    rf"    context:{os.linesep}"
                    rf"{os.linesep}"
                    rf" 1   \| \(ns reader-test\){os.linesep}"
                    rf" 2   \| {os.linesep}"
                    rf" 3   \| \(let \[a :b\]{os.linesep}"
                    rf" 4   \|   a{os.linesep}"
                    rf" 5 > \|"
                ),
                "".join(v),
            )


class TestComplex:
    @pytest.mark.parametrize(
        "v,raw",
        [
            (1j, "1J"),
            (100j, "100J"),
            (99_927_273j, "99927273J"),
            (0j, "0J"),
            (-1j, "-1J"),
            (-538_282j, "-538282J"),
        ],
    )
    def test_legal_complex(self, v: complex, raw: str):
        assert v == read_str_first(raw)

    @pytest.mark.parametrize("raw", ["1JJ", "1NJ"])
    def test_malformed_complex(self, raw: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(raw)

    @pytest.mark.parametrize(
        "v,raw",
        [
            (0.0j, "0.0J"),
            (0.0j, "0.J"),
            (0.093_873_72j, "0.09387372J"),
            (1.0j, "1.J"),
            (1.0j, "1.0J"),
            (1.332j, "1.332J"),
            (-1.332j, "-1.332J"),
            (-1.0j, "-1.0J"),
            (-0.332j, "-0.332J"),
        ],
    )
    def test_legal_float_complex(self, v: complex, raw: str):
        assert v == read_str_first(raw)

    @pytest.mark.parametrize("raw", ["1.0MJ", "22/7J", "22J/7"])
    def test_malformed_float_complex(self, raw: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(raw)


class TestInt:
    @pytest.mark.parametrize(
        "raw,v",
        [
            ("1", 1),
            ("100", 100),
            ("99927273", 99_927_273),
            ("0", 0),
            ("-0", 0),
            ("-1", -1),
            ("-538282", -538_282),
        ],
    )
    def test_legal_int(self, v: int, raw: str):
        assert v == read_str_first(raw)

    @pytest.mark.parametrize(
        "raw,v",
        [
            ("1N", 1),
            ("100N", 100),
            ("99927273N", 99_927_273),
            ("0N", 0),
            ("-1N", -1),
            ("-538282N", -538_282),
        ],
    )
    def test_legal_bigint(self, v: int, raw: str):
        assert v == read_str_first(raw)

    def test_malformed_bigint(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first("1NN")


class TestFloat:
    @pytest.mark.parametrize(
        "raw,v",
        [
            ("0.0", 0.0),
            ("-0.0", 0.0),
            ("0.09387372", 0.093_873_72),
            ("1.", 1.0),
            ("1.0", 1.0),
            ("1.332", 1.332),
            ("-1.332", -1.332),
            ("-1.", -1.0),
            ("-1.0", -1.0),
            ("-0.332", -0.332),
        ],
    )
    def test_legal_float(self, v: float, raw: str):
        assert v == read_str_first(raw)

    @pytest.mark.parametrize("raw", ["0..11", "0.111.9"])
    def test_malformed_float(self, raw: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(raw)


class TestOctalLiteral:
    @pytest.mark.parametrize(
        "raw,v",
        [
            ("00", 0o0),
            ("00N", 0o0),
            ("-00", -0o0),
            ("-03", -0o3),
            ("03", 0o3),
            ("0666", 0o666),
            ("-0666", -0o666),
            ("0666N", 0o666),
            ("-0666N", -0o666),
        ],
    )
    def test_legal_octal_literal(self, v: int, raw: str):
        assert v == read_str_first(raw)

    @pytest.mark.parametrize("raw", ["089", "01.", "0639"])
    def test_malformed_octal_literal(self, raw: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(raw)


class TestHexLiteral:
    @pytest.mark.parametrize(
        "raw,v",
        [
            ("0x3", 0x3),
            ("-0x3", -0x3),
            ("0X3", 0x3),
            ("0x0", 0x0),
            ("0x0N", 0x0),
            ("0xFACE", 0xFACE),
            ("-0xFACE", -0xFACE),
            ("0XFACE", 0xFACE),
            ("0xFACEN", 0xFACE),
            ("-0xFACEN", -0xFACE),
            ("0xface", 0xFACE),
            ("-0xface", -0xFACE),
            ("0Xface", 0xFACE),
            ("0xfaceN", 0xFACE),
            ("-0xfaceN", -0xFACE),
        ],
    )
    def test_legal_hex_literal(self, v: int, raw: str):
        assert v == read_str_first(raw)

    @pytest.mark.parametrize("raw", ["0x", "0x.", "0x3.", "0xFA-E"])
    def test_malformed_hex_literal(self, raw: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(raw)


class TestArbitraryBaseLiteral:
    @pytest.mark.parametrize(
        "raw,v",
        [
            ("2r0", int("0", base=2)),
            ("-2r0", -int("0", base=2)),
            ("16rFACE", 0xFACE),
            ("-16rFACE", -0xFACE),
        ],
    )
    def test_legal_arbitrary_base_literal(self, v: int, raw: str):
        assert v == read_str_first(raw)

    @pytest.mark.parametrize("raw", ["1r1", "37r42", "0r53", "6r799", "6r", "432r382"])
    def test_malformed_arbitrary_base_literal(self, raw: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(raw)


class TestScientificNotationLiteral:
    @pytest.mark.parametrize(
        "raw,v",
        [
            ("-0e4", -0e4),
            ("0e0", 0e0),
            ("0e12", 0e12),
            ("-2e-6", -2e-6),
            ("2e6", 2e6),
            ("2E6", 2e6),
            ("-2e6", -2e6),
            ("-2E6", -2e6),
            ("2.e6", 2.0e6),
            ("-2.e6", -2.0e6),
            ("3.14e8", 3.14e8),
            ("3.14e-8", 3.14e-8),
            ("0.443e12", 0.443e12),
        ],
    )
    def test_legal_scientific_notation_literal(self, v: int, raw: str):
        assert v == read_str_first(raw)

    @pytest.mark.parametrize("raw", ["2e-3.6", "2e--4"])
    def test_malformed_scientific_notation_literal(self, raw: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(raw)


class TestRatios:
    @pytest.mark.parametrize(
        "raw,v",
        [
            ("22/7", Fraction(22, 7)),
            ("-22/7", Fraction(-22, 7)),
            ("0/3", 0),
            ("1/3", Fraction(1, 3)),
        ],
    )
    def test_legal_arbitrary_base_literal(self, v: int, raw: str):
        assert v == read_str_first(raw)

    @pytest.mark.parametrize("raw", ["22/-7", "1/0"])
    def test_malformed_ratio(self, raw: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(raw)


class TestKeyword:
    @pytest.mark.parametrize(
        "v,raw",
        [
            ("kw", ":kw"),
            ("kebab-kw", ":kebab-kw"),
            ("underscore_kw", ":underscore_kw"),
            ("dotted.kw", ":dotted.kw"),
            ("kw?", ":kw?"),
            ("+", ":+"),
            ("?", ":?"),
            ("=", ":="),
            ("!", ":!"),
            ("-", ":-"),
            ("*", ":*"),
            ("/", ":/"),
            (">", ":>"),
            ("->", ":->"),
            ("->>", ":->>"),
            ("-->", ":-->"),
            ("--------------->", ":--------------->"),
            ("<", ":<"),
            ("<-", ":<-"),
            ("<<-", ":<<-"),
            ("<--", ":<--"),
            ("<body>", ":<body>"),
            ("*muffs*", ":*muffs*"),
            ("yay!", ":yay!"),
            ("*'", ":*'"),
            ("a:b", ":a:b"),
            ("#", ":#"),
            ("div#id", ":div#id"),
        ],
    )
    def test_legal_bare_keyword(self, v: str, raw: str):
        assert kw.keyword(v) == read_str_first(raw)

    @pytest.mark.parametrize(
        "k,ns,raw",
        [
            ("kw", "ns", ":ns/kw"),
            ("kw", "qualified.ns", ":qualified.ns/kw"),
            ("kw", "really.qualified.ns", ":really.qualified.ns/kw"),
            ("a:b", "ab", ":ab/a:b"),
            ("a:b", "a:b", ":a:b/a:b"),
            ("#", "html", ":html/#"),
        ],
    )
    def test_legal_ns_keyword(self, k: str, ns: str, raw: str):
        assert kw.keyword(k, ns=ns) == read_str_first(raw)

    @pytest.mark.parametrize(
        "v", ["://", ":ns//kw", ":some/ns/sym", ":ns/sym/", ":/kw"]
    )
    def test_illegal_keyword(self, v: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(v)

    def test_autoresolved_kw(self, test_ns: str, ns: runtime.Namespace):
        assert kw.keyword("kw", ns=test_ns) == read_str_first("::kw")

        new_ns = runtime.Namespace(sym.symbol("other.ns"))
        ns.add_alias(new_ns, sym.symbol("other"))
        assert kw.keyword("kw", ns="other.ns") == read_str_first("::other/kw")

        with pytest.raises(reader.SyntaxError):
            read_str_first("::third/kw")


@pytest.mark.parametrize("s,val", [("nil", None), ("true", True), ("false", False)])
def test_literals(s: str, val):
    assert read_str_first(s) is val


class TestSymbol:
    @pytest.mark.parametrize(
        "s",
        [
            "sym",
            "kebab-sym",
            "underscore_sym",
            "sym?",
            "+",
            "?",
            "=",
            "!",
            "-",
            "*",
            "/",
            ">",
            "->",
            "->>",
            "<",
            "<-",
            "<<-",
            "$",
            "<body>",
            "*mufs*",
            "yay!",
            ".interop",
            "ns.name",
            "*'",
            "a:b",
            "div#id",
        ],
    )
    def test_legal_bare_symbol(self, s: str):
        assert sym.symbol(s) == read_str_first(s)

    @pytest.mark.parametrize(
        "s,ns,raw",
        [
            ("sym", "ns", "ns/sym"),
            ("sym", "qualified.ns", "qualified.ns/sym"),
            ("sym", "really.qualified.ns", "really.qualified.ns/sym"),
            (".interop", "ns.second", "ns.second/.interop"),
            ("sy:m", "ns", "ns/sy:m"),
            ("sy:m", "n:s", "n:s/sy:m"),
        ],
    )
    def test_legal_ns_symbol(self, s: str, ns: str, raw: str):
        assert sym.symbol(s, ns=ns) == read_str_first(raw)

    @pytest.mark.parametrize(
        "v",
        [
            "//",
            "ns//sym",
            "some/ns/sym",
            "ns/sym/",
            "/sym",
            ".second.ns/name",
            "ns..third/name",
            "#",
            # This will raise because the default pushback depth of the
            # reader.StreamReader instance used by the reader is 5, so
            # we are unable to pushback more - characters consumed by
            # reader._read_num trying to parse a number.
            "------->",
        ],
    )
    def test_illegal_symbol(self, v: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(v)


class TestString:
    @pytest.mark.parametrize(
        "v,raw",
        [
            ("", '""'),
            ('"', r'"\""'),
            ("\\", r'"\\"'),
            ("\a", r'"\a"'),
            ("\b", r'"\b"'),
            ("\f", r'"\f"'),
            ("\n", r'"\n"'),
            ("\r", r'"\r"'),
            ("\t", r'"\t"'),
            ("\v", r'"\v"'),
            ("Hello,\nmy name is\tChris.", r'"Hello,\nmy name is\tChris."'),
            ("Regular string", '"Regular string"'),
            ("String with 'inner string'", "\"String with 'inner string'\""),
            ('String with "inner string"', r'"String with \"inner string\""'),
        ],
    )
    def test_legal_string(self, v: str, raw: str):
        assert v == read_str_first(raw)

    def test_invalid_escape(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first(r'"\q"')

    def test_missing_terminating_quote(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first('"Start of a string')


class TestFormatString:
    def test_must_include_quote(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first(r"#f []")

    @pytest.mark.parametrize(
        "v,raw",
        [
            ("", '#f ""'),
            ('"', r'#f "\""'),
            ("\\", r'#f "\\"'),
            ("\a", r'#f "\a"'),
            ("\b", r'#f "\b"'),
            ("\f", r'#f "\f"'),
            ("\n", r'#f "\n"'),
            ("\r", r'#f "\r"'),
            ("\t", r'#f "\t"'),
            ("\v", r'#f "\v"'),
            ("Hello,\nmy name is\tChris.", r'#f "Hello,\nmy name is\tChris."'),
            ("Regular string", '#f "Regular string"'),
            ("String with 'inner string'", "#f \"String with 'inner string'\""),
            ('String with "inner string"', r'#f "String with \"inner string\""'),
        ],
    )
    def test_legal_string_is_legal_fstring(self, v: str, raw: str):
        assert v == read_str_first(raw)

    @pytest.mark.parametrize(
        "v,raw",
        [
            (
                llist.l(
                    reader._STR, "[", kw.keyword("whitespace", ns="surrounded.by"), "]"
                ),
                '#f "[{  :surrounded.by/whitespace   }]""',
            ),
            (llist.l(reader._STR, "[", None, "]"), '#f "[{nil}]""'),
            (llist.l(reader._STR, "[", True, "]"), '#f "[{true}]""'),
            (llist.l(reader._STR, "[", False, "]"), '#f "[{false}]""'),
            (llist.l(reader._STR, "[", 0, "]"), '#f "[{0}]""'),
            (llist.l(reader._STR, "[", 0.1, "]"), '#f "[{0.1}]""'),
            (llist.l(reader._STR, "[", kw.keyword("a"), "]"), '#f "[{:a}]""'),
            (llist.l(reader._STR, "[", sym.symbol("sym"), "]"), '#f "[{sym}]""'),
            (
                llist.l(
                    reader._STR, "[", llist.l(reader._QUOTE, sym.symbol("sym")), "]"
                ),
                '#f "[{\'sym}]""',
            ),
            (llist.l(reader._STR, "[", vec.EMPTY, "]"), '#f "[{[]}]""'),
            (llist.l(reader._STR, "[", vec.v("string"), "]"), '#f "[{["string"]}]""'),
            (llist.l(reader._STR, "[", llist.EMPTY, "]"), '#f "[{()}]""'),
            (llist.l(reader._STR, "[", llist.l("string"), "]"), '#f "[{("string")}]""'),
            (llist.l(reader._STR, "[", lset.EMPTY, "]"), '#f "[{#{}}]""'),
            (llist.l(reader._STR, "[", lset.s("string"), "]"), '#f "[{#{"string"}}]""'),
            (llist.l(reader._STR, "[", lmap.EMPTY, "]"), '#f "[{{}}]""'),
            (
                llist.l(reader._STR, "[", lmap.map({kw.keyword("a"): "string"}), "]"),
                '#f "[{{:a "string"}}]""',
            ),
            ("{}", r'#f "\{}""'),
            ("{(inc 1)}", r'#f "\{(inc 1)}""'),
            ("[inner]", '#f "[{"inner"}]""'),
        ],
    )
    def test_legal_fstring(self, v: str, raw: str):
        assert v == read_str_first(raw)

    def test_only_one_expr_allowed(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first(r'#f "one {(+ 1 2) :a} three"')

    def test_invalid_escape(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first(r'#f "\q"')

    def test_missing_expression(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first('#f "some val {} with no expr"')

    def test_missing_terminating_quote(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first('#f "Start of a format string')


class TestByteString:
    def test_must_include_quote(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first(r"#b []")

    @pytest.mark.parametrize(
        "v,raw",
        [
            (b"", '#b""'),
            (b"", '#b ""'),
            (b'"', r'#b "\""'),
            (b"\\", r'#b "\\"'),
            (b"\a", r'#b "\a"'),
            (b"\b", r'#b "\b"'),
            (b"\f", r'#b "\f"'),
            (b"\n", r'#b "\n"'),
            (b"\r", r'#b "\r"'),
            (b"\t", r'#b "\t"'),
            (b"\v", r'#b "\v"'),
            (
                b"\x7f\x45\x4c\x46\x01\x01\x01\x00",
                r'#b "\x7f\x45\x4c\x46\x01\x01\x01\x00"',
            ),
            (b"\x7fELF\x01\x01\x01\x00", r'#b "\x7fELF\x01\x01\x01\x00"'),
            (b"Regular string but with bytes", '#b "Regular string but with bytes"'),
            (
                b"Byte string with 'inner string'",
                "#b \"Byte string with 'inner string'\"",
            ),
            (
                b'Byte string with "inner string"',
                r'#b "Byte string with \"inner string\""',
            ),
        ],
    )
    def test_legal_byte_string(self, v: str, raw: str):
        assert v == read_str_first(raw)

    def test_cannot_include_non_ascii(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first(rf'#b "{chr(432)}"')

    def test_invalid_escape_remains(self):
        assert rb"\q" == read_str_first(r'#b "\q"')

    @pytest.mark.parametrize("v", [r'#b "\xjj"', r'#b "\xf"', r'#b "\x"'])
    def test_invalid_hex_escape_sequence(self, v: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(v)

    def test_missing_terminating_quote(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first('#b "Start of a string')


@pytest.mark.parametrize("s", ["", " ", "\t"])
def test_whitespace(s: str):
    assert read_str_first(s) is None


def test_vector():
    with pytest.raises(reader.SyntaxError):
        read_str_first("[")

    assert read_str_first("[]") == vec.vector([])
    assert read_str_first("[:a]") == vec.v(kw.keyword("a"))
    assert read_str_first("[:a :b]") == vec.v(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("[:a :b :c]") == vec.v(
        kw.keyword("a"), kw.keyword("b"), kw.keyword("c")
    )
    assert read_str_first("[:a, :b]") == vec.v(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("[:a :b, :c]") == vec.v(
        kw.keyword("a"), kw.keyword("b"), kw.keyword("c")
    )
    assert read_str_first("[:a, :b, :c]") == vec.v(
        kw.keyword("a"), kw.keyword("b"), kw.keyword("c")
    )
    assert read_str_first('[1 :a "string"]') == vec.v(1, kw.keyword("a"), "string")
    assert read_str_first('[1, :a, "string"]') == vec.v(1, kw.keyword("a"), "string")
    assert read_str_first('[1.4, :a, "string"]') == vec.v(
        1.4, kw.keyword("a"), "string"
    )

    assert read_str_first("[\n]") == vec.EMPTY
    assert read_str_first("[       :a\n :b\n]") == vec.v(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("[:a :b\n]") == vec.v(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("[:a :b      ]") == vec.v(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("[\n;;comment\n]") == vec.EMPTY
    assert read_str_first("[:a :b\n;;comment\n]") == vec.v(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("[:a \n;;comment\n :b]") == vec.v(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("[\n#_[:a :b]\n]") == vec.EMPTY
    assert read_str_first("[:a :b\n#_[:a :b]\n]") == vec.v(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("[:a \n#_[:a :b]\n :b]") == vec.v(
        kw.keyword("a"), kw.keyword("b")
    )


def test_list():
    with pytest.raises(reader.SyntaxError):
        read_str_first("(")

    assert read_str_first("()") == llist.list([])
    assert read_str_first("(func-with-no-args)") == llist.l(
        sym.symbol("func-with-no-args")
    )
    assert read_str_first('(str/join "one string" " and another")') == llist.l(
        sym.symbol("join", ns="str"), "one string", " and another"
    )
    assert read_str_first("(map inc [1 2 3])") == llist.l(
        sym.symbol("map"), sym.symbol("inc"), vec.v(1, 2, 3)
    )
    assert read_str_first("(- -1 2)") == llist.l(sym.symbol("-"), -1, 2)

    assert read_str_first("(\n)") == llist.EMPTY
    assert read_str_first("(       :a\n :b\n)") == llist.l(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("(:a :b\n)") == llist.l(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("(:a :b      )") == llist.l(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("(\n;;comment\n)") == llist.EMPTY
    assert read_str_first("(:a :b\n;;comment\n)") == llist.l(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("(:a \n;;comment\n :b)") == llist.l(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("(\n#_[:a :b]\n)") == llist.EMPTY
    assert read_str_first("(:a :b\n#_[:a :b]\n)") == llist.l(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("(:a \n#_[:a :b]\n :b)") == llist.l(
        kw.keyword("a"), kw.keyword("b")
    )


def test_set():
    with pytest.raises(reader.SyntaxError):
        read_str_first("#{")

    assert read_str_first("#{}") == lset.set([])
    assert read_str_first("#{:a}") == lset.s(kw.keyword("a"))
    assert read_str_first("#{:a :b}") == lset.s(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("#{:a :b :c}") == lset.s(
        kw.keyword("a"), kw.keyword("b"), kw.keyword("c")
    )
    assert read_str_first('#{:a 1 "some string"}') == lset.s(
        kw.keyword("a"), 1, "some string"
    )

    assert read_str_first("#{\n}") == lset.EMPTY
    assert read_str_first("#{       :a\n :b\n}") == lset.s(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("#{:a :b\n}") == lset.s(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("#{:a :b      }") == lset.s(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("#{\n;;comment\n}") == lset.EMPTY
    assert read_str_first("#{:a :b\n;;comment\n}") == lset.s(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("#{:a \n;;comment\n :b}") == lset.s(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("#{\n#_[:a :b]\n}") == lset.EMPTY
    assert read_str_first("#{:a :b\n#_[:a :b]\n}") == lset.s(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("#{:a \n#_[:a :b]\n :b}") == lset.s(
        kw.keyword("a"), kw.keyword("b")
    )

    with pytest.raises(reader.SyntaxError):
        read_str_first("#{:a :b :b}")


def test_map():
    with pytest.raises(reader.SyntaxError):
        read_str_first("{")

    with pytest.raises(reader.SyntaxError):
        read_str_first("{#py [] :some-keyword}")

    assert read_str_first("{}") == lmap.map({})
    assert read_str_first("{:a 1}") == lmap.map({kw.keyword("a"): 1})
    assert read_str_first('{:a 1 :b "string"}') == lmap.map(
        {kw.keyword("a"): 1, kw.keyword("b"): "string"}
    )

    assert read_str_first("{\n}") == lmap.EMPTY
    assert read_str_first("{       :a\n :b\n}") == lmap.map(
        {kw.keyword("a"): kw.keyword("b")}
    )
    assert read_str_first("{:a :b\n}") == lmap.map({kw.keyword("a"): kw.keyword("b")})
    assert read_str_first("{:a :b      }") == lmap.map(
        {kw.keyword("a"): kw.keyword("b")}
    )
    assert read_str_first("{\n;;comment\n}") == lmap.EMPTY
    assert read_str_first("{:a :b\n;;comment\n}") == lmap.map(
        {kw.keyword("a"): kw.keyword("b")}
    )
    assert read_str_first("{:a \n;;comment\n :b}") == lmap.map(
        {kw.keyword("a"): kw.keyword("b")}
    )
    assert read_str_first("{\n#_[:a :b]\n}") == lmap.EMPTY
    assert read_str_first("{:a :b\n#_[:a :b]\n}") == lmap.map(
        {kw.keyword("a"): kw.keyword("b")}
    )
    assert read_str_first("{:a \n#_[:a :b]\n :b}") == lmap.map(
        {kw.keyword("a"): kw.keyword("b")}
    )

    with pytest.raises(reader.SyntaxError):
        read_str_first("{:a 1 :b 2 :a 3}")

    with pytest.raises(reader.SyntaxError):
        read_str_first("{:a}")

    with pytest.raises(reader.SyntaxError):
        read_str_first("{:a 1 :b}")


def test_namespaced_map(test_ns: str, ns: runtime.Namespace):
    assert lmap.map(
        {
            kw.keyword("name", ns="member"): "Chris",
            kw.keyword("gender", ns="person"): "M",
            kw.keyword("id"): 15,
        }
    ) == read_str_first('#:person {:member/name "Chris" :gender "M" :_/id 15}')
    assert lmap.map(
        {
            sym.symbol("name", ns="member"): "Chris",
            sym.symbol("gender", ns="person"): "M",
            sym.symbol("id"): 15,
        }
    ) == read_str_first('#:person{member/name "Chris" gender "M" _/id 15}')

    with pytest.raises(reader.SyntaxError):
        read_str_first('#:person/thing {member/name "Chris" gender "M" _/id 15}')

    assert lmap.map(
        {
            kw.keyword("name", ns="member"): "Chris",
            kw.keyword("gender", ns=test_ns): "M",
            kw.keyword("id"): 15,
        }
    ) == read_str_first('#:: {:member/name "Chris" :gender "M" :_/id 15}')
    assert lmap.map(
        {
            sym.symbol("name", ns="member"): "Chris",
            sym.symbol("gender", ns=test_ns): "M",
            sym.symbol("id"): 15,
        }
    ) == read_str_first('#::{member/name "Chris" gender "M" _/id 15}')

    assert lmap.map(
        {
            kw.keyword("name", ns="member"): "Chris",
            kw.keyword("gender", ns="person"): "M",
            kw.keyword("id"): 15,
            kw.keyword("address", ns="person"): lmap.map(
                {kw.keyword("city"): "New York"}
            ),
        }
    ) == read_str_first(
        """
        #:person {:member/name "Chris"
                  :gender "M"
                  :_/id 15
                  :address {:city "New York"}}"""
    )
    assert lmap.map(
        {
            kw.keyword("name", ns="member"): "Chris",
            kw.keyword("gender", ns="person"): "M",
            kw.keyword("id"): 15,
            kw.keyword("address", ns="person"): lmap.map(
                {kw.keyword("city", ns="address"): "New York"}
            ),
        }
    ) == read_str_first(
        """
            #:person {:member/name "Chris"
                      :gender "M"
                      :_/id 15
                      :address #:address{:city "New York"}}"""
    )


def test_quoted():
    assert read_str_first("'a") == llist.l(sym.symbol("quote"), sym.symbol("a"))
    assert read_str_first("'some.ns/sym") == llist.l(
        sym.symbol("quote"), sym.symbol("sym", ns="some.ns")
    )
    assert read_str_first("'(def a 3)") == llist.l(
        sym.symbol("quote"), llist.l(sym.symbol("def"), sym.symbol("a"), 3)
    )


class TestSyntaxQuote:
    def test_resolve_with_simple_resolver(self):
        resolver = lambda s: sym.symbol(s.name, ns="test-ns")
        assert llist.l(
            reader._SEQ,
            llist.l(
                reader._CONCAT,
                llist.l(
                    reader._LIST,
                    llist.l(sym.symbol("quote"), sym.symbol("my-symbol", ns="test-ns")),
                ),
            ),
        ) == read_str_first(
            "`(my-symbol)", resolver=resolver
        ), "Resolve fully qualified symbol in syntax quote"

    def test_resolve_with_standard_resolver(self, test_ns: str, ns: runtime.Namespace):
        assert llist.l(
            reader._SEQ,
            llist.l(
                reader._CONCAT,
                llist.l(
                    reader._LIST,
                    llist.l(sym.symbol("quote"), sym.symbol("my-symbol", ns=test_ns)),
                ),
            ),
        ) == read_str_first(
            "`(my-symbol)", resolver=runtime.resolve_alias
        ), "Resolve a symbol in the current namespace"

    def test_resolve_with_custom_resolver(self):
        def complex_resolver(s: sym.Symbol) -> sym.Symbol:
            if s.name == "other-symbol":
                return s
            return sym.symbol(s.name, ns="test-ns")

        assert llist.l(
            reader._SEQ,
            llist.l(
                reader._CONCAT,
                llist.l(
                    reader._LIST,
                    llist.l(sym.symbol("quote"), sym.symbol("my-symbol", ns="test-ns")),
                ),
                llist.l(
                    reader._LIST,
                    llist.l(sym.symbol("quote"), sym.symbol("other-symbol")),
                ),
            ),
        ) == read_str_first(
            "`(my-symbol other-symbol)", resolver=complex_resolver
        ), "Resolve multiple symbols together"

    def test_resolve_inner_forms_even_in_quote(self):
        assert llist.l(
            reader._SEQ,
            llist.l(
                reader._CONCAT,
                llist.l(
                    reader._LIST,
                    llist.l(
                        reader._SEQ,
                        llist.l(
                            reader._CONCAT,
                            llist.l(
                                reader._LIST,
                                llist.l(sym.symbol("quote"), sym.symbol("quote")),
                            ),
                            llist.l(
                                reader._LIST,
                                llist.l(sym.symbol("quote"), sym.symbol("my-symbol")),
                            ),
                        ),
                    ),
                ),
            ),
        ) == read_str_first("`('my-symbol)"), "Resolver inner forms, even in quote"

    def test_do_not_resolve_unquoted_quoted_symbols(self):
        assert llist.l(
            reader._SEQ,
            llist.l(
                reader._CONCAT,
                llist.l(
                    reader._LIST, llist.l(sym.symbol("quote"), sym.symbol("my-symbol"))
                ),
            ),
        ) == read_str_first("`(~'my-symbol)"), "Do not resolve unquoted quoted syms"

    def test_reader_var_macro_works_with_unquote(self):
        assert llist.l(
            reader._SEQ,
            llist.l(
                reader._CONCAT,
                llist.l(
                    reader._LIST,
                    llist.l(
                        reader._SEQ,
                        llist.l(
                            reader._CONCAT,
                            llist.l(
                                reader._LIST,
                                llist.l(sym.symbol("quote"), sym.symbol("var")),
                            ),
                            llist.l(
                                reader._LIST,
                                llist.l(sym.symbol("quote"), sym.symbol("a-symbol")),
                            ),
                        ),
                    ),
                ),
            ),
        ) == read_str_first("`(#'~'a-symbol)"), "Reader var macro works with unquote"

    @pytest.mark.parametrize(
        "code,v",
        [
            ("`&", llist.l(sym.symbol("quote"), sym.symbol("&"))),
            ("`nil", None),
            ("`true", True),
            ("`false", False),
            ("`.interop", llist.l(sym.symbol("quote"), sym.symbol(".interop"))),
        ],
    )
    def test_do_not_resolve_unnamespaced_special_symbols(self, code: str, v):
        assert v == read_str_first(code)

    @pytest.mark.parametrize(
        "code,v",
        [
            ("`test-ns/&", llist.l(sym.symbol("quote"), sym.symbol("&", ns="test-ns"))),
            (
                "`test-ns/nil",
                llist.l(sym.symbol("quote"), sym.symbol("nil", ns="test-ns")),
            ),
            (
                "`test-ns/true",
                llist.l(sym.symbol("quote"), sym.symbol("true", ns="test-ns")),
            ),
            (
                "`test-ns/false",
                llist.l(sym.symbol("quote"), sym.symbol("false", ns="test-ns")),
            ),
            (
                "`test-ns/.interop",
                llist.l(sym.symbol("quote"), sym.symbol(".interop", ns="test-ns")),
            ),
        ],
    )
    def test_resolve_namespaced_special_symbols(self, code: str, v):
        assert v == read_str_first(code)

    @pytest.mark.parametrize(
        "code,v",
        [
            ("`#py []", []),
            ("`#py [1 2 3]", [1, 2, 3]),
            ("`#py ()", tuple()),
            ("`#py (1 2 3)", (1, 2, 3)),
            ("`#py #{}", set()),
            ("`#py #{1 2 3}", {1, 2, 3}),
            ("`#py {}", {}),
            ("`#py {1 2 3 4}", {1: 2, 3: 4}),
            (
                '`#uuid "c28f97e2-15b3-445f-91f9-c57fc71c9556"',
                uuid.UUID("c28f97e2-15b3-445f-91f9-c57fc71c9556"),
            ),
            (
                '#inst "2024-11-19T15:55:58.000000+00:00"',
                datetime.datetime(
                    2024, 11, 19, 15, 55, 58, tzinfo=datetime.timezone.utc
                ),
            ),
        ],
    )
    def test_do_not_resolve_data_reader_tags(self, code: str, v):
        assert v == read_str_first(code)

    class TestGensym:
        @pytest.fixture
        def resolver(self) -> Resolver:
            return lambda s: sym.symbol(s.name, ns="test-ns")

        def test_lone_gensym(self, resolver: Resolver):
            gensym = read_str_first("`s#", resolver=resolver)
            assert isinstance(gensym, llist.PersistentList)
            assert gensym.first == reader._QUOTE
            genned_sym: sym.Symbol = gensym[1]
            assert genned_sym.name.startswith("s_")

        def test_multiple_gensyms(self, resolver: Resolver):
            # Verify that identical gensym forms resolve to the same
            # symbol inside the same syntax quote expansion
            multisym = read_str_first("`(s1# s2# s1#)", resolver=resolver)[1].rest

            multisym1 = multisym[0][1]
            assert reader._QUOTE == multisym1[0]
            genned_sym1: sym.Symbol = multisym1[1]
            assert genned_sym1.ns is None
            assert genned_sym1.name.startswith("s1_")
            assert "#" not in genned_sym1.name

            multisym2 = multisym[1][1]
            assert reader._QUOTE == multisym2[0]
            genned_sym2: sym.Symbol = multisym2[1]
            assert genned_sym2.ns is None
            assert genned_sym2.name.startswith("s2_")
            assert "#" not in genned_sym2.name

            multisym3 = multisym[2][1]
            assert reader._QUOTE == multisym3[0]
            genned_sym3: sym.Symbol = multisym3[1]
            assert genned_sym3.ns is None
            assert genned_sym3.name.startswith("s1_")
            assert "#" not in genned_sym3.name

            assert genned_sym1 == genned_sym3
            assert genned_sym1 != genned_sym2

        def test_cannot_gensym_outside_syntax_quote(self):
            # Gensym literals must appear inside of syntax quote
            with pytest.raises(reader.SyntaxError):
                read_str_first("s#")


def test_unquote():
    assert llist.l(reader._UNQUOTE, sym.symbol("my-symbol")) == read_str_first(
        "~my-symbol"
    )

    assert llist.l(
        sym.symbol("quote"),
        llist.l(
            sym.symbol("print"),
            llist.l(sym.symbol("unquote", ns="basilisp.core"), sym.symbol("val")),
        ),
    ) == read_str_first("'(print ~val)"), "Unquote a symbol in a quote"

    resolver = lambda s: sym.symbol(s.name, ns="test-ns")
    assert llist.l(
        reader._SEQ,
        llist.l(
            reader._CONCAT,
            llist.l(
                reader._LIST,
                llist.l(sym.symbol("quote"), sym.symbol("my-symbol", ns="test-ns")),
            ),
            llist.l(reader._LIST, sym.symbol("other-symbol")),
        ),
    ) == read_str_first(
        "`(my-symbol ~other-symbol)", resolver=resolver
    ), "Resolve multiple symbols together"


def test_unquote_splicing():
    assert llist.l(reader._UNQUOTE_SPLICING, sym.symbol("my-symbol")) == read_str_first(
        "~@my-symbol"
    )
    assert llist.l(reader._UNQUOTE_SPLICING, vec.v(1, 2, 3)) == read_str_first(
        "~@[1 2 3]"
    )
    assert llist.l(reader._UNQUOTE_SPLICING, llist.l(1, 2, 3)) == read_str_first(
        "~@(1 2 3)"
    )
    assert llist.l(reader._UNQUOTE_SPLICING, lset.s(1, 2, 3)) == read_str_first(
        "~@#{1 2 3}"
    )
    assert llist.l(reader._UNQUOTE_SPLICING, lmap.map({1: 2, 3: 4})) == read_str_first(
        "~@{1 2 3 4}"
    )

    with pytest.raises(reader.SyntaxError):
        read_str_first("`~@[1 2 3]")

    with pytest.raises(reader.SyntaxError):
        read_str_first("`~@(1 2 3)")

    assert llist.l(
        sym.symbol("quote"),
        llist.l(
            sym.symbol("print"),
            llist.l(sym.symbol("unquote-splicing", ns="basilisp.core"), vec.v(1, 2, 3)),
        ),
    ) == read_str_first("'(print ~@[1 2 3])"), "Unquote splice a collection in a quote"

    assert llist.l(
        sym.symbol("quote"),
        llist.l(
            sym.symbol("print"),
            llist.l(
                sym.symbol("unquote-splicing", ns="basilisp.core"), sym.symbol("c")
            ),
        ),
    ) == read_str_first("'(print ~@c)"), "Unquote-splice a symbol in a quote"

    resolver = lambda s: sym.symbol(s.name, ns="test-ns")
    assert llist.l(
        reader._SEQ,
        llist.l(
            reader._CONCAT,
            llist.l(
                reader._LIST,
                llist.l(sym.symbol("quote"), sym.symbol("my-symbol", ns="test-ns")),
            ),
            vec.v(1, 2, 3),
        ),
    ) == read_str_first(
        "`(my-symbol ~@[1 2 3])", resolver=resolver
    ), "Unquote-splice a collection in a syntax quote"

    assert llist.l(
        reader._SEQ,
        llist.l(
            reader._CONCAT,
            llist.l(
                reader._LIST,
                llist.l(sym.symbol("quote"), sym.symbol("my-symbol", ns="test-ns")),
            ),
            sym.symbol("a"),
        ),
    ) == read_str_first(
        "`(my-symbol ~@a)", resolver=resolver
    ), "Unquote-splice a collection in a syntax quote"


def test_var():
    assert read_str_first("#'a") == llist.l(sym.symbol("var"), sym.symbol("a"))
    assert read_str_first("#'some.ns/a") == llist.l(
        sym.symbol("var"), sym.symbol("a", ns="some.ns")
    )


class TestMetadata:
    @staticmethod
    def assert_is_submap(m, sub):
        for k, subv in sub.items():
            try:
                mv = m[k]
                if subv != mv:
                    pytest.fail(f"Map key {k}: {mv} != {subv}")
            except KeyError:
                pytest.fail(f"Missing key {k}")
        return True

    @pytest.mark.parametrize(
        "s,form,expected_meta",
        [
            (
                "^str s",
                sym.symbol("s"),
                lmap.map(
                    {
                        kw.keyword("tag"): sym.symbol("str"),
                        reader.READER_LINE_KW: 1,
                        reader.READER_END_LINE_KW: 1,
                        reader.READER_COL_KW: 5,
                        reader.READER_END_COL_KW: 6,
                    }
                ),
            ),
            (
                "^:dynamic *ns*",
                sym.symbol("*ns*"),
                lmap.map({kw.keyword("dynamic"): True}),
            ),
            (
                '^{:doc "If true, assert."} *assert*',
                sym.symbol("*assert*"),
                lmap.map({kw.keyword("doc"): "If true, assert."}),
            ),
            (
                "^[] {}",
                lmap.EMPTY,
                lmap.map({kw.keyword("param-tags"): vec.EMPTY}),
            ),
            (
                '^[:a b "c"] {}',
                lmap.EMPTY,
                lmap.map(
                    {
                        kw.keyword("param-tags"): vec.v(
                            kw.keyword("a"), sym.symbol("b"), "c"
                        )
                    }
                ),
            ),
            (
                "^:has-meta [:a]",
                vec.v(kw.keyword("a")),
                lmap.map({kw.keyword("has-meta"): True}),
            ),
            (
                "^:has-meta (:a)",
                llist.l(kw.keyword("a")),
                lmap.map({kw.keyword("has-meta"): True}),
            ),
            (
                '^:has-meta {:key "val"}',
                lmap.map({kw.keyword("key"): "val"}),
                lmap.map({kw.keyword("has-meta"): True}),
            ),
            (
                "^:has-meta #{:a}",
                lset.s(kw.keyword("a")),
                lmap.map({kw.keyword("has-meta"): True}),
            ),
            (
                '^:dynamic ^{:doc "If true, assert."} ^python/bool ^[:dynamic :muffs] *assert*',
                sym.symbol("*assert*"),
                lmap.map(
                    {
                        kw.keyword("dynamic"): True,
                        kw.keyword("doc"): "If true, assert.",
                        kw.keyword("tag"): sym.symbol("bool", ns="python"),
                        kw.keyword("param-tags"): vec.v(
                            kw.keyword("dynamic"), kw.keyword("muffs")
                        ),
                    }
                ),
            ),
            (
                "^{:always true} ^{:always false} *assert*",
                sym.symbol("*assert*"),
                lmap.map({kw.keyword("always"): True}),
            ),
        ],
    )
    def test_legal_reader_metadata(
        self, s: str, form, expected_meta: lmap.PersistentMap
    ):
        v = read_str_first(s)
        assert v == form
        self.assert_is_submap(v.meta, expected_meta)

    @pytest.mark.parametrize(
        "s",
        [
            "^35233 {}",
            "^583.28 {}",
            "^12.6J {}",
            "^22/7 {}",
            "^12.6M {}",
            "^true {}",
            "^false {}",
            "^nil {}",
            '^"String value" {}',
        ],
    )
    def test_syntax_error_attaching_unsupported_type_as_metadata(self, s: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(s)

    @pytest.mark.parametrize(
        "s",
        [
            "^:has-meta 35233",
            "^:has-meta 583.28",
            "^:has-meta 12.6J",
            "^:has-meta 22/7",
            "^:has-meta 12.6M",
            "^:has-meta :i-am-a-keyword",
            "^:has-meta true",
            "^:has-meta false",
            "^:has-meta nil",
            '^:has-meta "String value"',
        ],
    )
    def test_syntax_error_attaching_metadata_to_unsupported_type(self, s: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(s)


class TestCommentReaderMacro:
    @pytest.mark.parametrize(
        "s",
        [
            "#_       (a list)",
            "#_1",
            '#_"string"',
            "#_:keyword",
            "#_symbol",
            "#_[]",
            "#_{}",
            "#_()",
            "#_#{}",
            "#_ #_ {} {}",
        ],
    )
    def test_comment_suppresses_form_with_eof(self, s: str):
        with pytest.raises(EOFError):
            read_str_first(s, is_eof_error=True)

    @pytest.mark.parametrize(
        "s,v",
        [
            ("#_:kw1 :kw2", kw.keyword("kw2")),
            ("#_ #_ :kw1 :kw2 :kw3", kw.keyword("kw3")),
            ("(#_sym)", llist.EMPTY),
            ("(#_ #_ sym 1)", llist.EMPTY),
            ("(inc #_counter 5)", llist.l(sym.symbol("inc"), 5)),
            ("(#_inc dec #_counter 8)", llist.l(sym.symbol("dec"), 8)),
            ("[#_m]", vec.EMPTY),
            ("[#_m 1]", vec.v(1)),
            ("[#_m 1 #_2]", vec.v(1)),
            ("[#_m 1 2]", vec.v(1, 2)),
            ("[#_m 1 #_2 4]", vec.v(1, 4)),
            ("[#_m 1 #_2 4 #_5]", vec.v(1, 4)),
            ("#{#_m}", lset.EMPTY),
            ("#{#_m 1}", lset.s(1)),
            ("#{#_m 1 #_2}", lset.s(1)),
            ("#{#_m 1 2}", lset.s(1, 2)),
            ("#{#_m 1 #_2 4}", lset.s(1, 4)),
            ("#{#_m 1 #_2 4 #_5}", lset.s(1, 4)),
            ("{#_:key}", lmap.EMPTY),
            ('{#_:key #_"value"}', lmap.EMPTY),
            ('{#_ #_:key "value"}', lmap.EMPTY),
            ('{:key #_"other" "value"}', lmap.map({kw.keyword("key"): "value"})),
            ('{:key "value" #_"other"}', lmap.map({kw.keyword("key"): "value"})),
            ("{#_ #_:a 3 :b 5}", lmap.map({kw.keyword("b"): 5})),
        ],
    )
    def test_comment_suppresses_form_in_other_form(self, s: str, v):
        assert v == read_str_first(s)

    def test_comment_creates_syntax_error(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first('{:key #_"value"}')


def test_comment_line():
    assert None is read_str_first("; I'm a little comment short and stout")
    assert kw.keyword("kw2") == read_str_first(";; :kw1\n:kw2")
    assert llist.l(sym.symbol("form"), kw.keyword("keyword")) == read_str_first(
        """;; Comment
        (form :keyword)
        """
    )


def test_shebang_line():
    assert None is read_str_first("#! I'm a little shebang short and stout")
    assert kw.keyword("kw2") == read_str_first("#!/usr/bin/env basilisp run\n:kw2")
    assert llist.l(sym.symbol("form"), kw.keyword("keyword")) == read_str_first(
        """#!/usr/bin/env basilisp run
        (form :keyword)
        """
    )


class TestReaderConditional:
    @pytest.mark.parametrize(
        "v",
        [
            # Keyword feature
            "#?(:clj 1 :lpy 2 default 3)",
            "#?(:clj 1 lpy 2 :default 3)",
            "#?(clj 1 :lpy 2 :default 3)",
            # Duplicate feature
            "#?(:clj 1 :clj 2 :default 3)",
            "#?(:clj 1 :lpy 2 :clj 3)",
            # Invalid collection
            "#?[:clj 1 :lpy 2 :default 3]",
            "#?(:clj 1 :lpy 2 :default 3",
            "#?(:clj 1 :lpy 2 :default 3]",
            # Even number of forms
            "#?(:clj)",
            "#?(:clj 1 lpy)",
            "#?(clj 1 :lpy 2 :default)",
        ],
    )
    def test_invalid_basic_form_syntax(self, v: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(v)

    def test_basic_form(self):
        assert 2 == read_str_first("#?(:clj 1 :lpy 2 :default 3)")
        assert 2 == read_str_first("#?(:clj #_1 1 #_:lpy :lpy 2 :default 3)")
        assert 1 == read_str_first("#?(:default 1 :lpy 2)")
        assert None is read_str_first("#?(:clj 1 :cljs 2)")
        assert [[], (), {}, set()] == read_str_first(
            "#?(:cljs #js [] :lpy #py [#py [] #py () #py {} #py #{}] :default [])"
        )

    def test_basic_form_preserving(self):
        c = read_str_first("#?(:clj 1 :lpy 2 :default 3)", process_reader_cond=False)
        assert isinstance(c, reader.ReaderConditional)
        assert not c.is_splicing
        assert False is c.val_at(reader.READER_COND_SPLICING_KW)
        assert llist.l(
            kw.keyword("clj"), 1, kw.keyword("lpy"), 2, kw.keyword("default"), 3
        ) == c.val_at(reader.READER_COND_FORM_KW)
        assert "#?(:clj 1 :lpy 2 :default 3)" == c.lrepr()

    def test_form_preserving_with_unknown_data_readers(self):
        c = read_str_first(
            "#?(:cljs #js [] :lpy #py [] :default [])", process_reader_cond=False
        )
        assert isinstance(c, reader.ReaderConditional)
        assert not c.is_splicing
        assert False is c.val_at(reader.READER_COND_SPLICING_KW)
        assert llist.l(
            kw.keyword("cljs"),
            tagged_literal(sym.symbol("js"), vec.EMPTY),
            kw.keyword("lpy"),
            tagged_literal(sym.symbol("py"), vec.EMPTY),
            kw.keyword("default"),
            vec.EMPTY,
        ) == c.val_at(reader.READER_COND_FORM_KW)
        assert "#?(:cljs #js [] :lpy #py [] :default [])" == c.lrepr()

    def test_ignore_unknown_data_readers_in_non_selected_conditional(self):
        v = read_str_first("#?(:cljs #js [] :default [])")
        assert isinstance(v, vec.PersistentVector)
        assert v == vec.EMPTY

    @pytest.mark.parametrize(
        "s,expected",
        [
            (
                "#?(:cljs [#?(:lpy :py :default :other)] :default :none)",
                kw.keyword("none"),
            ),
            (
                "#?(:lpy [#?(:lpy :py :default :other)] :default :none)",
                vec.v(kw.keyword("py")),
            ),
            (
                "#?(:lpy [#?(:clj :py :default :other)] :default :none)",
                vec.v(kw.keyword("other")),
            ),
            (
                "#?(:cljs [#?@(:clj [1 2] :default [3 4])] :default :none)",
                kw.keyword("none"),
            ),
            (
                "#?(:lpy [#?@(:clj [1 2] :default [3 4])] :default :none)",
                vec.v(3, 4),
            ),
            (
                "#?(:lpy [#?@(:clj [1 2] :cljs [3 4])] :default :none)",
                vec.EMPTY,
            ),
            (
                "#?(#?@(:clj [:clj [1 2]] :lpy [:lpy [3 4]]) :default [])",
                vec.v(3, 4),
            ),
            (
                "#?(#?@(:clj [:clj [1 2]] :lpy [:cljs [3 4]]) :default [])",
                vec.EMPTY,
            ),
            (
                "#?(#?@(:clj [1 2]) :default :none)",
                kw.keyword("none"),
            ),
        ],
    )
    def test_nested_reader_conditionals(self, s: str, expected):
        assert expected == read_str_first(s)

    @pytest.mark.parametrize(
        "v",
        [
            # No splice context
            "#?@(:clj [1 2 3] :lpy [4 5 6] :default [7 8 9])",
            # Invalid splice collection
            "(#?@(:clj (1 2) :lpy (3 4)))",
            "(#?@(:clj #{1 2} :lpy #{3 4}))",
            "[#?@(:clj (1 2) :lpy (3 4))]",
            "[#?@(:clj #{1 2} :lpy #{3 4})]",
            "#{#?@(:clj (1 2) :lpy (3 4))}",
            "#{#?@(:clj #{1 2} :lpy #{3 4})}",
            "{#?@(:clj (1 2) :lpy (3 4))}",
            "{#?@(:clj #{1 2} :lpy #{3 4})}",
            # Invalid container collection
            "#?@[:clj 1 :lpy 2 :default 3]",
            "#?@(:clj 1 :lpy 2 :default 3",
            # Keyword feature
            "#?@(:clj [1] :lpy [2] default [3])",
            "#?@(:clj [1] lpy [2] :default [3])",
            "#?@(clj [1] :lpy [2] :default [3])",
            # Duplicate feature
            "#?@(:clj [1] :clj [2] :default [3])",
            "#?@(:clj [1] :lpy [2] :clj [3])",
            # Even number of forms
            "#?@(:clj)",
            "#?@(:clj [1] lpy)",
            "#?@(clj [1] :lpy [2] :default)",
            # Invalid splice connection (in nested reader conditional)
            "#?(#?@(:lpy (:lpy [])) :default :none)",
        ],
    )
    def test_splicing_form_syntax(self, v: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(v)

    def test_splicing_form_preserving(self):
        c = read_str_first(
            "#?@(:clj [1 2 3] :lpy [4 5 6] :default [7 8 9])", process_reader_cond=False
        )
        assert isinstance(c, reader.ReaderConditional)
        assert c.is_splicing
        assert True is c.val_at(reader.READER_COND_SPLICING_KW)
        assert llist.l(
            kw.keyword("clj"),
            vec.v(1, 2, 3),
            kw.keyword("lpy"),
            vec.v(4, 5, 6),
            kw.keyword("default"),
            vec.v(7, 8, 9),
        ) == c.val_at(reader.READER_COND_FORM_KW)
        assert "#?@(:clj [1 2 3] :lpy [4 5 6] :default [7 8 9])" == c.lrepr()

    def test_splicing_form(self):
        assert llist.l(1, 3, 5, 7) == read_str_first(
            "(1 #?@(:clj [2 4 6] :lpy [3 5 7]))"
        )
        assert vec.v(1, 3, 5, 7) == read_str_first("[1 #?@(:clj [2 4 6] :lpy [3 5 7])]")
        assert lset.s(1, 3, 5, 7) == read_str_first(
            "#{1 #?@(:clj [2 4 6] :lpy [3 5 7])}"
        )
        assert llist.l(1) == read_str_first("(1 #?@(:clj [2 4 6]))")
        assert vec.v(1) == read_str_first("[1 #?@(:clj [2 4 6])]")
        assert lset.s(1) == read_str_first("#{1 #?@(:clj [2 4 6])}")

    def test_splicing_form_in_maps(self):
        assert lmap.EMPTY == read_str_first("{#?@(:clj [:a 1])}")
        assert lmap.map({kw.keyword("b"): 2}) == read_str_first(
            "{#?@(:clj [:a 1] :lpy [:b 2])}"
        )
        assert lmap.map({kw.keyword("a"): 3, kw.keyword("c"): 4}) == read_str_first(
            "{:a #?@(:clj [1 :b 2] :lpy [3 :c 4])}"
        )
        assert lmap.map({kw.keyword("a"): 2, kw.keyword("e"): 5}) == read_str_first(
            "{#?@(:clj [:a 1 :b] :lpy [:a 2 :e]) 5}"
        )


class TestFunctionReaderMacro:
    @pytest.mark.parametrize(
        "code,v",
        [
            ("#()", llist.l(sym.symbol("fn*"), vec.v(), None)),
            (
                "#(identity %)",
                llist.l(
                    sym.symbol("fn*"),
                    vec.v(sym.symbol("arg-1")),
                    llist.l(sym.symbol("identity"), sym.symbol("arg-1")),
                ),
            ),
            (
                "#(identity %1)",
                llist.l(
                    sym.symbol("fn*"),
                    vec.v(sym.symbol("arg-1")),
                    llist.l(sym.symbol("identity"), sym.symbol("arg-1")),
                ),
            ),
            (
                "#(identity %& %1)",
                llist.l(
                    sym.symbol("fn*"),
                    vec.v(sym.symbol("arg-1"), sym.symbol("&"), sym.symbol("arg-rest")),
                    llist.l(
                        sym.symbol("identity"),
                        sym.symbol("arg-rest"),
                        sym.symbol("arg-1"),
                    ),
                ),
            ),
            (
                "#(identity %3)",
                llist.l(
                    sym.symbol("fn*"),
                    vec.v(
                        sym.symbol("arg-1"), sym.symbol("arg-2"), sym.symbol("arg-3")
                    ),
                    llist.l(sym.symbol("identity"), sym.symbol("arg-3")),
                ),
            ),
            (
                "#(identity %3 %&)",
                llist.l(
                    sym.symbol("fn*"),
                    vec.v(
                        sym.symbol("arg-1"),
                        sym.symbol("arg-2"),
                        sym.symbol("arg-3"),
                        sym.symbol("&"),
                        sym.symbol("arg-rest"),
                    ),
                    llist.l(
                        sym.symbol("identity"),
                        sym.symbol("arg-3"),
                        sym.symbol("arg-rest"),
                    ),
                ),
            ),
            (
                "#(identity {:arg %})",
                llist.l(
                    sym.symbol("fn*"),
                    vec.v(
                        sym.symbol("arg-1"),
                    ),
                    llist.l(
                        sym.symbol("identity"),
                        lmap.map({kw.keyword("arg"): sym.symbol("arg-1")}),
                    ),
                ),
            ),
            (
                "#(vec %&)",
                llist.l(
                    sym.symbol("fn*"),
                    vec.v(sym.symbol("&"), sym.symbol("arg-rest")),
                    llist.l(sym.symbol("vec"), sym.symbol("arg-rest")),
                ),
            ),
            (
                "#(vector %1 %&)",
                llist.l(
                    sym.symbol("fn*"),
                    vec.v(sym.symbol("arg-1"), sym.symbol("&"), sym.symbol("arg-rest")),
                    llist.l(
                        sym.symbol("vector"),
                        sym.symbol("arg-1"),
                        sym.symbol("arg-rest"),
                    ),
                ),
            ),
        ],
    )
    def test_function_reader_macro(self, code: str, v):
        assert v == read_str_first(code)

    @pytest.mark.parametrize("code", ["#(identity #(%1 %2))", "#app/ermagrd [1 2 3]"])
    def test_invalid_function_reader_macro(self, code: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(code)


def test_deref():
    assert read_str_first("@s") == llist.l(reader._DEREF, sym.symbol("s"))
    assert read_str_first("@ns/s") == llist.l(reader._DEREF, sym.symbol("s", ns="ns"))
    assert read_str_first("@(atom {})") == llist.l(
        reader._DEREF, llist.l(sym.symbol("atom"), lmap.EMPTY)
    )


def test_character_literal():
    assert "a" == read_str_first("\\a")
    assert "[" == read_str_first("\\[")
    assert "," == read_str_first("\\,")
    assert "^" == read_str_first("\\^")
    assert " " == read_str_first("\\ ")
    assert "Ω" == read_str_first("\\Ω")

    assert "Ω" == read_str_first("\\u03A9")

    assert " " == read_str_first("\\space")
    assert "\n" == read_str_first("\\newline")
    assert "\t" == read_str_first("\\tab")
    assert "\b" == read_str_first("\\backspace")
    assert "\f" == read_str_first("\\formfeed")
    assert "\r" == read_str_first("\\return")

    assert vec.v("a") == read_str_first("[\\a]")
    assert vec.v("]") == read_str_first("[\\]]")
    assert vec.v("Ω") == read_str_first("[\\Ω]")

    assert llist.l(sym.symbol("str"), "Ω") == read_str_first("(str \\u03A9)")

    assert vec.v(" ") == read_str_first("[\\space]")
    assert vec.v("\n") == read_str_first("[\\newline]")
    assert vec.v("\t") == read_str_first("[\\tab]")
    assert llist.l(sym.symbol("str"), "\b", "\f", "\r") == read_str_first(
        "(str \\backspace \\formfeed \\return)"
    )

    with pytest.raises(reader.SyntaxError):
        read_str_first("\\u03A9zzz")

    with pytest.raises(reader.SyntaxError):
        read_str_first("\\uFFFFFFFF")

    with pytest.raises(reader.SyntaxError):
        read_str_first("\\blah")


def test_decimal_literal():
    assert langutil.decimal_from_str("3.14") == read_str_first("3.14M")
    assert 3.14 == read_str_first("3.14")

    with pytest.raises(reader.SyntaxError):
        read_str_first("3.14MM")

    with pytest.raises(reader.SyntaxError):
        read_str_first("3.1M4")

    with pytest.raises(reader.SyntaxError):
        read_str_first("3M.14")


def test_fraction_literal():
    assert langutil.fraction(1, 7) == read_str_first("1/7")
    assert langutil.fraction(22, 7) == read_str_first("22/7")

    with pytest.raises(reader.SyntaxError):
        read_str_first("3/7N")

    with pytest.raises(reader.SyntaxError):
        read_str_first("3N/7")

    with pytest.raises(reader.SyntaxError):
        read_str_first("3.3/7")

    with pytest.raises(reader.SyntaxError):
        read_str_first("3/7.4")

    with pytest.raises(reader.SyntaxError):
        read_str_first("3/7/14")


class TestDataReaders:
    def test_inst_reader_literal(self):
        assert (
            read_str_first('#inst "2018-01-18T03:26:57.296-00:00"')
            == langutil.inst_from_str("2018-01-18T03:26:57.296-00:00")
            == datetime.datetime(
                2018, 1, 18, 3, 26, 57, 296000, tzinfo=datetime.timezone.utc
            )
        )

    def test_invalid_inst_reader_literal(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first('#inst "I am a little teapot short and stout"')

    @pytest.mark.parametrize(
        "code,v",
        [
            ("#queue ()", lqueue.EMPTY),
            ("#queue (1 2 3)", lqueue.q(1, 2, 3)),
            (
                '#queue ([1 2 3] :a "b" {:c :d})',
                lqueue.q(
                    vec.v(1, 2, 3),
                    kw.keyword("a"),
                    "b",
                    lmap.map({kw.keyword("c"): kw.keyword("d")}),
                ),
            ),
        ],
    )
    def test_queue_reader_literal(self, code: str, v):
        assert v == read_str_first(code)

    @pytest.mark.parametrize("code,pattern", [('#"hi"', "hi"), (r'#"\s"', r"\s")])
    def test_regex_reader_literal(self, code: str, pattern: str):
        assert read_str_first(code) == langutil.regex_from_str(pattern)
        assert read_str_first(r'#"\s"') == langutil.regex_from_str(r"\s")

    def test_invalid_regex_reader_literal(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first(r'#"\y"')

    def test_numeric_constant_literal(self):
        assert math.isnan(read_str_first("##NaN"))
        assert read_str_first("##Inf") == float("inf")
        assert read_str_first("##-Inf") == -float("inf")

    @pytest.mark.parametrize("code", ["##float/NaN", "##e"])
    def test_invalid_numeric_constant_literal(self, code: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(code)

    def test_uuid_reader_literal(self):
        assert read_str_first(
            '#uuid "4ba98ef0-0620-4966-af61-f0f6c2dbf230"'
        ) == langutil.uuid_from_str("4ba98ef0-0620-4966-af61-f0f6c2dbf230")

    def test_invalid_uuid_reader_literal(self):
        with pytest.raises(reader.SyntaxError):
            read_str_first('#uuid "I am a little teapot short and stout"')

    @pytest.mark.parametrize(
        "code,v",
        [
            ("#py []", []),
            ('#py [1 :a "str"]', [1, kw.keyword("a"), "str"]),
            ("#py ()", ()),
            ('#py (1 :a "str")', (1, kw.keyword("a"), "str")),
            ("#py {}", {}),
            (
                '#py {:a 1 :other "str"}',
                {kw.keyword("a"): 1, kw.keyword("other"): "str"},
            ),
            ("#py #{}", set()),
            ('#py #{1 :a "str"}', {1, kw.keyword("a"), "str"}),
        ],
    )
    def test_python_data_structure_literals(self, code: str, v):
        assert v == read_str_first(code)

    @pytest.mark.parametrize("code", ["#py :kw", '#py "s"', "#py 3"])
    def test_invalid_python_data_structure_literals(self, code: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(code)

    def test_data_readers_qualified_tag(self):
        assert "s" == read_str_first(
            '#foo/bar "s"',
            data_readers=lmap.map({sym.symbol("bar", ns="foo"): lambda v: v}),
        )

    def test_data_readers_simple_tag(self):
        assert "s" == read_str_first(
            '#bar "s"',
            data_readers=lmap.map({sym.symbol("bar"): lambda v: v}),
        )

    def test_default_data_reader_fn(self):
        tag = sym.symbol("bar", ns="foo")
        assert (tag, "s") == read_str_first(
            '#foo/bar "s"', default_data_reader_fn=lambda tag, v: (tag, v)
        )

    @pytest.mark.parametrize("code", ["#boop :hi", "#ns/boop :hi"])
    def test_not_found_tag_error(self, code: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(code)
