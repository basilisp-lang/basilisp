import io
import math
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
from basilisp.lang.interfaces import IPersistentSet


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
            )
        )
    except StopIteration:
        return None


def test_stream_reader():
    sreader = reader.StreamReader(io.StringIO("12345"))

    assert "1" == sreader.peek()
    assert (1, 1) == sreader.loc

    assert "2" == sreader.next_token()
    assert (1, 2) == sreader.loc

    assert "2" == sreader.peek()
    assert (1, 2) == sreader.loc

    sreader.pushback()
    assert "1" == sreader.peek()
    assert (1, 1) == sreader.loc

    assert "2" == sreader.next_token()
    assert (1, 2) == sreader.loc

    assert "3" == sreader.next_token()
    assert (1, 3) == sreader.loc

    assert "4" == sreader.next_token()
    assert (1, 4) == sreader.loc

    assert "5" == sreader.next_token()
    assert (1, 5) == sreader.loc

    assert "" == sreader.next_token()
    assert (1, 6) == sreader.loc


def test_stream_reader_loc():
    s = str("i=1\n" "b=2\n" "i")
    sreader = reader.StreamReader(io.StringIO(s))

    assert "i" == sreader.peek()
    assert (1, 1) == sreader.loc

    assert "=" == sreader.next_token()
    assert (1, 2) == sreader.loc

    assert "=" == sreader.peek()
    assert (1, 2) == sreader.loc

    sreader.pushback()
    assert "i" == sreader.peek()
    assert (1, 1) == sreader.loc

    assert "=" == sreader.next_token()
    assert (1, 2) == sreader.loc

    assert "1" == sreader.next_token()
    assert (1, 3) == sreader.loc

    assert "\n" == sreader.next_token()
    assert (2, 0) == sreader.loc

    assert "b" == sreader.next_token()
    assert (2, 1) == sreader.loc

    assert "=" == sreader.next_token()
    assert (2, 2) == sreader.loc

    assert "2" == sreader.next_token()
    assert (2, 3) == sreader.loc

    assert "\n" == sreader.next_token()
    assert (3, 0) == sreader.loc

    assert "i" == sreader.next_token()
    assert (3, 1) == sreader.loc

    assert "" == sreader.next_token()
    assert (3, 2) == sreader.loc


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
            (0.093_873_72j, "0.09387372J"),
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
            ("0.09387372", 0.093_873_72),
            ("1.0", 1.0),
            ("1.332", 1.332),
            ("-1.332", -1.332),
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


class TestKeyword:
    @pytest.mark.parametrize(
        "v,raw",
        [
            ("kw", ":kw"),
            ("kebab-kw", ":kebab-kw"),
            ("underscore_kw", ":underscore_kw"),
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
        ],
    )
    def test_legal_bare_symbol(self, v: str, raw: str):
        assert kw.keyword(v) == read_str_first(raw)

    @pytest.mark.parametrize(
        "k,ns,raw",
        [
            ("kw", "ns", ":ns/kw"),
            ("kw", "qualified.ns", ":qualified.ns/kw"),
            ("kw", "really.qualified.ns", ":really.qualified.ns/kw"),
        ],
    )
    def test_legal_ns_symbol(self, k: str, ns: str, raw: str):
        assert kw.keyword(k, ns=ns) == read_str_first(raw)

    @pytest.mark.parametrize(
        "v", ["://", ":ns//kw", ":some/ns/sym", ":ns/sym/", ":/kw", ":dotted.kw"]
    )
    def test_illegal_symbol(self, v: str):
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
            "ns.second/.interop",
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

    assert read_str_first("[\n]") == vec.PersistentVector.empty()
    assert read_str_first("[       :a\n :b\n]") == vec.v(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("[:a :b\n]") == vec.v(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("[:a :b      ]") == vec.v(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("[\n;;comment\n]") == vec.PersistentVector.empty()
    assert read_str_first("[:a :b\n;;comment\n]") == vec.v(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("[:a \n;;comment\n :b]") == vec.v(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("[\n#_[:a :b]\n]") == vec.PersistentVector.empty()
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

    assert read_str_first("(\n)") == llist.PersistentList.empty()
    assert read_str_first("(       :a\n :b\n)") == llist.l(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("(:a :b\n)") == llist.l(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("(:a :b      )") == llist.l(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("(\n;;comment\n)") == llist.PersistentList.empty()
    assert read_str_first("(:a :b\n;;comment\n)") == llist.l(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("(:a \n;;comment\n :b)") == llist.l(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("(\n#_[:a :b]\n)") == llist.PersistentList.empty()
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

    assert read_str_first("#{\n}") == lset.PersistentSet.empty()
    assert read_str_first("#{       :a\n :b\n}") == lset.s(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("#{:a :b\n}") == lset.s(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("#{:a :b      }") == lset.s(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first("#{\n;;comment\n}") == lset.PersistentSet.empty()
    assert read_str_first("#{:a :b\n;;comment\n}") == lset.s(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("#{:a \n;;comment\n :b}") == lset.s(
        kw.keyword("a"), kw.keyword("b")
    )
    assert read_str_first("#{\n#_[:a :b]\n}") == lset.PersistentSet.empty()
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

    assert read_str_first("{}") == lmap.map({})
    assert read_str_first("{:a 1}") == lmap.map({kw.keyword("a"): 1})
    assert read_str_first('{:a 1 :b "string"}') == lmap.map(
        {kw.keyword("a"): 1, kw.keyword("b"): "string"}
    )

    assert read_str_first("{\n}") == lmap.PersistentMap.empty()
    assert read_str_first("{       :a\n :b\n}") == lmap.map(
        {kw.keyword("a"): kw.keyword("b")}
    )
    assert read_str_first("{:a :b\n}") == lmap.map({kw.keyword("a"): kw.keyword("b")})
    assert read_str_first("{:a :b      }") == lmap.map(
        {kw.keyword("a"): kw.keyword("b")}
    )
    assert read_str_first("{\n;;comment\n}") == lmap.PersistentMap.empty()
    assert read_str_first("{:a :b\n;;comment\n}") == lmap.map(
        {kw.keyword("a"): kw.keyword("b")}
    )
    assert read_str_first("{:a \n;;comment\n :b}") == lmap.map(
        {kw.keyword("a"): kw.keyword("b")}
    )
    assert read_str_first("{\n#_[:a :b]\n}") == lmap.PersistentMap.empty()
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


def test_syntax_quoted(test_ns: str, ns: runtime.Namespace):
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
                reader._LIST, llist.l(sym.symbol("quote"), sym.symbol("other-symbol"))
            ),
        ),
    ) == read_str_first(
        "`(my-symbol other-symbol)", resolver=complex_resolver
    ), "Resolve multiple symbols together"

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

    assert llist.l(
        reader._SEQ,
        llist.l(
            reader._CONCAT,
            llist.l(
                reader._LIST, llist.l(sym.symbol("quote"), sym.symbol("my-symbol"))
            ),
        ),
    ) == read_str_first("`(~'my-symbol)"), "Do not resolve unquoted quoted syms"


def test_syntax_quote_gensym():
    resolver = lambda s: sym.symbol(s.name, ns="test-ns")

    gensym = read_str_first("`s#", resolver=resolver)
    assert isinstance(gensym, llist.PersistentList)
    assert gensym.first == reader._QUOTE
    genned_sym: sym.Symbol = gensym[1]
    assert genned_sym.name.startswith("s_")

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


def test_meta():
    def issubmap(m, sub):
        for k, subv in sub.items():
            try:
                mv = m[k]
                return subv == mv
            except KeyError:
                return False
        return False

    s = read_str_first("^str s")
    assert s == sym.symbol("s")
    assert issubmap(s.meta, lmap.map({kw.keyword("tag"): sym.symbol("str")}))

    s = read_str_first("^:dynamic *ns*")
    assert s == sym.symbol("*ns*")
    assert issubmap(s.meta, lmap.map({kw.keyword("dynamic"): True}))

    s = read_str_first('^{:doc "If true, assert."} *assert*')
    assert s == sym.symbol("*assert*")
    assert issubmap(s.meta, lmap.map({kw.keyword("doc"): "If true, assert."}))

    v = read_str_first("^:has-meta [:a]")
    assert v == vec.v(kw.keyword("a"))
    assert issubmap(v.meta, lmap.map({kw.keyword("has-meta"): True}))

    l = read_str_first("^:has-meta (:a)")
    assert l == llist.l(kw.keyword("a"))
    assert issubmap(l.meta, lmap.map({kw.keyword("has-meta"): True}))

    m = read_str_first('^:has-meta {:key "val"}')
    assert m == lmap.map({kw.keyword("key"): "val"})
    assert issubmap(m.meta, lmap.map({kw.keyword("has-meta"): True}))

    t = read_str_first("^:has-meta #{:a}")
    assert t == lset.s(kw.keyword("a"))
    assert issubmap(t.meta, lmap.map({kw.keyword("has-meta"): True}))

    s = read_str_first('^:dynamic ^{:doc "If true, assert."} *assert*')
    assert s == sym.symbol("*assert*")
    assert issubmap(
        s.meta,
        lmap.map({kw.keyword("dynamic"): True, kw.keyword("doc"): "If true, assert."}),
    )

    s = read_str_first("^{:always true} ^{:always false} *assert*")
    assert s == sym.symbol("*assert*")
    assert issubmap(s.meta, lmap.map({kw.keyword("always"): True}))


def test_invalid_meta_structure():
    with pytest.raises(reader.SyntaxError):
        read_str_first("^35233 {}")

    with pytest.raises(reader.SyntaxError):
        read_str_first("^583.28 {}")

    with pytest.raises(reader.SyntaxError):
        read_str_first("^true {}")

    with pytest.raises(reader.SyntaxError):
        read_str_first("^false {}")

    with pytest.raises(reader.SyntaxError):
        read_str_first("^nil {}")

    with pytest.raises(reader.SyntaxError):
        read_str_first('^"String value" {}')


def test_invalid_meta_attachment():
    with pytest.raises(reader.SyntaxError):
        read_str_first("^:has-meta 35233")

    with pytest.raises(reader.SyntaxError):
        read_str_first("^:has-meta 583.28")

    with pytest.raises(reader.SyntaxError):
        read_str_first("^:has-meta :i-am-a-keyword")

    with pytest.raises(reader.SyntaxError):
        read_str_first("^:has-meta true")

    with pytest.raises(reader.SyntaxError):
        read_str_first("^:has-meta false")

    with pytest.raises(reader.SyntaxError):
        read_str_first("^:has-meta nil")

    with pytest.raises(reader.SyntaxError):
        read_str_first('^:has-meta "String value"')


def test_comment_reader_macro():
    with pytest.raises(EOFError):
        read_str_first("#_       (a list)", is_eof_error=True)

    with pytest.raises(EOFError):
        read_str_first("#_1", is_eof_error=True)

    with pytest.raises(EOFError):
        read_str_first('#_"string"', is_eof_error=True)

    with pytest.raises(EOFError):
        read_str_first("#_:keyword", is_eof_error=True)

    with pytest.raises(EOFError):
        read_str_first("#_symbol", is_eof_error=True)

    with pytest.raises(EOFError):
        read_str_first("#_[]", is_eof_error=True)

    with pytest.raises(EOFError):
        read_str_first("#_{}", is_eof_error=True)

    with pytest.raises(EOFError):
        read_str_first("#_()", is_eof_error=True)

    with pytest.raises(EOFError):
        read_str_first("#_#{}", is_eof_error=True)

    assert kw.keyword("kw2") == read_str_first("#_:kw1 :kw2")

    assert llist.PersistentList.empty() == read_str_first("(#_sym)")
    assert llist.l(sym.symbol("inc"), 5) == read_str_first("(inc #_counter 5)")
    assert llist.l(sym.symbol("dec"), 8) == read_str_first("(#_inc dec #_counter 8)")

    assert vec.PersistentVector.empty() == read_str_first("[#_m]")
    assert vec.v(1) == read_str_first("[#_m 1]")
    assert vec.v(1) == read_str_first("[#_m 1 #_2]")
    assert vec.v(1, 2) == read_str_first("[#_m 1 2]")
    assert vec.v(1, 4) == read_str_first("[#_m 1 #_2 4]")
    assert vec.v(1, 4) == read_str_first("[#_m 1 #_2 4 #_5]")

    assert lset.PersistentSet.empty() == read_str_first("#{#_m}")
    assert lset.s(1) == read_str_first("#{#_m 1}")
    assert lset.s(1) == read_str_first("#{#_m 1 #_2}")
    assert lset.s(1, 2) == read_str_first("#{#_m 1 2}")
    assert lset.s(1, 4) == read_str_first("#{#_m 1 #_2 4}")
    assert lset.s(1, 4) == read_str_first("#{#_m 1 #_2 4 #_5}")

    assert lmap.PersistentMap.empty() == read_str_first("{#_:key}")
    assert lmap.PersistentMap.empty() == read_str_first('{#_:key #_"value"}')
    assert lmap.map({kw.keyword("key"): "value"}) == read_str_first(
        '{:key #_"other" "value"}'
    )
    assert lmap.map({kw.keyword("key"): "value"}) == read_str_first(
        '{:key "value" #_"other"}'
    )

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
    def test_basic_form_syntax(self, v: str):
        with pytest.raises(reader.SyntaxError):
            read_str_first(v)

    def test_basic_form(self):
        assert 2 == read_str_first("#?(:clj 1 :lpy 2 :default 3)")
        assert 1 == read_str_first("#?(:default 1 :lpy 2)")
        assert None is read_str_first("#?(:clj 1 :cljs 2)")

    def test_basic_form_preserving(self):
        c = read_str_first("#?(:clj 1 :lpy 2 :default 3)", process_reader_cond=False)
        assert isinstance(c, reader.ReaderConditional)
        assert not c.is_splicing
        assert False is c.val_at(reader.READER_COND_SPLICING_KW)
        assert llist.l(
            kw.keyword("clj"), 1, kw.keyword("lpy"), 2, kw.keyword("default"), 3
        ) == c.val_at(reader.READER_COND_FORM_KW)
        assert "#?(:clj 1 :lpy 2 :default 3)" == c.lrepr()

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
        assert lmap.PersistentMap.empty() == read_str_first("{#?@(:clj [:a 1])}")
        assert lmap.map({kw.keyword("b"): 2}) == read_str_first(
            "{#?@(:clj [:a 1] :lpy [:b 2])}"
        )
        assert lmap.map({kw.keyword("a"): 3, kw.keyword("c"): 4}) == read_str_first(
            "{:a #?@(:clj [1 :b 2] :lpy [3 :c 4])}"
        )
        assert lmap.map({kw.keyword("a"): 2, kw.keyword("e"): 5}) == read_str_first(
            "{#?@(:clj [:a 1 :b] :lpy [:a 2 :e]) 5}"
        )


def test_function_reader_macro():
    assert read_str_first("#()") == llist.l(sym.symbol("fn*"), vec.v(), None)
    assert read_str_first("#(identity %)") == llist.l(
        sym.symbol("fn*"),
        vec.v(sym.symbol("arg-1")),
        llist.l(sym.symbol("identity"), sym.symbol("arg-1")),
    )
    assert read_str_first("#(identity %1)") == llist.l(
        sym.symbol("fn*"),
        vec.v(sym.symbol("arg-1")),
        llist.l(sym.symbol("identity"), sym.symbol("arg-1")),
    )
    assert read_str_first("#(identity %& %1)") == llist.l(
        sym.symbol("fn*"),
        vec.v(sym.symbol("arg-1"), sym.symbol("&"), sym.symbol("arg-rest")),
        llist.l(sym.symbol("identity"), sym.symbol("arg-rest"), sym.symbol("arg-1")),
    )
    assert read_str_first("#(identity %3)") == llist.l(
        sym.symbol("fn*"),
        vec.v(sym.symbol("arg-1"), sym.symbol("arg-2"), sym.symbol("arg-3")),
        llist.l(sym.symbol("identity"), sym.symbol("arg-3")),
    )
    assert read_str_first("#(identity %3 %&)") == llist.l(
        sym.symbol("fn*"),
        vec.v(
            sym.symbol("arg-1"),
            sym.symbol("arg-2"),
            sym.symbol("arg-3"),
            sym.symbol("&"),
            sym.symbol("arg-rest"),
        ),
        llist.l(sym.symbol("identity"), sym.symbol("arg-3"), sym.symbol("arg-rest")),
    )
    assert read_str_first("#(identity {:arg %})") == llist.l(
        sym.symbol("fn*"),
        vec.v(
            sym.symbol("arg-1"),
        ),
        llist.l(
            sym.symbol("identity"), lmap.map({kw.keyword("arg"): sym.symbol("arg-1")})
        ),
    )

    with pytest.raises(reader.SyntaxError):
        read_str_first("#(identity #(%1 %2))")

    with pytest.raises(reader.SyntaxError):
        read_str_first("#app/ermagrd [1 2 3]")


def test_deref():
    assert read_str_first("@s") == llist.l(reader._DEREF, sym.symbol("s"))
    assert read_str_first("@ns/s") == llist.l(reader._DEREF, sym.symbol("s", ns="ns"))
    assert read_str_first("@(atom {})") == llist.l(
        reader._DEREF, llist.l(sym.symbol("atom"), lmap.PersistentMap.empty())
    )


def test_character_literal():
    assert "a" == read_str_first("\\a")
    assert "Ω" == read_str_first("\\Ω")

    assert "Ω" == read_str_first("\\u03A9")

    assert " " == read_str_first("\\space")
    assert "\n" == read_str_first("\\newline")
    assert "\t" == read_str_first("\\tab")
    assert "\b" == read_str_first("\\backspace")
    assert "\f" == read_str_first("\\formfeed")
    assert "\r" == read_str_first("\\return")

    assert vec.v("a") == read_str_first("[\\a]")
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


def test_inst_reader_literal():
    assert read_str_first(
        '#inst "2018-01-18T03:26:57.296-00:00"'
    ) == langutil.inst_from_str("2018-01-18T03:26:57.296-00:00")

    with pytest.raises(reader.SyntaxError):
        read_str_first('#inst "I am a little teapot short and stout"')


def test_queue_reader_literal():
    assert read_str_first("#queue ()") == lqueue.EMPTY
    assert read_str_first("#queue (1 2 3)") == lqueue.q(1, 2, 3)
    assert read_str_first('#queue ([1 2 3] :a "b" {:c :d})') == lqueue.q(
        vec.v(1, 2, 3),
        kw.keyword("a"),
        "b",
        lmap.map({kw.keyword("c"): kw.keyword("d")}),
    )


def test_regex_reader_literal():
    assert read_str_first('#"hi"') == langutil.regex_from_str("hi")
    assert read_str_first(r'#"\s"') == langutil.regex_from_str(r"\s")

    with pytest.raises(reader.SyntaxError):
        read_str_first(r'#"\y"')


def test_numeric_constant_literal():
    assert math.isnan(read_str_first("##NaN"))
    assert read_str_first("##Inf") == float("inf")
    assert read_str_first("##-Inf") == -float("inf")

    with pytest.raises(reader.SyntaxError):
        read_str_first("##float/NaN")

    with pytest.raises(reader.SyntaxError):
        read_str_first("##e")


def test_uuid_reader_literal():
    assert read_str_first(
        '#uuid "4ba98ef0-0620-4966-af61-f0f6c2dbf230"'
    ) == langutil.uuid_from_str("4ba98ef0-0620-4966-af61-f0f6c2dbf230")

    with pytest.raises(reader.SyntaxError):
        read_str_first('#uuid "I am a little teapot short and stout"')


def test_python_literals():
    assert [] == read_str_first("#py []")
    assert [1, kw.keyword("a"), "str"] == read_str_first('#py [1 :a "str"]')

    assert () == read_str_first("#py ()")
    assert (1, kw.keyword("a"), "str") == read_str_first('#py (1 :a "str")')

    assert {} == read_str_first("#py {}")
    assert {kw.keyword("a"): 1, kw.keyword("other"): "str"} == read_str_first(
        '#py {:a 1 :other "str"}'
    )

    assert set() == read_str_first("#py #{}")
    assert {1, kw.keyword("a"), "str"} == read_str_first('#py #{1 :a "str"}')

    with pytest.raises(reader.SyntaxError):
        read_str_first("#py :kw")

    with pytest.raises(reader.SyntaxError):
        read_str_first('#py "s"')

    with pytest.raises(reader.SyntaxError):
        read_str_first("#py 3")


def test_namespace_tags_allowed():
    assert "s" == read_str_first(
        '#foo/bar "s"',
        data_readers=lmap.map({sym.symbol("bar", ns="foo"): lambda v: v}),
    )


def test_non_namespaced_tags_reserved():
    with pytest.raises(TypeError):
        read_str_first(
            "#boop :hi", data_readers=lmap.map({kw.keyword("boop"): lambda v: v})
        )

    with pytest.raises(ValueError):
        read_str_first(
            "#boop :hi", data_readers=lmap.map({sym.symbol("boop"): lambda v: v})
        )


def test_not_found_tag_error():
    with pytest.raises(reader.SyntaxError):
        read_str_first("#boop :hi")

    with pytest.raises(reader.SyntaxError):
        read_str_first("#ns/boop :hi")
