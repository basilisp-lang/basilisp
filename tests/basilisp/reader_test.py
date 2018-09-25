import io

import pytest

import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.runtime as runtime
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.util as langutil
import basilisp.lang.vector as vec
import basilisp.reader as reader
from basilisp.lang.runtime import Var


@pytest.fixture
def test_ns() -> str:
    return "test"


@pytest.fixture
def ns_var(test_ns: str):
    runtime.init_ns_var(which_ns=runtime._CORE_NS)
    yield runtime.set_current_ns(test_ns)


def read_str_first(s, resolver: reader.Resolver = None, data_readers=None):
    """Read the first form from the input string. If no form
    is found, return None."""
    try:
        return next(reader.read_str(s, resolver=resolver, data_readers=data_readers))
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
    s = str(
        "i=1\n"
        "b=2\n"
        "i"
    )
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


def test_complex():
    assert read_str_first("1J") == 1J
    assert read_str_first("100J") == 100J
    assert read_str_first("99927273J") == 99927273J
    assert read_str_first("0J") == 0J
    assert read_str_first("-1J") == -1J
    assert read_str_first("-538282J") == -538282J

    with pytest.raises(reader.SyntaxError):
        read_str_first("1JJ")

    with pytest.raises(reader.SyntaxError):
        read_str_first("1NJ")

    assert read_str_first("0.0J") == 0.0J
    assert read_str_first("0.09387372J") == 0.09387372J
    assert read_str_first("1.0J") == 1.0J
    assert read_str_first("1.332J") == 1.332J
    assert read_str_first("-1.332J") == -1.332J
    assert read_str_first("-1.0J") == -1.0J
    assert read_str_first("-0.332J") == -0.332J

    with pytest.raises(reader.SyntaxError):
        read_str_first("1.0MJ")

    with pytest.raises(reader.SyntaxError):
        read_str_first("22/7J")

    with pytest.raises(reader.SyntaxError):
        read_str_first("22J/7")


def test_int():
    assert read_str_first("1") == 1
    assert read_str_first("100") == 100
    assert read_str_first("99927273") == 99927273
    assert read_str_first("0") == 0
    assert read_str_first("-1") == -1
    assert read_str_first("-538282") == -538282

    assert read_str_first("1N") == 1
    assert read_str_first("100N") == 100
    assert read_str_first("99927273N") == 99927273
    assert read_str_first("0N") == 0
    assert read_str_first("-1N") == -1
    assert read_str_first("-538282N") == -538282

    with pytest.raises(reader.SyntaxError):
        read_str_first("1NN")


def test_float():
    assert read_str_first("0.0") == 0.0
    assert read_str_first("0.09387372") == 0.09387372
    assert read_str_first("1.0") == 1.0
    assert read_str_first("1.332") == 1.332
    assert read_str_first("-1.332") == -1.332
    assert read_str_first("-1.0") == -1.0
    assert read_str_first("-0.332") == -0.332

    with pytest.raises(reader.SyntaxError):
        read_str_first("0..11")

    with pytest.raises(reader.SyntaxError):
        read_str_first("0.111.9")


def test_kw():
    assert kw.keyword("kw") == read_str_first(":kw")
    assert kw.keyword("kebab-kw") == read_str_first(":kebab-kw")
    assert kw.keyword("underscore_kw") == read_str_first(":underscore_kw")
    assert kw.keyword("kw?") == read_str_first(":kw?")
    assert kw.keyword("+") == read_str_first(":+")
    assert kw.keyword("?") == read_str_first(":?")
    assert kw.keyword("=") == read_str_first(":=")
    assert kw.keyword("!") == read_str_first(":!")
    assert kw.keyword("-") == read_str_first(":-")
    assert kw.keyword("*") == read_str_first(":*")
    assert kw.keyword("/") == read_str_first(":/")
    assert kw.keyword(">") == read_str_first(":>")
    assert kw.keyword("->") == read_str_first(":->")
    assert kw.keyword("->>") == read_str_first(":->>")
    assert kw.keyword("-->") == read_str_first(":-->")
    assert kw.keyword("--------------->") == read_str_first(":--------------->")
    assert kw.keyword("<") == read_str_first(":<")
    assert kw.keyword("<-") == read_str_first(":<-")
    assert kw.keyword("<--") == read_str_first(":<--")
    assert kw.keyword("<body>") == read_str_first(":<body>")
    assert kw.keyword("*muffs*") == read_str_first(":*muffs*")
    assert kw.keyword("yay!") == read_str_first(":yay!")

    assert kw.keyword("kw", ns="ns") == read_str_first(":ns/kw")
    assert kw.keyword("kw", ns="qualified.ns") == read_str_first(
        ":qualified.ns/kw")
    assert kw.keyword("kw", ns="really.qualified.ns") == read_str_first(
        ":really.qualified.ns/kw")

    with pytest.raises(reader.SyntaxError):
        read_str_first("://")

    with pytest.raises(reader.SyntaxError):
        read_str_first(":ns//kw")

    with pytest.raises(reader.SyntaxError):
        read_str_first(":some/ns/sym")

    with pytest.raises(reader.SyntaxError):
        read_str_first(":ns/sym/")

    with pytest.raises(reader.SyntaxError):
        read_str_first(":/kw")

    with pytest.raises(reader.SyntaxError):
        read_str_first(":dotted.kw")


def test_literals():
    assert read_str_first("nil") is None
    assert read_str_first("true") is True
    assert read_str_first("false") is False


def test_symbol():
    assert sym.symbol("sym") == read_str_first("sym")
    assert sym.symbol("kebab-sym") == read_str_first("kebab-sym")
    assert sym.symbol("underscore_sym") == read_str_first("underscore_sym")
    assert sym.symbol("sym?") == read_str_first("sym?")
    assert sym.symbol("+") == read_str_first("+")
    assert sym.symbol("?") == read_str_first("?")
    assert sym.symbol("=") == read_str_first("=")
    assert sym.symbol("!") == read_str_first("!")
    assert sym.symbol("-") == read_str_first("-")
    assert sym.symbol("*") == read_str_first("*")
    assert sym.symbol("/") == read_str_first("/")
    assert sym.symbol(">") == read_str_first(">")
    assert sym.symbol("->") == read_str_first("->")
    assert sym.symbol("->>") == read_str_first("->>")
    assert sym.symbol("-->") == read_str_first("-->")
    assert sym.symbol("<") == read_str_first("<")
    assert sym.symbol("<-") == read_str_first("<-")
    assert sym.symbol("<--") == read_str_first("<--")
    assert sym.symbol("<body>") == read_str_first("<body>")
    assert sym.symbol("*muffs*") == read_str_first("*muffs*")
    assert sym.symbol("yay!") == read_str_first("yay!")

    assert sym.symbol("sym", ns="ns") == read_str_first("ns/sym")
    assert sym.symbol(
        "sym", ns="qualified.ns") == read_str_first("qualified.ns/sym")
    assert sym.symbol(
        "sym", ns="really.qualified.ns") == read_str_first("really.qualified.ns/sym")

    with pytest.raises(reader.SyntaxError):
        read_str_first("//")

    with pytest.raises(reader.SyntaxError):
        read_str_first("ns//sym")

    with pytest.raises(reader.SyntaxError):
        read_str_first("some/ns/sym")

    with pytest.raises(reader.SyntaxError):
        read_str_first("ns/sym/")

    with pytest.raises(reader.SyntaxError):
        read_str_first("/sym")

    with pytest.raises(reader.SyntaxError):
        # This will raise because the default pushback depth of the
        # reader.StreamReader instance used by the reader is 5, so
        # we are unable to pushback more - characters consumed by
        # reader._read_num trying to parse a number.
        read_str_first("------->")


def test_str():
    assert '' == read_str_first('""')

    assert "\"" == read_str_first(r'"\""')
    assert "\\" == read_str_first(r'"\\"')
    assert "\a" == read_str_first(r'"\a"')
    assert "\b" == read_str_first(r'"\b"')
    assert "\f" == read_str_first(r'"\f"')
    assert "\n" == read_str_first(r'"\n"')
    assert "\r" == read_str_first(r'"\r"')
    assert "\t" == read_str_first(r'"\t"')
    assert "\v" == read_str_first(r'"\v"')

    with pytest.raises(reader.SyntaxError):
        read_str_first(r'"\q"')

    assert "Hello,\nmy name is\tChris." == read_str_first(r'"Hello,\nmy name is\tChris."')

    assert 'Regular string' == read_str_first('"Regular string"')
    assert "String with 'inner string'" == read_str_first(
        '"String with \'inner string\'"')
    assert 'String with "inner string"' == read_str_first(
        r'"String with \"inner string\""')

    with pytest.raises(reader.SyntaxError):
        read_str_first('"Start of a string')


def test_whitespace():
    assert read_str_first('') is None
    assert read_str_first(' ') is None
    assert read_str_first('\t') is None


def test_vector():
    with pytest.raises(reader.SyntaxError):
        read_str_first('[')

    assert read_str_first('[]') == vec.vector([])
    assert read_str_first('[:a]') == vec.v(kw.keyword("a"))
    assert read_str_first('[:a :b]') == vec.v(kw.keyword("a"), kw.keyword("b"))
    assert read_str_first('[:a :b :c]') == vec.v(
        kw.keyword("a"), kw.keyword("b"), kw.keyword("c"))
    assert read_str_first('[:a, :b]') == vec.v(
        kw.keyword("a"), kw.keyword("b"))
    assert read_str_first('[:a :b, :c]') == vec.v(
        kw.keyword("a"), kw.keyword("b"), kw.keyword("c"))
    assert read_str_first('[:a, :b, :c]') == vec.v(
        kw.keyword("a"), kw.keyword("b"), kw.keyword("c"))
    assert read_str_first('[1 :a "string"]') == vec.v(1, kw.keyword("a"),
                                                      "string")
    assert read_str_first('[1, :a, "string"]') == vec.v(
        1, kw.keyword("a"), "string")
    assert read_str_first('[1.4, :a, "string"]') == vec.v(
        1.4, kw.keyword("a"), "string")


def test_list():
    with pytest.raises(reader.SyntaxError):
        read_str_first('(')

    assert read_str_first('()') == llist.list([])
    assert read_str_first('(func-with-no-args)') == llist.l(
        sym.symbol("func-with-no-args"))
    assert read_str_first('(str/join "one string" " and another")') == llist.l(
        sym.symbol("join", ns='str'), "one string", " and another")
    assert read_str_first('(map inc [1 2 3])') == llist.l(
        sym.symbol("map"), sym.symbol("inc"), vec.v(1, 2, 3))
    assert read_str_first('(- -1 2)') == llist.l(sym.symbol("-"), -1, 2)


def test_set():
    with pytest.raises(reader.SyntaxError):
        read_str_first('#{')

    assert read_str_first('#{}') == lset.set([])
    assert read_str_first('#{:a}') == lset.s(kw.keyword("a"))
    assert read_str_first('#{:a :b}') == lset.s(
        kw.keyword("a"), kw.keyword("b"))
    assert read_str_first('#{:a :b :c}') == lset.s(
        kw.keyword("a"), kw.keyword("b"), kw.keyword("c"))
    assert read_str_first('#{:a 1 "some string"}') == lset.s(
        kw.keyword("a"), 1, "some string")

    with pytest.raises(reader.SyntaxError):
        read_str_first('#{:a :b :b}')


def test_map():
    with pytest.raises(reader.SyntaxError):
        read_str_first('{')

    assert read_str_first('{}') == lmap.map({})
    assert read_str_first('{:a 1}') == lmap.map({kw.keyword('a'): 1})
    assert read_str_first('{:a 1 :b "string"}') == lmap.map({
        kw.keyword('a'):
            1,
        kw.keyword('b'):
            "string"
    })

    with pytest.raises(reader.SyntaxError):
        read_str_first('{:a 1 :b 2 :a 3}')

    with pytest.raises(reader.SyntaxError):
        read_str_first('{:a}')

    with pytest.raises(reader.SyntaxError):
        read_str_first('{:a 1 :b}')


def test_quoted():
    assert read_str_first("'a") == llist.l(
        sym.symbol('quote'), sym.symbol('a'))
    assert read_str_first("'some.ns/sym") == llist.l(
        sym.symbol('quote'), sym.symbol('sym', ns='some.ns'))
    assert read_str_first("'(def a 3)") == llist.l(
        sym.symbol('quote'), llist.l(sym.symbol('def'), sym.symbol('a'), 3))


def test_syntax_quoted(test_ns: str, ns_var: Var):
    resolver = lambda s: sym.symbol(s.name, ns='test-ns')
    assert llist.l(reader._SEQ,
                   llist.l(reader._CONCAT,
                           llist.l(reader._LIST,
                                   llist.l(sym.symbol('quote'), sym.symbol('my-symbol', ns='test-ns'))))
                   ) == read_str_first('`(my-symbol)', resolver=resolver
                                       ), "Resolve fully qualified symbol in syntax quote"

    assert llist.l(reader._SEQ,
                   llist.l(reader._CONCAT,
                           llist.l(reader._LIST,
                                   llist.l(sym.symbol('quote'), sym.symbol('my-symbol', ns=test_ns))))
                   ) == read_str_first('`(my-symbol)',
                                       resolver=runtime.resolve_alias), "Resolve a symbol in the current namespace"

    def complex_resolver(s: sym.Symbol) -> sym.Symbol:
        if s.name == 'other-symbol':
            return s
        return sym.symbol(s.name, ns='test-ns')

    assert llist.l(reader._SEQ,
                   llist.l(reader._CONCAT,
                           llist.l(reader._LIST,
                                   llist.l(sym.symbol('quote'), sym.symbol('my-symbol', ns='test-ns'))),
                           llist.l(reader._LIST,
                                   llist.l(sym.symbol('quote'), sym.symbol('other-symbol'))))
                   ) == read_str_first(
        '`(my-symbol other-symbol)', resolver=complex_resolver), "Resolve multiple symbols together"

    assert llist.l(reader._SEQ,
                   llist.l(reader._CONCAT,
                           llist.l(reader._LIST,
                                   llist.l(reader._SEQ,
                                           llist.l(reader._CONCAT,
                                                   llist.l(reader._LIST,
                                                           llist.l(sym.symbol('quote'),
                                                                   sym.symbol('quote'))),
                                                   llist.l(reader._LIST,
                                                           llist.l(sym.symbol('quote'), sym.symbol('my-symbol'))))
                                           )
                                   ))) == read_str_first("`('my-symbol)"), "Resolver inner forms, even in quote"

    assert llist.l(reader._SEQ,
                   llist.l(reader._CONCAT,
                           llist.l(reader._LIST,
                                   llist.l(sym.symbol('quote'), sym.symbol('my-symbol'))))
                   ) == read_str_first("`(~'my-symbol)"), "Do not resolve unquoted quoted syms"


def test_syntax_quote_gensym():
    gensym = read_str_first("`s#")
    assert isinstance(gensym, llist.List)
    assert gensym.first == reader._QUOTE
    genned_sym: sym.Symbol = gensym[1]
    assert genned_sym.name.startswith("s_")

    # Verify that identical gensym forms resolve to the same
    # symbol inside the same syntax quote expansion
    multisym = read_str_first("`(s1# s2# s1#)")[1].rest

    multisym1 = multisym[0][1]
    assert reader._QUOTE == multisym1[0]
    genned_sym1: sym.Symbol = multisym1[1]
    assert genned_sym1.name.startswith("s1_")

    multisym2 = multisym[1][1]
    assert reader._QUOTE == multisym2[0]
    genned_sym2: sym.Symbol = multisym2[1]
    assert genned_sym2.name.startswith("s2_")

    multisym3 = multisym[2][1]
    assert reader._QUOTE == multisym3[0]
    genned_sym3: sym.Symbol = multisym3[1]
    assert genned_sym3.name.startswith("s1_")

    assert genned_sym1 == genned_sym3
    assert genned_sym1 != genned_sym2

    # Gensym literals must appear inside of syntax quote
    with pytest.raises(reader.SyntaxError):
        read_str_first("s#")


def test_unquote():
    assert llist.l(reader._UNQUOTE, sym.symbol('my-symbol')) == read_str_first('~my-symbol')

    assert llist.l(sym.symbol('quote'),
                   llist.l(sym.symbol('print'),
                           llist.l(sym.symbol('unquote', ns='basilisp.core'), sym.symbol('val')))
                   ) == read_str_first("'(print ~val)"), "Unquote a symbol in a quote"

    resolver = lambda s: sym.symbol(s.name, ns='test-ns')
    assert llist.l(reader._SEQ,
                   llist.l(reader._CONCAT,
                           llist.l(reader._LIST,
                                   llist.l(sym.symbol('quote'), sym.symbol('my-symbol', ns='test-ns'))),
                           llist.l(reader._LIST, sym.symbol('other-symbol')))
                   ) == read_str_first(
        '`(my-symbol ~other-symbol)', resolver=resolver), "Resolve multiple symbols together"


def test_unquote_splicing():
    assert llist.l(reader._UNQUOTE_SPLICING, sym.symbol('my-symbol')) == read_str_first('~@my-symbol')
    assert llist.l(reader._UNQUOTE_SPLICING, vec.v(1, 2, 3)) == read_str_first('~@[1 2 3]')
    assert llist.l(reader._UNQUOTE_SPLICING, llist.l(1, 2, 3)) == read_str_first('~@(1 2 3)')
    assert llist.l(reader._UNQUOTE_SPLICING, lset.s(1, 2, 3)) == read_str_first('~@#{1 2 3}')
    assert llist.l(reader._UNQUOTE_SPLICING, lmap.map({1: 2, 3: 4})) == read_str_first('~@{1 2 3 4}')

    with pytest.raises(reader.SyntaxError):
        read_str_first('`~@[1 2 3]')

    with pytest.raises(reader.SyntaxError):
        read_str_first('`~@(1 2 3)')

    assert llist.l(sym.symbol('quote'), llist.l(sym.symbol('print'), llist.l(
        sym.symbol('unquote-splicing', ns='basilisp.core'), vec.v(1, 2, 3)))
                   ) == read_str_first("'(print ~@[1 2 3])"), "Unquote splice a collection in a quote"

    assert llist.l(sym.symbol('quote'), llist.l(sym.symbol('print'), llist.l(
        sym.symbol('unquote-splicing', ns='basilisp.core'), sym.symbol('c')))
                   ) == read_str_first("'(print ~@c)"), "Unquote-splice a symbol in a quote"

    resolver = lambda s: sym.symbol(s.name, ns='test-ns')
    assert llist.l(reader._SEQ,
                   llist.l(reader._CONCAT,
                           llist.l(reader._LIST,
                                   llist.l(sym.symbol('quote'), sym.symbol('my-symbol', ns='test-ns'))),
                           vec.v(1, 2, 3))
                   ) == read_str_first(
        '`(my-symbol ~@[1 2 3])', resolver=resolver), "Unquote-splice a collection in a syntax quote"

    assert llist.l(reader._SEQ,
                   llist.l(reader._CONCAT,
                           llist.l(reader._LIST,
                                   llist.l(sym.symbol('quote'), sym.symbol('my-symbol', ns='test-ns'))),
                           sym.symbol('a'))
                   ) == read_str_first(
        '`(my-symbol ~@a)', resolver=resolver), "Unquote-splice a collection in a syntax quote"


def test_var():
    assert read_str_first("#'a") == llist.l(sym.symbol('var'), sym.symbol('a'))
    assert read_str_first("#'some.ns/a") == llist.l(
        sym.symbol('var'), sym.symbol('a', ns='some.ns'))


def test_interop_call():
    assert read_str_first('(. "STRING" lower)') == llist.l(
        sym.symbol('.'), "STRING", sym.symbol('lower'))
    assert read_str_first('(.lower "STRING")') == llist.l(
        sym.symbol('.'), "STRING", sym.symbol('lower'))
    assert read_str_first('(.split "www.google.com" ".")') == llist.l(
        sym.symbol('.'), "www.google.com", sym.symbol('split'), ".")
    assert read_str_first('(. "www.google.com" split ".")') == llist.l(
        sym.symbol('.'), "www.google.com", sym.symbol('split'), ".")

    with pytest.raises(reader.SyntaxError):
        read_str_first('(."non-symbol" symbol)')


def test_interop_prop():
    assert read_str_first("(. sym -name)") == llist.l(
        sym.symbol('.-'), sym.symbol('sym'), sym.symbol('name'))
    assert read_str_first('(.-algorithm encoder)') == llist.l(
        sym.symbol('.-'), sym.symbol('encoder'), sym.symbol('algorithm'))

    with pytest.raises(reader.SyntaxError):
        read_str_first('(.- name sym)')

    with pytest.raises(reader.SyntaxError):
        read_str_first('(.-"string" sym)')


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
    assert s == sym.symbol('s')
    assert issubmap(s.meta, lmap.map({kw.keyword('tag'): sym.symbol('str')}))

    s = read_str_first("^:dynamic *ns*")
    assert s == sym.symbol('*ns*')
    assert issubmap(s.meta, lmap.map({kw.keyword('dynamic'): True}))

    s = read_str_first('^{:doc "If true, assert."} *assert*')
    assert s == sym.symbol('*assert*')
    assert issubmap(s.meta, lmap.map({kw.keyword('doc'): "If true, assert."}))

    v = read_str_first("^:has-meta [:a]")
    assert v == vec.v(kw.keyword('a'))
    assert issubmap(v.meta, lmap.map({kw.keyword('has-meta'): True}))

    l = read_str_first('^:has-meta (:a)')
    assert l == llist.l(kw.keyword('a'))
    assert issubmap(l.meta, lmap.map({kw.keyword('has-meta'): True}))

    m = read_str_first('^:has-meta {:key "val"}')
    assert m == lmap.map({kw.keyword('key'): "val"})
    assert issubmap(m.meta, lmap.map({kw.keyword('has-meta'): True}))

    t = read_str_first('^:has-meta #{:a}')
    assert t == lset.s(kw.keyword('a'))
    assert issubmap(t.meta, lmap.map({kw.keyword('has-meta'): True}))

    s = read_str_first('^:dynamic ^{:doc "If true, assert."} *assert*')
    assert s == sym.symbol('*assert*')
    assert issubmap(s.meta, lmap.map({
        kw.keyword('dynamic'): True,
        kw.keyword('doc'): "If true, assert."
    }))

    s = read_str_first('^{:always true} ^{:always false} *assert*')
    assert s == sym.symbol('*assert*')
    assert issubmap(s.meta, lmap.map({kw.keyword('always'): True}))


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
    assert None is read_str_first('#_       (a list)')
    assert None is read_str_first('#_1')
    assert None is read_str_first('#_"string"')
    assert None is read_str_first('#_:keyword')
    assert None is read_str_first('#_symbol')
    assert kw.keyword('kw2') == read_str_first('#_:kw1 :kw2')

    assert llist.List.empty() == read_str_first('(#_sym)')
    assert llist.l(sym.symbol('inc'), 5) == read_str_first('(inc #_counter 5)')
    assert llist.l(sym.symbol('dec'), 8) == read_str_first('(#_inc dec #_counter 8)')

    assert vec.Vector.empty() == read_str_first('[#_m]')
    assert vec.v(1) == read_str_first('[#_m 1]')
    assert vec.v(1) == read_str_first('[#_m 1 #_2]')
    assert vec.v(1, 2) == read_str_first('[#_m 1 2]')
    assert vec.v(1, 4) == read_str_first('[#_m 1 #_2 4]')
    assert vec.v(1, 4) == read_str_first('[#_m 1 #_2 4 #_5]')

    assert lset.Set.empty() == read_str_first('#{#_m}')
    assert lset.s(1) == read_str_first('#{#_m 1}')
    assert lset.s(1) == read_str_first('#{#_m 1 #_2}')
    assert lset.s(1, 2) == read_str_first('#{#_m 1 2}')
    assert lset.s(1, 4) == read_str_first('#{#_m 1 #_2 4}')
    assert lset.s(1, 4) == read_str_first('#{#_m 1 #_2 4 #_5}')

    assert lmap.Map.empty() == read_str_first('{#_:key}')
    assert lmap.Map.empty() == read_str_first('{#_:key #_"value"}')
    assert lmap.map({kw.keyword('key'): "value"}) == read_str_first(
        '{:key #_"other" "value"}')
    assert lmap.map({kw.keyword('key'): "value"}) == read_str_first(
        '{:key "value" #_"other"}')

    with pytest.raises(reader.SyntaxError):
        read_str_first('{:key #_"value"}')


def test_comment_line():
    assert read_str_first("; I'm a little comment short and stout") is None
    assert read_str_first(";; :kw1\n:kw2") == kw.keyword('kw2')
    assert read_str_first(""";; Comment
        (form :keyword)
        """) == llist.l(sym.symbol('form'), kw.keyword('keyword'))


def test_function_reader_macro():
    assert read_str_first("#()") == llist.l(sym.symbol('fn*'), vec.v(), None)
    assert read_str_first("#(identity %)") == llist.l(
        sym.symbol('fn*'), vec.v(sym.symbol('arg-1')),
        llist.l(sym.symbol('identity'), sym.symbol('arg-1')))
    assert read_str_first("#(identity %1)") == llist.l(
        sym.symbol('fn*'), vec.v(sym.symbol('arg-1')),
        llist.l(sym.symbol('identity'), sym.symbol('arg-1')))
    assert read_str_first("#(identity %& %1)") == llist.l(
        sym.symbol('fn*'),
        vec.v(sym.symbol('arg-1'), sym.symbol('&'), sym.symbol('arg-rest')),
        llist.l(
            sym.symbol('identity'), sym.symbol('arg-rest'),
            sym.symbol('arg-1')))
    assert read_str_first("#(identity %3)") == llist.l(
        sym.symbol('fn*'),
        vec.v(sym.symbol('arg-1'), sym.symbol('arg-2'), sym.symbol('arg-3')),
        llist.l(sym.symbol('identity'), sym.symbol('arg-3')))
    assert read_str_first("#(identity %3 %&)") == llist.l(
        sym.symbol('fn*'),
        vec.v(
            sym.symbol('arg-1'), sym.symbol('arg-2'), sym.symbol('arg-3'),
            sym.symbol('&'), sym.symbol('arg-rest')),
        llist.l(
            sym.symbol('identity'), sym.symbol('arg-3'),
            sym.symbol('arg-rest')))

    with pytest.raises(reader.SyntaxError):
        read_str_first("#(identity #(%1 %2))")

    with pytest.raises(reader.SyntaxError):
        read_str_first("#app/ermagrd [1 2 3]")


def test_deref():
    assert read_str_first("@s") == llist.l(reader._DEREF, sym.symbol('s'))
    assert read_str_first("@ns/s") == llist.l(reader._DEREF, sym.symbol('s', ns='ns'))
    assert read_str_first("@(atom {})") == llist.l(reader._DEREF, llist.l(sym.symbol('atom'), lmap.Map.empty()))


def test_character_literal():
    assert "a" == read_str_first('\\a')
    assert "Ω" == read_str_first('\\Ω')

    assert "Ω" == read_str_first('\\u03A9')

    assert " " == read_str_first('\\space')
    assert "\n" == read_str_first('\\newline')
    assert "\t" == read_str_first('\\tab')
    assert "\b" == read_str_first('\\backspace')
    assert "\f" == read_str_first('\\formfeed')
    assert "\r" == read_str_first('\\return')

    assert vec.v("a") == read_str_first('[\\a]')
    assert vec.v("Ω") == read_str_first('[\\Ω]')

    assert llist.l(sym.symbol("str"), "Ω") == read_str_first('(str \\u03A9)')

    assert vec.v(" ") == read_str_first('[\\space]')
    assert vec.v("\n") == read_str_first('[\\newline]')
    assert vec.v("\t") == read_str_first('[\\tab]')
    assert llist.l(sym.symbol("str"), "\b", "\f", "\r") == read_str_first('(str \\backspace \\formfeed \\return)')

    with pytest.raises(reader.SyntaxError):
        read_str_first('\\u03A9zzz')

    with pytest.raises(reader.SyntaxError):
        read_str_first('\\uFFFFFFFF')

    with pytest.raises(reader.SyntaxError):
        read_str_first('\\blah')


def test_decimal_literal():
    assert langutil.decimal_from_str('3.14') == read_str_first('3.14M')
    assert 3.14 == read_str_first('3.14')

    with pytest.raises(reader.SyntaxError):
        read_str_first('3.14MM')

    with pytest.raises(reader.SyntaxError):
        read_str_first('3.1M4')

    with pytest.raises(reader.SyntaxError):
        read_str_first('3M.14')


def test_fraction_literal():
    assert langutil.fraction(1, 7) == read_str_first('1/7')
    assert langutil.fraction(22, 7) == read_str_first('22/7')

    with pytest.raises(reader.SyntaxError):
        read_str_first('3/7N')

    with pytest.raises(reader.SyntaxError):
        read_str_first('3N/7')

    with pytest.raises(reader.SyntaxError):
        read_str_first('3.3/7')

    with pytest.raises(reader.SyntaxError):
        read_str_first('3/7.4')

    with pytest.raises(reader.SyntaxError):
        read_str_first('3/7/14')


def test_inst_reader_literal():
    assert read_str_first(
        '#inst "2018-01-18T03:26:57.296-00:00"') == langutil.inst_from_str(
        "2018-01-18T03:26:57.296-00:00")

    with pytest.raises(reader.SyntaxError):
        read_str_first('#inst "I am a little teapot short and stout"')


def test_regex_reader_literal():
    assert read_str_first('#"hi"') == langutil.regex_from_str("hi")
    assert read_str_first(r'#"\s"') == langutil.regex_from_str(r"\s")

    with pytest.raises(reader.SyntaxError):
        read_str_first(r'#"\y"')


def test_uuid_reader_literal():
    assert read_str_first('#uuid "4ba98ef0-0620-4966-af61-f0f6c2dbf230"'
                          ) == langutil.uuid_from_str(
        "4ba98ef0-0620-4966-af61-f0f6c2dbf230")

    with pytest.raises(reader.SyntaxError):
        read_str_first('#uuid "I am a little teapot short and stout"')


def test_non_namespaced_tags_reserved():
    with pytest.raises(TypeError):
        read_str_first("#boop :hi", data_readers=lmap.map({kw.keyword("boop"): lambda v: v}))

    with pytest.raises(ValueError):
        read_str_first("#boop :hi", data_readers=lmap.map({sym.symbol("boop"): lambda v: v}))


def test_not_found_tag_error():
    with pytest.raises(reader.SyntaxError):
        read_str_first("#boop :hi")

    with pytest.raises(reader.SyntaxError):
        read_str_first("#ns/boop :hi")
