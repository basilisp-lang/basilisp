import io
import pytest
import apylisp.lang.float as lfloat
import apylisp.lang.integer as integer
import apylisp.lang.keyword as kw
import apylisp.lang.list as llist
import apylisp.lang.map as lmap
import apylisp.lang.set as lset
import apylisp.lang.string as string
import apylisp.lang.symbol as sym
import apylisp.lang.vector as vec
import apylisp.reader as reader


def test_stream_reader():
    sreader = reader.StreamReader(io.StringIO("12345"))
    assert sreader.peek() == "1"
    assert sreader.next_token() == "2"
    assert sreader.peek() == "2"
    sreader.pushback()
    assert sreader.peek() == "1"
    assert sreader.next_token() == "2"
    assert sreader.next_token() == "3"
    assert sreader.next_token() == "4"
    assert sreader.next_token() == "5"
    assert sreader.next_token() == ""


def test_int():
    assert reader.read_str("1") == integer.integer("1")
    assert reader.read_str("100") == integer.integer("100")
    assert reader.read_str("99927273") == integer.integer("99927273")
    assert reader.read_str("0") == integer.integer("0")
    assert reader.read_str("-1") == integer.integer("-1")
    assert reader.read_str("-538282") == integer.integer("-538282")


def test_float():
    assert reader.read_str("0.0") == lfloat.lfloat("0.0")
    assert reader.read_str("0.09387372") == lfloat.lfloat("0.09387372")
    assert reader.read_str("1.0") == lfloat.lfloat("1.0")
    assert reader.read_str("1.332") == lfloat.lfloat("1.332")
    assert reader.read_str("-1.332") == lfloat.lfloat("-1.332")
    assert reader.read_str("-1.0") == lfloat.lfloat("-1.0")
    assert reader.read_str("-0.332") == lfloat.lfloat("-0.332")

    with pytest.raises(reader.SyntaxError):
        reader.read_str("0..11")

    with pytest.raises(reader.SyntaxError):
        reader.read_str("0.111.9")


def test_kw():
    assert reader.read_str(":kw") == kw.keyword("kw")
    assert reader.read_str(":kebab-kw") == kw.keyword("kebab-kw")
    assert reader.read_str(":underscore_kw") == kw.keyword("underscore_kw")
    assert reader.read_str(":kw?") == kw.keyword("kw?")
    assert reader.read_str(":+") == kw.keyword("+")
    assert reader.read_str(":?") == kw.keyword("?")
    assert reader.read_str(":=") == kw.keyword("=")
    assert reader.read_str(":!") == kw.keyword("!")
    assert reader.read_str(":-") == kw.keyword("-")
    assert reader.read_str(":*") == kw.keyword("*")
    assert reader.read_str(":*muffs*") == kw.keyword("*muffs*")
    assert reader.read_str(":yay!") == kw.keyword("yay!")

    assert reader.read_str(":ns/kw") == kw.keyword("kw", ns="ns")
    assert reader.read_str(":qualified.ns/kw") == kw.keyword(
        "kw", ns="qualified.ns")
    assert reader.read_str(":really.qualified.ns/kw") == kw.keyword(
        "kw", ns="really.qualified.ns")

    with pytest.raises(reader.SyntaxError):
        reader.read_str(":ns//kw")

    with pytest.raises(reader.SyntaxError):
        reader.read_str(":/kw")

    with pytest.raises(reader.SyntaxError):
        reader.read_str(":dotted.kw")


def test_literals():
    assert reader.read_str("nil") == None
    assert reader.read_str("true") == True
    assert reader.read_str("false") == False


def test_symbol():
    assert reader.read_str("sym") == sym.symbol("sym")
    assert reader.read_str("kebab-sym") == sym.symbol("kebab-sym")
    assert reader.read_str("underscore_sym") == sym.symbol("underscore_sym")
    assert reader.read_str("sym?") == sym.symbol("sym?")
    assert reader.read_str("+") == sym.symbol("+")
    assert reader.read_str("?") == sym.symbol("?")
    assert reader.read_str("=") == sym.symbol("=")
    assert reader.read_str("!") == sym.symbol("!")
    assert reader.read_str("-") == sym.symbol("-")
    assert reader.read_str("*") == sym.symbol("*")
    assert reader.read_str("*muffs*") == sym.symbol("*muffs*")
    assert reader.read_str("yay!") == sym.symbol("yay!")

    assert reader.read_str("ns/sym") == sym.symbol("sym", ns="ns")
    assert reader.read_str("qualified.ns/sym") == sym.symbol(
        "sym", ns="qualified.ns")
    assert reader.read_str("really.qualified.ns/sym") == sym.symbol(
        "sym", ns="really.qualified.ns")

    with pytest.raises(reader.SyntaxError):
        reader.read_str("ns//sym")

    with pytest.raises(reader.SyntaxError):
        reader.read_str("/sym")

    with pytest.raises(reader.SyntaxError):
        reader.read_str("dotted.symbol")


def test_str():
    assert reader.read_str('""') == string.string('')
    assert reader.read_str('"Regular string"') == string.string(
        'Regular string')
    assert reader.read_str('"String with \'inner string\'"') == string.string(
        "String with 'inner string'")
    assert reader.read_str('"String with \\"inner string\\""'
                           ) == string.string('String with "inner string"')

    with pytest.raises(reader.SyntaxError):
        reader.read_str('"Start of a string')


def test_whitespace():
    assert reader.read_str('') == None
    assert reader.read_str(' ') == None
    assert reader.read_str('\t') == None


def test_vector():
    assert reader.read_str('[]') == vec.vector([])
    assert reader.read_str('[:a]') == vec.v(kw.keyword("a"))
    assert reader.read_str('[:a :b]') == vec.v(
        kw.keyword("a"), kw.keyword("b"))
    assert reader.read_str('[:a :b :c]') == vec.v(
        kw.keyword("a"), kw.keyword("b"), kw.keyword("c"))
    assert reader.read_str('[:a, :b]') == vec.v(
        kw.keyword("a"), kw.keyword("b"))
    assert reader.read_str('[:a :b, :c]') == vec.v(
        kw.keyword("a"), kw.keyword("b"), kw.keyword("c"))
    assert reader.read_str('[:a, :b, :c]') == vec.v(
        kw.keyword("a"), kw.keyword("b"), kw.keyword("c"))
    assert reader.read_str('[1 :a "string"]') == vec.v(
        integer.integer("1"), kw.keyword("a"), string.string("string"))
    assert reader.read_str('[1, :a, "string"]') == vec.v(
        integer.integer("1"), kw.keyword("a"), string.string("string"))
    assert reader.read_str('[1.4, :a, "string"]') == vec.v(
        lfloat.lfloat("1.4"), kw.keyword("a"), string.string("string"))


def test_list():
    assert reader.read_str('()') == llist.list([])
    assert reader.read_str('(func-with-no-args)') == llist.l(
        sym.symbol("func-with-no-args"))
    assert reader.read_str(
        '(str/join "one string" " and another")') == llist.l(
            sym.symbol("join", ns='str'),
            string.string("one string"), string.string(" and another"))
    assert reader.read_str('(map inc [1 2 3])') == llist.l(
        sym.symbol("map"),
        sym.symbol("inc"),
        vec.v(
            integer.integer("1"), integer.integer("2"), integer.integer("3")))
    assert reader.read_str('(- -1 2)') == llist.l(
        sym.symbol("-"), integer.integer("-1"), integer.integer("2"))


def test_set():
    assert reader.read_str('#{}') == lset.set([])
    assert reader.read_str('#{:a}') == lset.s(kw.keyword("a"))
    assert reader.read_str('#{:a :b}') == lset.s(
        kw.keyword("a"), kw.keyword("b"))
    assert reader.read_str('#{:a :b :c}') == lset.s(
        kw.keyword("a"), kw.keyword("b"), kw.keyword("c"))
    assert reader.read_str('#{:a 1 "some string"}') == lset.s(
        kw.keyword("a"), integer.integer("1"), string.string("some string"))

    with pytest.raises(reader.SyntaxError):
        reader.read_str('#{:a :b :b}')


def test_map():
    assert reader.read_str('{}') == lmap.map({})
    assert reader.read_str('{:a 1}') == lmap.map({
        kw.keyword('a'):
        integer.integer("1")
    })
    assert reader.read_str('{:a 1 :b "string"}') == lmap.map({
        kw.keyword('a'):
        integer.integer("1"),
        kw.keyword('b'):
        string.string("string")
    })

    with pytest.raises(reader.SyntaxError):
        reader.read_str('{:a 1 :b 2 :a 3}')

    with pytest.raises(reader.SyntaxError):
        reader.read_str('{:a}')

    with pytest.raises(reader.SyntaxError):
        reader.read_str('{:a 1 :b}')
