import io
import pytest
import basilisp.lang.keyword as kw
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.set as lset
import basilisp.lang.symbol as sym
import basilisp.lang.util as langutil
import basilisp.lang.vector as vec
import basilisp.reader as reader


def read_str_first(s):
    """Read the first form from the input string. If no form
    is found, return None."""
    form = reader.read_str(s)
    return None if form is None else form.first


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
    assert read_str_first("1") == 1
    assert read_str_first("100") == 100
    assert read_str_first("99927273") == 99927273
    assert read_str_first("0") == 0
    assert read_str_first("-1") == -1
    assert read_str_first("-538282") == -538282


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
    assert read_str_first(":kw") == kw.keyword("kw")
    assert read_str_first(":kebab-kw") == kw.keyword("kebab-kw")
    assert read_str_first(":underscore_kw") == kw.keyword("underscore_kw")
    assert read_str_first(":kw?") == kw.keyword("kw?")
    assert read_str_first(":+") == kw.keyword("+")
    assert read_str_first(":?") == kw.keyword("?")
    assert read_str_first(":=") == kw.keyword("=")
    assert read_str_first(":!") == kw.keyword("!")
    assert read_str_first(":-") == kw.keyword("-")
    assert read_str_first(":*") == kw.keyword("*")
    assert read_str_first(":*muffs*") == kw.keyword("*muffs*")
    assert read_str_first(":yay!") == kw.keyword("yay!")

    assert read_str_first(":ns/kw") == kw.keyword("kw", ns="ns")
    assert read_str_first(":qualified.ns/kw") == kw.keyword(
        "kw", ns="qualified.ns")
    assert read_str_first(":really.qualified.ns/kw") == kw.keyword(
        "kw", ns="really.qualified.ns")

    with pytest.raises(reader.SyntaxError):
        read_str_first(":ns//kw")

    with pytest.raises(reader.SyntaxError):
        read_str_first(":/kw")

    with pytest.raises(reader.SyntaxError):
        read_str_first(":dotted.kw")


def test_literals():
    assert read_str_first("nil") is None
    assert read_str_first("true") is True
    assert read_str_first("false") is False


def test_symbol():
    assert read_str_first("sym") == sym.symbol("sym")
    assert read_str_first("kebab-sym") == sym.symbol("kebab-sym")
    assert read_str_first("underscore_sym") == sym.symbol("underscore_sym")
    assert read_str_first("sym?") == sym.symbol("sym?")
    assert read_str_first("+") == sym.symbol("+")
    assert read_str_first("?") == sym.symbol("?")
    assert read_str_first("=") == sym.symbol("=")
    assert read_str_first("!") == sym.symbol("!")
    assert read_str_first("-") == sym.symbol("-")
    assert read_str_first("*") == sym.symbol("*")
    assert read_str_first("*muffs*") == sym.symbol("*muffs*")
    assert read_str_first("yay!") == sym.symbol("yay!")

    assert read_str_first("ns/sym") == sym.symbol("sym", ns="ns")
    assert read_str_first("qualified.ns/sym") == sym.symbol(
        "sym", ns="qualified.ns")
    assert read_str_first("really.qualified.ns/sym") == sym.symbol(
        "sym", ns="really.qualified.ns")

    with pytest.raises(reader.SyntaxError):
        read_str_first("ns//sym")

    with pytest.raises(reader.SyntaxError):
        read_str_first("/sym")


def test_str():
    assert read_str_first('""') == ''
    assert read_str_first('"Regular string"') == 'Regular string'
    assert read_str_first(
        '"String with \'inner string\'"') == "String with 'inner string'"
    assert read_str_first(
        r'"String with \"inner string\""') == 'String with "inner string"'

    with pytest.raises(reader.SyntaxError):
        read_str_first('"Start of a string')


def test_whitespace():
    assert read_str_first('') is None
    assert read_str_first(' ') is None
    assert read_str_first('\t') is None


def test_vector():
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
    assert read_str_first('()') == llist.list([])
    assert read_str_first('(func-with-no-args)') == llist.l(
        sym.symbol("func-with-no-args"))
    assert read_str_first('(str/join "one string" " and another")') == llist.l(
        sym.symbol("join", ns='str'), "one string", " and another")
    assert read_str_first('(map inc [1 2 3])') == llist.l(
        sym.symbol("map"), sym.symbol("inc"), vec.v(1, 2, 3))
    assert read_str_first('(- -1 2)') == llist.l(sym.symbol("-"), -1, 2)


def test_set():
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


def test_meta():
    s = read_str_first("^str s")
    assert s == sym.symbol('s')
    assert s.meta == lmap.map({kw.keyword('tag'): sym.symbol('str')})

    s = read_str_first("^:dynamic *ns*")
    assert s == sym.symbol('*ns*')
    assert s.meta == lmap.map({kw.keyword('dynamic'): True})

    s = read_str_first('^{:doc "If true, assert."} *assert*')
    assert s == sym.symbol('*assert*')
    assert s.meta == lmap.map({kw.keyword('doc'): "If true, assert."})

    v = read_str_first("^:has-meta [:a]")
    assert v == vec.v(kw.keyword('a'))
    assert v.meta == lmap.map({kw.keyword('has-meta'): True})

    l = read_str_first('^:has-meta (:a)')
    assert l == llist.l(kw.keyword('a'))
    assert l.meta == lmap.map({kw.keyword('has-meta'): True})

    m = read_str_first('^:has-meta {:key "val"}')
    assert m == lmap.map({kw.keyword('key'): "val"})
    assert m.meta == lmap.map({kw.keyword('has-meta'): True})

    t = read_str_first('^:has-meta #{:a}')
    assert t == lset.s(kw.keyword('a'))
    assert t.meta == lmap.map({kw.keyword('has-meta'): True})

    s = read_str_first('^:dynamic ^{:doc "If true, assert."} *assert*')
    assert s == sym.symbol('*assert*')
    assert s.meta == lmap.map({
        kw.keyword('dynamic'): True,
        kw.keyword('doc'): "If true, assert."
    })

    s = read_str_first('^{:always true} ^{:always false} *assert*')
    assert s == sym.symbol('*assert*')
    assert s.meta == lmap.map({kw.keyword('always'): True})


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
    assert read_str_first('#_       (a list)') is None
    assert read_str_first('#_:keyword') is None
    assert read_str_first('#_:kw1 :kw2') == kw.keyword('kw2')


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


def test_regex_reader_literal():
    assert read_str_first('#"hi"') == langutil.regex_from_str("hi")
    assert read_str_first('#"\s"') == langutil.regex_from_str(r"\s")

    with pytest.raises(reader.SyntaxError):
        read_str_first('#"\y"')


def test_inst_reader_literal():
    assert read_str_first(
        '#inst "2018-01-18T03:26:57.296-00:00"') == langutil.inst_from_str(
            "2018-01-18T03:26:57.296-00:00")

    with pytest.raises(reader.SyntaxError):
        read_str_first('#inst "I am a little teapot short and stout"')


def test_uuid_reader_literal():
    assert read_str_first('#uuid "4ba98ef0-0620-4966-af61-f0f6c2dbf230"'
                          ) == langutil.uuid_from_str(
                              "4ba98ef0-0620-4966-af61-f0f6c2dbf230")

    with pytest.raises(reader.SyntaxError):
        read_str_first('#uuid "I am a little teapot short and stout"')


def test_non_namespaced_tags_reserved():
    with pytest.raises(reader.SyntaxError):
        read_str_first("#boop :hi")
