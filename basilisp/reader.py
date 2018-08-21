import collections
import contextlib
import functools
import io
import re
import uuid
from datetime import datetime
from typing import (Deque, List, Tuple, Optional, Collection, Callable, Any, Union, MutableMapping, Pattern, Iterable,
                    TypeVar, cast, Dict)

from functional import seq

import basilisp.lang.keyword as keyword
import basilisp.lang.list as llist
import basilisp.lang.map as lmap
import basilisp.lang.meta as lmeta
import basilisp.lang.set as lset
import basilisp.lang.symbol as symbol
import basilisp.lang.util as langutil
import basilisp.lang.vector as vector
import basilisp.walker as walk
from basilisp.lang.typing import LispForm, IterableLispForm
from basilisp.util import Maybe

ns_name_chars = re.compile(r'\w|-|\+|\*|\?|/|\=|\\|!|&|%|>|<')
begin_num_chars = re.compile('[0-9\-]')
num_chars = re.compile('[0-9]')
whitespace_chars = re.compile('[\s,]')
newline_chars = re.compile('(\r\n|\r|\n)')
fn_macro_args = re.compile('(%)(&|[0-9])?')

GenSymEnvironment = Dict[str, symbol.Symbol]
Resolver = Callable[[symbol.Symbol], symbol.Symbol]
LispReaderFn = Callable[["ReaderContext"], LispForm]
W = TypeVar('W', bound=LispReaderFn)

READER_LINE_KW = keyword.keyword('line', ns='basilisp.reader')
READER_COL_KW = keyword.keyword('col', ns='basilisp.reader')

_AMPERSAND = symbol.symbol("&")
_FN = symbol.symbol("fn*")
_INTEROP_CALL = symbol.symbol(".")
_INTEROP_PROP = symbol.symbol(".-")
_QUOTE = symbol.symbol("quote")
_VAR = symbol.symbol("var")

_APPLY = symbol.symbol('apply', ns='basilisp.core')
_CONCAT = symbol.symbol('concat', ns='basilisp.core')
_HASH_MAP = symbol.symbol('hash-map', ns='basilisp.core')
_HASH_SET = symbol.symbol('hash-set', ns='basilisp.core')
_LIST = symbol.symbol('list', ns='basilisp.core')
_SEQ = symbol.symbol('seq', ns='basilisp.core')
_UNQUOTE = symbol.symbol('unquote', ns='basilisp.core')
_UNQUOTE_SPLICING = symbol.symbol('unquote-splicing', ns='basilisp.core')
_VECTOR = symbol.symbol('vector', ns='basilisp.core')


class SyntaxError(Exception):  # pylint:disable=redefined-builtin
    pass


class StreamReader:
    """A simple stream reader with n-character lookahead."""
    DEFAULT_INDEX = -2

    __slots__ = ('_stream', '_pushback_depth', '_idx', '_buffer', '_line', '_col')

    def __init__(self, stream: io.TextIOBase, pushback_depth: int = 5) -> None:
        self._stream = stream
        self._pushback_depth = pushback_depth
        self._idx = -2
        init_buffer = [self._stream.read(1), self._stream.read(1)]
        self._buffer = collections.deque(init_buffer, pushback_depth)
        self._line = collections.deque([1], pushback_depth)
        self._col = collections.deque([1], pushback_depth)

        for c in init_buffer[1:]:
            self._update_loc(c)

    @property
    def col(self):
        return self._col[self._idx]

    @property
    def line(self):
        return self._line[self._idx]

    @property
    def loc(self):
        return self.line, self.col

    def _update_loc(self, c):
        """Update the internal line and column buffers after a new character
        is added.

        The column number is set to 0, so the first character on the next line
        is column number 1."""
        if newline_chars.match(c):
            self._col.append(0)
            self._line.append(self._line[-1] + 1)
        else:
            self._col.append(self._col[-1] + 1)
            self._line.append(self._line[-1])

    def peek(self) -> str:
        """Peek at the next character in the stream."""
        return self._buffer[self._idx]

    def pushback(self) -> None:
        """Push one character back onto the stream, allowing it to be
        read again."""
        if abs(self._idx - 1) > self._pushback_depth:
            raise IndexError("Exceeded pushback depth")
        self._idx -= 1

    def advance(self) -> str:
        """Advance the current token pointer by one and return the
        previous token value from before advancing the counter."""
        cur = self.peek()
        self.next_token()
        return cur

    def next_token(self) -> str:
        """Advance the stream forward by one character and return the
        next token in the stream."""
        if self._idx < StreamReader.DEFAULT_INDEX:
            self._idx += 1
        else:
            c = self._stream.read(1)
            self._update_loc(c)
            self._buffer.append(c)
        return self.peek()


class ReaderContext:
    __slots__ = ('_reader', '_resolve', '_in_anon_fn', '_syntax_quoted', '_gensym_env')

    def __init__(self, reader: StreamReader, resolver: Resolver = None) -> None:
        self._reader = reader
        self._resolve = Maybe(resolver).or_else_get(lambda x: x)
        self._in_anon_fn: Deque[bool] = collections.deque([])
        self._syntax_quoted: Deque[bool] = collections.deque([])
        self._gensym_env: Deque[GenSymEnvironment] = collections.deque([])

    @property
    def reader(self) -> StreamReader:
        return self._reader

    def resolve(self, sym: symbol.Symbol) -> symbol.Symbol:
        return self._resolve(sym)

    @contextlib.contextmanager
    def in_anon_fn(self):
        self._in_anon_fn.append(True)
        yield
        self._in_anon_fn.pop()

    @property
    def is_in_anon_fn(self) -> bool:
        try:
            return self._in_anon_fn[-1] is True
        except IndexError:
            return False

    @property
    def gensym_env(self) -> GenSymEnvironment:
        return self._gensym_env[-1]

    @contextlib.contextmanager
    def syntax_quoted(self):
        self._syntax_quoted.append(True)
        self._gensym_env.append({})
        yield
        self._gensym_env.pop()
        self._syntax_quoted.pop()

    @contextlib.contextmanager
    def unquoted(self):
        self._syntax_quoted.append(False)
        yield
        self._syntax_quoted.pop()

    @property
    def is_syntax_quoted(self) -> bool:
        try:
            return self._syntax_quoted[-1] is True
        except IndexError:
            return False


__EOF = 'EOF'


def _with_loc(f: W) -> W:
    """Wrap a reader function in a decorator to supply line and column
    information along with relevant forms."""

    @functools.wraps(f)
    def with_lineno_and_col(ctx):
        meta = lmap.map({READER_LINE_KW: ctx.reader.line, READER_COL_KW: ctx.reader.col})
        v = f(ctx)
        try:
            return v.with_meta(meta)  # type: ignore
        except AttributeError:
            return v

    return cast(W, with_lineno_and_col)


def _read_namespaced(ctx: ReaderContext, allowed_suffix: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Read a namespaced token from the input stream."""
    ns: List[str] = []
    name: List[str] = []
    reader = ctx.reader
    has_ns = False
    while True:
        token = reader.peek()
        if token == '/':
            reader.next_token()
            if has_ns:
                raise SyntaxError("Found '/'; expected word character")
            elif len(name) == 0:
                name.append('/')
            else:
                if '/' in name:
                    raise SyntaxError("Found '/' after '/'")
                has_ns = True
                ns = name
                name = []
        elif ns_name_chars.match(token):
            reader.next_token()
            name.append(token)
        elif allowed_suffix is not None and token == allowed_suffix:
            reader.next_token()
            name.append(token)
        elif not has_ns and token == '.':
            reader.next_token()
            name.append(token)
        else:
            break

    ns_str = None if not has_ns else ''.join(ns)
    name_str = ''.join(name)

    # A small exception for the symbol '/ used for division
    if ns_str is None:
        if '/' in name_str and name_str != '/':
            raise SyntaxError("'/' character disallowed in names")

    assert ns_str is None or len(ns_str) > 0

    return ns_str, name_str


def _read_coll(ctx: ReaderContext, f: Callable[[Collection[Any]], Union[
        llist.List, lset.Set, vector.Vector]], end_token: str, coll_name: str):
    """Read a collection from the input stream and create the
    collection using f."""
    coll: List = []
    reader = ctx.reader
    while True:
        token = reader.peek()
        if token == '':
            raise SyntaxError(f"Unexpected EOF in {coll_name}")
        if whitespace_chars.match(token):
            reader.advance()
            continue
        if token == end_token:
            reader.next_token()
            return f(coll)
        elem = _read_next(ctx)
        coll.append(elem)


def _consume_whitespace(reader: StreamReader) -> None:
    token = reader.peek()
    while whitespace_chars.match(token):
        token = reader.advance()


def _read_interop(ctx: ReaderContext, end_token: str) -> llist.List:
    """Read a Python interop call or property access.

    The instance member access syntax permits the following iterations:

      (. instance member & args)
      (.member instance & args)

      (. instance -property)
      (.-property instance)

    This function always dynamically rewrites everything into one of two
    canonical formats:

      (. instance member & args)
      (.- instance property)

    By using just two canonical forms, it will be much easier to parse
    and compile Python interop code."""
    reader = ctx.reader
    start = reader.advance()
    assert start == '.'
    seq: List[LispForm] = []

    token = reader.peek()
    if whitespace_chars.match(token):
        instance = _read_next(ctx)
        member = _read_next(ctx)
        if not isinstance(member, symbol.Symbol):
            raise SyntaxError(f"Expected Symbol; found {type(member)}")
        is_property = member.name.startswith('-')
        if is_property:
            seq.append(_INTEROP_PROP)
            member = symbol.symbol(member.name[1:])
        else:
            seq.append(_INTEROP_CALL)
        seq.append(instance)
        seq.append(member)
    elif token == '-':
        reader.advance()
        seq.append(_INTEROP_PROP)
        if whitespace_chars.match(reader.peek()):
            raise SyntaxError(f"Expected Symbol; found whitespace")
        member = _read_next(ctx)
        if not isinstance(member, symbol.Symbol):
            raise SyntaxError(f"Expected Symbol; found {type(member)}")
        instance = _read_next(ctx)
        seq.append(instance)
        seq.append(member)
    else:
        assert not whitespace_chars.match(token)
        seq.append(_INTEROP_CALL)
        member = _read_next(ctx)
        instance = _read_next(ctx)
        if not isinstance(member, symbol.Symbol):
            raise SyntaxError(f"Expected Symbol; found {type(member)}")
        seq.append(instance)
        seq.append(member)

    while True:
        token = reader.peek()
        if token == '':
            raise SyntaxError(f"Unexpected EOF in list")
        if token == end_token:
            reader.next_token()
            return llist.list(seq)
        elem = _read_next(ctx)
        seq.append(elem)


@_with_loc
def _read_list(ctx: ReaderContext) -> llist.List:
    """Read a list element from the input stream."""
    start = ctx.reader.advance()
    assert start == '('
    if ctx.reader.peek() == '.':
        return _read_interop(ctx, ')')
    return _read_coll(ctx, llist.list, ')', 'list')


@_with_loc
def _read_vector(ctx: ReaderContext) -> vector.Vector:
    """Read a vector element from the input stream."""
    start = ctx.reader.advance()
    assert start == '['
    return _read_coll(ctx, vector.vector, ']', 'vector')


@_with_loc
def _read_set(ctx: ReaderContext) -> lset.Set:
    """Return a set from the input stream."""
    start = ctx.reader.advance()
    assert start == '{'

    def set_if_valid(s: Collection) -> lset.Set:
        if len(s) != len(set(s)):
            raise SyntaxError("Duplicated values in set")
        return lset.set(s)

    return _read_coll(ctx, set_if_valid, '}', 'set')


@_with_loc
def _read_map(ctx: ReaderContext) -> lmap.Map:
    """Return a map from the input stream."""
    reader = ctx.reader
    start = reader.advance()
    assert start == '{'
    d: MutableMapping[Any, Any] = {}
    while True:
        if reader.peek() == '}':
            reader.next_token()
            break
        k = _read_next(ctx)
        if reader.peek() == '}':
            raise SyntaxError("Unexpected token '}'; expected map value")
        v = _read_next(ctx)
        if k in d:
            raise SyntaxError("Duplicate key '{}' in map literal".format(k))
        d[k] = v

    return lmap.map(d)


# Due to some ambiguities that arise in parsing symbols, numbers, and the
# special keywords `true`, `false`, and `nil`, we have to have a looser
# type defined for the return from these reader functions.
MaybeSymbol = Union[bool, None, symbol.Symbol]
MaybeNumber = Union[float, int, MaybeSymbol]


def _read_num(ctx: ReaderContext) -> MaybeNumber:
    """Return a numeric (integer or float) from the input stream."""
    chars: List[str] = []
    reader = ctx.reader
    is_float = False
    while True:
        token = reader.peek()
        if token == '-':
            following_token = reader.next_token()
            if not begin_num_chars.match(following_token):
                reader.pushback()
                try:
                    for _ in chars:
                        reader.pushback()
                except IndexError:
                    raise SyntaxError("Requested to pushback too many characters onto StreamReader")
                return _read_sym(ctx)
            chars.append(token)
            continue
        elif token == '.':
            if is_float:
                raise SyntaxError(
                    "Found extra '.' in float; expected decimal portion")
            is_float = True
        elif not num_chars.match(token):
            break
        reader.next_token()
        chars.append(token)

    if len(chars) == 0:
        raise SyntaxError("Expected integer or float")

    s = ''.join(chars)
    return float(s) if is_float else int(s)


def _read_str(ctx: ReaderContext) -> str:
    """Return a string from the input stream."""
    s: List[str] = []
    reader = ctx.reader
    while True:
        prev = reader.peek()
        token = reader.next_token()
        if token == '':
            raise SyntaxError("Unexpected EOF in string")
        if token == "\\":
            token = reader.next_token()
            if token == '"':
                s.append('"')
                continue
            else:
                s.append("\\")
        if token == '"' and not prev == "\\":
            reader.next_token()
            return ''.join(s)
        s.append(token)


@_with_loc
def _read_sym(ctx: ReaderContext) -> MaybeSymbol:
    """Return a symbol from the input stream.

    If a symbol appears in a syntax quoted form, the reader will attempt
    to resolve the symbol using the resolver in the ReaderContext `ctx`.
    The resolver will look into the current namespace for an alias or
    namespace matching the symbol's namespace."""
    ns, name = _read_namespaced(ctx, allowed_suffix='#')
    if not ctx.is_syntax_quoted and name.endswith('#'):
        raise SyntaxError("Gensym may not appear outside syntax quote")
    if ns is None:
        if name == 'nil':
            return None
        elif name == 'true':
            return True
        elif name == 'false':
            return False
    if ctx.is_syntax_quoted:
        return ctx.resolve(symbol.symbol(name, ns))
    return symbol.symbol(name, ns=ns)


def _read_kw(ctx: ReaderContext) -> keyword.Keyword:
    """Return a keyword from the input stream."""
    start = ctx.reader.advance()
    assert start == ':'
    ns, name = _read_namespaced(ctx)
    if '.' in name:
        raise SyntaxError("Found '.' in keyword name")
    return keyword.keyword(name, ns=ns)


def _read_meta(ctx: ReaderContext) -> lmeta.Meta:
    """Read metadata and apply that to the next object in the
    input stream."""
    start = ctx.reader.advance()
    assert start == '^'
    meta = _read_next(ctx)

    meta_map = None
    if isinstance(meta, symbol.Symbol):
        meta_map = lmap.map({keyword.keyword('tag'): meta})
    elif isinstance(meta, keyword.Keyword):
        meta_map = lmap.map({meta: True})
    elif isinstance(meta, lmap.Map):
        meta_map = meta
    else:
        raise SyntaxError(
            f"Expected symbol, keyword, or map for metadata, not {type(meta)}")

    obj_with_meta = _read_next(ctx)
    try:
        return obj_with_meta.with_meta(meta_map)  # type: ignore
    except AttributeError:
        raise SyntaxError(
            f"Can not attach metadata to object of type {type(obj_with_meta)}")


def _read_function(ctx: ReaderContext) -> llist.List:
    """Read a function reader macro from the input stream."""
    if ctx.is_in_anon_fn:
        raise SyntaxError(f"Nested #() definitions not allowed")

    with ctx.in_anon_fn():
        form = _read_list(ctx)
    arg_set = set()

    def arg_suffix(arg_num):
        if arg_num is None:
            return '1'
        elif arg_num == '&':
            return 'rest'
        else:
            return arg_num

    def sym_replacement(arg_num):
        suffix = arg_suffix(arg_num)
        return symbol.symbol(f'arg-{suffix}')

    def identify_and_replace(f):
        if isinstance(f, symbol.Symbol):
            if f.ns is None:
                match = fn_macro_args.match(f.name)
                if match is not None:
                    arg_num = match.group(2)
                    suffix = arg_suffix(arg_num)
                    arg_set.add(suffix)
                    return sym_replacement(arg_num)
        return f

    body = walk.postwalk(identify_and_replace, form) if len(form) > 0 else None

    arg_list: List[symbol.Symbol] = []
    numbered_args = sorted(map(int, filter(lambda k: k != 'rest', arg_set)))
    if len(numbered_args) > 0:
        max_arg = max(numbered_args)
        arg_list = [sym_replacement(str(i)) for i in range(1, max_arg + 1)]
        if 'rest' in arg_set:
            arg_list.append(_AMPERSAND)
            arg_list.append(sym_replacement('rest'))

    return llist.l(_FN, vector.vector(arg_list), body)


def _read_quoted(ctx: ReaderContext) -> llist.List:
    """Read a quoted form from the input stream."""
    start = ctx.reader.advance()
    assert start == "'"
    next_form = _read_next(ctx)
    return llist.l(_QUOTE, next_form)


def _is_unquote(form: LispForm) -> bool:
    """Return True if this form is unquote."""
    try:
        return form.first == _UNQUOTE  # type: ignore
    except AttributeError:
        return False


def _is_unquote_splicing(form: LispForm) -> bool:
    """Return True if this form is unquote-splicing."""
    try:
        return form.first == _UNQUOTE_SPLICING  # type: ignore
    except AttributeError:
        return False


def _expand_syntax_quote(ctx: ReaderContext, form: IterableLispForm) -> Iterable[LispForm]:
    """Expand syntax quoted forms to handle unquoting and unquote-splicing.

    The unquoted form (unquote x) becomes:
        (list x)

    The unquote-spliced form (unquote-splicing x) becomes
        x

    All other forms are recursively processed as by _process_syntax_quoted_form
    and are returned as:
        (list form)"""
    expanded = []

    for elem in form:
        if _is_unquote(elem):
            expanded.append(llist.l(_LIST, elem[1]))
        elif _is_unquote_splicing(elem):
            expanded.append(elem[1])
        else:
            expanded.append(llist.l(_LIST, _process_syntax_quoted_form(ctx, elem)))

    return expanded


def _process_syntax_quoted_form(ctx: ReaderContext, form: LispForm) -> LispForm:
    """Post-process syntax quoted forms to generate forms that can be assembled
    into the correct types at runtime.

    Lists are turned into:
        (basilisp.core/seq
         (basilisp.core/concat [& rest]))

    Vectors are turned into:
        (basilisp.core/apply
         basilisp.core/vector
         (basilisp.core.concat/ [& rest]))

    Sets are turned into:
        (basilisp.core/apply
         basilisp.core/hash-set
         (basilisp.core.concat/ [& rest]))

    Maps are turned into:
        (basilisp.core/apply
         basilisp.core/hash-map
         (basilisp.core.concat/ [& rest]))

    The child forms (called rest above) are processed by _expand_syntax_quote.

    All other forms are passed through without modification."""
    lconcat = lambda v: llist.list(v).cons(_CONCAT)
    if _is_unquote(form):
        return form[1]  # type: ignore
    elif _is_unquote_splicing(form):
        raise SyntaxError("Cannot splice outside collection")
    elif isinstance(form, llist.List):
        return llist.l(_SEQ, lconcat(_expand_syntax_quote(ctx, form)))
    elif isinstance(form, vector.Vector):
        return llist.l(_APPLY, _VECTOR, lconcat(_expand_syntax_quote(ctx, form)))
    elif isinstance(form, lset.Set):
        return llist.l(_APPLY, _HASH_SET, lconcat(_expand_syntax_quote(ctx, form)))
    elif isinstance(form, lmap.Map):
        flat_kvs = seq(form.items()).flatten().to_list()
        return llist.l(_APPLY, _HASH_MAP, lconcat(_expand_syntax_quote(ctx, flat_kvs)))
    elif isinstance(form, symbol.Symbol):
        if form.ns is None and form.name.endswith("#"):
            try:
                return llist.l(_QUOTE, ctx.gensym_env[form.name])
            except KeyError:
                genned = symbol.symbol(langutil.genname(form.name[:-1]))
                ctx.gensym_env[form.name] = genned
                return llist.l(_QUOTE, genned)
        return llist.l(_QUOTE, form)
    else:
        return form


def _read_syntax_quoted(ctx: ReaderContext) -> LispForm:
    """Read a syntax-quote and set the syntax-quoting state in the reader."""
    start = ctx.reader.advance()
    assert start == "`"

    with ctx.syntax_quoted():
        return _process_syntax_quoted_form(ctx, _read_next(ctx))


def _read_unquote(ctx: ReaderContext) -> LispForm:
    """Read an unquoted form and handle any special logic of unquoting.

    Unquoted forms can take two, well... forms:

      `~form` is read as `(unquote form)` and any nested forms are read
      literally and passed along to the compiler untouched.

      `~@form` is read as `(unquote-splicing form)` which tells the compiler
      to splice in the contents of a sequential form such as a list or
      vector into the final compiled form. This helps macro writers create
      longer forms such as function calls, function bodies, or data structures
      with the contents of another collection they have."""
    start = ctx.reader.advance()
    assert start == "~"

    with ctx.unquoted():
        next_char = ctx.reader.peek()
        if next_char == '@':
            ctx.reader.advance()
            next_form = _read_next(ctx)
            return llist.l(_UNQUOTE_SPLICING, next_form)
        else:
            next_form = _read_next(ctx)
            return llist.l(_UNQUOTE, next_form)


def _read_regex(ctx: ReaderContext) -> Pattern:
    """Read a regex reader macro from the input stream."""
    s = _read_str(ctx)
    try:
        return langutil.regex_from_str(s)
    except re.error:
        raise SyntaxError(f"Unrecognized regex pattern syntax: {s}")


def _inst_from_str(inst_str: str) -> datetime:
    try:
        return langutil.inst_from_str(inst_str)
    except (ValueError, OverflowError):
        raise SyntaxError(f"Unrecognized date/time syntax: {inst_str}")


def _uuid_from_str(uuid_str: str) -> uuid.UUID:
    try:
        return langutil.uuid_from_str(uuid_str)
    except (ValueError, TypeError):
        raise SyntaxError(f"Unrecognized UUID format: {uuid_str}")


_DATA_READERS = lmap.map({
    symbol.symbol('inst'): _inst_from_str,
    symbol.symbol('uuid'): _uuid_from_str
})


def _read_reader_macro(ctx: ReaderContext) -> LispForm:
    """Return a data structure evaluated as a reader
    macro from the input stream."""
    start = ctx.reader.advance()
    assert start == '#'
    token = ctx.reader.peek()
    if token == '{':
        return _read_set(ctx)
    elif token == "(":
        return _read_function(ctx)
    elif token == "'":
        ctx.reader.advance()
        s = _read_sym(ctx)
        return llist.l(_VAR, s)
    elif token == '"':
        return _read_regex(ctx)
    elif token == "_":
        ctx.reader.advance()
        _read_next(ctx)  # Ignore the entire next form
        return _read_next(ctx)
    elif ns_name_chars.match(token):
        s = _read_sym(ctx)
        assert isinstance(s, symbol.Symbol)
        if s.ns is None and s not in _DATA_READERS:
            raise SyntaxError(
                f"Non-namespaced tags are reserved by the reader (#{s} not found)"
            )

        v = _read_next(ctx)
        if s in _DATA_READERS:
            f = _DATA_READERS[s]
            return f(v)
        else:
            raise SyntaxError(f"No data reader found for tag #{s}")

    raise SyntaxError(f"Unexpected token '{token}' in reader macro")


def _read_comment(ctx: ReaderContext) -> LispForm:
    """Read (and ignore) a single-line comment from the input stream.
    Return the next form after the next line break."""
    reader = ctx.reader
    start = reader.advance()
    assert start == ';'
    while True:
        token = reader.peek()
        if newline_chars.match(token):
            reader.advance()
            return _read_next(ctx)
        if token == '':
            return __EOF
        reader.advance()


def _read_next(ctx: ReaderContext) -> LispForm:  # noqa: C901
    """Read the next full token from the input stream."""
    reader = ctx.reader
    token = reader.peek()
    if token == '(':
        return _read_list(ctx)
    elif token == '[':
        return _read_vector(ctx)
    elif token == '{':
        return _read_map(ctx)
    elif begin_num_chars.match(token):
        return _read_num(ctx)
    elif whitespace_chars.match(token):
        reader.next_token()
        return _read_next(ctx)
    elif token == ':':
        return _read_kw(ctx)
    elif token == '"':
        return _read_str(ctx)
    elif token == "'":
        return _read_quoted(ctx)
    elif ns_name_chars.match(token):
        return _read_sym(ctx)
    elif token == '#':
        return _read_reader_macro(ctx)
    elif token == '^':
        return _read_meta(ctx)  # type: ignore
    elif token == ';':
        return _read_comment(ctx)
    elif token == '`':
        return _read_syntax_quoted(ctx)
    elif token == '~':
        return _read_unquote(ctx)
    elif token == '':
        return __EOF
    else:
        raise SyntaxError("Unexpected token '{token}'".format(token=token))


def read(stream, resolver: Resolver = None) -> Iterable[LispForm]:
    """Read the contents of a stream as a Lisp expression.

    The caller is responsible for closing the input stream."""
    reader = StreamReader(stream)
    ctx = ReaderContext(reader, resolver=resolver)
    while True:
        expr = _read_next(ctx)
        if expr is __EOF:
            return
        yield expr


def read_str(s: str, resolver: Resolver = None) -> Iterable[LispForm]:
    """Read the contents of a string as a Lisp expression."""
    with io.StringIO(s) as buf:
        yield from read(buf, resolver=resolver)


def read_file(filename: str, resolver: Resolver = None) -> Iterable[LispForm]:
    """Read the contents of a file as a Lisp expression."""
    with open(filename) as f:
        yield from read(f, resolver=resolver)
