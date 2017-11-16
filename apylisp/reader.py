import collections
import io
import re
import threading
import apylisp.lang.float as lfloat
import apylisp.lang.integer as integer
import apylisp.lang.keyword as keyword
import apylisp.lang.list as llist
import apylisp.lang.map as lmap
import apylisp.lang.set as lset
import apylisp.lang.string as string
import apylisp.lang.symbol as symbol
import apylisp.lang.vector as vector
from apylisp.util import spy

ns_name_chars = re.compile(r'\w|-|\+|\*|\?|/|\=|\\|!')
begin_num_chars = re.compile('[0-9\-]')
num_chars = re.compile('[0-9]')
whitespace_chars = re.compile('[\s,]')


class SyntaxError(Exception):
    pass


class StreamReader:
    """A simple stream reader with n-character lookahead."""
    DEFAULT_INDEX = -2

    def __init__(self, stream, pushback_depth=5):
        self._stream = stream
        self._pushback_depth = pushback_depth
        self._idx = -2
        init_buffer = [self._stream.read(1), self._stream.read(1)]
        self._buffer = collections.deque(init_buffer, pushback_depth)

    def peek(self):
        """Peek at the next character in the stream."""
        return self._buffer[self._idx]

    def pushback(self):
        """Push one character back onto the stream, allowing it to be
        read again."""
        if abs(self._idx - 1) > self._pushback_depth:
            raise IndexError("Exceeded pushback depth")
        self._idx -= 1

    def advance(self):
        """Advance the current token pointer by one and return the
        previous token value from before advancing the counter."""
        cur = self.peek()
        self.next_token()
        return cur

    def next_token(self):
        """Advance the stream forward by one character and return the
        next token in the stream."""
        if self._idx < StreamReader.DEFAULT_INDEX:
            self._idx += 1
        else:
            c = self._stream.read(1)
            self._buffer.append(c)

        return self.peek()


__EOF = 'EOF'


def _read_namespaced(reader):
    """Read a namespaced token from the input stream."""
    ns = []
    name = []
    has_ns = False
    while True:
        token = reader.peek()
        if token == '/':
            reader.next_token()
            if has_ns:
                raise SyntaxError("Found '/'; expected word character")
            has_ns = True
            ns = name
            name = []
            if len(ns) == 0:
                raise SyntaxError("Found ''; expected namespace")
        elif ns_name_chars.match(token):
            reader.next_token()
            name.append(token)
        elif not has_ns and token == '.':
            reader.next_token()
            name.append(token)
        else:
            ns = None if not has_ns else ''.join(ns)
            name = ''.join(name)
            break

    if '.' in name:
        raise SyntaxError("Found '.' in symbol or keyword name")

    return ns, name


def _read_coll(reader, f, end_token, coll_name):
    """Read a collection from the input stream and create the
    collection using f."""
    coll = []
    while True:
        token = reader.peek()
        if token == '':
            raise SyntaxError(f"Unexpected EOF in {coll_name}")
        if token == end_token:
            reader.next_token()
            return f(coll)
        elem = _read_next(reader)
        coll.append(elem)


def _read_list(reader):
    """Read a list element from the input stream."""
    start = reader.advance()
    assert start == '('
    return _read_coll(reader, llist.list, ')', 'list')


def _read_vector(reader):
    """Read a vector element from the input stream."""
    start = reader.advance()
    assert start == '['
    return _read_coll(reader, vector.vector, ']', 'vector')


def _read_set(reader):
    """Return a set from the input stream."""
    start = reader.advance()
    assert start == '{'

    def set_if_valid(s):
        if len(s) != len(set(s)):
            raise SyntaxError("Duplicated values in set")
        return lset.set(s)

    return _read_coll(reader, set_if_valid, '}', 'set')


def _read_map(reader):
    """Return a map from the input stream."""
    start = reader.advance()
    assert start == '{'
    d = {}
    while True:
        if reader.peek() == '}':
            reader.next_token()
            break
        k = _read_next(reader)
        if reader.peek() == '}':
            raise SyntaxError("Unexpected token '}'; expected map value")
        v = _read_next(reader)
        if k in d:
            raise SyntaxError("Duplicate key '{}' in map literal".format(k))
        d[k] = v

    return lmap.map(d)


def _read_num(reader):
    """Return a numeric (integer or float) from the input stream."""
    chars = []
    is_float = False
    while True:
        token = reader.peek()
        if token == '-':
            following_token = reader.next_token()
            if not begin_num_chars.match(following_token):
                reader.pushback()
                return _read_sym(reader)
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
    return lfloat.lfloat(s) if is_float else integer.integer(s)


def _read_str(reader):
    """Return a string from the input stream."""
    s = []
    while True:
        prev = reader.peek()
        token = reader.next_token()
        if token == '':
            raise SyntaxError("Unexpected EOF in string")
        if token == "\\":
            continue
        if token == '"' and not prev == "\\":
            reader.next_token()
            return string.string(''.join(s))
        s.append(token)


def _read_sym(reader):
    """Return a symbol from the input stream."""
    ns, name = _read_namespaced(reader)
    if ns == None:
        if name == 'nil':
            return None
        elif name == 'true':
            return True
        elif name == 'false':
            return False
    return symbol.symbol(name, ns=ns)


def _read_kw(reader):
    """Return a keyword from the input stream."""
    start = reader.advance()
    assert start == ':'
    ns, name = _read_namespaced(reader)
    return keyword.keyword(name, ns=ns)


def _read_reader_macro(reader):
    """Return a data structure evaluated as a reader
    macro from the input stream."""
    start = reader.advance()
    assert start == '#'
    token = reader.peek()
    if token == '{':
        return _read_set(reader)

    raise SyntaxError(f"Unexpected token '{token}' in reader macro")


def _read_next(reader):
    """Read the next full token from the input stream."""
    token = reader.peek()
    if token == '(':
        return _read_list(reader)
    elif token == '[':
        return _read_vector(reader)
    elif token == '{':
        return _read_map(reader)
    elif begin_num_chars.match(token):
        return _read_num(reader)
    elif whitespace_chars.match(token):
        reader.next_token()
        return _read_next(reader)
    elif token == ':':
        return _read_kw(reader)
    elif token == '"':
        return _read_str(reader)
    elif ns_name_chars.match(token):
        return _read_sym(reader)
    elif token == '#':
        return _read_reader_macro(reader)
    elif token == '':
        return __EOF
    else:
        raise SyntaxError("Unexpected token '{token}'".format(token=token))


def read(stream):
    """Read the contents of a stream as a Lisp expression.

    The caller is responsible for closing the input stream."""
    reader = StreamReader(stream)
    data = []
    while True:
        expr = _read_next(reader)
        if expr is __EOF:
            break
        data.append(expr)

    if len(data) == 0:
        return None
    elif len(data) == 1:
        return data[0]

    return llist.list(data)


def read_str(s):
    """Read the contents of a string as a Lisp expression."""
    with io.StringIO(s) as buf:
        return read(buf)


def read_file(filename):
    """Read the contents of a file as a Lisp expression."""
    with open(filename) as f:
        return read(f)
