import io
import re
import threading
import lang.float as lfloat
import lang.integer as integer
import lang.keyword as keyword
import lang.list as llist
import lang.string as string
import lang.symbol as symbol
import lang.vector as vector


ns_name_chars = re.compile(r'\w|-|\+|\*|\?|/|\=|\\|!')
num_chars = re.compile('[0-9]')
whitespace_chars = re.compile('\s')


class SyntaxError(Exception):
    pass


class StreamReader:
    """A simple stream reader with 1 character lookahead."""
    def __init__(self, stream):
        self._stream = stream
        self._prev = self._stream.read(1)
        self._cur = self._stream.read(1)

    def peek(self):
        """Peek at the next character in the stream."""
        return self._prev

    def next_token(self):
        """Advance the stream forward by one character."""
        self._prev = self._cur
        self._cur = self._stream.read(1)
        return self._prev


__EOF = 'EOF'


def read(stream, is_module=False):
    """Read the contents of a stream as a Lisp expression."""
    reader = StreamReader(stream)

    def read_namespaced():
        """Read a namespaced token from the input stream."""
        ns = []
        name = []
        has_ns = False
        while True:
            token = reader.peek()
            if token == '/':
                reader.reader.next_token()
                if has_ns:
                    raise SyntaxError("Found '/'; expected word character")
                has_ns = True
                ns = name
                name = []
            elif ns_name_chars.match(token):
                reader.next_token()
                name.append(token)
            else:
                ns = None if not has_ns else ''.join(ns)
                name = ''.join(name)
                return ns, name

    def read_next():
        """Read the next full token from the input stream."""
        token = reader.peek()
        if token == '(':
            l = []
            while True:
                token = reader.peek()
                if token == '':
                    raise SyntaxError("Unexpected EOF in list")
                if token == ')':
                    reader.next_token()
                    return llist.list(l)
                reader.next_token()
                l.append(read_next())
        elif token == '[':
            v = []
            while True:
                token = reader.peek()
                if token == '':
                    raise SyntaxError("Unexpected EOF in vector")
                if token == '':
                    reader.next_token()
                    return vector.vector(v)
                reader.next_token()
                v.append(read_next())
        elif num_chars.match(token):
            chars = [token]
            while True:
                token = reader.peek()
                if not num_chars.match(token):
                    return integer.integer(''.join(chars))
                reader.next_token()
                chars.append(token)
        elif whitespace_chars.match(token):
            reader.next_token()
            return read_next()
        elif token == ':':
            token = reader.next_token()
            ns, name = read_namespaced()
            return keyword.keyword(name, ns=ns)
        elif token == '"':
            s = []
            while True:
                prev = reader.peek()
                token = reader.next_token()
                if token == '':
                    raise SyntaxError("Unexpected EOF in string")
                if token == "\\":
                    prev = '\\'
                    continue
                if token == '"' and not prev == "\\":
                    reader.next_token()
                    return string.string(''.join(s))
                s.append(token)
        elif ns_name_chars.match(token):
            ns, name = read_namespaced()
            return symbol.symbol(name, ns=ns)
        elif token == '':
            return __EOF
        else:
            raise SyntaxError("Unexpected token '{token}'".format(token=token))

    data = []
    while True:
        expr = read_next()
        if expr is __EOF:
            break
        if len(data) > 1 and not is_module:
            raise SyntaxError("Unexpected expr '{expr}'".format(expr=repr(expr)))
        data.append(expr)

    if len(data) == 0:
        return None
    elif not is_module:
        return data[0]

    return llist.list(data)


def read_str(s, is_module=False):
    """Read the contents of a string as a Lisp expression."""
    return read(io.StringIO(s), is_module=is_module)

