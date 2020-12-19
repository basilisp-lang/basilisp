import collections
import contextlib
import decimal
import functools
import io
import re
import uuid
from datetime import datetime
from fractions import Fraction
from itertools import chain
from typing import (
    Any,
    Callable,
    Collection,
    Deque,
    Dict,
    Hashable,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Pattern,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import attr

from basilisp.lang import keyword as kw
from basilisp.lang import list as llist
from basilisp.lang import map as lmap
from basilisp.lang import queue as lqueue
from basilisp.lang import set as lset
from basilisp.lang import symbol as sym
from basilisp.lang import util as langutil
from basilisp.lang import vector as vec
from basilisp.lang.interfaces import (
    ILispObject,
    ILookup,
    IMeta,
    IPersistentList,
    IPersistentMap,
    IPersistentSet,
    IPersistentVector,
    IRecord,
    IType,
    IWithMeta,
)
from basilisp.lang.obj import seq_lrepr as _seq_lrepr
from basilisp.lang.runtime import (
    READER_COND_DEFAULT_FEATURE_SET,
    Namespace,
    Var,
    get_current_ns,
    lrepr,
)
from basilisp.lang.typing import IterableLispForm, LispForm, ReaderForm
from basilisp.lang.util import munge
from basilisp.util import Maybe, partition

ns_name_chars = re.compile(r"\w|-|\+|\*|\?|/|\=|\\|!|&|%|>|<|\$|\.")
alphanumeric_chars = re.compile(r"\w")
begin_num_chars = re.compile(r"[0-9\-]")
num_chars = re.compile("[0-9]")
whitespace_chars = re.compile(r"[\s,]")
newline_chars = re.compile("(\r\n|\r|\n)")
fn_macro_args = re.compile("(%)(&|[0-9])?")
unicode_char = re.compile(r"u(\w+)")

DataReaders = Optional[lmap.PersistentMap]
GenSymEnvironment = MutableMapping[str, sym.Symbol]
Resolver = Callable[[sym.Symbol], sym.Symbol]
LispReaderFn = Callable[["ReaderContext"], LispForm]
W = TypeVar("W", bound=LispReaderFn)

READER_LINE_KW = kw.keyword("line", ns="basilisp.lang.reader")
READER_COL_KW = kw.keyword("col", ns="basilisp.lang.reader")

READER_COND_FORM_KW = kw.keyword("form")
READER_COND_SPLICING_KW = kw.keyword("splicing?")

_AMPERSAND = sym.symbol("&")
_FN = sym.symbol("fn*")
_INTEROP_CALL = sym.symbol(".")
_INTEROP_PROP = sym.symbol(".-")
_QUOTE = sym.symbol("quote")
_VAR = sym.symbol("var")

_APPLY = sym.symbol("apply", ns="basilisp.core")
_CONCAT = sym.symbol("concat", ns="basilisp.core")
_DEREF = sym.symbol("deref", ns="basilisp.core")
_HASH_MAP = sym.symbol("hash-map", ns="basilisp.core")
_HASH_SET = sym.symbol("hash-set", ns="basilisp.core")
_LIST = sym.symbol("list", ns="basilisp.core")
_SEQ = sym.symbol("seq", ns="basilisp.core")
_UNQUOTE = sym.symbol("unquote", ns="basilisp.core")
_UNQUOTE_SPLICING = sym.symbol("unquote-splicing", ns="basilisp.core")
_VECTOR = sym.symbol("vector", ns="basilisp.core")


class Comment:
    pass


COMMENT = Comment()

LispReaderForm = Union[ReaderForm, Comment, "ReaderConditional"]
RawReaderForm = Union[ReaderForm, "ReaderConditional"]


# pylint:disable=redefined-builtin
@attr.s(auto_attribs=True, repr=False, slots=True, str=False)
class SyntaxError(Exception):
    message: str
    line: Optional[int] = None
    col: Optional[int] = None
    filename: Optional[str] = None

    def __repr__(self):
        return (
            f"basilisp.lang.reader.SyntaxError({self.message}, {self.line},"
            f"{self.col}, filename={self.filename})"
        )

    def __str__(self):
        keys: Dict[str, Union[str, int]] = {}
        if self.filename is not None:
            keys["file"] = self.filename
        if self.line is not None and self.col is not None:
            keys["line"] = self.line
            keys["col"] = self.col
        if not keys:
            return self.message
        else:
            details = ", ".join(f"{key}: {val}" for key, val in keys.items())
            return f"{self.message} ({details})"


class UnexpectedEOFError(SyntaxError):
    """Syntax Error type raised when the reader encounters an unexpected EOF
    reading a form.

    Useful for cases such as the REPL reader, where unexpected EOF errors
    likely indicate the user is trying to enter a multiline form."""


class StreamReader:
    """A simple stream reader with n-character lookahead."""

    DEFAULT_INDEX = -2

    __slots__ = ("_stream", "_pushback_depth", "_idx", "_buffer", "_line", "_col")

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
    def name(self) -> Optional[None]:
        return getattr(self._stream, "name", None)

    @property
    def col(self) -> int:
        return self._col[self._idx]

    @property
    def line(self) -> int:
        return self._line[self._idx]

    @property
    def loc(self) -> Tuple[int, int]:
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


@functools.singledispatch
def _py_from_lisp(
    form: Union[
        llist.PersistentList,
        lmap.PersistentMap,
        lset.PersistentSet,
        vec.PersistentVector,
    ]
) -> ReaderForm:
    raise SyntaxError(f"Unrecognized Python type: {type(form)}")


@_py_from_lisp.register(IPersistentList)
def _py_tuple_from_list(form: llist.PersistentList) -> tuple:
    return tuple(form)


@_py_from_lisp.register(IPersistentMap)
def _py_dict_from_map(form: lmap.PersistentMap) -> dict:
    return dict(form)


@_py_from_lisp.register(IPersistentSet)
def _py_set_from_set(form: lset.PersistentSet) -> set:
    return set(form)


@_py_from_lisp.register(IPersistentVector)
def _py_list_from_vec(form: vec.PersistentVector) -> list:
    return list(form)


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


class ReaderContext:
    _DATA_READERS = lmap.map(
        {
            sym.symbol("inst"): _inst_from_str,
            sym.symbol("py"): _py_from_lisp,
            sym.symbol("queue"): lqueue.queue,
            sym.symbol("uuid"): _uuid_from_str,
        }
    )

    __slots__ = (
        "_data_readers",
        "_features",
        "_process_reader_cond",
        "_reader",
        "_resolve",
        "_in_anon_fn",
        "_syntax_quoted",
        "_gensym_env",
        "_eof",
    )

    def __init__(  # pylint: disable=too-many-arguments
        self,
        reader: StreamReader,
        resolver: Resolver = None,
        data_readers: DataReaders = None,
        eof: Any = None,
        features: Optional[IPersistentSet[kw.Keyword]] = None,
        process_reader_cond: bool = True,
    ) -> None:
        data_readers = Maybe(data_readers).or_else_get(lmap.PersistentMap.empty())
        for reader_sym in data_readers.keys():
            if not isinstance(reader_sym, sym.Symbol):
                raise TypeError("Expected symbol for data reader tag")
            if not reader_sym.ns:
                raise ValueError("Non-namespaced tags are reserved by the reader")

        self._data_readers = ReaderContext._DATA_READERS.update_with(
            lambda l, r: l,  # Do not allow callers to overwrite existing builtin readers
            data_readers,
        )
        self._features = (
            features if features is not None else READER_COND_DEFAULT_FEATURE_SET
        )
        self._process_reader_cond = process_reader_cond
        self._reader = reader
        self._resolve = Maybe(resolver).or_else_get(lambda x: x)
        self._in_anon_fn: Deque[bool] = collections.deque([])
        self._syntax_quoted: Deque[bool] = collections.deque([])
        self._gensym_env: Deque[GenSymEnvironment] = collections.deque([])
        self._eof = eof

    @property
    def data_readers(self) -> lmap.PersistentMap:
        return self._data_readers

    @property
    def eof(self) -> Any:
        return self._eof

    @property
    def reader_features(self) -> IPersistentSet[kw.Keyword]:
        return self._features

    @property
    def should_process_reader_cond(self) -> bool:
        return self._process_reader_cond

    @property
    def reader(self) -> StreamReader:
        return self._reader

    def resolve(self, sym: sym.Symbol) -> sym.Symbol:
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

    def syntax_error(self, msg: str) -> SyntaxError:
        """Return a SyntaxError with the given message, hydrated with filename, line,
        and column metadata from the reader if it exists."""
        return SyntaxError(
            msg, line=self.reader.line, col=self.reader.col, filename=self.reader.name
        )

    def eof_error(self, msg: str) -> UnexpectedEOFError:
        """Return an UnexpectedEOFError with the given message, hydrated with filename,
        line, and column metadata from the reader if it exists."""
        return UnexpectedEOFError(
            msg, line=self.reader.line, col=self.reader.col, filename=self.reader.name
        )


class ReaderConditional(ILookup[kw.Keyword, ReaderForm], ILispObject):
    FEATURE_NOT_PRESENT = object()

    __slots__ = ("_form", "_feature_vec", "_is_splicing")

    def __init__(
        self,
        form: llist.PersistentList[Tuple[kw.Keyword, ReaderForm]],
        is_splicing: bool = False,
    ):
        self._form = form
        self._feature_vec = self._compile_feature_vec(form)
        self._is_splicing = is_splicing

    @staticmethod
    def _compile_feature_vec(form: IPersistentList[Tuple[kw.Keyword, ReaderForm]]):
        found_features: Set[kw.Keyword] = set()
        feature_list: List[Tuple[kw.Keyword, ReaderForm]] = []

        try:
            for k, v in partition(form, 2):
                if not isinstance(k, kw.Keyword):
                    raise SyntaxError(
                        f"Reader conditional features must be keywords, not {type(k)}"
                    )
                if k in found_features:
                    raise SyntaxError(
                        f"Duplicate feature '{k}' in reader conditional literal"
                    )
                found_features.add(k)
                feature_list.append((k, v))
        except ValueError:
            raise SyntaxError(
                "Reader conditionals must contain an even number of forms"
            )

        return vec.vector(feature_list)

    def val_at(
        self, k: kw.Keyword, default: Optional[ReaderForm] = None
    ) -> Optional[ReaderForm]:
        if k == READER_COND_FORM_KW:
            return self._form
        elif k == READER_COND_SPLICING_KW:
            return self._is_splicing
        else:
            return default

    @property
    def is_splicing(self):
        return self._is_splicing

    def select_feature(
        self, features: IPersistentSet[kw.Keyword]
    ) -> Union[ReaderForm, object]:
        for k, form in self._feature_vec:
            if k in features:
                return form
        return self.FEATURE_NOT_PRESENT

    def _lrepr(self, **kwargs) -> str:
        return _seq_lrepr(
            chain.from_iterable(self._feature_vec),
            "#?@(" if self.is_splicing else "#?(",
            ")",
            **kwargs,
        )


EOF = object()


def _with_loc(f: W) -> W:
    """Wrap a reader function in a decorator to supply line and column
    information along with relevant forms."""

    @functools.wraps(f)
    def with_lineno_and_col(ctx, **kwargs):
        line, col = ctx.reader.line, ctx.reader.col
        v = f(ctx, **kwargs)  # type: ignore[call-arg]
        if isinstance(v, IWithMeta):
            new_meta = lmap.map({READER_LINE_KW: line, READER_COL_KW: col})
            old_meta = v.meta
            return v.with_meta(
                old_meta.cons(new_meta) if old_meta is not None else new_meta
            )
        else:
            return v

    return cast(W, with_lineno_and_col)


def _read_namespaced(
    ctx: ReaderContext, allowed_suffix: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """Read a namespaced token from the input stream."""
    ns: List[str] = []
    name: List[str] = []
    reader = ctx.reader
    has_ns = False
    while True:
        token = reader.peek()
        if token == "/":
            reader.next_token()
            if has_ns:
                raise ctx.syntax_error("Found '/'; expected word character")
            elif len(name) == 0:
                name.append("/")
            else:
                if "/" in name:
                    raise ctx.syntax_error("Found '/' after a previous '/'")
                has_ns = True
                ns = name
                name = []
        elif ns_name_chars.match(token):
            reader.next_token()
            name.append(token)
        elif allowed_suffix is not None and token == allowed_suffix:
            reader.next_token()
            name.append(token)
        else:
            break

    ns_str = None if not has_ns else "".join(ns)
    name_str = "".join(name)

    # A small exception for the symbol '/ used for division
    if ns_str is None:
        if "/" in name_str and name_str != "/":
            raise ctx.syntax_error("'/' character disallowed in names")

    assert ns_str is None or len(ns_str) > 0

    return ns_str, name_str


def _read_coll(
    ctx: ReaderContext,
    f: Callable[
        [Collection[Any]],
        Union[llist.PersistentList, lset.PersistentSet, vec.PersistentVector],
    ],
    end_token: str,
    coll_name: str,
):
    """Read a collection from the input stream and create the
    collection using f."""
    coll: List = []
    reader = ctx.reader
    while True:
        token = reader.peek()
        if token == "":
            raise ctx.eof_error(f"Unexpected EOF in {coll_name}")
        if whitespace_chars.match(token):
            reader.advance()
            continue
        if token == end_token:
            reader.next_token()
            return f(coll)
        elem = _read_next(ctx)
        if elem is COMMENT or isinstance(elem, Comment):
            continue
        elif _should_splice_reader_conditional(ctx, elem):
            assert isinstance(elem, ReaderConditional)
            selected_feature = elem.select_feature(ctx.reader_features)
            if selected_feature is ReaderConditional.FEATURE_NOT_PRESENT:
                continue
            elif isinstance(selected_feature, vec.PersistentVector):
                coll.extend(selected_feature)
            else:
                raise ctx.syntax_error(
                    "Expecting Vector for splicing reader conditional "
                    f"form; got {type(selected_feature)}"
                )
        else:
            assert (
                not isinstance(elem, ReaderConditional)
                or not ctx.should_process_reader_cond
            ), "Reader conditionals must be processed if specified"
            coll.append(elem)


@_with_loc
def _read_list(ctx: ReaderContext) -> llist.PersistentList:
    """Read a list element from the input stream."""
    start = ctx.reader.advance()
    assert start == "("
    return _read_coll(ctx, llist.list, ")", "list")


@_with_loc
def _read_vector(ctx: ReaderContext) -> vec.PersistentVector:
    """Read a vector element from the input stream."""
    start = ctx.reader.advance()
    assert start == "["
    return _read_coll(ctx, vec.vector, "]", "vector")


@_with_loc
def _read_set(ctx: ReaderContext) -> lset.PersistentSet:
    """Return a set from the input stream."""
    start = ctx.reader.advance()
    assert start == "{"

    def set_if_valid(s: Collection) -> lset.PersistentSet:
        coll_set = set(s)
        if len(s) != len(coll_set):
            dupes = ", ".join(
                lrepr(k) for k, v in collections.Counter(s).items() if v > 1
            )
            raise ctx.syntax_error(f"Duplicated values in set: {dupes}")
        return lset.set(s)

    return _read_coll(ctx, set_if_valid, "}", "set")


def __read_map_elems(ctx: ReaderContext) -> Iterable[RawReaderForm]:
    """Return an iterable of map contents, potentially splicing in values from
    reader conditionals."""
    reader = ctx.reader
    while True:
        token = reader.peek()
        if token == "":
            raise ctx.eof_error("Unexpected EOF in map}")
        if whitespace_chars.match(token):
            reader.advance()
            continue
        if token == "}":
            reader.next_token()
            return
        v = _read_next(ctx)
        if v is COMMENT or isinstance(v, Comment):
            continue
        elif _should_splice_reader_conditional(ctx, v):
            assert isinstance(v, ReaderConditional)
            selected_feature = v.select_feature(ctx.reader_features)
            if selected_feature is ReaderConditional.FEATURE_NOT_PRESENT:
                continue
            elif isinstance(selected_feature, vec.PersistentVector):
                yield from selected_feature
            else:
                raise ctx.syntax_error(
                    "Expecting Vector for splicing reader conditional "
                    f"form; got {type(selected_feature)}"
                )
        else:
            assert (
                not isinstance(v, ReaderConditional)
                or not ctx.should_process_reader_cond
            ), "Reader conditionals must be processed if specified"
            yield v


def _map_key_processor(
    namespace: Optional[str],
) -> Callable[[Hashable], Hashable]:
    """Return a map key processor.

    If no `namespace` is provided, return an identity function. If a `namespace`
    is given, return a function that can apply namespaces to un-namespaced
    keyword and symbol values."""
    if namespace is None:
        return lambda v: v

    def process_key(k: Any) -> Any:
        if isinstance(k, kw.Keyword):
            if k.ns is None:
                return kw.keyword(k.name, ns=namespace)
            if k.ns == "_":
                return kw.keyword(k.name)
        if isinstance(k, sym.Symbol):
            if k.ns is None:
                return sym.symbol(k.name, ns=namespace)
            if k.ns == "_":
                return sym.symbol(k.name)
        return k

    return process_key


@_with_loc
def _read_map(
    ctx: ReaderContext, namespace: Optional[str] = None
) -> lmap.PersistentMap:
    """Return a map from the input stream."""
    reader = ctx.reader
    start = reader.advance()
    assert start == "{"
    d: MutableMapping[Any, Any] = {}
    process_key = _map_key_processor(namespace)
    try:
        for k, v in partition(list(__read_map_elems(ctx)), 2):
            k = process_key(k)
            if k in d:
                raise ctx.syntax_error(f"Duplicate key '{k}' in map literal")
            d[k] = v
    except ValueError:
        raise ctx.syntax_error("Unexpected token '}'; expected map value")
    else:
        return lmap.map(d)


def _read_namespaced_map(ctx: ReaderContext) -> lmap.PersistentMap:
    """Read a namespaced map from the input stream."""
    start = ctx.reader.peek()
    assert start == ":"
    ctx.reader.advance()
    if ctx.reader.peek() == ":":
        ctx.reader.advance()
        current_ns = get_current_ns()
        map_ns = current_ns.name
    else:
        kw_ns, map_ns = _read_namespaced(ctx)
        if kw_ns is not None:
            raise ctx.syntax_error(
                f"Invalid map namespace '{kw_ns}/{map_ns}'; namespaces for maps must "
                "be specified as keywords without namespaces"
            )

    token = ctx.reader.peek()
    while whitespace_chars.match(token):
        token = ctx.reader.next_token()

    return _read_map(ctx, namespace=map_ns)


# Due to some ambiguities that arise in parsing symbols, numbers, and the
# special keywords `true`, `false`, and `nil`, we have to have a looser
# type defined for the return from these reader functions.
MaybeSymbol = Union[bool, None, sym.Symbol]
MaybeNumber = Union[complex, decimal.Decimal, float, Fraction, int, MaybeSymbol]


def _read_num(  # noqa: C901  # pylint: disable=too-many-statements
    ctx: ReaderContext,
) -> MaybeNumber:
    """Return a numeric (complex, Decimal, float, int, Fraction) from the input stream."""
    chars: List[str] = []
    reader = ctx.reader

    is_complex = False
    is_decimal = False
    is_float = False
    is_integer = False
    is_ratio = False
    while True:
        token = reader.peek()
        if token == "-":
            following_token = reader.next_token()
            if not begin_num_chars.match(following_token):
                reader.pushback()
                try:
                    for _ in chars:
                        reader.pushback()
                except IndexError:
                    raise ctx.syntax_error(
                        "Requested to pushback too many characters onto StreamReader"
                    )
                return _read_sym(ctx)
            chars.append(token)
            continue
        elif token == ".":
            if is_float:
                raise ctx.syntax_error(
                    "Found extra '.' in float; expected decimal portion"
                )
            is_float = True
        elif token == "J":
            if is_complex:
                raise ctx.syntax_error("Found extra 'J' suffix in complex literal")
            is_complex = True
        elif token == "M":
            if is_decimal:
                raise ctx.syntax_error("Found extra 'M' suffix in decimal literal")
            is_decimal = True
        elif token == "N":
            if is_integer:
                raise ctx.syntax_error("Found extra 'N' suffix in integer literal")
            is_integer = True
        elif token == "/":
            if is_ratio:
                raise ctx.syntax_error("Found extra '/' in ratio literal")
            is_ratio = True
        elif not num_chars.match(token):
            break
        reader.next_token()
        chars.append(token)

    assert len(chars) > 0, "Must have at least one digit in integer or float"

    s = "".join(chars)
    if (
        sum(
            [
                is_complex and is_decimal,
                is_complex and is_integer,
                is_complex and is_ratio,
                is_decimal or is_float,
                is_integer,
                is_ratio,
            ]
        )
        > 1
    ):
        raise ctx.syntax_error(f"Invalid number format: {s}")

    if is_complex:
        imaginary = float(s[:-1]) if is_float else int(s[:-1])
        return complex(0, imaginary)
    elif is_decimal:
        try:
            return decimal.Decimal(s[:-1])
        except decimal.InvalidOperation:
            raise ctx.syntax_error(f"Invalid number format: {s}") from None
    elif is_float:
        return float(s)
    elif is_ratio:
        assert "/" in s, "Ratio must contain one '/' character"
        num, denominator = s.split("/")
        return Fraction(numerator=int(num), denominator=int(denominator))
    elif is_integer:
        return int(s[:-1])
    return int(s)


_STR_ESCAPE_CHARS = {
    '"': '"',
    "\\": "\\",
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
}


def _read_str(ctx: ReaderContext, allow_arbitrary_escapes: bool = False) -> str:
    """Return a string from the input stream.

    If allow_arbitrary_escapes is True, do not throw a SyntaxError if an
    unknown escape sequence is encountered."""
    s: List[str] = []
    reader = ctx.reader
    while True:
        token = reader.next_token()
        if token == "":
            raise ctx.eof_error("Unexpected EOF in string")
        if token == "\\":
            token = reader.next_token()
            escape_char = _STR_ESCAPE_CHARS.get(token, None)
            if escape_char:
                s.append(escape_char)
                continue
            if allow_arbitrary_escapes:
                s.append("\\")
            else:
                raise ctx.syntax_error(f"Unknown escape sequence: \\{token}")
        if token == '"':
            reader.next_token()
            return "".join(s)
        s.append(token)


@_with_loc
def _read_sym(ctx: ReaderContext) -> MaybeSymbol:
    """Return a symbol from the input stream.

    If a symbol appears in a syntax quoted form, the reader will attempt
    to resolve the symbol using the resolver in the ReaderContext `ctx`.
    The resolver will look into the current namespace for an alias or
    namespace matching the symbol's namespace."""
    ns, name = _read_namespaced(ctx, allowed_suffix="#")
    if not ctx.is_syntax_quoted and name.endswith("#"):
        raise ctx.syntax_error("Gensym may not appear outside syntax quote")
    if ns is not None:
        if any(map(lambda s: len(s) == 0, ns.split("."))):
            raise ctx.syntax_error(
                "All '.' separated segments of a namespace "
                "must contain at least one character."
            )
    if name.startswith(".") and ns is not None:
        raise ctx.syntax_error("Symbols starting with '.' may not have a namespace")
    if ns is None:
        if name == "nil":
            return None
        elif name == "true":
            return True
        elif name == "false":
            return False
    if ctx.is_syntax_quoted and not name.endswith("#"):
        return ctx.resolve(sym.symbol(name, ns))
    return sym.symbol(name, ns=ns)


def _read_kw(ctx: ReaderContext) -> kw.Keyword:
    """Return a keyword from the input stream."""
    start = ctx.reader.advance()
    assert start == ":"
    if ctx.reader.peek() == ":":
        ctx.reader.advance()
        should_autoresolve = True
    else:
        should_autoresolve = False

    ns, name = _read_namespaced(ctx)
    if "." in name:
        raise ctx.syntax_error("Found '.' in keyword name")

    if should_autoresolve:
        current_ns = get_current_ns()
        if ns is not None:
            aliased_ns = current_ns.aliases.get(sym.symbol(ns))
            if aliased_ns is None:
                raise ctx.syntax_error(f"Cannot resolve namespace alias '{ns}'")
            ns = aliased_ns.name
        else:
            ns = current_ns.name
        return kw.keyword(name, ns=ns)

    return kw.keyword(name, ns=ns)


def _read_meta(ctx: ReaderContext) -> IMeta:
    """Read metadata and apply that to the next object in the
    input stream."""
    start = ctx.reader.advance()
    assert start == "^"
    meta = _read_next_consuming_comment(ctx)

    meta_map: Optional[lmap.PersistentMap[LispForm, LispForm]]
    if isinstance(meta, sym.Symbol):
        meta_map = lmap.map({kw.keyword("tag"): meta})
    elif isinstance(meta, kw.Keyword):
        meta_map = lmap.map({meta: True})
    elif isinstance(meta, lmap.PersistentMap):
        meta_map = meta
    else:
        raise ctx.syntax_error(
            f"Expected symbol, keyword, or map for metadata, not {type(meta)}"
        )

    obj_with_meta = _read_next_consuming_comment(ctx)
    if isinstance(obj_with_meta, IWithMeta):
        new_meta = (
            obj_with_meta.meta.cons(meta_map)
            if obj_with_meta.meta is not None
            else meta_map
        )
        return obj_with_meta.with_meta(new_meta)
    else:
        raise ctx.syntax_error(
            f"Can not attach metadata to object of type {type(obj_with_meta)}"
        )


def _walk(inner_f, outer_f, form):
    """Walk an arbitrary, possibly nested data structure, applying inner_f to each
    element of form and then applying outer_f to the resulting form."""
    if isinstance(form, IPersistentList):
        return outer_f(llist.list(map(inner_f, form)))
    elif isinstance(form, IPersistentVector):
        return outer_f(vec.vector(map(inner_f, form)))
    elif isinstance(form, IPersistentMap):
        return outer_f(lmap.hash_map(*chain.from_iterable(map(inner_f, form.seq()))))
    elif isinstance(form, IPersistentSet):
        return outer_f(lset.set(map(inner_f, form)))
    else:
        return outer_f(form)


def _postwalk(f, form):
    """ "Walk form using depth-first, post-order traversal, applying f to each form
    and replacing form with its result."""
    inner_f = functools.partial(_postwalk, f)
    return _walk(inner_f, f, form)


@_with_loc
def _read_function(ctx: ReaderContext) -> llist.PersistentList:
    """Read a function reader macro from the input stream."""
    if ctx.is_in_anon_fn:
        raise ctx.syntax_error("Nested #() definitions not allowed")

    with ctx.in_anon_fn():
        form = _read_list(ctx)
    arg_set = set()

    def arg_suffix(arg_num):
        if arg_num is None:
            return "1"
        elif arg_num == "&":
            return "rest"
        else:
            return arg_num

    def sym_replacement(arg_num):
        suffix = arg_suffix(arg_num)
        return sym.symbol(f"arg-{suffix}")

    def identify_and_replace(f):
        if isinstance(f, sym.Symbol):
            if f.ns is None:
                match = fn_macro_args.match(f.name)
                if match is not None:
                    arg_num = match.group(2)
                    suffix = arg_suffix(arg_num)
                    arg_set.add(suffix)
                    return sym_replacement(arg_num)
        return f

    body = _postwalk(identify_and_replace, form) if len(form) > 0 else None

    arg_list: List[sym.Symbol] = []
    numbered_args = sorted(map(int, filter(lambda k: k != "rest", arg_set)))
    if len(numbered_args) > 0:
        max_arg = max(numbered_args)
        arg_list = [sym_replacement(str(i)) for i in range(1, max_arg + 1)]
        if "rest" in arg_set:
            arg_list.append(_AMPERSAND)
            arg_list.append(sym_replacement("rest"))

    return llist.l(_FN, vec.vector(arg_list), body)


@_with_loc
def _read_quoted(ctx: ReaderContext) -> llist.PersistentList:
    """Read a quoted form from the input stream."""
    start = ctx.reader.advance()
    assert start == "'"
    next_form = _read_next_consuming_comment(ctx)
    return llist.l(_QUOTE, next_form)


def _is_unquote(form: RawReaderForm) -> bool:
    """Return True if this form is unquote."""
    try:
        return form.first == _UNQUOTE  # type: ignore
    except AttributeError:
        return False


def _is_unquote_splicing(form: RawReaderForm) -> bool:
    """Return True if this form is unquote-splicing."""
    try:
        return form.first == _UNQUOTE_SPLICING  # type: ignore
    except AttributeError:
        return False


def _expand_syntax_quote(
    ctx: ReaderContext, form: IterableLispForm
) -> Iterable[LispForm]:
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


def _process_syntax_quoted_form(
    ctx: ReaderContext, form: RawReaderForm
) -> RawReaderForm:
    """Post-process syntax quoted forms to generate forms that can be assembled
    into the correct types at runtime.

    Lists are turned into:
        (basilisp.core/seq
         (basilisp.core/concat [& rest]))

    Vectors are turned into:
        (basilisp.core/apply
         basilisp.core/vector
         (basilisp.core/concat [& rest]))

    Sets are turned into:
        (basilisp.core/apply
         basilisp.core/hash-set
         (basilisp.core/concat [& rest]))

    Maps are turned into:
        (basilisp.core/apply
         basilisp.core/hash-map
         (basilisp.core/concat [& rest]))

    The child forms (called rest above) are processed by _expand_syntax_quote.

    All other forms are passed through without modification."""
    lconcat = lambda v: llist.list(v).cons(_CONCAT)
    if _is_unquote(form):
        return form[1]  # type: ignore
    elif _is_unquote_splicing(form):
        raise ctx.syntax_error("Cannot splice outside collection")
    elif isinstance(form, llist.PersistentList):
        return llist.l(_SEQ, lconcat(_expand_syntax_quote(ctx, form)))
    elif isinstance(form, vec.PersistentVector):
        return llist.l(_APPLY, _VECTOR, lconcat(_expand_syntax_quote(ctx, form)))
    elif isinstance(form, lset.PersistentSet):
        return llist.l(_APPLY, _HASH_SET, lconcat(_expand_syntax_quote(ctx, form)))
    elif isinstance(form, lmap.PersistentMap):
        flat_kvs = list(chain.from_iterable(form.items()))
        return llist.l(_APPLY, _HASH_MAP, lconcat(_expand_syntax_quote(ctx, flat_kvs)))  # type: ignore
    elif isinstance(form, sym.Symbol):
        if form.ns is None and form.name.endswith("#"):
            try:
                return llist.l(_QUOTE, ctx.gensym_env[form.name])
            except KeyError:
                genned = sym.symbol(langutil.genname(form.name[:-1])).with_meta(
                    form.meta
                )
                ctx.gensym_env[form.name] = genned
                return llist.l(_QUOTE, genned)
        return llist.l(_QUOTE, form)
    else:
        return form


def _read_syntax_quoted(ctx: ReaderContext) -> RawReaderForm:
    """Read a syntax-quote and set the syntax-quoting state in the reader."""
    start = ctx.reader.advance()
    assert start == "`"

    with ctx.syntax_quoted():
        return _process_syntax_quoted_form(ctx, _read_next_consuming_comment(ctx))


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
        if next_char == "@":
            ctx.reader.advance()
            next_form = _read_next_consuming_comment(ctx)
            return llist.l(_UNQUOTE_SPLICING, next_form)
        else:
            next_form = _read_next_consuming_comment(ctx)
            return llist.l(_UNQUOTE, next_form)


@_with_loc
def _read_deref(ctx: ReaderContext) -> LispForm:
    """Read a derefed form from the input stream."""
    start = ctx.reader.advance()
    assert start == "@"
    next_form = _read_next_consuming_comment(ctx)
    return llist.l(_DEREF, next_form)


_SPECIAL_CHARS = {
    "newline": "\n",
    "space": " ",
    "tab": "\t",
    "formfeed": "\f",
    "backspace": "\b",
    "return": "\r",
}


def _read_character(ctx: ReaderContext) -> str:
    """Read a character literal from the input stream.

    Character literals may appear as:
      - \\a \\b \\c etc will yield 'a', 'b', and 'c' respectively

      - \\newline, \\space, \\tab, \\formfeed, \\backspace, \\return yield
        the named characters

      - \\uXXXX yield the unicode digit corresponding to the code
        point named by the hex digits XXXX"""
    start = ctx.reader.advance()
    assert start == "\\"

    s: List[str] = []
    reader = ctx.reader
    token = reader.peek()
    while True:
        if token == "" or whitespace_chars.match(token):
            break
        if not alphanumeric_chars.match(token):
            break
        s.append(token)
        token = reader.next_token()

    char = "".join(s)
    special = _SPECIAL_CHARS.get(char, None)
    if special is not None:
        return special

    match = unicode_char.match(char)
    if match is not None:
        try:
            return chr(int(f"0x{match.group(1)}", 16))
        except (ValueError, OverflowError):
            raise ctx.syntax_error(f"Unsupported character \\u{char}") from None

    if len(char) > 1:
        raise ctx.syntax_error(f"Unsupported character \\{char}")

    return char


def _read_regex(ctx: ReaderContext) -> Pattern:
    """Read a regex reader macro from the input stream."""
    s = _read_str(ctx, allow_arbitrary_escapes=True)
    try:
        return langutil.regex_from_str(s)
    except re.error:
        raise ctx.syntax_error(f"Unrecognized regex pattern syntax: {s}")


_NUMERIC_CONSTANTS = {
    "NaN": float("nan"),
    "Inf": float("inf"),
    "-Inf": -float("inf"),
}


def _read_numeric_constant(ctx: ReaderContext) -> float:
    start = ctx.reader.advance()
    assert start == "#"
    ns, name = _read_namespaced(ctx)
    if ns is not None:
        raise ctx.syntax_error(f"Unrecognized numeric constant: '##{ns}/{name}'")
    c = _NUMERIC_CONSTANTS.get(name)
    if c is None:
        raise ctx.syntax_error(f"Unrecognized numeric constant: '##{name}'")
    return c


def _should_splice_reader_conditional(ctx: ReaderContext, form: LispReaderForm) -> bool:
    """Return True if and only if form is a ReaderConditional which should be spliced
    into a surrounding collection context."""
    return (
        isinstance(form, ReaderConditional)
        and ctx.should_process_reader_cond
        and form.is_splicing
    )


def _read_reader_conditional_preserving(ctx: ReaderContext) -> ReaderConditional:
    """Read a reader conditional form and return the unprocessed reader
    conditional object."""
    reader = ctx.reader
    start = reader.advance()
    assert start == "?"
    token = reader.peek()

    if token == "@":
        is_splicing = True
        ctx.reader.advance()
    elif token == "(":
        is_splicing = False
    else:
        raise ctx.syntax_error(
            f"Unexpected token '{token}'; expected opening "
            "'(' for reader conditional"
        )

    open_token = reader.advance()
    if open_token != "(":
        raise ctx.syntax_error(
            f"Expected opening '(' for reader conditional; got '{open_token}'"
        )

    feature_list = _read_coll(ctx, llist.list, ")", "reader conditional")
    assert isinstance(feature_list, llist.PersistentList)
    return ReaderConditional(feature_list, is_splicing=is_splicing)


def _read_reader_conditional(ctx: ReaderContext) -> LispReaderForm:
    """Read a reader conditional form and either return it or process it and
    return the resulting form.

    If the reader is not set to process the reader conditional, it will always
    be returned as a ReaderConditional object.

    If the reader is set to process reader conditionals, only non-splicing reader
    conditionals are processed here. If no matching feature is found in a
    non-splicing reader conditional, a comment will be emitted (which is ultimately
    discarded downstream in the reader).

    Splicing reader conditionals are processed in the respective collection readers."""
    reader_cond = _read_reader_conditional_preserving(ctx)
    if ctx.should_process_reader_cond and not reader_cond.is_splicing:
        form = reader_cond.select_feature(ctx.reader_features)
        return cast(
            LispReaderForm,
            COMMENT if form is ReaderConditional.FEATURE_NOT_PRESENT else form,
        )
    else:
        return reader_cond


def _load_record_or_type(
    ctx: ReaderContext, s: sym.Symbol, v: LispReaderForm
) -> Union[IRecord, IType]:
    """Attempt to load the constructor named by `s` and construct a new
    record or type instance from the vector or map following name."""
    assert s.ns is None, "Record reader macro cannot have namespace"
    assert "." in s.name, "Record names must appear fully qualified"

    ns_name, rec = s.name.rsplit(".", maxsplit=1)
    ns_sym = sym.symbol(ns_name)
    ns = Namespace.get(ns_sym)
    if ns is None:
        raise ctx.syntax_error(f"Namespace {ns_name} does not exist")

    rectype = getattr(ns.module, munge(rec), None)
    if rectype is None:
        raise ctx.syntax_error(f"Record or type {s} does not exist")

    if isinstance(v, vec.PersistentVector):
        if issubclass(rectype, (IRecord, IType)):
            posfactory = Var.find_in_ns(ns_sym, sym.symbol(f"->{rec}"))
            assert (
                posfactory is not None
            ), "Record and Type must have positional factories"
            return posfactory.value(*v)
        else:
            raise ctx.syntax_error(f"Var {s} is not a Record or Type")
    elif isinstance(v, lmap.PersistentMap):
        if issubclass(rectype, IRecord):
            mapfactory = Var.find_in_ns(ns_sym, sym.symbol(f"map->{rec}"))
            assert mapfactory is not None, "Record must have map factory"
            return mapfactory.value(v)
        else:
            raise ctx.syntax_error(f"Var {s} is not a Record type")
    else:
        raise ctx.syntax_error("Records may only be constructed from Vectors and Maps")


def _read_reader_macro(ctx: ReaderContext) -> LispReaderForm:
    """Return a data structure evaluated as a reader
    macro from the input stream."""
    start = ctx.reader.advance()
    assert start == "#"
    token = ctx.reader.peek()
    if token == "{":
        return _read_set(ctx)
    elif token == "(":
        return _read_function(ctx)
    elif token == ":":
        return _read_namespaced_map(ctx)
    elif token == "'":
        ctx.reader.advance()
        s = _read_sym(ctx)
        return llist.l(_VAR, s)
    elif token == '"':
        return _read_regex(ctx)
    elif token == "_":
        ctx.reader.advance()
        _read_next(ctx)  # Ignore the entire next form
        return COMMENT
    elif token == "!":
        return _read_comment(ctx)
    elif token == "?":
        return _read_reader_conditional(ctx)
    elif token == "#":
        return _read_numeric_constant(ctx)
    elif ns_name_chars.match(token):
        s = _read_sym(ctx)
        assert isinstance(s, sym.Symbol)
        v = _read_next_consuming_comment(ctx)
        if s in ctx.data_readers:
            f = ctx.data_readers[s]
            try:
                return f(v)
            except SyntaxError as e:
                raise ctx.syntax_error(e.message).with_traceback(e.__traceback__)
        elif s.ns is None and "." in s.name:
            return _load_record_or_type(ctx, s, v)
        else:
            raise ctx.syntax_error(f"No data reader found for tag #{s}")

    raise ctx.syntax_error(f"Unexpected token '{token}' in reader macro")


def _read_comment(ctx: ReaderContext) -> LispReaderForm:
    """Read (and ignore) a single-line comment from the input stream.
    Return the next form after the next line break."""
    reader = ctx.reader
    start = reader.advance()
    assert start in {";", "!"}
    while True:
        token = reader.peek()
        if newline_chars.match(token):
            reader.advance()
            return COMMENT
        if token == "":
            return ctx.eof
        reader.advance()


def _read_next_consuming_comment(ctx: ReaderContext) -> RawReaderForm:
    """Read the next full form from the input stream, consuming any
    reader comments completely."""
    while True:
        v = _read_next(ctx)
        if v is ctx.eof:
            return ctx.eof
        if v is COMMENT or isinstance(v, Comment):
            continue
        return v


def _read_next_consuming_whitespace(ctx: ReaderContext) -> LispReaderForm:
    """Read the next full form from the input stream, consuming any whitespace."""
    reader = ctx.reader
    token = reader.peek()
    while whitespace_chars.match(token):
        token = reader.next_token()
    return _read_next(ctx)


def _read_next(ctx: ReaderContext) -> LispReaderForm:  # noqa: C901 MC0001
    """Read the next full form from the input stream."""
    reader = ctx.reader
    token = reader.peek()
    if token == "(":
        return _read_list(ctx)
    elif token == "[":
        return _read_vector(ctx)
    elif token == "{":
        return _read_map(ctx)
    elif begin_num_chars.match(token):
        return _read_num(ctx)
    elif whitespace_chars.match(token):
        return _read_next_consuming_whitespace(ctx)
    elif token == ":":
        return _read_kw(ctx)
    elif token == '"':
        return _read_str(ctx)
    elif token == "'":
        return _read_quoted(ctx)
    elif token == "\\":
        return _read_character(ctx)
    elif ns_name_chars.match(token):
        return _read_sym(ctx)
    elif token == "#":
        return _read_reader_macro(ctx)
    elif token == "^":
        return _read_meta(ctx)  # type: ignore
    elif token == ";":
        return _read_comment(ctx)
    elif token == "`":
        return _read_syntax_quoted(ctx)
    elif token == "~":
        return _read_unquote(ctx)
    elif token == "@":
        return _read_deref(ctx)
    elif token == "":
        return ctx.eof
    else:
        raise ctx.syntax_error(f"Unexpected token '{token}'")


def read(  # pylint: disable=too-many-arguments
    stream,
    resolver: Resolver = None,
    data_readers: DataReaders = None,
    eof: Any = EOF,
    is_eof_error: bool = False,
    features: Optional[IPersistentSet[kw.Keyword]] = None,
    process_reader_cond: bool = True,
) -> Iterable[RawReaderForm]:
    """Read the contents of a stream as a Lisp expression.

    Callers may optionally specify a namespace resolver, which will be used
    to adjudicate the fully-qualified name of symbols appearing inside of
    a syntax quote.

    Callers may optionally specify a map of custom data readers that will
    be used to resolve values in reader macros. Data reader tags specified
    by callers must be namespaced symbols; non-namespaced symbols are
    reserved by the reader. Data reader functions must be functions taking
    one argument and returning a value.

    Callers may specify whether or not reader conditional forms are processed
    or passed through raw (default: processed). Callers may provide a set of
    keyword "features" which the reader will use to determine which branches
    of reader conditional forms to read if reader conditionals are to be
    processed. If none are specified, then the `:default` and `:lpy` features
    are provided.

    The caller is responsible for closing the input stream."""
    reader = StreamReader(stream)
    ctx = ReaderContext(
        reader,
        resolver=resolver,
        data_readers=data_readers,
        eof=eof,
        features=features,
        process_reader_cond=process_reader_cond,
    )
    while True:
        expr = _read_next(ctx)
        if expr is ctx.eof:
            if is_eof_error:
                raise EOFError
            return
        if expr is COMMENT or isinstance(expr, Comment):
            continue
        if isinstance(expr, ReaderConditional) and ctx.should_process_reader_cond:
            raise ctx.syntax_error(
                f"Unexpected reader conditional '{repr(expr)})'; "
                "reader is configured to process reader conditionals"
            )
        yield expr


def read_str(  # pylint: disable=too-many-arguments
    s: str,
    resolver: Resolver = None,
    data_readers: DataReaders = None,
    eof: Any = EOF,
    is_eof_error: bool = False,
    features: Optional[IPersistentSet[kw.Keyword]] = None,
    process_reader_cond: bool = True,
) -> Iterable[RawReaderForm]:
    """Read the contents of a string as a Lisp expression.

    Keyword arguments to this function have the same meanings as those of
    basilisp.lang.reader.read."""
    with io.StringIO(s) as buf:
        yield from read(
            buf,
            resolver=resolver,
            data_readers=data_readers,
            eof=eof,
            is_eof_error=is_eof_error,
            features=features,
            process_reader_cond=process_reader_cond,
        )


def read_file(  # pylint: disable=too-many-arguments
    filename: str,
    resolver: Resolver = None,
    data_readers: DataReaders = None,
    eof: Any = EOF,
    is_eof_error: bool = False,
    features: Optional[IPersistentSet[kw.Keyword]] = None,
    process_reader_cond: bool = True,
) -> Iterable[RawReaderForm]:
    """Read the contents of a file as a Lisp expression.

    Keyword arguments to this function have the same meanings as those of
    basilisp.lang.reader.read."""
    with open(filename) as f:
        yield from read(
            f,
            resolver=resolver,
            data_readers=data_readers,
            eof=eof,
            is_eof_error=is_eof_error,
            features=features,
            process_reader_cond=process_reader_cond,
        )
