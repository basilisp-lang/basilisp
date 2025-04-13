# pylint: disable=too-many-branches,too-many-lines,too-many-return-statements

import collections
import contextlib
import decimal
import functools
import io
import os
import re
import uuid
from collections.abc import Collection, Iterable, MutableMapping, Sequence
from datetime import datetime
from fractions import Fraction
from itertools import chain
from re import Pattern
from types import TracebackType
from typing import Any, Callable, NoReturn, Optional, TypeVar, Union, cast

import attr
from typing_extensions import Unpack

from basilisp.lang import keyword as kw
from basilisp.lang import list as llist
from basilisp.lang import map as lmap
from basilisp.lang import queue as lqueue
from basilisp.lang import set as lset
from basilisp.lang import symbol as sym
from basilisp.lang import util as langutil
from basilisp.lang import vector as vec
from basilisp.lang.exception import format_exception
from basilisp.lang.interfaces import (
    ILispObject,
    ILookup,
    IMeta,
    IPersistentList,
    IPersistentMap,
    IPersistentSet,
    IPersistentVector,
    IRecord,
    ISeq,
    IType,
    IWithMeta,
)
from basilisp.lang.obj import PrintSettings
from basilisp.lang.obj import seq_lrepr as _seq_lrepr
from basilisp.lang.runtime import (
    READER_COND_DEFAULT_FEATURE_SET,
    Namespace,
    Var,
    get_current_ns,
    lrepr,
)
from basilisp.lang.source import format_source_context
from basilisp.lang.tagged import TaggedLiteral, tagged_literal
from basilisp.lang.typing import IterableLispForm, LispForm, ReaderForm
from basilisp.lang.util import munge
from basilisp.util import Maybe, partition

ns_name_chars = re.compile(r"\w|-|\+|\*|\?|/|\=|\\|!|&|%|>|<|\$|:|\.")
alphanumeric_chars = re.compile(r"\w")
begin_num_chars = re.compile(r"[0-9\-]")
maybe_num_chars = re.compile(r"[0-9A-Za-z/\.]")
integer_literal = re.compile(r"(-?(?:\d|[1-9]\d+))N?")
float_literal = re.compile(r"(-?(?:\d|[1-9]\d+)(?:\.\d*)?)M?")
complex_literal = re.compile(r"-?(\d+(?:\.\d*)?)J")
arbitrary_base_literal = re.compile(r"-?(\d{1,2})r([0-9A-Za-z]+)")
octal_literal = re.compile("-?0([0-7]+)N?")
hex_literal = re.compile("-?0[Xx]([0-9A-Fa-f]+)N?")
ratio_literal = re.compile(r"(-?\d+)/(\d+)")
scientific_notation_literal = re.compile(r"-?(\d+(?:\.\d*)?)[Ee](-?\d+)")
whitespace_chars = re.compile(r"[\s,]")
newline_chars = re.compile("(\r\n|\r|\n)")
fn_macro_args = re.compile("(%)(&|[0-9])?")
unicode_char = re.compile(r"u(\w+)")

DataReaderFn = Callable[[Any], Any]
DataReaders = lmap.PersistentMap[sym.Symbol, DataReaderFn]
GenSymEnvironment = MutableMapping[str, sym.Symbol]
Resolver = Callable[[sym.Symbol], sym.Symbol]
LispReaderFn = Callable[["ReaderContext"], LispForm]
W = TypeVar("W", bound=LispReaderFn)

READER_LINE_KW = kw.keyword("line", ns="basilisp.lang.reader")
READER_COL_KW = kw.keyword("col", ns="basilisp.lang.reader")
READER_END_LINE_KW = kw.keyword("end-line", ns="basilisp.lang.reader")
READER_END_COL_KW = kw.keyword("end-col", ns="basilisp.lang.reader")

READER_TAG_KW = kw.keyword("tag")
READER_PARAM_TAGS_KW = kw.keyword("param-tags")

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
DefaultDataReaderFn = Callable[[sym.Symbol, RawReaderForm], Any]


# pylint:disable=redefined-builtin
@attr.define(repr=False, str=False)
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
        keys: dict[str, Union[str, int]] = {}
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


@format_exception.register(SyntaxError)
def format_syntax_error(  # pylint: disable=unused-argument
    e: SyntaxError,
    tp: Optional[type[Exception]] = None,
    tb: Optional[TracebackType] = None,
    disable_color: Optional[bool] = None,
) -> list[str]:
    """If `disable_color` is True, no color formatting will be applied to the source
    code."""

    context_exc: Optional[BaseException] = e.__cause__

    lines = [os.linesep]
    if context_exc is not None:
        lines.append(f"  exception: {type(context_exc)} from {type(e)}{os.linesep}")
    else:
        lines.append(f"  exception: {type(e)}{os.linesep}")
    if context_exc is None:
        lines.append(f"    message: {e.message}{os.linesep}")
    else:
        lines.append(f"    message: {e.message}: {context_exc}{os.linesep}")

    if e.line is not None and e.col is not None:
        line_num = f"{e.line}:{e.col}"
    elif e.line is not None:
        line_num = str(e.line)
    else:
        line_num = ""

    if e.filename is not None:
        lines.append(
            f"   location: {e.filename}:{line_num or 'NO_SOURCE_LINE'}{os.linesep}"
        )
    elif line_num:
        lines.append(f"       line: {line_num}{os.linesep}")

    # Print context source lines around the error. Use the current exception to
    # derive source lines, but use the inner cause exception to place a marker
    # around the error.
    if (
        e.filename is not None
        and e.line is not None
        and (
            context_lines := format_source_context(
                e.filename, e.line, disable_color=disable_color
            )
        )
    ):
        lines.append(f"    context:{os.linesep}")
        lines.append(os.linesep)
        lines.extend(context_lines)

    return lines


class UnexpectedEOFError(SyntaxError):
    """Syntax Error type raised when the reader encounters an unexpected EOF
    reading a form.

    Useful for cases such as the REPL reader, where unexpected EOF errors
    likely indicate the user is trying to enter a multiline form."""


class StreamReader:
    """A simple stream reader with n-character lookahead."""

    DEFAULT_INDEX = -2

    __slots__ = ("_stream", "_pushback_depth", "_idx", "_buffer", "_line", "_col")

    def __init__(
        self,
        stream: io.TextIOBase,
        pushback_depth: int = 5,
        init_line: Optional[int] = None,
        init_column: Optional[int] = None,
    ) -> None:
        """`init_line` and `init_column` refer to where the `stream`
        starts in the broader context, defaulting to 1 and 0
        respectively if not provided."""
        init_line = init_line if init_line is not None else 1
        init_column = init_column if init_column is not None else 0
        self._stream = stream
        self._pushback_depth = pushback_depth
        self._idx = -2
        self._line = collections.deque([init_line], pushback_depth)
        self._col = collections.deque([init_column], pushback_depth)
        self._buffer = collections.deque([self._stream.read(1)], pushback_depth)

        # Load up an extra character
        self._buffer.append(self._stream.read(1))
        self._update_loc()

    @property
    def name(self) -> Optional[str]:
        return getattr(self._stream, "name", None)

    @property
    def col(self) -> int:
        """Return the column of the character returned by `peek`."""
        return self._col[self._idx]

    @property
    def line(self) -> int:
        """Return the line of the character returned by `peek`."""
        return self._line[self._idx]

    @property
    def loc(self) -> tuple[int, int]:
        """Return the location of the character returned by `peek` as a tuple of
        (line, col)."""
        return self.line, self.col

    def _update_loc(self):
        """Update the internal line and column buffers after a new character is
        added."""
        if self._buffer[-2] == "\n" or (
            self._buffer[-2] == "\r" and self._buffer[-1] != "\n"
        ):
            self._col.append(0)
            self._line.append(self._line[-1] + 1)
        else:
            self._col.append(self._col[-1] + 1)
            self._line.append(self._line[-1])

    def peek(self) -> str:
        """Peek at the next character in the stream."""
        return self._buffer[self._idx]

    def pushback(self) -> None:
        """Push one character back onto the stream, allowing it to be read again."""
        if abs(self._idx - 1) > self._pushback_depth:
            raise IndexError("Exceeded pushback depth")
        self._idx -= 1

    def advance(self) -> str:
        """Advance the current character pointer by one and return the previous
        character value from before advancing the counter.

        Equivalent to calling `peek`, then `next_char`, then returning the result of
        the previous `peek`."""
        cur = self.peek()
        self.next_char()
        return cur

    def next_char(self) -> str:
        """Advance the stream forward by one character and return the next character
        in the stream."""
        if self._idx < StreamReader.DEFAULT_INDEX:
            self._idx += 1
        else:
            c = self._stream.read(1)
            self._buffer.append(c)
            self._update_loc()
        return self.peek()


@functools.singledispatch
def _py_from_lisp(form: object) -> ReaderForm:
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
    except (ValueError, OverflowError) as e:
        raise SyntaxError(f"Unrecognized date/time syntax: {inst_str}") from e


def _uuid_from_str(uuid_str: str) -> uuid.UUID:
    try:
        return langutil.uuid_from_str(uuid_str)
    except (ValueError, TypeError) as e:
        raise SyntaxError(f"Unrecognized UUID format: {uuid_str}") from e


def _raise_unknown_tag(s: sym.Symbol, v: LispReaderForm) -> NoReturn:
    raise SyntaxError(f"No data reader found for tag #{s}")


class ReaderContext:
    _DATA_READERS: DataReaders = lmap.map(
        {
            sym.symbol("inst"): _inst_from_str,
            sym.symbol("py"): _py_from_lisp,
            sym.symbol("queue"): lqueue.queue,
            sym.symbol("uuid"): _uuid_from_str,
        }
    )

    __slots__ = (
        "_data_readers",
        "_default_data_reader_fn",
        "_features",
        "_process_reader_cond",
        "_process_tagged_literals",
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
        resolver: Optional[Resolver] = None,
        data_readers: Optional[DataReaders] = None,
        eof: Any = None,
        features: Optional[IPersistentSet[kw.Keyword]] = None,
        process_reader_cond: bool = True,
        default_data_reader_fn: Optional[DefaultDataReaderFn] = None,
    ) -> None:
        self._data_readers = Maybe(data_readers).or_else_get(lmap.EMPTY)
        self._default_data_reader_fn = Maybe(default_data_reader_fn).or_else_get(
            _raise_unknown_tag
        )
        self._features = Maybe(features).or_else_get(READER_COND_DEFAULT_FEATURE_SET)
        self._process_reader_cond = process_reader_cond
        self._reader = reader
        self._resolve = Maybe(resolver).or_else_get(lambda x: x)
        self._process_tagged_literals: collections.deque[bool] = collections.deque([])
        self._in_anon_fn: collections.deque[bool] = collections.deque([])
        self._syntax_quoted: collections.deque[bool] = collections.deque([])
        self._gensym_env: collections.deque[GenSymEnvironment] = collections.deque([])
        self._eof = eof

    @property
    def data_readers(self) -> DataReaders:
        return self._data_readers

    @property
    def default_data_reader_fn(self) -> DefaultDataReaderFn:
        return self._default_data_reader_fn

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

    @contextlib.contextmanager
    def process_tagged_literals(self, v: bool):
        self._process_tagged_literals.append(v)
        yield
        self._process_tagged_literals.pop()

    @property
    def should_process_tagged_literals(self) -> bool:
        try:
            return self._process_tagged_literals[-1] is True
        except IndexError:
            return True

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
        form: llist.PersistentList[tuple[kw.Keyword, ReaderForm]],
        is_splicing: bool = False,
    ):
        self._form = form
        self._feature_vec = self._compile_feature_vec(form)
        self._is_splicing = is_splicing

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ReaderConditional):
            return NotImplemented
        return self._form == other._form and self._is_splicing == other._is_splicing

    @staticmethod
    def _compile_feature_vec(form: IPersistentList[tuple[kw.Keyword, ReaderForm]]):
        found_features: set[kw.Keyword] = set()
        feature_list: list[tuple[kw.Keyword, ReaderForm]] = []

        try:
            for k, v in partition(
                cast(Sequence[tuple[kw.Keyword, ReaderForm]], form), 2
            ):
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
        except ValueError as e:
            raise SyntaxError(
                "Reader conditionals must contain an even number of forms"
            ) from e

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

    def _lrepr(self, **kwargs: Unpack[PrintSettings]) -> str:
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
        v = f(ctx, **kwargs)
        end_line, end_col = ctx.reader.line, ctx.reader.col
        if isinstance(v, IWithMeta):
            new_meta = lmap.map(
                {
                    READER_LINE_KW: line,
                    READER_COL_KW: col,
                    READER_END_LINE_KW: end_line,
                    READER_END_COL_KW: end_col,
                }
            )
            old_meta = v.meta
            return v.with_meta(
                old_meta.cons(new_meta) if old_meta is not None else new_meta
            )
        else:
            return v

    return cast(W, with_lineno_and_col)


def _read_namespaced(
    ctx: ReaderContext, allowed_suffix: Optional[str] = None
) -> tuple[Optional[str], str]:
    """Read a namespaced token (keyword or symbol) from the input stream."""
    ns: list[str] = []
    name: list[str] = []
    reader = ctx.reader
    has_ns = False
    while True:
        char = reader.peek()
        if char == "/":
            reader.next_char()
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
        elif ns_name_chars.match(char) or (name and char == "'") or char == "#":
            reader.next_char()
            name.append(char)
        elif allowed_suffix is not None and char == allowed_suffix:
            reader.next_char()
            name.append(char)
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
    end_char: str,
    coll_name: str,
):
    """Read a collection from the input stream and create the
    collection using f."""
    coll: list = []
    reader = ctx.reader
    while True:
        char = reader.peek()
        if char == "":
            raise ctx.eof_error(f"Unexpected EOF in {coll_name}")
        if whitespace_chars.match(char):
            reader.advance()
            continue
        if char == end_char:
            reader.next_char()
            return f(coll)
        elem = _read_next(ctx)
        if elem is COMMENT or isinstance(elem, Comment):
            continue
        elif _should_splice_reader_conditional(ctx, elem):
            assert isinstance(elem, ReaderConditional)
            selected_feature = _select_reader_conditional_branch(ctx, elem)
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
        char = reader.peek()
        if char == "":
            raise ctx.eof_error("Unexpected EOF in map}")
        if whitespace_chars.match(char):
            reader.advance()
            continue
        if char == "}":
            reader.next_char()
            return
        v = _read_next(ctx)
        if v is COMMENT or isinstance(v, Comment):
            continue
        elif _should_splice_reader_conditional(ctx, v):
            assert isinstance(v, ReaderConditional)
            selected_feature = _select_reader_conditional_branch(ctx, v)
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
) -> Callable[[Any], Any]:
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
        # pylint: disable=redefined-loop-name
        for k, v in partition(list(__read_map_elems(ctx)), 2):
            k = process_key(k)
            try:
                if k in d:
                    raise ctx.syntax_error(f"Duplicate key '{k}' in map literal")  # type: ignore[str-bytes-safe]
            except TypeError as e:
                raise ctx.syntax_error("Map keys must be hashable") from e
            else:
                d[k] = v
    except ValueError as e:
        raise ctx.syntax_error("Unexpected char '}'; expected map value") from e
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

    char = ctx.reader.peek()
    while whitespace_chars.match(char):
        char = ctx.reader.next_char()

    return _read_map(ctx, namespace=map_ns)


# Due to some ambiguities that arise in parsing symbols, numbers, and the
# special keywords `true`, `false`, and `nil`, we have to have a looser
# type defined for the return from these reader functions.
MaybeSymbol = Union[bool, None, sym.Symbol]
MaybeNumber = Union[complex, decimal.Decimal, float, Fraction, int, MaybeSymbol]


def _read_num(  # noqa: C901  # pylint: disable=too-many-locals,too-many-statements
    ctx: ReaderContext,
) -> MaybeNumber:
    """Return a numeric (complex, Decimal, float, int, Fraction) from the input stream."""
    chars: list[str] = []
    reader = ctx.reader

    while True:
        char = reader.peek()
        if char == "-":
            following_char = reader.next_char()
            if not begin_num_chars.match(following_char):
                reader.pushback()
                try:
                    for _ in chars:
                        reader.pushback()
                except IndexError as e:
                    raise ctx.syntax_error(
                        "Requested to pushback too many characters onto StreamReader"
                    ) from e
                return _read_sym(ctx)
            chars.append(char)
            continue
        elif not maybe_num_chars.match(char):
            break
        reader.next_char()
        chars.append(char)

    assert len(chars) > 0, "Must have at least one digit in number"

    s = "".join(chars)
    neg = s.startswith("-")

    if (match := integer_literal.fullmatch(s)) is not None:
        return int(match.group(1))
    elif (match := float_literal.fullmatch(s)) is not None:
        if s.endswith("M"):
            try:
                return decimal.Decimal(match.group(1))
            except decimal.InvalidOperation:  # pragma: no cover
                raise ctx.syntax_error(f"Invalid number format: {s}") from None
        else:
            return float(match.group(1))
    elif (match := octal_literal.fullmatch(s)) is not None:
        v = int(match.group(1), base=8)
        return -v if neg else v
    elif (match := hex_literal.fullmatch(s)) is not None:
        v = int(match.group(1), base=16)
        return -v if neg else v
    elif (match := ratio_literal.fullmatch(s)) is not None:
        num, denominator = match.groups()
        if (numerator := int(num)) == 0:
            return 0
        try:
            return Fraction(numerator=numerator, denominator=int(denominator))
        except ZeroDivisionError as e:
            raise ctx.syntax_error(f"Invalid ratio format: {s}") from e
    elif (match := scientific_notation_literal.fullmatch(s)) is not None:
        sig = float(m) if "." in (m := match.group(1)) else int(m)
        exp = int(match.group(2))
        res = sig * (10**exp)
        return -res if neg else res
    elif (match := arbitrary_base_literal.fullmatch(s)) is not None:
        base = int(match.group(1))
        if not 2 <= base <= 36:
            raise ctx.syntax_error(
                f"Invalid base {base} for integer literal {s}: must be between 2 and 36"
            )
        try:
            v = int(match.group(2), base=base)
        except ValueError as e:
            raise ctx.syntax_error(f"Invalid number format: {s}") from e
        else:
            return -v if neg else v
    elif (match := complex_literal.fullmatch(s)) is not None:
        imaginary_raw = match.group(1)
        imaginary = float(imaginary_raw) if "." in imaginary_raw else int(imaginary_raw)
        return complex(0, -imaginary if neg else imaginary)
    raise ctx.syntax_error(f"Invalid number format: {s}")


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
    """Return a UTF-8 encoded string from the input stream.

    If allow_arbitrary_escapes is True, do not throw a SyntaxError if an
    unknown escape sequence is encountered."""
    s: list[str] = []
    reader = ctx.reader
    while True:
        char = reader.next_char()
        if char == "":
            raise ctx.eof_error("Unexpected EOF in string")
        if char == "\\":
            char = reader.next_char()
            escape_char = _STR_ESCAPE_CHARS.get(char, None)
            if escape_char:
                s.append(escape_char)
                continue
            if allow_arbitrary_escapes:
                s.append("\\")
            else:
                raise ctx.syntax_error(f"Unknown escape sequence: \\{char}")
        if char == '"':
            reader.next_char()
            return "".join(s)
        s.append(char)


_BYTES_ESCAPE_CHARS = {
    '"': b'"',
    "\\": b"\\",
    "a": b"\a",
    "b": b"\b",
    "f": b"\f",
    "n": b"\n",
    "r": b"\r",
    "t": b"\t",
    "v": b"\v",
}


def _read_hex_byte(ctx: ReaderContext) -> bytes:
    """Read a byte with a 2 digit hex code such as `\\xff`."""
    reader = ctx.reader
    c1 = reader.next_char()
    c2 = reader.next_char()
    try:
        return bytes([int("".join(["0x", c1, c2]), base=16)])
    except ValueError as e:
        raise ctx.syntax_error(
            f"Invalid byte representation for base 16: 0x{c1}{c2}"
        ) from e


def _read_byte_str(ctx: ReaderContext) -> bytes:
    """Return a byte string from the input stream.

    Byte strings have the same restrictions and semantics as byte literals in Python.
    Individual characters must be within the ASCII range or must be valid escape sequences.
    """
    reader = ctx.reader

    char = reader.peek()
    while whitespace_chars.match(char):
        char = reader.next_char()

    if char != '"':
        raise ctx.syntax_error(f"Expected '\"'; got '{char}' instead")

    b: list[bytes] = []
    while True:
        char = reader.next_char()
        if char == "":
            raise ctx.eof_error("Unexpected EOF in byte string")
        if ord(char) < 1 or ord(char) > 127:
            raise ctx.eof_error("Byte strings must contain only ASCII characters")
        if char == "\\":
            char = reader.next_char()
            escape_char = _BYTES_ESCAPE_CHARS.get(char, None)
            if escape_char:
                b.append(escape_char)
                continue
            elif char == "x":
                b.append(_read_hex_byte(ctx))
                continue
            else:
                # In Python, invalid escape sequences entered into byte strings are
                # retained with backslash for debugging purposes, so we do the same.
                b.append(b"\\")
                b.append(char.encode("utf-8"))
                continue
        if char == '"':
            reader.next_char()
            return b"".join(b)
        b.append(char.encode("utf-8"))


@_with_loc
def _read_sym(ctx: ReaderContext, is_reader_macro_sym: bool = False) -> MaybeSymbol:
    """Return a symbol from the input stream.

    If a symbol appears in a syntax quoted form, the reader will attempt
    to resolve the symbol using the resolver in the ReaderContext `ctx`.
    The resolver will look into the current namespace for an alias or
    namespace matching the symbol's namespace. If no namespace is specififed
    for the symbol, it will be assigned to the current namespace, unless the
    symbol is `&`."""
    ns, name = _read_namespaced(ctx, allowed_suffix="#")
    if not ctx.is_syntax_quoted and name.endswith("#"):
        raise ctx.syntax_error("Gensym may not appear outside syntax quote")
    if ns is not None:
        if any(map(lambda s: len(s) == 0, ns.split("."))):
            raise ctx.syntax_error(
                "All '.' separated segments of a namespace "
                "must contain at least one character."
            )
    if ns is None:
        if name == "nil":
            return None
        elif name == "true":
            return True
        elif name == "false":
            return False
        elif name == "&":
            return _AMPERSAND
        elif name.startswith("."):
            return sym.symbol(name)
    if ctx.is_syntax_quoted and not name.endswith("#") and not is_reader_macro_sym:
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
        meta_map = lmap.map({READER_TAG_KW: meta})
    elif isinstance(meta, kw.Keyword):
        meta_map = lmap.map({meta: True})
    elif isinstance(meta, lmap.PersistentMap):
        meta_map = meta
    elif isinstance(meta, vec.PersistentVector):
        meta_map = lmap.map({READER_PARAM_TAGS_KW: meta})
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


@functools.singledispatch
def _walk(form, _, outer_f):
    """Walk an arbitrary, possibly nested data structure, applying inner_f to each
    element of form and then applying outer_f to the resulting form."""
    return outer_f(form)


@_walk.register(IPersistentList)
@_walk.register(ISeq)
def _walk_ipersistentlist(form: Union[IPersistentList, ISeq], inner_f, outer_f):
    coll = llist.list(map(inner_f, form))
    if isinstance(form, IMeta) and form.meta is not None:
        coll = coll.with_meta(form.meta)
    return outer_f(coll)


@_walk.register(IPersistentVector)
def _walk_ipersistentvector(form: IPersistentVector, inner_f, outer_f):
    coll = vec.vector(map(inner_f, form))
    if isinstance(form, IMeta) and form.meta is not None:
        coll = coll.with_meta(form.meta)
    return outer_f(coll)


@_walk.register(IPersistentMap)
def _walk_ipersistentmap(form: IPersistentMap, inner_f, outer_f):
    coll = lmap.hash_map(*chain.from_iterable(map(inner_f, form.seq() or ())))
    if isinstance(form, IMeta) and form.meta is not None:
        coll = coll.with_meta(form.meta)
    return outer_f(coll)


@_walk.register(IPersistentSet)
def _walk_ipersistentset(form: IPersistentSet, inner_f, outer_f):
    coll = lset.set(map(inner_f, form))
    if isinstance(form, IMeta) and form.meta is not None:
        coll = coll.with_meta(form.meta)
    return outer_f(coll)


def _postwalk(f, form):
    """Walk form using depth-first, post-order traversal, applying f to each form
    and replacing form with its result."""
    inner_f = functools.partial(_postwalk, f)
    return _walk(form, inner_f, f)


@_with_loc
def _read_function(ctx: ReaderContext) -> llist.PersistentList:
    """Read a function reader macro from the input stream."""
    if ctx.is_in_anon_fn:
        raise ctx.syntax_error("Nested #() definitions not allowed")

    current_ns = get_current_ns()

    with ctx.in_anon_fn():
        form = _read_list(ctx)
    arg_set = set()

    def arg_suffix(arg_num: Optional[str]) -> str:
        if arg_num is None:
            return "1"
        elif arg_num == "&":
            return "rest"
        else:
            return arg_num

    def sym_replacement(arg_num: Optional[str]) -> sym.Symbol:
        suffix = arg_suffix(arg_num)
        if ctx.is_syntax_quoted:
            suffix = f"{suffix}#"
        return sym.symbol(f"arg-{suffix}")

    def identify_and_replace(f):
        if isinstance(f, sym.Symbol):
            # Checking against the current namespace is generally only used for
            # when anonymous function definitions are syntax quoted. Arguments
            # are resolved in terms of the current namespace, so we simply check
            # if the symbol namespace matches the current runtime namespace.
            if f.ns is None or f.ns == current_ns.name:
                match = fn_macro_args.match(f.name)
                if match is not None:
                    arg_num = match.group(2)
                    suffix = arg_suffix(arg_num)
                    arg_set.add(suffix)
                    return sym_replacement(arg_num)
        return f

    body = _postwalk(identify_and_replace, form) if len(form) > 0 else None

    arg_list: list[sym.Symbol] = []
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
      - \\a \\$ \\[ etc will yield 'a', '$', and '[' respectively

      - \\newline, \\space, \\tab, \\formfeed, \\backspace, \\return yield
        the named characters

      - \\uXXXX yield the unicode digit corresponding to the code
        point named by the hex digits XXXX"""
    start = ctx.reader.advance()
    assert start == "\\"

    s: list[str] = []
    reader = ctx.reader
    char = reader.peek()
    is_first_char = True
    while True:
        if char == "" or (not is_first_char and not alphanumeric_chars.match(char)):
            break
        s.append(char)
        char = reader.next_char()
        is_first_char = False

    character = "".join(s)
    special = _SPECIAL_CHARS.get(character, None)
    if special is not None:
        return special

    match = unicode_char.match(character)
    if match is not None:
        try:
            return chr(int(f"0x{match.group(1)}", 16))
        except (ValueError, OverflowError):
            raise ctx.syntax_error(f"Unsupported character \\u{character}") from None

    if len(character) > 1:
        raise ctx.syntax_error(f"Unsupported character \\{character}")

    return character


def _read_regex(ctx: ReaderContext) -> Pattern:
    """Read a regex reader macro from the input stream."""
    s = _read_str(ctx, allow_arbitrary_escapes=True)
    try:
        return langutil.regex_from_str(s)
    except re.error as e:
        raise ctx.syntax_error(f"Unrecognized regex pattern syntax: {s}") from e


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


def _select_reader_conditional_branch(
    ctx: ReaderContext, reader_cond: ReaderConditional
) -> LispReaderForm:
    """Select the reader conditional branch by feature and then resolve any tagged
    literals for the selected feature."""

    def resolve_tagged_literals(form: LispReaderForm):
        if isinstance(form, TaggedLiteral):
            resolved = _postwalk(resolve_tagged_literals, form.form)
            return _resolve_tagged_literal(ctx, form.tag, resolved)
        return form

    return _postwalk(
        resolve_tagged_literals, reader_cond.select_feature(ctx.reader_features)
    )


def _should_splice_reader_conditional(ctx: ReaderContext, form: LispReaderForm) -> bool:
    """Return True if and only if form is a ReaderConditional which should be spliced
    into a surrounding collection context."""
    return (
        isinstance(form, ReaderConditional)
        and ctx.should_process_reader_cond
        and form.is_splicing
    )


def _read_reader_conditional_preserving(
    ctx: ReaderContext, is_splicing: bool
) -> ReaderConditional:
    """Read a reader conditional form and return the reader conditional object."""
    coll: list = []
    reader = ctx.reader
    while True:
        char = reader.peek()
        if char == "":
            raise ctx.eof_error("Unexpected EOF in reader conditional")
        if whitespace_chars.match(char):
            reader.advance()
            continue
        if char == ")":
            reader.next_char()
            return ReaderConditional(llist.list(coll), is_splicing=is_splicing)

        with ctx.process_tagged_literals(False):
            elem = _read_next(ctx)

        if elem is COMMENT or isinstance(elem, Comment):
            continue
        elif _should_splice_reader_conditional(ctx, elem):
            assert isinstance(elem, ReaderConditional)
            selected_feature = _select_reader_conditional_branch(ctx, elem)
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
    reader = ctx.reader
    start = reader.advance()
    assert start == "?"
    char = reader.peek()

    if char == "@":
        is_splicing = True
        ctx.reader.advance()
    elif char == "(":
        is_splicing = False
    else:
        raise ctx.syntax_error(
            f"Unexpected char '{char}'; expected opening '(' for reader conditional"
        )

    open_char = reader.advance()
    if open_char != "(":
        raise ctx.syntax_error(
            f"Expected opening '(' for reader conditional; got '{open_char}'"
        )

    reader_cond = _read_reader_conditional_preserving(ctx, is_splicing)
    if ctx.should_process_reader_cond and not reader_cond.is_splicing:
        form = _select_reader_conditional_branch(ctx, reader_cond)
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


def _resolve_tagged_literal(
    ctx: ReaderContext, s: sym.Symbol, v: RawReaderForm
) -> LispReaderForm:
    """Resolve a tagged literal into whatever value is returned by the associated data reader."""
    data_reader = None
    if s in ctx.data_readers:
        data_reader = ctx.data_readers[s]
    elif s in ReaderContext._DATA_READERS:
        data_reader = ReaderContext._DATA_READERS[s]

    if data_reader is not None:
        try:
            return data_reader(v)
        except SyntaxError as e:
            raise ctx.syntax_error(e.message).with_traceback(e.__traceback__) from None
    elif s.ns is None and "." in s.name:
        return _load_record_or_type(ctx, s, v)
    else:
        try:
            return ctx.default_data_reader_fn(s, v)
        except SyntaxError as e:
            raise ctx.syntax_error(e.message).with_traceback(e.__traceback__) from None


def _read_reader_macro(ctx: ReaderContext) -> LispReaderForm:
    """Return a data structure evaluated as a reader macro from the input stream."""
    start = ctx.reader.advance()
    assert start == "#"
    char = ctx.reader.peek()
    if char == "{":
        return _read_set(ctx)
    elif char == "(":
        return _read_function(ctx)
    elif char == ":":
        return _read_namespaced_map(ctx)
    elif char == "'":
        ctx.reader.advance()
        char_next = ctx.reader.peek()
        if char_next == "~":
            s = _read_unquote(ctx)
        else:
            s = _read_sym(ctx)
        return llist.l(_VAR, s)
    elif char == '"':
        return _read_regex(ctx)
    elif char == "_":
        ctx.reader.advance()
        _read_next_consuming_comment(ctx)  # Ignore the entire next form
        return COMMENT
    elif char == "!":
        return _read_comment(ctx)
    elif char == "?":
        try:
            return _read_reader_conditional(ctx)
        except SyntaxError as e:
            raise ctx.syntax_error(e.message).with_traceback(e.__traceback__) from None
    elif char == "#":
        return _read_numeric_constant(ctx)
    elif ns_name_chars.match(char):
        s = _read_sym(ctx, is_reader_macro_sym=True)
        assert isinstance(s, sym.Symbol)
        if s.ns is None and s.name == "b":
            return _read_byte_str(ctx)

        v = _read_next_consuming_comment(ctx)

        if not ctx.should_process_tagged_literals:
            return tagged_literal(s, v)

        return _resolve_tagged_literal(ctx, s, v)

    raise ctx.syntax_error(f"Unexpected char '{char}' in reader macro")


def _read_comment(ctx: ReaderContext) -> LispReaderForm:
    """Read (and ignore) a single-line comment from the input stream.
    Return the next form after the next line break."""
    reader = ctx.reader
    start = reader.advance()
    assert start in {";", "!"}
    while True:
        char = reader.peek()
        if newline_chars.match(char):
            reader.advance()
            return COMMENT
        if char == "":
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
    char = reader.peek()
    while whitespace_chars.match(char):
        char = reader.next_char()
    return _read_next(ctx)


def _read_next(ctx: ReaderContext) -> LispReaderForm:  # noqa: C901
    """Read the next full form from the input stream."""
    reader = ctx.reader
    char = reader.peek()
    if char == "(":
        return _read_list(ctx)
    elif char == "[":
        return _read_vector(ctx)
    elif char == "{":
        return _read_map(ctx)
    elif begin_num_chars.match(char):
        return _read_num(ctx)
    elif whitespace_chars.match(char):
        return _read_next_consuming_whitespace(ctx)
    elif char == ":":
        return _read_kw(ctx)
    elif char == '"':
        return _read_str(ctx)
    elif char == "'":
        return _read_quoted(ctx)
    elif char == "\\":
        return _read_character(ctx)
    elif ns_name_chars.match(char):
        return _read_sym(ctx)
    elif char == "#":
        return _read_reader_macro(ctx)
    elif char == "^":
        return _read_meta(ctx)  # type: ignore
    elif char == ";":
        return _read_comment(ctx)
    elif char == "`":
        return _read_syntax_quoted(ctx)
    elif char == "~":
        return _read_unquote(ctx)
    elif char == "@":
        return _read_deref(ctx)
    elif char == "":
        return ctx.eof
    else:
        raise ctx.syntax_error(f"Unexpected char '{char}'")


def syntax_quote(  # pylint: disable=too-many-arguments
    form: RawReaderForm,
    resolver: Optional[Resolver] = None,
    data_readers: Optional[DataReaders] = None,
    eof: Any = EOF,
    features: Optional[IPersistentSet[kw.Keyword]] = None,
    process_reader_cond: bool = True,
    default_data_reader_fn: Optional[DefaultDataReaderFn] = None,
):
    """Return the syntax quoted version of a form."""
    # the buffer is unused here, but is necessary to create a ReaderContext
    with io.StringIO("") as buf:
        ctx = ReaderContext(
            StreamReader(buf),
            resolver=resolver,
            data_readers=data_readers,
            eof=eof,
            features=features,
            process_reader_cond=process_reader_cond,
            default_data_reader_fn=default_data_reader_fn,
        )
        return _process_syntax_quoted_form(ctx, form)


def read(  # pylint: disable=too-many-arguments
    stream,
    resolver: Optional[Resolver] = None,
    data_readers: Optional[DataReaders] = None,
    eof: Any = EOF,
    is_eof_error: bool = False,
    features: Optional[IPersistentSet[kw.Keyword]] = None,
    process_reader_cond: bool = True,
    default_data_reader_fn: Optional[DefaultDataReaderFn] = None,
    init_line: Optional[int] = None,
    init_column: Optional[int] = None,
) -> Iterable[RawReaderForm]:
    """Read the contents of a stream as a Lisp expression.

    The optional `init_line` and `init_column` specify where the
    `stream` location metadata starts in the broader context, if not
    from the start.

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
    reader = StreamReader(stream, init_line=init_line, init_column=init_column)
    ctx = ReaderContext(
        reader,
        resolver=resolver,
        data_readers=data_readers,
        eof=eof,
        features=features,
        process_reader_cond=process_reader_cond,
        default_data_reader_fn=default_data_reader_fn,
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
    resolver: Optional[Resolver] = None,
    data_readers: Optional[DataReaders] = None,
    eof: Any = EOF,
    is_eof_error: bool = False,
    features: Optional[IPersistentSet[kw.Keyword]] = None,
    process_reader_cond: bool = True,
    default_data_reader_fn: Optional[DefaultDataReaderFn] = None,
    init_line: Optional[int] = None,
    init_column: Optional[int] = None,
) -> Iterable[RawReaderForm]:
    """Read the contents of a string as a Lisp expression.

    The optional `init_line` and `init_column` specify where the
    `stream` location metadata starts in the broader context, if not
    from the beginning.

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
            default_data_reader_fn=default_data_reader_fn,
            init_line=init_line,
            init_column=init_column,
        )


def read_file(  # pylint: disable=too-many-arguments
    filename: str,
    resolver: Optional[Resolver] = None,
    data_readers: Optional[DataReaders] = None,
    eof: Any = EOF,
    is_eof_error: bool = False,
    features: Optional[IPersistentSet[kw.Keyword]] = None,
    process_reader_cond: bool = True,
    default_data_reader_fn: Optional[DefaultDataReaderFn] = None,
) -> Iterable[RawReaderForm]:
    """Read the contents of a file as a Lisp expression.

    Keyword arguments to this function have the same meanings as those of
    basilisp.lang.reader.read."""
    with open(filename, encoding="utf-8") as f:
        yield from read(
            f,
            resolver=resolver,
            data_readers=data_readers,
            eof=eof,
            is_eof_error=is_eof_error,
            features=features,
            process_reader_cond=process_reader_cond,
            default_data_reader_fn=default_data_reader_fn,
        )
