import argparse
import importlib.metadata
import io
import os
import pathlib
import sys
import textwrap
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Optional, Union

from basilisp import main as basilisp
from basilisp.lang import compiler as compiler
from basilisp.lang import keyword as kw
from basilisp.lang import list as llist
from basilisp.lang import map as lmap
from basilisp.lang import reader as reader
from basilisp.lang import runtime as runtime
from basilisp.lang import symbol as sym
from basilisp.lang import vector as vec
from basilisp.lang.exception import print_exception
from basilisp.lang.typing import CompilerOpts
from basilisp.lang.util import munge
from basilisp.prompt import get_prompter

CLI_INPUT_FILE_PATH = "<CLI Input>"
REPL_INPUT_FILE_PATH = "<REPL Input>"
REPL_NS = "basilisp.repl"
NREPL_SERVER_NS = "basilisp.contrib.nrepl-server"
STDIN_INPUT_FILE_PATH = "<stdin>"
STDIN_FILE_NAME = "-"

BOOL_TRUE = frozenset({"true", "t", "1", "yes", "y"})
BOOL_FALSE = frozenset({"false", "f", "0", "no", "n"})

DEFAULT_COMPILER_OPTS = {k.name: v for k, v in compiler.compiler_opts().items()}


def eval_stream(stream, ctx: compiler.CompilerContext, ns: runtime.Namespace):
    """Evaluate the forms in stdin into a Python module AST node."""
    last = None
    for form in reader.read(stream, resolver=runtime.resolve_alias):
        assert not isinstance(form, reader.ReaderConditional)
        last = compiler.compile_and_exec_form(form, ctx, ns)
    return last


def eval_str(s: str, ctx: compiler.CompilerContext, ns: runtime.Namespace, eof: Any):
    """Evaluate the forms in a string into a Python module AST node."""
    last = eof
    for form in reader.read_str(s, resolver=runtime.resolve_alias, eof=eof):
        assert not isinstance(form, reader.ReaderConditional)
        last = compiler.compile_and_exec_form(form, ctx, ns)
    return last


def eval_file(filename: str, ctx: compiler.CompilerContext, ns: runtime.Namespace):
    """Evaluate a file with the given name into a Python module AST node."""
    if (path := Path(filename)).exists():
        return compiler.load_file(path, ctx, ns)
    else:
        raise FileNotFoundError(f"Error: The file {filename} does not exist.")


def bootstrap_repl(ctx: compiler.CompilerContext, which_ns: str) -> types.ModuleType:
    """Bootstrap the REPL with a few useful vars and returned the bootstrapped
    module so it's functions can be used by the REPL command."""
    which_ns_sym = sym.symbol(which_ns)
    ns = runtime.Namespace.get_or_create(which_ns_sym)
    compiler.compile_and_exec_form(
        llist.l(
            sym.symbol("ns", ns=runtime.CORE_NS),
            which_ns_sym,
            llist.l(kw.keyword("use"), sym.symbol(REPL_NS)),
        ),
        ctx,
        ns,
    )
    return importlib.import_module(REPL_NS)


def init_path(args: argparse.Namespace, unsafe_path: str = "") -> None:
    """Prepend any import group arguments to `sys.path`, including `unsafe_path` (which
    defaults to the empty string) if --include-unsafe-path is specified."""

    def prepend_once(path: str) -> None:
        if path in sys.path:
            return
        sys.path.insert(0, path)

    for pth in args.include_path or []:
        p = pathlib.Path(pth).resolve()
        prepend_once(str(p))

    if args.include_unsafe_path:
        prepend_once(unsafe_path)


def _to_bool(v: Optional[str]) -> Optional[bool]:
    """Coerce a string argument to a boolean value, if possible."""
    if v is None:
        return v
    elif v.lower() in BOOL_TRUE:
        return True
    elif v.lower() in BOOL_FALSE:
        return False
    else:
        raise argparse.ArgumentTypeError("Unable to coerce flag value to boolean.")


def _set_envvar_action(
    var: str, parent: type[argparse.Action] = argparse.Action
) -> type[argparse.Action]:
    """Return an argparse.Action instance (deriving from `parent`) that sets the value
    as the default value of the environment variable `var`."""

    class EnvVarSetterAction(parent):  # type: ignore
        def __call__(  # pylint: disable=signature-differs
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: Any,
            option_string: str,
        ):
            os.environ.setdefault(var, str(values))

    return EnvVarSetterAction


def _add_compiler_arg_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "compiler arguments",
        description=(
            "The compiler arguments below can be used to tweak warnings emitted by the "
            "compiler during compilation and in some cases, tweak emitted code. Note "
            "that Basilisp, like Python, aggressively caches compiled namespaces so "
            "you may need to disable namespace caching or modify your file to see the "
            "compiler argument changes take effect."
        ),
    )
    group.add_argument(
        "--generate-auto-inlines",
        action="store",
        nargs="?",
        const=os.getenv("BASILISP_GENERATE_AUTO_INLINES"),
        type=_to_bool,
        help=(
            "if true, the compiler will attempt to generate inline function defs "
            "for functions with a boolean `^:inline` meta key (env: "
            "BASILISP_GENERATE_AUTO_INLINES; default: "
            f"{DEFAULT_COMPILER_OPTS['generate-auto-inlines']})"
        ),
    )
    group.add_argument(
        "--inline-functions",
        action="store",
        nargs="?",
        const=os.getenv("BASILISP_INLINE_FUNCTIONS"),
        type=_to_bool,
        help=(
            "if true, the compiler will attempt to inline functions with an `^:inline` "
            "function definition at their invocation site (env: "
            "BASILISP_INLINE_FUNCTIONS; default: "
            f"{DEFAULT_COMPILER_OPTS['inline-functions']})"
        ),
    )
    group.add_argument(
        "--warn-on-arity-mismatch",
        action="store",
        nargs="?",
        const=os.getenv("BASILISP_WARN_ON_ARITY_MISMATCH"),
        type=_to_bool,
        help=(
            "if true, emit warnings if a Basilisp function invocation is detected with "
            "an unsupported number of arguments "
            "(env: BASILISP_WARN_ON_ARITY_MISMATCH; default: "
            f"{DEFAULT_COMPILER_OPTS['warn-on-arity-mismatch']})"
        ),
    )
    group.add_argument(
        "--warn-on-shadowed-name",
        action="store",
        nargs="?",
        const=os.getenv("BASILISP_WARN_ON_SHADOWED_NAME"),
        type=_to_bool,
        help=(
            "if true, emit warnings if a local name is shadowed by another local "
            "name (env: BASILISP_WARN_ON_SHADOWED_NAME; default: "
            f"{DEFAULT_COMPILER_OPTS['warn-on-shadowed-name']})"
        ),
    )
    group.add_argument(
        "--warn-on-shadowed-var",
        action="store",
        nargs="?",
        const=os.getenv("BASILISP_WARN_ON_SHADOWED_VAR"),
        type=_to_bool,
        help=(
            "if true, emit warnings if a Var name is shadowed by a local name "
            "(env: BASILISP_WARN_ON_SHADOWED_VAR; default: "
            f"{DEFAULT_COMPILER_OPTS['warn-on-shadowed-var']})"
        ),
    )
    group.add_argument(
        "--warn-on-unused-names",
        action="store",
        nargs="?",
        const=os.getenv("BASILISP_WARN_ON_UNUSED_NAMES"),
        type=_to_bool,
        help=(
            "if true, emit warnings if a local name is bound and unused "
            "(env: BASILISP_WARN_ON_UNUSED_NAMES; default: "
            f"{DEFAULT_COMPILER_OPTS['warn-on-unused-names']})"
        ),
    )
    group.add_argument(
        "--warn-on-non-dynamic-set",
        action="store",
        nargs="?",
        const=os.getenv("BASILISP_WARN_ON_NON_DYNAMIC_SET"),
        type=_to_bool,
        help=(
            "if true, emit warnings if the compiler detects an attempt to set! "
            "a Var which is not marked as ^:dynamic (env: "
            "BASILISP_WARN_ON_NON_DYNAMIC_SET; default: "
            f"{DEFAULT_COMPILER_OPTS['warn-on-non-dynamic-set']})"
        ),
    )
    group.add_argument(
        "--use-var-indirection",
        action="store",
        nargs="?",
        const=os.getenv("BASILISP_USE_VAR_INDIRECTION"),
        type=_to_bool,
        help=(
            "if true, all Var accesses will be performed via Var indirection "
            "(env: BASILISP_USE_VAR_INDIRECTION; default: "
            f"{DEFAULT_COMPILER_OPTS['use-var-indirection']})"
        ),
    )
    group.add_argument(
        "--warn-on-var-indirection",
        action="store",
        nargs="?",
        const=os.getenv("BASILISP_WARN_ON_VAR_INDIRECTION"),
        type=_to_bool,
        help=(
            "if true, emit warnings if a Var reference cannot be direct linked "
            "(env: BASILISP_WARN_ON_VAR_INDIRECTION; default: "
            f"{DEFAULT_COMPILER_OPTS['warn-on-var-indirection']})"
        ),
    )


def _compiler_opts(args: argparse.Namespace) -> CompilerOpts:
    return compiler.compiler_opts(
        generate_auto_inlines=args.generate_auto_inlines,
        inline_functions=args.inline_functions,
        warn_on_arity_mismatch=args.warn_on_arity_mismatch,
        warn_on_shadowed_name=args.warn_on_shadowed_name,
        warn_on_shadowed_var=args.warn_on_shadowed_var,
        warn_on_non_dynamic_set=args.warn_on_non_dynamic_set,
        warn_on_unused_names=args.warn_on_unused_names,
        use_var_indirection=args.use_var_indirection,
        warn_on_var_indirection=args.warn_on_var_indirection,
    )


def _add_debug_arg_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("debug options")
    group.add_argument(
        "--disable-ns-cache",
        action=_set_envvar_action(
            "BASILISP_DO_NOT_CACHE_NAMESPACES", parent=argparse._StoreAction
        ),
        nargs="?",
        const=True,
        type=_to_bool,
        help=(
            "if true, disable attempting to load cached namespaces "
            "(env: BASILISP_DO_NOT_CACHE_NAMESPACES; default: false)"
        ),
    )
    group.add_argument(
        "--enable-logger",
        action=_set_envvar_action(
            "BASILISP_USE_DEV_LOGGER", parent=argparse._StoreAction
        ),
        nargs="?",
        const=True,
        type=_to_bool,
        help=(
            "if true, enable the Basilisp root logger "
            "(env: BASILISP_USE_DEV_LOGGER; default: false)"
        ),
    )
    group.add_argument(
        "-l",
        "--log-level",
        action=_set_envvar_action(
            "BASILISP_LOGGING_LEVEL", parent=argparse._StoreAction
        ),
        type=lambda s: s.upper(),
        default="WARNING",
        help=(
            "the logging level for logs emitted by the Basilisp compiler and runtime "
            "(env: BASILISP_LOGGING_LEVEL; default: WARNING)"
        ),
    )
    group.add_argument(
        "--emit-generated-python",
        action=_set_envvar_action(
            "BASILISP_EMIT_GENERATED_PYTHON", parent=argparse._StoreAction
        ),
        nargs="?",
        const=True,
        type=_to_bool,
        help=(
            "if true, store generated Python code in `*generated-python*` dynamic "
            "Vars within each namespace (env: BASILISP_EMIT_GENERATED_PYTHON; "
            "default: true)"
        ),
    )


def _add_import_arg_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "path options",
        description=(
            "The path options below can be used to control how Basilisp (and Python) "
            "find your code."
        ),
    )
    group.add_argument(
        "--include-unsafe-path",
        action="store",
        nargs="?",
        const=True,
        default=os.getenv("BASILISP_INCLUDE_UNSAFE_PATH", "true"),
        type=_to_bool,
        help=(
            "if true, automatically prepend a potentially unsafe path to `sys.path`; "
            "setting `--include-unsafe-path=false` is the Basilisp equivalent to "
            "setting PYTHONSAFEPATH to a non-empty string for CPython's REPL "
            "(env: BASILISP_INCLUDE_UNSAFE_PATH; default: true)"
        ),
    )
    group.add_argument(
        "-p",
        "--include-path",
        action="append",
        help=(
            "path to prepend to `sys.path`; may be specified more than once to "
            "include multiple paths (env: PYTHONPATH)"
        ),
    )


def _add_runtime_arg_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "runtime arguments",
        description=(
            "The runtime arguments below affect reader and execution time features."
        ),
    )
    group.add_argument(
        "--data-readers-entry-points",
        action=_set_envvar_action(
            "BASILISP_USE_DATA_READERS_ENTRY_POINT", parent=argparse._StoreAction
        ),
        nargs="?",
        const=_to_bool(os.getenv("BASILISP_USE_DATA_READERS_ENTRY_POINT", "true")),
        type=_to_bool,
        help=(
            "if true, Load data readers from importlib entry points in the "
            '"basilisp_data_readers" group (env: '
            "BASILISP_USE_DATA_READERS_ENTRY_POINT; default: true)"
        ),
    )


Handler = Union[
    Callable[[argparse.ArgumentParser, argparse.Namespace], None],
    Callable[[argparse.ArgumentParser, argparse.Namespace, list[str]], None],
]


def _subcommand(
    subcommand: str,
    *,
    help: Optional[str] = None,  # pylint: disable=redefined-builtin
    description: Optional[str] = None,
    handler: Handler,
    allows_extra: bool = False,
) -> Callable[
    [Callable[[argparse.ArgumentParser], None]],
    Callable[["argparse._SubParsersAction"], None],
]:
    def _wrap_add_subcommand(
        f: Callable[[argparse.ArgumentParser], None],
    ) -> Callable[["argparse._SubParsersAction"], None]:
        def _wrapped_subcommand(subparsers: "argparse._SubParsersAction"):
            parser = subparsers.add_parser(
                subcommand, help=help, description=description
            )
            parser.set_defaults(handler=handler)
            parser.set_defaults(allows_extra=allows_extra)
            f(parser)

        return _wrapped_subcommand

    return _wrap_add_subcommand


def bootstrap_basilisp_installation(_, args: argparse.Namespace) -> None:
    if args.quiet:
        print_ = lambda v: v
    else:
        print_ = print

    if args.uninstall:
        if not (
            removed := basilisp.unbootstrap_python(site_packages=args.site_packages)
        ):
            print_("No Basilisp bootstrap files were found.")
        else:
            if removed is not None:
                print_(f"Removed '{removed}'")
    else:
        path = basilisp.bootstrap_python(site_packages=args.site_packages)
        print_(
            f"(Added {path})\n\n"
            "Your Python installation has been bootstrapped! You can undo this at any "
            "time with with `basilisp bootstrap --uninstall`."
        )


@_subcommand(
    "bootstrap",
    help="bootstrap the Python installation to allow importing Basilisp namespaces",
    description=textwrap.dedent(
        """Bootstrap the Python installation to allow importing Basilisp namespaces"
        without requiring an additional bootstrapping step.

        Python installations are bootstrapped by installing a `basilispbootstrap.pth`
        file in your `site-packages` directory. Python installations execute `*.pth`
        files found at startup.

        Bootstrapping your Python installation in this way can help avoid needing to
        perform manual bootstrapping from Python code within your application.

        On the first startup, Basilisp will compile `basilisp.core` to byte code
        which could take up to 30 seconds in some cases depending on your system and
        which version of Python you are using. Subsequent startups should be
        considerably faster so long as you allow Basilisp to cache bytecode for
        namespaces."""
    ),
    handler=bootstrap_basilisp_installation,
)
def _add_bootstrap_subcommand(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="if true, remove any `.pth` files installed by Basilisp in all site-packages directories",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="if true, do not print out any",
    )
    # Allow specifying the "site-packages" directories via CLI argument for testing.
    # Not intended to be used by end users.
    parser.add_argument(
        "--site-packages",
        help=argparse.SUPPRESS,
    )


def nrepl_server(
    _,
    args: argparse.Namespace,
) -> None:
    basilisp.init(_compiler_opts(args))
    init_path(args)
    nrepl_server_mod = importlib.import_module(munge(NREPL_SERVER_NS))
    nrepl_server_mod.start_server__BANG__(
        lmap.map(
            {
                kw.keyword("host"): args.host,
                kw.keyword("port"): args.port,
                kw.keyword("nrepl-port-file"): args.port_filepath,
            }
        )
    )


@_subcommand(
    "nrepl-server",
    help="start the nREPL server",
    description="Start the nREPL server.",
    handler=nrepl_server,
)
def _add_nrepl_server_subcommand(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="the interface address to bind to, defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        default=0,
        type=int,
        help="the port to connect to, defaults to 0 (random available port).",
    )
    parser.add_argument(
        "--port-filepath",
        default=".nrepl-port",
        help='the file path where the server port number is output to, defaults to ".nrepl-port".',
    )
    _add_compiler_arg_group(parser)
    _add_import_arg_group(parser)
    _add_runtime_arg_group(parser)
    _add_debug_arg_group(parser)


def repl(
    _,
    args: argparse.Namespace,
) -> None:
    opts = _compiler_opts(args)
    basilisp.init(opts)
    init_path(args)
    ctx = compiler.CompilerContext(filename=REPL_INPUT_FILE_PATH, opts=opts)
    prompter = get_prompter()
    eof = object()

    # Bind user-settable dynamic Vars to their existing value to allow users to
    # conveniently (set! *var* val) at the REPL without needing `binding`.
    with runtime.bindings(
        {
            var: var.value
            for var in map(
                lambda name: runtime.Var.find_safe(
                    sym.symbol(name, ns=runtime.CORE_NS)
                ),
                [
                    "*e",
                    "*1",
                    "*2",
                    "*3",
                    "*assert*",
                    "*data-readers*",
                    "*resolver*",
                    runtime.PRINT_DUP_VAR_NAME,
                    runtime.PRINT_LEVEL_VAR_NAME,
                    runtime.PRINT_READABLY_VAR_NAME,
                    runtime.PRINT_LEVEL_VAR_NAME,
                    runtime.PRINT_META_VAR_NAME,
                    runtime.PRINT_NAMESPACE_MAPS_VAR_NAME,
                ],
            )
        }
    ):
        repl_module = bootstrap_repl(ctx, args.default_ns)
        ns_var = runtime.set_current_ns(args.default_ns)

        while True:
            ns: runtime.Namespace = ns_var.value
            try:
                lsrc = prompter.prompt(f"{ns.name}=> ")
            except EOFError:
                break
            except KeyboardInterrupt:  # pragma: no cover
                print("")
                continue

            if len(lsrc) == 0:
                continue

            try:
                result = eval_str(lsrc, ctx, ns, eof)
                if result is eof:  # pragma: no cover
                    continue
                prompter.print(runtime.lrepr(result))
                repl_module.mark_repl_result(result)
            except reader.SyntaxError as e:
                print_exception(e, reader.SyntaxError, e.__traceback__)
                repl_module.mark_exception(e)
                continue
            except compiler.CompilerException as e:
                print_exception(e, compiler.CompilerException, e.__traceback__)
                repl_module.mark_exception(e)
                continue
            except Exception as e:  # pylint: disable=broad-exception-caught
                print_exception(e, Exception, e.__traceback__)
                repl_module.mark_exception(e)
                continue


@_subcommand(
    "repl",
    help="start the Basilisp REPL",
    description="Start a Basilisp REPL.",
    handler=repl,
)
def _add_repl_subcommand(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--default-ns",
        default=runtime.REPL_DEFAULT_NS,
        help="default namespace to use for the REPL",
    )
    _add_compiler_arg_group(parser)
    _add_import_arg_group(parser)
    _add_runtime_arg_group(parser)
    _add_debug_arg_group(parser)


def run(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    target = args.file_or_ns_or_code
    if args.load_namespace:
        if args.in_ns is not None:
            parser.error(
                "argument --in-ns: not allowed with argument -n/--load-namespace"
            )
        in_ns = runtime.REPL_DEFAULT_NS
    else:
        in_ns = target if args.in_ns is not None else runtime.REPL_DEFAULT_NS

    opts = _compiler_opts(args)
    basilisp.init(opts)
    ctx = compiler.CompilerContext(
        filename=(
            CLI_INPUT_FILE_PATH
            if args.code
            else (STDIN_INPUT_FILE_PATH if target == STDIN_FILE_NAME else target)
        ),
        opts=opts,
    )
    eof = object()

    core_ns = runtime.Namespace.get(runtime.CORE_NS_SYM)
    assert core_ns is not None

    with runtime.ns_bindings(in_ns) as ns:
        ns.refer_all(core_ns)

        if args.args:
            cli_args_var = core_ns.find(sym.symbol(runtime.COMMAND_LINE_ARGS_VAR_NAME))
            assert cli_args_var is not None
            cli_args_var.bind_root(vec.vector(args.args))

        if args.code:
            init_path(args)
            eval_str(target, ctx, ns, eof)
        elif args.load_namespace:
            # Set the requested namespace as the *main-ns*
            main_ns_var = core_ns.find(sym.symbol(runtime.MAIN_NS_VAR_NAME))
            assert main_ns_var is not None
            main_ns_var.bind_root(sym.symbol(target))

            init_path(args)
            importlib.import_module(munge(target))
        elif target == STDIN_FILE_NAME:
            init_path(args)
            eval_stream(io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8"), ctx, ns)
        else:
            init_path(args, unsafe_path=str(pathlib.Path(target).resolve().parent))
            eval_file(target, ctx, ns)


@_subcommand(
    "run",
    help="run a Basilisp script or code or namespace",
    description=textwrap.dedent(
        """Run a Basilisp script or a line of code or load a Basilisp namespace.

        If `-c` is provided, execute the line of code as given. If `-n` is given,
        interpret `file_or_ns_or_code` as a fully qualified Basilisp namespace
        relative to `sys.path`. Otherwise, execute the file as a script relative to
        the current working directory.

        `*main-ns*` will be set to the value provided for `-n`. In all other cases,
        it will be `nil`."""
    ),
    handler=run,
)
def _add_run_subcommand(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "file_or_ns_or_code",
        help=(
            "file path to a Basilisp file, a string of Basilisp code, or a fully "
            "qualified Basilisp namespace name"
        ),
    )

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "-c",
        "--code",
        action="store_true",
        help="if provided, treat argument as a string of code",
    )
    grp.add_argument(
        "-n",
        "--load-namespace",
        action="store_true",
        help="if provided, treat argument as the name of a namespace",
    )

    parser.add_argument(
        "--in-ns",
        help="namespace to use for the code (default: basilisp.user); ignored when `-n` is used",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="command line args made accessible to the script as basilisp.core/*command-line-args*",
    )
    _add_compiler_arg_group(parser)
    _add_import_arg_group(parser)
    _add_runtime_arg_group(parser)
    _add_debug_arg_group(parser)


def test(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    extra: list[str],
) -> None:  # pragma: no cover
    init_path(args)
    basilisp.init(_compiler_opts(args))
    # parse_known_args leaves the `--` separator as the first element if it is present
    # but retaining that causes PyTest to interpret all the arguments as positional
    if extra and extra[0] == "--":
        extra = extra[1:]
    try:
        import pytest
    except (ImportError, ModuleNotFoundError):
        parser.error(
            "Cannot run tests without dependency PyTest. Please install PyTest and try again.",
        )
    else:
        sys.exit(pytest.main(args=list(extra)))


@_subcommand(
    "test",
    help="run tests in a Basilisp project",
    description=textwrap.dedent(
        """Run tests in a Basilisp project.

        Any options not recognized by Basilisp and all positional arguments will
        be collected and passed on to PyTest. It is possible to directly signal
        the end of option processing using an explicit `--` as in:

            `basilisp test -p other_dir -- -k vector`

        This can be useful to also directly execute PyTest commands with Basilisp.
        For instance, you can directly print the PyTest command-line help text using:

            `basilisp test -- -h`

        If all options are unambiguous (e.g. they are only either used by Basilisp
        or by PyTest), then you can omit the `--`:

            `basilisp test -k vector -p other_dir`

        Returns the PyTest exit code as the exit code."""
    ),
    handler=test,
    allows_extra=True,
)
def _add_test_subcommand(parser: argparse.ArgumentParser) -> None:
    _add_compiler_arg_group(parser)
    _add_import_arg_group(parser)
    _add_runtime_arg_group(parser)
    _add_debug_arg_group(parser)


def version(_, __) -> None:
    v = importlib.metadata.version("basilisp")
    print(f"Basilisp {v}")


@_subcommand("version", help="print the version of Basilisp", handler=version)
def _add_version_subcommand(_: argparse.ArgumentParser) -> None:
    pass


def run_script():
    """Entrypoint to run the Basilisp script named by `sys.argv[1]` as by the
    `basilisp run` subcommand.

    This is provided as a shim for platforms where shebang lines cannot contain more
    than one argument and thus `#!/usr/bin/env basilisp run` would be non-functional.

    The current process is replaced as by `os.execvp`."""
    # os.exec* functions do not perform shell expansion, so we must do so manually.
    script_path = Path(sys.argv[1]).resolve()
    args = ["basilisp", "run", str(script_path)]
    # Collect arguments sent to the script and pass them onto `basilisp run`
    if rest := sys.argv[2:]:
        args.append("--")
        args.extend(rest)
    os.execvp("basilisp", args)  # nosec B606, B607


def invoke_cli(args: Optional[Sequence[str]] = None) -> None:
    """Entrypoint to run the Basilisp CLI."""
    parser = argparse.ArgumentParser(
        description="Basilisp is a Lisp dialect inspired by Clojure targeting Python 3."
    )

    subparsers = parser.add_subparsers(help="sub-commands")
    _add_bootstrap_subcommand(subparsers)
    _add_nrepl_server_subcommand(subparsers)
    _add_repl_subcommand(subparsers)
    _add_run_subcommand(subparsers)
    _add_test_subcommand(subparsers)
    _add_version_subcommand(subparsers)

    parsed_args, extra = parser.parse_known_args(args=args)
    allows_extra = getattr(parsed_args, "allows_extra", False)
    if extra and not allows_extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    elif hasattr(parsed_args, "handler"):
        if allows_extra:
            parsed_args.handler(parser, parsed_args, extra)
        else:
            parsed_args.handler(parser, parsed_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    invoke_cli()
