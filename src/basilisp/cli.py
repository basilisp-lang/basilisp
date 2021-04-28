import argparse
import importlib
import io
import os
import sys
import traceback
import types
from typing import Any, Callable, Optional, Sequence

from basilisp import main as basilisp
from basilisp.lang import compiler as compiler
from basilisp.lang import reader as reader
from basilisp.lang import runtime as runtime
from basilisp.lang import symbol as sym
from basilisp.prompt import get_prompter

CLI_INPUT_FILE_PATH = "<CLI Input>"
REPL_INPUT_FILE_PATH = "<REPL Input>"
REPL_NS = "basilisp.repl"
STDIN_INPUT_FILE_PATH = "<stdin>"
STDIN_FILE_NAME = "-"


def eval_file(filename: str, ctx: compiler.CompilerContext, ns: runtime.Namespace):
    """Evaluate a file with the given name into a Python module AST node."""
    last = None
    for form in reader.read_file(filename, resolver=runtime.resolve_alias):
        assert not isinstance(form, reader.ReaderConditional)
        last = compiler.compile_and_exec_form(form, ctx, ns)
    return last


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


def bootstrap_repl(ctx: compiler.CompilerContext, which_ns: str) -> types.ModuleType:
    """Bootstrap the REPL with a few useful vars and returned the bootstrapped
    module so it's functions can be used by the REPL command."""
    ns = runtime.Namespace.get_or_create(sym.symbol(which_ns))
    eval_str(f"(ns {sym.symbol(which_ns)} (:use basilisp.repl))", ctx, ns, object())
    return importlib.import_module(REPL_NS)


def _add_compiler_arg_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("compiler arguments")
    group.add_argument(
        "--warn-on-shadowed-name",
        action="store_const",
        const=True,
        default=os.getenv("BASILISP_WARN_ON_SHADOWED_NAME"),
        help="if provided, emit warnings if a local name is shadowed by another local name",
    )
    group.add_argument(
        "--warn-on-shadowed-var",
        action="store_const",
        const=True,
        default=os.getenv("BASILISP_WARN_ON_SHADOWED_VAR"),
        help="if provided, emit warnings if a Var name is shadowed by a local name",
    )
    group.add_argument(
        "--warn-on-unused-names",
        action="store_const",
        const=True,
        default=os.getenv("BASILISP_WARN_ON_UNUSED_NAMES"),
        help="if provided, emit warnings if a local name is bound and unused",
    )
    group.add_argument(
        "--use-var-indirection",
        action="store_const",
        const=True,
        default=os.getenv("BASILISP_USE_VAR_INDIRECTION"),
        help="if provided, all Var accesses will be performed via Var indirection",
    )
    group.add_argument(
        "--warn-on-var-indirection",
        action="store_const",
        const=True,
        default=os.getenv("BASILISP_WARN_ON_VAR_INDIRECTION"),
        help="if provided, emit warnings if a Var reference cannot be direct linked",
    )


Handler = Callable[[argparse.ArgumentParser, argparse.Namespace], None]


def _subcommand(
    subcommand: str,
    *,
    help: Optional[str] = None,
    description: Optional[str] = None,
    handler: Handler,
):
    def _wrap_add_subcommand(f: Callable[[argparse.ArgumentParser], None]):
        def _wrapped_subcommand(subparsers: "argparse._SubParsersAction"):
            parser = subparsers.add_parser(
                subcommand, help=help, description=description
            )
            parser.set_defaults(handler=handler)
            f(parser)

        return _wrapped_subcommand

    return _wrap_add_subcommand


def repl(  # pylint: disable=too-many-arguments,too-many-locals
    _,
    args: argparse.Namespace,
):
    opts = compiler.compiler_opts(
        warn_on_shadowed_name=args.warn_on_shadowed_name,
        warn_on_shadowed_var=args.warn_on_shadowed_var,
        warn_on_unused_names=args.warn_on_unused_names,
        use_var_indirection=args.use_var_indirection,
        warn_on_var_indirection=args.warn_on_var_indirection,
    )
    basilisp.init(opts)
    ctx = compiler.CompilerContext(filename=REPL_INPUT_FILE_PATH, opts=opts)
    repl_module = bootstrap_repl(ctx, args.default_ns)
    ns_var = runtime.set_current_ns(args.default_ns)
    prompter = get_prompter()
    eof = object()
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
            repl_module.mark_repl_result(result)  # type: ignore
        except reader.SyntaxError as e:
            traceback.print_exception(reader.SyntaxError, e, e.__traceback__)
            repl_module.mark_exception(e)  # type: ignore
            continue
        except compiler.CompilerException as e:
            traceback.print_exception(compiler.CompilerException, e, e.__traceback__)
            repl_module.mark_exception(e)  # type: ignore
            continue
        except Exception as e:
            traceback.print_exception(Exception, e, e.__traceback__)
            repl_module.mark_exception(e)  # type: ignore
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


def run(  # pylint: disable=too-many-arguments
    _,
    args: argparse.Namespace,
):
    opts = compiler.compiler_opts(
        warn_on_shadowed_name=args.warn_on_shadowed_name,
        warn_on_shadowed_var=args.warn_on_shadowed_var,
        warn_on_unused_names=args.warn_on_unused_names,
        use_var_indirection=args.use_var_indirection,
        warn_on_var_indirection=args.warn_on_var_indirection,
    )
    basilisp.init(opts)
    ctx = compiler.CompilerContext(
        filename=CLI_INPUT_FILE_PATH
        if args.code
        else (
            STDIN_INPUT_FILE_PATH
            if args.file_or_code == STDIN_FILE_NAME
            else args.file_or_code
        ),
        opts=opts,
    )
    eof = object()

    core_ns = runtime.Namespace.get(runtime.CORE_NS_SYM)
    assert core_ns is not None

    with runtime.ns_bindings(args.in_ns) as ns:
        ns.refer_all(core_ns)

        if args.code:
            print(runtime.lrepr(eval_str(args.file_or_code, ctx, ns, eof)))
        elif args.file_or_code == STDIN_FILE_NAME:
            print(
                runtime.lrepr(
                    eval_stream(
                        io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8"), ctx, ns
                    )
                )
            )
        else:
            print(runtime.lrepr(eval_file(args.file_or_code, ctx, ns)))


@_subcommand(
    "run",
    help="run a Basilisp script or code",
    description="Run a Basilisp script or a line of code, if it is provided.",
    handler=run,
)
def _add_run_subcommand(parser: argparse.ArgumentParser):
    parser.add_argument(
        "file_or_code",
        help="file path to a Basilisp file or, if -c is provided, a string of Basilisp code",
    )
    parser.add_argument(
        "-c",
        "--code",
        action="store_true",
        help="if provided, treat argument as a string of code",
    )
    parser.add_argument(
        "--in-ns", default=runtime.REPL_DEFAULT_NS, help="namespace to use for the code"
    )
    _add_compiler_arg_group(parser)


def test(parser: argparse.ArgumentParser, args: argparse.Namespace):  # pragma: no cover
    try:
        import pytest
    except (ImportError, ModuleNotFoundError):
        parser.error(
            "Cannot run tests without dependency PyTest. Please install PyTest and try again.",
        )
    else:
        pytest.main(args=list(args.args))


@_subcommand(
    "test",
    help="run tests in a Basilisp project",
    description="Run tests in a Basilisp project.",
    handler=test,
)
def _add_test_subcommand(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("args", nargs=-1)


def version(parser: argparse.ArgumentParser, _):
    from basilisp.__version__ import __version__

    parser.exit(status=0, message=f"Basilisp {__version__}")


@_subcommand("version", help="print the version of Basilisp", handler=version)
def _add_version_subcommand(_: argparse.ArgumentParser) -> None:
    pass


def invoke_cli(args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Basilisp is a Lisp dialect inspired by Clojure targeting Python 3."
    )

    subparsers = parser.add_subparsers(help="sub-commands")
    _add_repl_subcommand(subparsers)
    _add_run_subcommand(subparsers)
    _add_test_subcommand(subparsers)
    _add_version_subcommand(subparsers)

    args = parser.parse_args(args=args)
    if hasattr(args, "handler"):
        args.handler(parser, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    invoke_cli()
