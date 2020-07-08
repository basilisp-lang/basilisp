import importlib
import traceback
import types
from typing import Any

import click
import pytest

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


@click.group()
def cli():
    """Basilisp is a Lisp dialect inspired by Clojure targeting Python 3."""


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


@cli.command(short_help="start the Basilisp REPL")
@click.option(
    "--default-ns",
    default=runtime.REPL_DEFAULT_NS,
    help="default namespace to use for the REPL",
)
@click.option(
    "--warn-on-shadowed-name",
    default=None,
    is_flag=True,
    envvar="BASILISP_WARN_ON_SHADOWED_NAME",
    help="if provided, emit warnings if a local name is shadowed by another local name",
)
@click.option(
    "--warn-on-shadowed-var",
    default=None,
    is_flag=True,
    envvar="BASILISP_WARN_ON_SHADOWED_VAR",
    help="if provided, emit warnings if a Var name is shadowed by a local name",
)
@click.option(
    "--warn-on-unused-names",
    default=None,
    is_flag=True,
    envvar="BASILISP_WARN_ON_UNUSED_NAMES",
    help="if provided, emit warnings if a local name is bound and unused",
)
@click.option(
    "--use-var-indirection",
    default=None,
    is_flag=True,
    envvar="BASILISP_USE_VAR_INDIRECTION",
    help="if provided, all Var accesses will be performed via Var indirection",
)
@click.option(
    "--warn-on-var-indirection",
    default=None,
    is_flag=True,
    envvar="BASILISP_WARN_ON_VAR_INDIRECTION",
    help="if provided, emit warnings if a Var reference cannot be direct linked",
)
def repl(  # pylint: disable=too-many-arguments,too-many-locals
    default_ns,
    warn_on_shadowed_name,
    warn_on_shadowed_var,
    warn_on_unused_names,
    use_var_indirection,
    warn_on_var_indirection,
):
    opts = compiler.compiler_opts(
        warn_on_shadowed_name=warn_on_shadowed_name,
        warn_on_shadowed_var=warn_on_shadowed_var,
        warn_on_unused_names=warn_on_unused_names,
        use_var_indirection=use_var_indirection,
        warn_on_var_indirection=warn_on_var_indirection,
    )
    basilisp.init(opts)
    ctx = compiler.CompilerContext(filename=REPL_INPUT_FILE_PATH, opts=opts)
    repl_module = bootstrap_repl(ctx, default_ns)
    ns_var = runtime.set_current_ns(default_ns)
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


@cli.command(short_help="run a Basilisp script or code")
@click.argument("file-or-code")
@click.option(
    "-c", "--code", is_flag=True, help="if provided, treat argument as a string of code"
)
@click.option(
    "--in-ns", default=runtime.REPL_DEFAULT_NS, help="namespace to use for the code"
)
@click.option(
    "--warn-on-shadowed-name",
    default=None,
    is_flag=True,
    envvar="BASILISP_WARN_ON_SHADOWED_NAME",
    help="if provided, emit warnings if a local name is shadowed by another local name",
)
@click.option(
    "--warn-on-shadowed-var",
    default=None,
    is_flag=True,
    envvar="BASILISP_WARN_ON_SHADOWED_VAR",
    help="if provided, emit warnings if a Var name is shadowed by a local name",
)
@click.option(
    "--warn-on-unused-names",
    default=None,
    is_flag=True,
    envvar="BASILISP_WARN_ON_UNUSED_NAMES",
    help="if provided, emit warnings if a local name is bound and unused",
)
@click.option(
    "--use-var-indirection",
    default=None,
    is_flag=True,
    envvar="BASILISP_USE_VAR_INDIRECTION",
    help="if provided, all Var accesses will be performed via Var indirection",
)
@click.option(
    "--warn-on-var-indirection",
    default=None,
    is_flag=True,
    envvar="BASILISP_WARN_ON_VAR_INDIRECTION",
    help="if provided, emit warnings if a Var reference cannot be direct linked",
)
def run(  # pylint: disable=too-many-arguments
    file_or_code,
    code,
    in_ns,
    warn_on_shadowed_name,
    warn_on_shadowed_var,
    warn_on_unused_names,
    use_var_indirection,
    warn_on_var_indirection,
):
    """Run a Basilisp script or a line of code, if it is provided."""
    opts = compiler.compiler_opts(
        warn_on_shadowed_name=warn_on_shadowed_name,
        warn_on_shadowed_var=warn_on_shadowed_var,
        warn_on_unused_names=warn_on_unused_names,
        use_var_indirection=use_var_indirection,
        warn_on_var_indirection=warn_on_var_indirection,
    )
    basilisp.init(opts)
    ctx = compiler.CompilerContext(
        filename=CLI_INPUT_FILE_PATH
        if code
        else (
            STDIN_INPUT_FILE_PATH if file_or_code == STDIN_FILE_NAME else file_or_code
        ),
        opts=opts,
    )
    eof = object()

    core_ns = runtime.Namespace.get(runtime.CORE_NS_SYM)
    assert core_ns is not None

    with runtime.ns_bindings(in_ns) as ns:
        ns.refer_all(core_ns)

        if code:
            print(runtime.lrepr(eval_str(file_or_code, ctx, ns, eof)))
        elif file_or_code == STDIN_FILE_NAME:
            print(runtime.lrepr(eval_stream(click.get_text_stream("stdin"), ctx, ns)))
        else:
            print(runtime.lrepr(eval_file(file_or_code, ctx, ns)))


@cli.command(short_help="run tests in a Basilisp project")
@click.argument("args", nargs=-1)
def test(args):  # pragma: no cover
    """Run tests in a Basilisp project."""
    pytest.main(args=list(args))


@cli.command(short_help="print the version of Basilisp")
def version():
    from basilisp.__version__ import __version__

    print(f"Basilisp {__version__}")


if __name__ == "__main__":
    cli()
