import importlib
import traceback
import types
from typing import Any

import click
import pytest

import basilisp.lang.compiler as compiler
import basilisp.lang.reader as reader
import basilisp.lang.runtime as runtime
import basilisp.lang.symbol as sym
import basilisp.main as basilisp
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
    repl_ns = runtime.Namespace.get_or_create(sym.symbol(REPL_NS))
    ns = runtime.Namespace.get_or_create(sym.symbol(which_ns))
    compiler.set_compiler_context(ns, ctx)
    core_ns = runtime.Namespace.get(sym.symbol(runtime.CORE_NS))
    assert core_ns is not None
    ns.refer_all(core_ns)
    repl_module = importlib.import_module(REPL_NS)
    ns.add_alias(repl_ns, sym.symbol(REPL_NS))
    ns.refer_all(repl_ns)
    return repl_module


@cli.command(short_help="start the Basilisp REPL")
@click.option(
    "--default-ns",
    default=runtime.REPL_DEFAULT_NS,
    help="default namespace to use for the REPL",
)
@click.option(
    "--use-var-indirection",
    default=False,
    is_flag=True,
    envvar="BASILISP_USE_VAR_INDIRECTION",
    help="if provided, all Var accesses will be performed via Var indirection",
)
@click.option(
    "--warn-on-shadowed-name",
    default=False,
    is_flag=True,
    envvar="BASILISP_WARN_ON_SHADOWED_NAME",
    help="if provided, emit warnings if a local name is shadowed by another local name",
)
@click.option(
    "--warn-on-shadowed-var",
    default=False,
    is_flag=True,
    envvar="BASILISP_WARN_ON_SHADOWED_VAR",
    help="if provided, emit warnings if a Var name is shadowed by a local name",
)
@click.option(
    "--warn-on-var-indirection",
    default=True,
    is_flag=True,
    envvar="BASILISP_WARN_ON_VAR_INDIRECTION",
    help="if provided, emit warnings if a Var reference cannot be direct linked",
)
def repl(
    default_ns,
    use_var_indirection,
    warn_on_shadowed_name,
    warn_on_shadowed_var,
    warn_on_var_indirection,
):
    basilisp.init()
    ctx = compiler.CompilerContext(
        filename=REPL_INPUT_FILE_PATH,
        opts={
            compiler.WARN_ON_SHADOWED_NAME: warn_on_shadowed_name,
            compiler.WARN_ON_SHADOWED_VAR: warn_on_shadowed_var,
            compiler.USE_VAR_INDIRECTION: use_var_indirection,
            compiler.WARN_ON_VAR_INDIRECTION: warn_on_var_indirection,
        },
    )
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
    "--use-var-indirection",
    default=False,
    is_flag=True,
    envvar="BASILISP_USE_VAR_INDIRECTION",
    help="if provided, all Var accesses will be performed via Var indirection",
)
@click.option(
    "--warn-on-shadowed-name",
    default=False,
    is_flag=True,
    envvar="BASILISP_WARN_ON_SHADOWED_NAME",
    help="if provided, emit warnings if a local name is shadowed by another local name",
)
@click.option(
    "--warn-on-shadowed-var",
    default=False,
    is_flag=True,
    envvar="BASILISP_WARN_ON_SHADOWED_VAR",
    help="if provided, emit warnings if a Var name is shadowed by a local name",
)
@click.option(
    "--warn-on-var-indirection",
    default=True,
    is_flag=True,
    envvar="BASILISP_WARN_ON_VAR_INDIRECTION",
    help="if provided, emit warnings if a Var reference cannot be direct linked",
)
def run(  # pylint: disable=too-many-arguments
    file_or_code,
    code,
    in_ns,
    use_var_indirection,
    warn_on_shadowed_name,
    warn_on_shadowed_var,
    warn_on_var_indirection,
):
    """Run a Basilisp script or a line of code, if it is provided."""
    basilisp.init()
    ctx = compiler.CompilerContext(
        filename=CLI_INPUT_FILE_PATH
        if code
        else (
            STDIN_INPUT_FILE_PATH if file_or_code == STDIN_FILE_NAME else file_or_code
        ),
        opts={
            compiler.WARN_ON_SHADOWED_NAME: warn_on_shadowed_name,
            compiler.WARN_ON_SHADOWED_VAR: warn_on_shadowed_var,
            compiler.USE_VAR_INDIRECTION: use_var_indirection,
            compiler.WARN_ON_VAR_INDIRECTION: warn_on_var_indirection,
        },
    )
    eof = object()

    core_ns = runtime.Namespace.get(sym.symbol(runtime.CORE_NS))
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
