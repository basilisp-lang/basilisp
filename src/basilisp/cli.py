import importlib

# noinspection PyUnresolvedReferences
import readline  # noqa: F401
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


@click.group()
def cli():
    """Basilisp is a Lisp dialect inspired by Clojure targeting Python 3."""
    pass


def eval_file(filename: str, ctx: compiler.CompilerContext, module: types.ModuleType):
    """Evaluate a file with the given name into a Python module AST node."""
    last = None
    for form in reader.read_file(filename, resolver=runtime.resolve_alias):
        last = compiler.compile_and_exec_form(form, ctx, module, filename)
    return last


def eval_str(s: str, ctx: compiler.CompilerContext, module: types.ModuleType, eof: Any):
    """Evaluate the forms in a string into a Python module AST node."""
    last = eof
    for form in reader.read_str(s, resolver=runtime.resolve_alias, eof=eof):
        last = compiler.compile_and_exec_form(
            form, ctx, module, source_filename="REPL Input"
        )
    return last


def bootstrap_repl(which_ns: str) -> types.ModuleType:
    """Bootstrap the REPL with a few useful vars and returned the
    bootstrapped module so it's functions can be used by the REPL
    command."""
    repl_ns = runtime.Namespace.get_or_create(sym.symbol("basilisp.repl"))
    ns = runtime.Namespace.get_or_create(sym.symbol(which_ns))
    repl_module = importlib.import_module("basilisp.repl")
    ns.add_alias(sym.symbol("basilisp.repl"), repl_ns)
    ns.refer_all(repl_ns)
    return repl_module


@cli.command(short_help="start the Basilisp REPL")
@click.option(
    "--default-ns",
    default=runtime._REPL_DEFAULT_NS,
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
    repl_module = bootstrap_repl(default_ns)
    ctx = compiler.CompilerContext(
        {
            compiler.USE_VAR_INDIRECTION: use_var_indirection,
            compiler.WARN_ON_SHADOWED_NAME: warn_on_shadowed_name,
            compiler.WARN_ON_SHADOWED_VAR: warn_on_shadowed_var,
            compiler.WARN_ON_VAR_INDIRECTION: warn_on_var_indirection,
        }
    )
    ns_var = runtime.set_current_ns(default_ns)
    eof = object()
    while True:
        ns: runtime.Namespace = ns_var.value
        try:
            lsrc = input(f"{ns.name}=> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print("")
            continue

        if len(lsrc) == 0:
            continue

        try:
            result = eval_str(lsrc, ctx, ns.module, eof)
            if result is eof:
                continue
            print(compiler.lrepr(result))
            repl_module.mark_repl_result(result)
        except reader.SyntaxError as e:
            traceback.print_exception(reader.SyntaxError, e, e.__traceback__)
            repl_module.mark_exception(e)
            continue
        except compiler.CompilerException as e:
            traceback.print_exception(compiler.CompilerException, e, e.__traceback__)
            repl_module.mark_exception(e)
            continue
        except Exception as e:
            traceback.print_exception(Exception, e, e.__traceback__)
            repl_module.mark_exception(e)
            continue


@cli.command(short_help="run a Basilisp script or code")
@click.argument("file-or-code")
@click.option(
    "-c", "--code", is_flag=True, help="if provided, treat argument as a string of code"
)
@click.option(
    "--in-ns", default=runtime._REPL_DEFAULT_NS, help="namespace to use for the code"
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
        {
            compiler.USE_VAR_INDIRECTION: use_var_indirection,
            compiler.WARN_ON_SHADOWED_NAME: warn_on_shadowed_name,
            compiler.WARN_ON_SHADOWED_VAR: warn_on_shadowed_var,
            compiler.WARN_ON_VAR_INDIRECTION: warn_on_var_indirection,
        }
    )
    eof = object()

    with runtime.ns_bindings(in_ns) as ns:
        if code:
            print(compiler.lrepr(eval_str(file_or_code, ctx, ns.module, eof)))
        else:
            print(compiler.lrepr(eval_file(file_or_code, ctx, ns.module)))


@cli.command(short_help="run tests in a Basilisp project")
@click.argument("args", nargs=-1)
def test(args):
    """Run tests in a Basilisp project."""
    pytest.main(args=list(args))


if __name__ == "__main__":
    cli()
