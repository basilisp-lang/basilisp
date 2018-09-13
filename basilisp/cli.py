import importlib
# noinspection PyUnresolvedReferences
import readline  # noqa: F401
import traceback
import types

import click

import basilisp.compiler as compiler
import basilisp.lang.runtime as runtime
import basilisp.lang.symbol as sym
import basilisp.main as basilisp
import basilisp.reader as reader
from basilisp.util import Maybe


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


def eval_str(s: str, ctx: compiler.CompilerContext, module: types.ModuleType):
    """Evaluate the forms in a string into a Python module AST node."""
    last = None
    for form in reader.read_str(s, resolver=runtime.resolve_alias):
        last = compiler.compile_and_exec_form(form, ctx, module, source_filename='REPL Input')
    return last


def bootstrap_repl(which_ns: str) -> types.ModuleType:
    """Bootstrap the REPL with a few useful vars and returned the
    bootstrapped module so it's functions can be used by the REPL
    command."""
    repl_ns = runtime.Namespace.get_or_create(sym.symbol('basilisp.repl'))
    ns = runtime.Namespace.get_or_create(sym.symbol(which_ns))
    repl_module = importlib.import_module('basilisp.repl')
    ns.add_alias(sym.symbol('basilisp.repl'), repl_ns)
    for name in ['*1', '*2', '*3', '*e', 'doc', 'pydoc', 'source']:
        ns.intern(sym.symbol(name),
                  Maybe(runtime.Var.find(sym.symbol(name, ns='basilisp.repl'))).or_else_raise(
                      lambda: runtime.RuntimeException(
                          f"Var basilisp.repl/{name} not found!")))  # pylint: disable=cell-var-from-loop
    return repl_module


@cli.command(short_help='start the Basilisp REPL')
@click.option('--default-ns', default=runtime._REPL_DEFAULT_NS, help='default namespace to use for the REPL')
def repl(default_ns):
    basilisp.init()
    repl_module = bootstrap_repl(default_ns)
    ctx = compiler.CompilerContext()
    ns_var = runtime.set_current_ns(default_ns)
    while True:
        ns: runtime.Namespace = ns_var.value
        try:
            lsrc = input(f'{ns.name}=> ')
        except EOFError:
            break
        except KeyboardInterrupt:
            print('')
            continue

        if len(lsrc) == 0:
            continue

        try:
            result = eval_str(lsrc, ctx, ns.module)
            print(compiler.lrepr(result))
            repl_module.mark_repl_result(result)
        except reader.SyntaxError as e:
            traceback.print_exception(reader.SyntaxError, e, e.__traceback__)
            repl_module.mark_exception(e)
            continue
        except compiler.CompilerException as e:
            traceback.print_exception(compiler.CompilerException, e,
                                      e.__traceback__)
            repl_module.mark_exception(e)
            continue
        except Exception as e:
            traceback.print_exception(Exception, e, e.__traceback__)
            repl_module.mark_exception(e)
            continue


@cli.command(short_help='run a Basilisp script or code')
@click.argument('file-or-code')
@click.option('-c', '--code', is_flag=True, help='if provided, treat argument as a string of code')
@click.option('--in-ns', default=runtime._REPL_DEFAULT_NS, help='namespace to use for the code')
def run(file_or_code, code, in_ns):
    """Run a Basilisp script or a line of code, if it is provided."""
    basilisp.init()
    ctx = compiler.CompilerContext()
    ns_var = runtime.set_current_ns(in_ns)
    ns: runtime.Namespace = ns_var.value

    if code:
        print(compiler.lrepr(eval_str(file_or_code, ctx, ns.module)))
    else:
        print(compiler.lrepr(eval_file(file_or_code, ctx, ns.module)))


if __name__ == "__main__":
    cli()
