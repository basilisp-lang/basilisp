import importlib
# noinspection PyUnresolvedReferences
import readline  # noqa: F401
import traceback
import types

import basilisp.compiler as compiler
import basilisp.importer as importer
import basilisp.lang.runtime as runtime
import basilisp.reader as reader


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


def repl(default_ns=runtime._REPL_DEFAULT_NS):
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
            print(compiler.lrepr(eval_str(lsrc, ctx, ns.module)))
        except reader.SyntaxError as e:
            traceback.print_exception(reader.SyntaxError, e, e.__traceback__)
            continue
        except compiler.CompilerException as e:
            traceback.print_exception(compiler.CompilerException, e,
                                      e.__traceback__)
            continue
        except Exception as e:
            traceback.print_exception(Exception, e, e.__traceback__)
            continue


def init():
    """Initialize the runtime environment for evaluation."""
    runtime.init_ns_var()
    runtime.bootstrap()
    importer.hook_imports()
    importlib.import_module('basilisp.core')


if __name__ == "__main__":
    init()
    repl()
