import traceback

import basilisp.compiler as compiler
import basilisp.lang.runtime as runtime
import basilisp.reader as reader


def eval_file(filename: str, ctx: compiler.CompilerContext):
    """Evaluate a file with the given name into a Python module AST node."""
    last = None
    for form in reader.read_file(filename, resolver=runtime.resolve_alias):
        last = compiler.compile_form(form, ctx=ctx)
    return last


def eval_str(s: str, ctx: compiler.CompilerContext):
    """Evaluate the forms in a string into a Python module AST node."""
    last = None
    for form in reader.read_str(s, resolver=runtime.resolve_alias):
        last = compiler.compile_form(form, ctx=ctx)
    return last


def import_core_ns(ctx: compiler.CompilerContext):
    core_ns_filename = runtime.core_resource()
    eval_file(core_ns_filename, ctx)


def repl(default_ns=runtime._REPL_DEFAULT_NS):
    ctx = compiler.CompilerContext()
    runtime.bootstrap()
    import_core_ns(ctx)
    ns_var = runtime.set_current_ns(default_ns)
    while True:
        try:
            lsrc = input(f'{ns_var.value.name}=> ')
        except EOFError:
            break
        except KeyboardInterrupt:
            print('')
            continue

        if len(lsrc) == 0:
            continue

        try:
            print(compiler.lrepr(eval_str(lsrc, ctx)))
        except reader.SyntaxError as e:
            traceback.print_exception(reader.SyntaxError, e, e.__traceback__)
            continue
        except compiler.CompilerException as e:
            traceback.print_exception(compiler.CompilerException, e,
                                      e.__traceback__)
            continue


if __name__ == "__main__":
    runtime.init_ns_var()
    repl()
