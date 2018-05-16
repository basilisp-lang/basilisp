import traceback

import basilisp.compiler as compiler
import basilisp.lang.namespace as namespace
import basilisp.lang.runtime as runtime
import basilisp.lang.symbol as sym
import basilisp.reader as reader


def entrypoint(filename, default_ns=runtime._REPL_DEFAULT_NS):
    ns = namespace.Namespace(sym.Symbol(default_ns))
    return


def import_core_ns():
    core_ns_filename = runtime.core_resource()
    core_ns_fn = '__basilisp_core__'
    compiler.compile_file(core_ns_filename, wrapped_fn_name=core_ns_fn)


def repl(default_ns=runtime._REPL_DEFAULT_NS):
    runtime.bootstrap()
    import_core_ns()
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
            print(compiler.lrepr(compiler.compile_str(lsrc)))
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
