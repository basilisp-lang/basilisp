import readline
import traceback
import apylisp.compiler as compiler
import apylisp.reader as reader
import apylisp.lang.namespace as namespace
import apylisp.lang.runtime as runtime
import apylisp.lang.symbol as sym


def entrypoint(filename, default_ns=runtime._REPL_DEFAULT_NS):
    ns = namespace.Namespace(sym.Symbol(default_ns))
    return


def import_core_ns():
    core_ns_filename = runtime.core_resource()
    core_ns_fn = '__apylisp_core__'
    core_ast = compiler.compile_file(
        core_ns_filename, wrapped_fn_name=core_ns_fn)
    print(compiler.to_py_str(core_ast))
    compiler.exec_ast(core_ast, expr_fn=core_ns_fn)


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
            ast = compiler.compile_str(lsrc)
            if ast is None:
                continue
        except reader.SyntaxError as e:
            traceback.print_exception(reader.SyntaxError, e, e.__traceback__)
            continue

        if runtime.print_generated_python():
            print(compiler.to_py_str(ast))

        print(compiler.lrepr(compiler.exec_ast(ast)))


if __name__ == "__main__":
    runtime.init_ns_var()
    repl()
