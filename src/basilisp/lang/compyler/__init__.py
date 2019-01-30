import ast
import itertools
import types
from typing import Optional, Callable, Any, Iterable, List

from astor import code_gen as codegen

import basilisp.lang.runtime as runtime
from basilisp.lang.compyler.generator import (
    GeneratorContext,
    GeneratedPyAST,
    expressionize as _expressionize,
    gen_py_ast,
    py_module_preamble,
    statementize as _statementize,
)
from basilisp.lang.compyler.parser import ParserContext, parse_ast
from basilisp.lang.typing import LispForm
from basilisp.lang.util import genname

_DEFAULT_FN = "__lisp_expr__"


class CompilerException(Exception):
    pass


def to_py_source(t: ast.AST, outfile: str) -> None:
    source = codegen.to_source(t)
    with open(outfile, mode="w") as f:
        f.writelines(source)


def to_py_str(t: ast.AST) -> str:
    """Return a string of the Python code which would generate the input
    AST node."""
    return codegen.to_source(t)


BytecodeCollector = Optional[Callable[[types.CodeType], None]]


def compile_and_exec_form(  # pylint: disable= too-many-arguments
    form: LispForm,
    pctx: ParserContext,
    gctx: GeneratorContext,
    module: types.ModuleType,
    wrapped_fn_name: str = _DEFAULT_FN,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> Any:
    """Compile and execute the given form. This function will be most useful
    for the REPL and testing purposes. Returns the result of the executed expression.

    Callers may override the wrapped function name, which is used by the
    REPL to evaluate the result of an expression and print it back out."""
    if form is None:
        return None

    if not module.__basilisp_bootstrapped__:  # type: ignore
        _bootstrap_module(gctx, module)

    final_wrapped_name = genname(wrapped_fn_name)

    lisp_ast = parse_ast(pctx, form)
    py_ast = gen_py_ast(gctx, lisp_ast)
    form_ast = list(
        map(
            _statementize,
            itertools.chain(
                py_ast.dependencies,
                [_expressionize(GeneratedPyAST(node=py_ast.node), final_wrapped_name)],
            ),
        )
    )

    ast_module = ast.Module(body=form_ast)
    ast.fix_missing_locations(ast_module)

    if runtime.print_generated_python():
        print(to_py_str(ast_module))
    else:
        runtime.add_generated_python(to_py_str(ast_module))

    bytecode = compile(ast_module, gctx.filename, "exec")
    if collect_bytecode:
        collect_bytecode(bytecode)
    exec(bytecode, module.__dict__)
    return getattr(module, final_wrapped_name)()


def _incremental_compile_module(
    py_ast: GeneratedPyAST,
    mod: types.ModuleType,
    source_filename: str,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> None:
    """Incrementally compile a stream of AST nodes in module mod.

    The source_filename will be passed to Python's native compile.

    Incremental compilation is an integral part of generating a Python module
    during the same process as macro-expansion."""
    module_body = list(
        map(_statementize, itertools.chain(py_ast.dependencies, [py_ast.node]))
    )

    module = ast.Module(body=list(module_body))
    ast.fix_missing_locations(module)

    if runtime.print_generated_python():
        print(to_py_str(module))
    else:
        runtime.add_generated_python(to_py_str(module))

    bytecode = compile(module, source_filename, "exec")
    if collect_bytecode:
        collect_bytecode(bytecode)
    exec(bytecode, mod.__dict__)


def _bootstrap_module(
    gctx: GeneratorContext,
    mod: types.ModuleType,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> None:
    """Bootstrap a new module with imports and other boilerplate."""
    _incremental_compile_module(
        py_module_preamble(gctx),
        mod,
        source_filename=gctx.filename,
        collect_bytecode=collect_bytecode,
    )
    mod.__basilisp_bootstrapped__ = True  # type: ignore


def compile_module(
    forms: Iterable[LispForm],
    pctx: ParserContext,
    gctx: GeneratorContext,
    module: types.ModuleType,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> None:
    """Compile an entire Basilisp module into Python bytecode which can be
    executed as a Python module.

    This function is designed to generate bytecode which can be used for the
    Basilisp import machinery, to allow callers to import Basilisp modules from
    Python code.
    """
    _bootstrap_module(gctx, module)

    for form in forms:
        nodes = gen_py_ast(gctx, parse_ast(pctx, form))
        _incremental_compile_module(
            nodes,
            module,
            source_filename=gctx.filename,
            collect_bytecode=collect_bytecode,
        )


def compile_bytecode(
    code: List[types.CodeType], gctx: GeneratorContext, module: types.ModuleType
) -> None:
    """Compile cached bytecode into the given module.

    The Basilisp import hook attempts to cache bytecode while compiling Basilisp
    namespaces. When the cached bytecode is reloaded from disk, it needs to be
    compiled within a bootstrapped module. This function bootstraps the module
    and then proceeds to compile a collection of bytecodes into the module."""
    _bootstrap_module(gctx, module)
    for bytecode in code:
        exec(bytecode, module.__dict__)
