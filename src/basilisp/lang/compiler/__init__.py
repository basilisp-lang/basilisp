import itertools
import os
import types
from typing import Any, Callable, Iterable, List, Mapping, Optional

from astor import code_gen as codegen

import basilisp._pyast as ast
import basilisp.lang.runtime as runtime
from basilisp.lang.compiler.analyzer import (  # noqa
    AnalyzerContext,
    WARN_ON_SHADOWED_NAME,
    WARN_ON_SHADOWED_VAR,
    WARN_ON_UNUSED_NAMES,
    analyze_form,
    macroexpand,
    macroexpand_1,
)
from basilisp.lang.compiler.exception import CompilerException, CompilerPhase  # noqa
from basilisp.lang.compiler.generator import (  # noqa
    GeneratedPyAST,
    GeneratorContext,
    USE_VAR_INDIRECTION,
    WARN_ON_VAR_INDIRECTION,
    expressionize as _expressionize,
    gen_py_ast,
    py_module_preamble,
    statementize as _statementize,
)
from basilisp.lang.compiler.optimizer import PythonASTOptimizer
from basilisp.lang.typing import ReaderForm
from basilisp.lang.util import genname

_DEFAULT_FN = "__lisp_expr__"


def to_py_str(t: ast.AST) -> str:
    """Return a string of the Python code which would generate the input
    AST node."""
    return codegen.to_source(t)


BytecodeCollector = Optional[Callable[[types.CodeType], None]]


class CompilerContext:
    __slots__ = ("_filename", "_actx", "_gctx", "_optimizer")

    def __init__(self, filename: str, opts: Optional[Mapping[str, bool]] = None):
        self._filename = filename
        self._actx = AnalyzerContext(filename=filename, opts=opts)
        self._gctx = GeneratorContext(filename=filename, opts=opts)
        self._optimizer = PythonASTOptimizer()

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def analyzer_context(self) -> AnalyzerContext:
        return self._actx

    @property
    def generator_context(self) -> GeneratorContext:
        return self._gctx

    @property
    def py_ast_optimizer(self) -> PythonASTOptimizer:
        return self._optimizer


def _emit_ast_string(module: ast.AST) -> None:  # pragma: no cover
    """Emit the generated Python AST string either to standard out or to the
    *generated-python* dynamic Var for the current namespace. If the
    BASILISP_EMIT_GENERATED_PYTHON env var is not set True, this method is a
    no-op."""
    # TODO: eventually, this default should become "false" but during this
    #       period of heavy development, having it set to "true" by default
    #       is tremendously useful
    if os.getenv("BASILISP_EMIT_GENERATED_PYTHON", "true") != "true":
        return

    if runtime.print_generated_python():
        print(to_py_str(module))
    else:
        runtime.add_generated_python(to_py_str(module))


def compile_and_exec_form(  # pylint: disable= too-many-arguments
    form: ReaderForm,
    ctx: CompilerContext,
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
        _bootstrap_module(ctx.generator_context, ctx.py_ast_optimizer, module)

    final_wrapped_name = genname(wrapped_fn_name)

    lisp_ast = analyze_form(ctx.analyzer_context, form)
    py_ast = gen_py_ast(ctx.generator_context, lisp_ast)
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
    ast_module = ctx.py_ast_optimizer.visit(ast_module)
    ast.fix_missing_locations(ast_module)

    _emit_ast_string(ast_module)

    bytecode = compile(ast_module, ctx.filename, "exec")
    if collect_bytecode:
        collect_bytecode(bytecode)
    exec(bytecode, module.__dict__)
    return getattr(module, final_wrapped_name)()


def _incremental_compile_module(
    optimizer: PythonASTOptimizer,
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
    module = optimizer.visit(module)
    ast.fix_missing_locations(module)

    _emit_ast_string(module)

    bytecode = compile(module, source_filename, "exec")
    if collect_bytecode:
        collect_bytecode(bytecode)
    exec(bytecode, mod.__dict__)


def _bootstrap_module(
    gctx: GeneratorContext,
    optimizer: PythonASTOptimizer,
    mod: types.ModuleType,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> None:
    """Bootstrap a new module with imports and other boilerplate."""
    _incremental_compile_module(
        optimizer,
        py_module_preamble(gctx),
        mod,
        source_filename=gctx.filename,
        collect_bytecode=collect_bytecode,
    )
    mod.__basilisp_bootstrapped__ = True  # type: ignore


def compile_module(
    forms: Iterable[ReaderForm],
    ctx: CompilerContext,
    module: types.ModuleType,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> None:
    """Compile an entire Basilisp module into Python bytecode which can be
    executed as a Python module.

    This function is designed to generate bytecode which can be used for the
    Basilisp import machinery, to allow callers to import Basilisp modules from
    Python code.
    """
    _bootstrap_module(ctx.generator_context, ctx.py_ast_optimizer, module)

    for form in forms:
        nodes = gen_py_ast(
            ctx.generator_context, analyze_form(ctx.analyzer_context, form)
        )
        _incremental_compile_module(
            ctx.py_ast_optimizer,
            nodes,
            module,
            source_filename=ctx.filename,
            collect_bytecode=collect_bytecode,
        )


def compile_bytecode(
    code: List[types.CodeType],
    gctx: GeneratorContext,
    optimizer: PythonASTOptimizer,
    module: types.ModuleType,
) -> None:
    """Compile cached bytecode into the given module.

    The Basilisp import hook attempts to cache bytecode while compiling Basilisp
    namespaces. When the cached bytecode is reloaded from disk, it needs to be
    compiled within a bootstrapped module. This function bootstraps the module
    and then proceeds to compile a collection of bytecodes into the module."""
    _bootstrap_module(gctx, optimizer, module)
    for bytecode in code:
        exec(bytecode, module.__dict__)
