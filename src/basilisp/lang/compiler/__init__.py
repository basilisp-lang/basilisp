import ast
import itertools
import os
import types
from ast import unparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Optional

from basilisp.lang import list as llist
from basilisp.lang import map as lmap
from basilisp.lang import runtime as runtime
from basilisp.lang import symbol as sym
from basilisp.lang.compiler.analyzer import (  # noqa
    GENERATE_AUTO_INLINES,
    INLINE_FUNCTIONS,
    WARN_ON_ARITY_MISMATCH,
    WARN_ON_NON_DYNAMIC_SET,
    WARN_ON_SHADOWED_NAME,
    WARN_ON_SHADOWED_VAR,
    WARN_ON_UNUSED_NAMES,
    AnalyzerContext,
    analyze_form,
    macroexpand,
    macroexpand_1,
)
from basilisp.lang.compiler.constants import SpecialForm
from basilisp.lang.compiler.exception import CompilerException, CompilerPhase  # noqa
from basilisp.lang.compiler.generator import (
    USE_VAR_INDIRECTION,
    WARN_ON_VAR_INDIRECTION,
    GeneratedPyAST,
    GeneratorContext,
)
from basilisp.lang.compiler.generator import expressionize as _expressionize  # noqa
from basilisp.lang.compiler.generator import (
    gen_py_ast,
    py_module_preamble,
)
from basilisp.lang.compiler.generator import statementize as _statementize
from basilisp.lang.compiler.optimizer import PythonASTOptimizer
from basilisp.lang.interfaces import ISeq
from basilisp.lang.runtime import BasilispModule
from basilisp.lang.typing import CompilerOpts, ReaderForm
from basilisp.lang.util import genname
from basilisp.util import Maybe

_DEFAULT_FN = "__lisp_expr__"


def to_py_str(t: ast.AST) -> str:
    """Return a string of the Python code which would generate the input
    AST node."""
    return unparse(t) + "\n\n"


BytecodeCollector = Callable[[types.CodeType], None]


class CompilerContext:
    __slots__ = ("_filename", "_actx", "_gctx", "_optimizer")

    def __init__(self, filename: str, opts: Optional[CompilerOpts] = None):
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


def compiler_opts(  # pylint: disable=too-many-arguments
    generate_auto_inlines: Optional[bool] = None,
    inline_functions: Optional[bool] = None,
    warn_on_arity_mismatch: Optional[bool] = None,
    warn_on_shadowed_name: Optional[bool] = None,
    warn_on_shadowed_var: Optional[bool] = None,
    warn_on_unused_names: Optional[bool] = None,
    warn_on_non_dynamic_set: Optional[bool] = None,
    use_var_indirection: Optional[bool] = None,
    warn_on_var_indirection: Optional[bool] = None,
) -> CompilerOpts:
    """Return a map of compiler options with defaults applied."""
    return lmap.map(
        {
            # Analyzer options
            GENERATE_AUTO_INLINES: Maybe(generate_auto_inlines).or_else_get(True),
            INLINE_FUNCTIONS: Maybe(inline_functions).or_else_get(True),
            WARN_ON_ARITY_MISMATCH: Maybe(warn_on_arity_mismatch).or_else_get(True),
            WARN_ON_SHADOWED_NAME: Maybe(warn_on_shadowed_name).or_else_get(False),
            WARN_ON_SHADOWED_VAR: Maybe(warn_on_shadowed_var).or_else_get(False),
            WARN_ON_UNUSED_NAMES: Maybe(warn_on_unused_names).or_else_get(True),
            WARN_ON_NON_DYNAMIC_SET: Maybe(warn_on_non_dynamic_set).or_else_get(True),
            # Generator options
            USE_VAR_INDIRECTION: Maybe(use_var_indirection).or_else_get(False),
            WARN_ON_VAR_INDIRECTION: Maybe(warn_on_var_indirection).or_else_get(True),
        }
    )


def _emit_ast_string(
    module: ast.AST,
) -> None:  # pragma: no cover
    """Emit the generated Python AST string either to standard out or to the
    *generated-python* dynamic Var for the current namespace. If the
    BASILISP_EMIT_GENERATED_PYTHON env var is not set True, this method is a
    no-op."""
    # TODO: eventually, this default should become "false" but during this
    #       period of heavy development, having it set to "true" by default
    #       is tremendously useful
    if os.getenv("BASILISP_EMIT_GENERATED_PYTHON", "true").lower() != "true":
        return

    if runtime.print_generated_python():
        print(to_py_str(module))
    else:
        runtime.add_generated_python(to_py_str(module))


def _flatmap_forms(forms: Iterable[ReaderForm]) -> Iterable[ReaderForm]:
    """Flatmap over an iterable of forms, unrolling any top-level `do` forms"""
    for form in forms:
        if isinstance(form, ISeq) and form.first == SpecialForm.DO:
            yield from _flatmap_forms(form.rest)
        else:
            yield form


_sentinel = object()


def compile_and_exec_form(
    form: ReaderForm,
    ctx: CompilerContext,
    ns: runtime.Namespace,
    wrapped_fn_name: str = _DEFAULT_FN,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> Any:
    """Compile and execute the given form. This function will be most useful
    for the REPL and testing purposes. Returns the result of the executed expression.

    Callers may override the wrapped function name, which is used by the
    REPL to evaluate the result of an expression and print it back out."""
    if form is None:
        return None

    if not ns.module.__basilisp_bootstrapped__:
        _bootstrap_module(ctx.generator_context, ctx.py_ast_optimizer, ns.module)

    last = _sentinel
    for unrolled_form in _flatmap_forms([form]):
        final_wrapped_name = genname(wrapped_fn_name)
        lisp_ast = analyze_form(ctx.analyzer_context, unrolled_form)
        py_ast = gen_py_ast(ctx.generator_context, lisp_ast)
        form_ast = list(
            map(
                _statementize,
                itertools.chain(
                    py_ast.dependencies,
                    [
                        _expressionize(
                            GeneratedPyAST(node=py_ast.node), final_wrapped_name
                        )
                    ],
                ),
            )
        )

        ast_module = ast.Module(body=form_ast, type_ignores=[])
        ast_module = ctx.py_ast_optimizer.visit(ast_module)
        ast.fix_missing_locations(ast_module)

        _emit_ast_string(ast_module)

        bytecode = compile(ast_module, ctx.filename, "exec")
        if collect_bytecode:
            collect_bytecode(bytecode)
        exec(bytecode, ns.module.__dict__)  # pylint: disable=exec-used  # nosec 6102
        try:
            last = getattr(ns.module, final_wrapped_name)()
        finally:
            del ns.module.__dict__[final_wrapped_name]

    assert last is not _sentinel, "Must compile at least one form"
    return last


def _incremental_compile_module(
    optimizer: PythonASTOptimizer,
    py_ast: GeneratedPyAST,
    module: BasilispModule,
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

    module_ast = ast.Module(body=list(module_body), type_ignores=[])
    module_ast = optimizer.visit(module_ast)
    ast.fix_missing_locations(module_ast)

    _emit_ast_string(module_ast)

    bytecode = compile(module_ast, source_filename, "exec")
    if collect_bytecode:
        collect_bytecode(bytecode)
    exec(bytecode, module.__dict__)  # pylint: disable=exec-used  # nosec 6102


def _bootstrap_module(
    gctx: GeneratorContext,
    optimizer: PythonASTOptimizer,
    module: BasilispModule,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> None:
    """Bootstrap a new module with imports and other boilerplate."""
    _incremental_compile_module(
        optimizer,
        py_module_preamble(),
        module,
        source_filename=gctx.filename,
        collect_bytecode=collect_bytecode,
    )
    module.__basilisp_bootstrapped__ = True


def compile_module(
    forms: Iterable[ReaderForm],
    ctx: CompilerContext,
    module: BasilispModule,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> None:
    """Compile an entire Basilisp module into Python bytecode which can be
    executed as a Python module.

    This function is designed to generate bytecode which can be used for the
    Basilisp import machinery, to allow callers to import Basilisp modules from
    Python code.
    """
    _bootstrap_module(ctx.generator_context, ctx.py_ast_optimizer, module)

    for form in _flatmap_forms(forms):
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
    code: list[types.CodeType],
    gctx: GeneratorContext,
    optimizer: PythonASTOptimizer,
    module: BasilispModule,
) -> None:
    """Compile cached bytecode into the given module.

    The Basilisp import hook attempts to cache bytecode while compiling Basilisp
    namespaces. When the cached bytecode is reloaded from disk, it needs to be
    compiled within a bootstrapped module. This function bootstraps the module
    and then proceeds to compile a collection of bytecodes into the module."""
    _bootstrap_module(gctx, optimizer, module)
    for bytecode in code:
        exec(bytecode, module.__dict__)  # pylint: disable=exec-used  # nosec 6102


_LOAD_SYM = sym.symbol("load", ns=runtime.CORE_NS)
_LOAD_FILE_SYM = sym.symbol("load-file", ns=runtime.CORE_NS)


def load(
    path: str,
    ctx: CompilerContext,
    ns: runtime.Namespace,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> Any:
    """Call :lpy:fn:`basilisp.core/load` with the given ``path``, returning the
    result."""
    return compile_and_exec_form(
        llist.l(_LOAD_SYM, path), ctx, ns, collect_bytecode=collect_bytecode
    )


def load_file(
    path: Path,
    ctx: CompilerContext,
    ns: runtime.Namespace,
    collect_bytecode: Optional[BytecodeCollector] = None,
) -> Any:
    """Call :lpy:fn:`basilisp.core/load-file` with the given ``path``, returning the
    result."""
    return compile_and_exec_form(
        llist.l(_LOAD_FILE_SYM, path.as_posix()),
        ctx,
        ns,
        collect_bytecode=collect_bytecode,
    )
