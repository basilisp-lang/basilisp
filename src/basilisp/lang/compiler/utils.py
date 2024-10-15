import ast
import sys
from functools import partial

if sys.version_info >= (3, 12):
    ast_AsyncFunctionDef = partial(ast.AsyncFunctionDef, type_params=[])
    ast_ClassDef = partial(ast.ClassDef, type_params=[])
    ast_FunctionDef = partial(ast.FunctionDef, type_params=[])
else:
    ast_AsyncFunctionDef = ast.AsyncFunctionDef
    ast_ClassDef = ast.ClassDef
    ast_FunctionDef = ast.FunctionDef


__all__ = ("ast_ClassDef", "ast_FunctionDef")
