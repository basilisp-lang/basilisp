from collections import deque
from contextlib import contextmanager
from typing import Deque, Iterable, List, Optional, Set

from basilisp import _pyast as ast


def _filter_dead_code(nodes: Iterable[ast.AST]) -> List[ast.AST]:
    """Return a list of body nodes, trimming out unreachable code (any
    statements appearing after `break`, `continue`, `raise`, and `return`
    nodes)."""
    new_nodes: List[ast.AST] = []
    for node in nodes:
        if isinstance(node, (ast.Break, ast.Continue, ast.Raise, ast.Return)):
            new_nodes.append(node)
            break
        new_nodes.append(node)
    return new_nodes


class PythonASTOptimizer(ast.NodeTransformer):
    __slots__ = ("_global_ctx",)

    def __init__(self):
        self._global_ctx: Deque[Set[str]] = deque([set()])

    @contextmanager
    def _new_global_context(self):
        """Context manager which sets a new Python `global` context."""
        self._global_ctx.append(set())
        try:
            yield
        finally:
            self._global_ctx.pop()

    @property
    def _global_context(self) -> Set[str]:
        """Return the current Python `global` context."""
        return self._global_ctx[-1]

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Optional[ast.AST]:
        """Eliminate dead code from except handler bodies."""
        new_node = self.generic_visit(node)
        assert isinstance(new_node, ast.ExceptHandler)
        return ast.copy_location(
            ast.ExceptHandler(
                type=new_node.type,
                name=new_node.name,
                body=_filter_dead_code(new_node.body),
            ),
            new_node,
        )

    def visit_Expr(self, node: ast.Expr) -> Optional[ast.Expr]:
        """Eliminate no-op constant expressions which are in the tree
        as standalone statements."""
        if isinstance(node.value, (ast.Constant, ast.Name)):
            return None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Optional[ast.AST]:
        """Eliminate dead code from function bodies."""
        with self._new_global_context():
            new_node = self.generic_visit(node)
        assert isinstance(new_node, ast.FunctionDef)
        return ast.copy_location(
            ast.FunctionDef(
                name=new_node.name,
                args=new_node.args,
                body=_filter_dead_code(new_node.body),
                decorator_list=new_node.decorator_list,
                returns=new_node.returns,
            ),
            new_node,
        )

    def visit_Global(self, node: ast.Global) -> Optional[ast.Global]:
        """Eliminate redundant name declarations inside a Python `global` statement.

        Python `global` statements may only refer to a name prior to its declaration.
        Global contexts track names in prior `global` declarations and eliminate
        redundant names in `global` declarations. If all of the names in the current
        `global` statement are redundant, the entire node will be omitted."""
        new_names = set(node.names) - self._global_context
        self._global_context.update(new_names)
        return (
            ast.copy_location(ast.Global(names=list(new_names)), node)
            if new_names
            else None
        )

    def visit_If(self, node: ast.If) -> Optional[ast.AST]:
        """Eliminate dead code from if/elif bodies.

        If the new `if` statement `body` is empty after eliminating dead code, replace
        the body with the `orelse` body and negate the `if` condition.

        If both the `body` and `orelse` body are empty, eliminate the node from the
        tree."""
        new_node = self.generic_visit(node)
        assert isinstance(new_node, ast.If)

        new_body = _filter_dead_code(new_node.body)
        new_orelse = _filter_dead_code(new_node.orelse)

        if new_body:
            ifstmt = ast.If(
                test=new_node.test,
                body=new_body,
                orelse=new_orelse,
            )
        elif new_orelse:
            ifstmt = ast.If(
                test=ast.UnaryOp(op=ast.Not(), operand=new_node.test),
                body=new_orelse,
                orelse=[],
            )
        else:
            return None

        return ast.copy_location(ifstmt, new_node)

    def visit_While(self, node: ast.While) -> Optional[ast.AST]:
        """Eliminate dead code from while bodies."""
        new_node = self.generic_visit(node)
        assert isinstance(new_node, ast.While)
        return ast.copy_location(
            ast.While(
                test=new_node.test,
                body=_filter_dead_code(new_node.body),
                orelse=_filter_dead_code(new_node.orelse),
            ),
            new_node,
        )

    def visit_Try(self, node: ast.Try) -> Optional[ast.AST]:
        """Eliminate dead code from except try bodies."""
        new_node = self.generic_visit(node)
        assert isinstance(new_node, ast.Try)
        return ast.copy_location(
            ast.Try(
                body=_filter_dead_code(new_node.body),
                handlers=new_node.handlers,
                orelse=_filter_dead_code(new_node.orelse),
                finalbody=_filter_dead_code(new_node.finalbody),
            ),
            new_node,
        )
