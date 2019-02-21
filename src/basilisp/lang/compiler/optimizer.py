import ast
from typing import Optional


class PythonASTOptimizer(ast.NodeTransformer):
    def visit_Expr(self, node: ast.Expr) -> Optional[ast.Expr]:
        """Eliminate no-op constant expressions which are in the tree
        as standalone statements."""
        if isinstance(
            node.value,
            (
                ast.Constant,  # type: ignore
                ast.Name,
                ast.NameConstant,
                ast.Num,
                ast.Str,
            ),
        ):
            return None
        return node
