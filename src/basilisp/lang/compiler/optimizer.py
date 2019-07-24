from typing import Iterable, List, Optional

import basilisp._pyast as ast


def _filter_dead_code(nodes: Iterable[ast.AST]) -> List[ast.AST]:
    """Return a list of body nodes, trimming out unreachable code (any
    statements appearing after `break`, `continue`, and `return` nodes)."""
    new_nodes: List[ast.AST] = []
    for node in nodes:
        if isinstance(node, (ast.Break, ast.Continue, ast.Return)):
            new_nodes.append(node)
            break
        new_nodes.append(node)
    return new_nodes


class PythonASTOptimizer(ast.NodeTransformer):
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

    def visit_If(self, node: ast.If) -> Optional[ast.AST]:
        """Eliminate dead code from if/elif bodies."""
        new_node = self.generic_visit(node)
        assert isinstance(new_node, ast.If)
        return ast.copy_location(
            ast.If(
                test=new_node.test,
                body=_filter_dead_code(new_node.body),
                orelse=_filter_dead_code(new_node.orelse),
            ),
            new_node,
        )

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
