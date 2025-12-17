import ast

class sgrepVisitor(ast.NodeVisitor):
    def __init__(self):
        self.scope_stack = []
        self.chunks = []

    def _get_context_header(self):
        return " > ".join(self.scope_stack)

    def visit_ClassDef(self, node):
        self.scope_stack.append(f"Class: {node.name}")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node):
        self.scope_stack.append(f"Function: {node.name}")
        func_source = ast.get_source_segment(self.source_code, node)
        header = self._get_context_header()
        self.chunks.append({
            "context": header,
            "code": func_source,
            "line_start": node.lineno
        })
        self.generic_visit(node)
        self.scope_stack.pop()

    def parse(self, source_code):
        self.source_code = source_code
        tree = ast.parse(source_code)
        self.visit(tree)
        return self.chunks
