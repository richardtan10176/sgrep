import ast


class sgrepVisitor(ast.NodeVisitor):
    """Chunks a Python source file into one embeddable record per function.

    Classes are not chunks themselves; they only shape the context header of
    the methods inside them ("Class: Foo > Function: bar").
    """

    def __init__(self):
        self.scope_stack = []
        self.chunks = []
        self.source_code = None

    def _context_header(self, name: str) -> str:
        return " > ".join(self.scope_stack + [f"Function: {name}"])

    def visit_ClassDef(self, node):
        self.scope_stack.append(f"Class: {node.name}")
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node):
        func_source = ast.get_source_segment(self.source_code, node)
        if func_source is None:
            return

        self.chunks.append({
            "context": self._context_header(node.name),
            "code": func_source,
            "line_start": node.lineno,
            "line_end": node.end_lineno,
        })
        # deliberately not recursing: this chunk's source already contains
        # every nested function and method, so descending would index the same
        # text two or three times and let the duplicates crowd out the top-k

    visit_AsyncFunctionDef = visit_FunctionDef

    def parse(self, source_code):
        self.source_code = source_code
        self.chunks = []
        self.scope_stack = []
        tree = ast.parse(source_code)
        self.visit(tree)
        return self.chunks
