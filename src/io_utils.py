"""JSONL readers and file identities used by the data pipelines."""

import ast
import hashlib
import json
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_jsonl(path: str | Path):
    """Yield the zero-based file line number and record for each nonblank line."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line.strip():
                yield line_index, json.loads(line)


class _WithoutDocstrings(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
        return node

    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip


def source_code_identity(paths) -> dict:
    """Hash Python syntax without comments, formatting, or docstrings.

    Callers must include every local module that affects their calculation.
    External library versions are recorded separately by each pipeline.
    """
    files = {}
    for path in sorted(map(Path, paths)):
        if path.name in files:
            raise ValueError(f"Duplicate source filename: {path.name}")
        tree = _WithoutDocstrings().visit(ast.parse(path.read_text(encoding="utf-8")))
        syntax = ast.dump(tree, include_attributes=False)
        files[path.name] = hashlib.sha256(syntax.encode("utf-8")).hexdigest()
    return {"format": "python-ast-v1", "files": files}
