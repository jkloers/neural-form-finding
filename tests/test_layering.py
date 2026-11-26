"""The package layering the repo depends on: nff -> nothing above it; nff_paper -> nff; legacy -> nobody.

    nff/        the research pipeline. Must never import nff_paper or legacy.
    nff_paper/  the article and the deck. May import nff, never legacy.
    legacy/     frozen archive. Nothing imports it.

Imports are read from the AST, so lazy imports inside functions count and docstrings do not.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

FORBIDDEN = {
    "nff": ("nff_paper", "legacy"),
    "nff_paper": ("legacy",),
}


def _imported_modules(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.lineno, node.module


def _python_files(package: str):
    for path in sorted((ROOT / package).rglob("*.py")):
        if "private" in path.parts or "__pycache__" in path.parts:
            continue
        yield path


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_package_does_not_import_upward(package):
    violations = []
    for path in _python_files(package):
        for lineno, module in _imported_modules(path):
            top = module.split(".")[0]
            if top in FORBIDDEN[package]:
                violations.append(f"{path.relative_to(ROOT)}:{lineno} imports {module}")
    assert not violations, "layering violated:\n" + "\n".join(violations)
