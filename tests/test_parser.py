"""
Tests for the Python AST parser.

Tests focus on:
  - Entity extraction (files, functions, classes)
  - Relationship detection (imports, calls, inheritance, containment)
  - Parameter counting, docstring detection, complexity estimation
  - Error resilience (syntax errors should not crash the parser)
"""

from __future__ import annotations

import textwrap
import tempfile
from pathlib import Path

import pytest

from graphguard.parser.python_parser import PythonParser, ParseResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal in-memory repo for testing."""
    return tmp_path


def _write(tmp_repo: Path, rel_path: str, code: str) -> Path:
    p = tmp_repo / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(code), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Helper: parse a single snippet
# ---------------------------------------------------------------------------

def _parse_snippet(tmp_path: Path, code: str, filename: str = "module.py") -> ParseResult:
    _write(tmp_path, filename, code)
    return PythonParser().parse(tmp_path)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

class TestEntityExtraction:
    def test_file_entity_created(self, tmp_repo: Path) -> None:
        result = _parse_snippet(tmp_repo, "x = 1\n")
        file_entities = [e for e in result.entities if e.entity_type == "file"]
        assert len(file_entities) == 1

    def test_function_extracted(self, tmp_repo: Path) -> None:
        result = _parse_snippet(tmp_repo, "def greet(name): pass\n")
        funcs = [e for e in result.entities if e.entity_type == "function"]
        assert any(e.name == "greet" for e in funcs)

    def test_class_extracted(self, tmp_repo: Path) -> None:
        result = _parse_snippet(tmp_repo, "class Dog:\n    pass\n")
        classes = [e for e in result.entities if e.entity_type == "class"]
        assert any(e.name == "Dog" for e in classes)

    def test_async_function_extracted(self, tmp_repo: Path) -> None:
        result = _parse_snippet(tmp_repo, "async def fetch(): pass\n")
        funcs = [e for e in result.entities if e.entity_type == "function"]
        assert any(e.name == "fetch" for e in funcs)

    def test_multiple_files_parsed(self, tmp_repo: Path) -> None:
        _write(tmp_repo, "a.py", "def foo(): pass\n")
        _write(tmp_repo, "b.py", "def bar(): pass\n")
        result = PythonParser().parse(tmp_repo)
        file_entities = [e for e in result.entities if e.entity_type == "file"]
        assert len(file_entities) == 2

    def test_skips_pycache(self, tmp_repo: Path) -> None:
        _write(tmp_repo, "good.py", "x = 1\n")
        _write(tmp_repo, "__pycache__/cached.py", "x = 1\n")
        result = PythonParser().parse(tmp_repo)
        paths = [e.file_path for e in result.entities]
        assert not any("__pycache__" in p for p in paths)


# ---------------------------------------------------------------------------
# Code metrics
# ---------------------------------------------------------------------------

class TestCodeMetrics:
    def test_param_count(self, tmp_repo: Path) -> None:
        result = _parse_snippet(tmp_repo, "def add(a, b, c): pass\n")
        func = next(e for e in result.entities if e.name == "add")
        assert func.num_params == 3

    def test_self_excluded_from_params(self, tmp_repo: Path) -> None:
        code = "class Foo:\n    def method(self, x): pass\n"
        result = _parse_snippet(tmp_repo, code)
        method = next(e for e in result.entities if e.name == "method")
        assert method.num_params == 1  # only x, not self

    def test_docstring_detected(self, tmp_repo: Path) -> None:
        code = 'def greet():\n    """Say hello."""\n    pass\n'
        result = _parse_snippet(tmp_repo, code)
        func = next(e for e in result.entities if e.name == "greet")
        assert func.has_docstring is True

    def test_no_docstring_detected(self, tmp_repo: Path) -> None:
        result = _parse_snippet(tmp_repo, "def greet():\n    pass\n")
        func = next(e for e in result.entities if e.name == "greet")
        assert func.has_docstring is False

    def test_complexity_increases_with_branches(self, tmp_repo: Path) -> None:
        code = textwrap.dedent("""\
            def process(x):
                if x > 0:
                    for i in range(x):
                        while i > 0:
                            i -= 1
                return x
        """)
        result = _parse_snippet(tmp_repo, code)
        func = next(e for e in result.entities if e.name == "process")
        # 1 base + if + for + while = 4
        assert func.complexity >= 4

    def test_simple_function_complexity_is_one(self, tmp_repo: Path) -> None:
        result = _parse_snippet(tmp_repo, "def identity(x): return x\n")
        func = next(e for e in result.entities if e.name == "identity")
        assert func.complexity == 1


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

class TestRelationships:
    def test_import_relationship(self, tmp_repo: Path) -> None:
        result = _parse_snippet(tmp_repo, "import os\n")
        imports = [r for r in result.relationships if r.relationship_type == "imports"]
        assert any("os" in r.target_id for r in imports)

    def test_from_import_relationship(self, tmp_repo: Path) -> None:
        result = _parse_snippet(tmp_repo, "from pathlib import Path\n")
        imports = [r for r in result.relationships if r.relationship_type == "imports"]
        assert any("pathlib" in r.target_id for r in imports)

    def test_containment_relationship(self, tmp_repo: Path) -> None:
        result = _parse_snippet(tmp_repo, "def foo(): pass\n")
        contains = [r for r in result.relationships if r.relationship_type == "contains"]
        assert len(contains) == 1
        assert "foo" in contains[0].target_id

    def test_class_containment(self, tmp_repo: Path) -> None:
        result = _parse_snippet(tmp_repo, "class Dog:\n    pass\n")
        contains = [r for r in result.relationships if r.relationship_type == "contains"]
        assert any("Dog" in r.target_id for r in contains)

    def test_inheritance_relationship(self, tmp_repo: Path) -> None:
        code = "class Animal:\n    pass\nclass Dog(Animal):\n    pass\n"
        result = _parse_snippet(tmp_repo, code)
        inherits = [r for r in result.relationships if r.relationship_type == "inherits"]
        assert len(inherits) == 1
        assert "Dog" in inherits[0].source_id
        assert "Animal" in inherits[0].target_id

    def test_internal_call_relationship(self, tmp_repo: Path) -> None:
        code = "def helper(): pass\ndef main():\n    helper()\n"
        result = _parse_snippet(tmp_repo, code)
        calls = [r for r in result.relationships if r.relationship_type == "calls"]
        assert any("helper" in r.target_id for r in calls)

    def test_imported_base_resolves_to_module(self, tmp_repo: Path) -> None:
        """A class inheriting from an imported symbol points at the module,
        not a fabricated local class node (regression test)."""
        code = "from enum import Enum\nclass Color(Enum):\n    RED = 1\n"
        result = _parse_snippet(tmp_repo, code)
        inherits = [r for r in result.relationships if r.relationship_type == "inherits"]
        assert len(inherits) == 1
        assert inherits[0].target_id == "module::enum"
        # And no fake class:: node should be invented for Enum
        bad = [e for e in result.entities if e.node_id.endswith("::Enum")]
        assert bad == []

    def test_dotted_base_resolves_to_module(self, tmp_repo: Path) -> None:
        code = "import abc\nclass Base(abc.ABC):\n    pass\n"
        result = _parse_snippet(tmp_repo, code)
        inherits = [r for r in result.relationships if r.relationship_type == "inherits"]
        assert any(r.target_id == "module::abc" for r in inherits)


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------

class TestErrorResilience:
    def test_syntax_error_file_skipped(self, tmp_repo: Path) -> None:
        _write(tmp_repo, "broken.py", "def (: \n  !!invalid\n")
        _write(tmp_repo, "good.py", "x = 1\n")
        result = PythonParser().parse(tmp_repo)
        assert len(result.errors) >= 1
        # Good file should still be parsed
        file_paths = [e.file_path for e in result.entities if e.entity_type == "file"]
        assert any("good.py" in p for p in file_paths)

    def test_empty_file_handled(self, tmp_repo: Path) -> None:
        _write(tmp_repo, "empty.py", "")
        result = PythonParser().parse(tmp_repo)
        assert len(result.errors) == 0
