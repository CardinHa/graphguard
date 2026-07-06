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

from graphguard.parser.python_parser import (
    PythonParser,
    ParseResult,
    _path_to_module,
    _resolve_relative_module,
)


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


# ---------------------------------------------------------------------------
# Path/module helpers
# ---------------------------------------------------------------------------

class TestPathHelpers:
    def test_path_to_module_with_src_prefix(self) -> None:
        assert _path_to_module("src/requests/utils.py") == "requests.utils"

    def test_path_to_module_no_prefix(self) -> None:
        assert _path_to_module("mypackage/core.py") == "mypackage.core"

    def test_path_to_module_init(self) -> None:
        assert _path_to_module("src/requests/__init__.py") == "requests"

    def test_path_to_module_nested(self) -> None:
        assert _path_to_module("src/a/b/c.py") == "a.b.c"

    def test_resolve_relative_same_package(self) -> None:
        # from . import utils  in requests/auth.py
        assert _resolve_relative_module("src/requests/auth.py", None, 1) == "requests"

    def test_resolve_relative_submodule(self) -> None:
        # from .utils import func  in requests/auth.py
        assert _resolve_relative_module("src/requests/auth.py", "utils", 1) == "requests.utils"

    def test_resolve_relative_parent(self) -> None:
        # from .. import core  in requests/packages/auth.py
        assert _resolve_relative_module("src/requests/packages/auth.py", "core", 2) == "requests.core"

    def test_resolve_relative_from_init(self) -> None:
        # from . import utils  in requests/__init__.py
        assert _resolve_relative_module("src/requests/__init__.py", "utils", 1) == "requests.utils"


# ---------------------------------------------------------------------------
# Cross-file call resolution
# ---------------------------------------------------------------------------

class TestCrossFileCallResolution:
    def test_direct_import_call_resolved(self, tmp_repo: Path) -> None:
        """from helper_mod import helper; def main(): helper() -> cross-file calls edge."""
        _write(tmp_repo, "helper_mod.py", "def helper():\n    pass\n")
        _write(tmp_repo, "caller_mod.py",
               "from helper_mod import helper\ndef main():\n    helper()\n")
        result = PythonParser().parse(tmp_repo)
        calls = [r for r in result.relationships if r.relationship_type == "calls"]
        cross = [
            r for r in calls
            if "main" in r.source_id and "helper" in r.target_id and "helper_mod" in r.target_id
        ]
        assert len(cross) == 1, f"Expected 1 cross-file call edge, got {len(cross)}: {cross}"

    def test_attribute_call_resolved(self, tmp_repo: Path) -> None:
        """import utils; utils.process() -> calls edge to utils.process."""
        _write(tmp_repo, "utils.py", "def process():\n    pass\n")
        _write(tmp_repo, "main.py",
               "from utils import process\ndef run():\n    process()\n")
        result = PythonParser().parse(tmp_repo)
        calls = [r for r in result.relationships if r.relationship_type == "calls"]
        assert any("process" in r.target_id and "utils" in r.target_id for r in calls)

    def test_relative_import_submodule_call(self, tmp_repo: Path) -> None:
        """from . import utils; utils.process() -> cross-file calls edge in a package."""
        (tmp_repo / "mypkg").mkdir()
        _write(tmp_repo, "mypkg/__init__.py", "")
        _write(tmp_repo, "mypkg/utils.py", "def process():\n    pass\n")
        _write(tmp_repo, "mypkg/main.py",
               "from . import utils\ndef run():\n    utils.process()\n")
        result = PythonParser().parse(tmp_repo)
        calls = [r for r in result.relationships if r.relationship_type == "calls"]
        assert any("process" in r.target_id for r in calls), \
            f"Expected calls edge to process, got: {[r.target_id for r in calls]}"

    def test_relative_symbol_import_call(self, tmp_repo: Path) -> None:
        """from .utils import process; process() -> cross-file calls edge."""
        (tmp_repo / "pkg").mkdir()
        _write(tmp_repo, "pkg/__init__.py", "")
        _write(tmp_repo, "pkg/utils.py", "def process():\n    pass\n")
        _write(tmp_repo, "pkg/main.py",
               "from .utils import process\ndef run():\n    process()\n")
        result = PythonParser().parse(tmp_repo)
        calls = [r for r in result.relationships if r.relationship_type == "calls"]
        assert any("process" in r.target_id and "utils" in r.target_id for r in calls), \
            f"Expected cross-file edge to utils.process, calls: {[r.target_id for r in calls]}"

    def test_external_library_call_not_resolved(self, tmp_repo: Path) -> None:
        """Calls to external libraries should not create spurious entity nodes."""
        _write(tmp_repo, "consumer.py",
               "import os\ndef run():\n    os.path.join('a', 'b')\n")
        result = PythonParser().parse(tmp_repo)
        entity_ids = {e.node_id for e in result.entities}
        # No fake function node for os.path.join should appear
        assert not any("func::consumer.py::join" in nid for nid in entity_ids)

    def test_no_self_loop_calls(self, tmp_repo: Path) -> None:
        """Recursive calls should not create self-loop edges."""
        _write(tmp_repo, "module.py",
               "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)\n")
        result = PythonParser().parse(tmp_repo)
        calls = [r for r in result.relationships if r.relationship_type == "calls"]
        # Recursive call: source and target should be the same node — but
        # _add_rel skips src == tgt, so there should be no self-loop
        self_loops = [r for r in calls if r.source_id == r.target_id]
        assert self_loops == []

    def test_forward_reference_call_resolved(self, tmp_repo: Path) -> None:
        """def a(): b() defined BEFORE def b(): ... must still produce the
        a -> b calls edge (regression test: calls must not depend on
        definition order within a file)."""
        code = "def a():\n    b()\n\ndef b():\n    pass\n"
        result = _parse_snippet(tmp_repo, code)
        calls = [r for r in result.relationships if r.relationship_type == "calls"]
        assert any(
            r.source_id.endswith("::a") and r.target_id.endswith("::b")
            for r in calls
        ), f"Expected forward-reference a->b call edge, got: {[(r.source_id, r.target_id) for r in calls]}"

    def test_nested_function_call_not_double_counted(self, tmp_repo: Path) -> None:
        """A call inside a nested function must be attributed to the nested
        function only — not also to the enclosing outer function."""
        code = textwrap.dedent("""\
            def helper():
                pass

            def outer():
                def inner():
                    helper()
                inner()
        """)
        result = _parse_snippet(tmp_repo, code)
        calls = [r for r in result.relationships if r.relationship_type == "calls"]
        outer_to_helper = [
            r for r in calls
            if r.source_id.endswith("::outer") and r.target_id.endswith("::helper")
        ]
        inner_to_helper = [
            r for r in calls
            if r.source_id.endswith("::inner") and r.target_id.endswith("::helper")
        ]
        outer_to_inner = [
            r for r in calls
            if r.source_id.endswith("::outer") and r.target_id.endswith("::inner")
        ]
        assert outer_to_helper == [], "helper() was double-counted against the outer function"
        assert len(inner_to_helper) == 1
        assert len(outer_to_inner) == 1

    def test_absolute_submodule_import_attribute_call_resolved(self, tmp_repo: Path) -> None:
        """from pkg import submodule; submodule.func() must resolve into
        pkg/submodule.py, not be looked up (and dropped/misattributed) in
        pkg/__init__.py."""
        (tmp_repo / "pkg").mkdir()
        _write(tmp_repo, "pkg/__init__.py", "")
        _write(tmp_repo, "pkg/submodule.py", "def func():\n    pass\n")
        _write(tmp_repo, "pkg/main.py",
               "from pkg import submodule\ndef run():\n    submodule.func()\n")
        result = PythonParser().parse(tmp_repo)
        calls = [r for r in result.relationships if r.relationship_type == "calls"]
        assert any(r.target_id == "func::pkg/submodule.py::func" for r in calls), (
            f"Expected edge into pkg/submodule.py::func, got: "
            f"{[r.target_id for r in calls]}"
        )

    def test_dotted_import_attribute_call_resolved(self, tmp_repo: Path) -> None:
        """import a.b; a.b.func() must resolve into a/b.py. `import a.b`
        binds the name `a`, not the literal string "a.b" — the parser must
        key the binding accordingly."""
        (tmp_repo / "a").mkdir()
        _write(tmp_repo, "a/__init__.py", "")
        _write(tmp_repo, "a/b.py", "def func():\n    pass\n")
        _write(tmp_repo, "a/main.py",
               "import a.b\ndef run():\n    a.b.func()\n")
        result = PythonParser().parse(tmp_repo)
        calls = [r for r in result.relationships if r.relationship_type == "calls"]
        assert any(r.target_id == "func::a/b.py::func" for r in calls), (
            f"Expected edge into a/b.py::func, got: {[r.target_id for r in calls]}"
        )
