"""
AST-based Python repository parser.

Converts a directory of .py files into ParsedEntity and ParsedRelationship
objects that the graph builder later consumes.

Node ID format
--------------
  file::rel/path/to/file.py
  func::rel/path/to/file.py::function_name
  class::rel/path/to/file.py::ClassName
  module::module_name          (external / third-party import)
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from graphguard.utils.config import SKIP_DIRS
from graphguard.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParsedEntity:
    """A code entity extracted from a Python source file."""
    node_id: str            # globally unique identifier
    name: str
    entity_type: str        # "file" | "function" | "class" | "module"
    file_path: str          # repo-relative path
    line_number: int
    lines_of_code: int = 0
    num_params: int = 0
    has_docstring: bool = False
    complexity: int = 1     # cyclomatic complexity estimate (functions only)


@dataclass
class ParsedRelationship:
    """A directed dependency between two code entities."""
    source_id: str
    target_id: str
    relationship_type: str  # "imports" | "calls" | "inherits" | "contains"
    file_path: str
    line_number: int = 0


@dataclass
class ParseResult:
    """Aggregated output from parsing an entire repository."""
    entities: list[ParsedEntity] = field(default_factory=list)
    relationships: list[ParsedRelationship] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # (caller_id, symbol_name, full_module, lineno, submodule_hint) — resolved
    # in a second pass. submodule_hint is a candidate dotted module path to
    # try before full_module, used when the imported name might itself be a
    # submodule (e.g. `from pkg import submodule; submodule.func()`).
    pending_calls: list[tuple[str, str, str, int, str]] = field(default_factory=list)

    def entity_map(self) -> dict[str, ParsedEntity]:
        return {e.node_id: e for e in self.entities}


# ---------------------------------------------------------------------------
# Helpers for node IDs
# ---------------------------------------------------------------------------

def _file_id(rel_path: str) -> str:
    return f"file::{rel_path}"


def _func_id(rel_path: str, name: str) -> str:
    return f"func::{rel_path}::{name}"


def _class_id(rel_path: str, name: str) -> str:
    return f"class::{rel_path}::{name}"


def _module_id(name: str) -> str:
    return f"module::{name}"


def _path_to_module(rel_path: str) -> str:
    """Convert a repo-relative file path to a dotted module name.

    Examples
    --------
    src/requests/utils.py  ->  requests.utils
    requests/__init__.py   ->  requests
    mypackage/core.py      ->  mypackage.core
    """
    p = Path(rel_path).with_suffix("")
    parts = list(p.parts)
    if parts and parts[0] in ("src", "lib", "source"):
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else ""


def _resolve_relative_module(
    rel_path: str, module: Optional[str], level: int
) -> str:
    """Resolve a relative import to an absolute dotted module name.

    Parameters
    ----------
    rel_path : repo-relative path of the file containing the import
    module   : the ``from X`` part after stripping dots (None for ``from . import Y``)
    level    : number of leading dots (1 = current package, 2 = parent, ...)

    Examples (file = src/requests/auth.py)
    ----------------------------------------
    from . import utils        -> requests.utils   (level=1, module=None, name="utils")
    from .utils import func    -> requests.utils   (level=1, module="utils")
    from .. import core        -> core             (level=2, module="core")
    """
    p = Path(rel_path).with_suffix("")
    parts = list(p.parts)
    if parts and parts[0] in ("src", "lib", "source"):
        parts = parts[1:]
    # Strip the filename (including __init__) — both regular modules and package
    # __init__.py files live in the same directory, so both strip the last part.
    pkg_parts = list(parts[:-1])
    up = level - 1
    if up > 0:
        pkg_parts = pkg_parts[:-up] if len(pkg_parts) > up else []
    base = ".".join(pkg_parts)
    return f"{base}.{module}" if module else base


# ---------------------------------------------------------------------------
# Cyclomatic complexity estimator
# ---------------------------------------------------------------------------

_BRANCH_NODES = (
    ast.If, ast.For, ast.While, ast.Try,
    ast.ExceptHandler, ast.With, ast.Assert,
    ast.comprehension,
)

# BoolOp (and/or) adds one path per additional operand
_BOOL_EXTRA = ast.BoolOp


_NESTED_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _own_calls(node: ast.AST) -> list[ast.Call]:
    """
    Return every ``ast.Call`` that belongs to ``node`` itself, skipping calls
    that live inside a nested function/coroutine definition.

    Plain ``ast.walk(node)`` descends into nested ``def``s too, so a call
    inside an inner function would be attributed to *both* the inner
    function and its outer enclosing function — double counting the same
    call as two separate edges. Stopping the walk at nested function
    boundaries (their calls are collected separately, when the visitor
    processes that nested function) fixes the double count while still
    picking up calls nested inside non-function constructs (if/for/with/
    lambda/comprehensions/nested calls in arguments, etc.).
    """
    calls: list[ast.Call] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _NESTED_SCOPE_NODES):
            continue
        if isinstance(child, ast.Call):
            calls.append(child)
        calls.extend(_own_calls(child))
    return calls


def _cyclomatic_complexity(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Estimate McCabe cyclomatic complexity: 1 + number of branching points."""
    count = 1
    for node in ast.walk(func_node):
        if isinstance(node, _BRANCH_NODES):
            count += 1
        elif isinstance(node, _BOOL_EXTRA):
            # each additional operand beyond the first adds a path
            count += len(node.values) - 1
    return count


# ---------------------------------------------------------------------------
# Per-file AST visitor
# ---------------------------------------------------------------------------

class _FileVisitor(ast.NodeVisitor):
    """Walks a single file's AST and collects entities + relationships."""

    def __init__(self, rel_path: str, source_lines: list[str]) -> None:
        self.rel_path = rel_path
        self.source_lines = source_lines
        self.file_id = _file_id(rel_path)

        self.entities: list[ParsedEntity] = []
        self.relationships: list[ParsedRelationship] = []

        # Track names defined in this file for call resolution
        self._defined_funcs: dict[str, str] = {}   # name -> node_id
        self._defined_classes: dict[str, str] = {} # name -> node_id

        # Import alias → top-level module name (for module-edge fallback)
        self._import_map: dict[str, str] = {}
        # Imported symbol/alias → full dotted module path
        # e.g. `from requests.utils import func`  -> {"func": "requests.utils"}
        # e.g. `from . import utils` in pkg/auth.py -> {"utils": "pkg.utils"}
        # e.g. `from enum import Enum`              -> {"Enum": "enum"}
        self._imported_names: dict[str, str] = {}
        # Bound name → candidate dotted path if the imported name is itself
        # a submodule, e.g. `from pkg import submodule` -> {"submodule": "pkg.submodule"}.
        # Used to disambiguate attribute calls like `submodule.func()`, which
        # must resolve against pkg/submodule.py, not pkg/__init__.py.
        self._submodule_hints: dict[str, str] = {}
        # Calls to imported symbols that need cross-file resolution
        self._pending_calls: list[tuple[str, str, str, int, str]] = []
        # Same-file calls to names not yet defined at the point of the call —
        # may be forward references to a function defined later in this file.
        # Resolved once the whole file has been walked (see
        # _resolve_pending_same_file_calls), making this a genuine two-pass
        # per-file resolution instead of definition-order-dependent.
        self._pending_same_file_calls: list[tuple[str, str, int]] = []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _loc(self, node: ast.AST) -> int:
        """Lines of code for a function or class body."""
        if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
            return node.end_lineno - node.lineno + 1  # type: ignore[attr-defined]
        return 0

    def _has_docstring(self, node: ast.FunctionDef | ast.ClassDef | ast.Module) -> bool:
        return (
            bool(node.body)
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        )

    def _add_rel(
        self,
        src: str,
        tgt: str,
        rel_type: str,
        lineno: int = 0,
    ) -> None:
        if src != tgt:
            self.relationships.append(
                ParsedRelationship(
                    source_id=src,
                    target_id=tgt,
                    relationship_type=rel_type,
                    file_path=self.rel_path,
                    line_number=lineno,
                )
            )

    # ------------------------------------------------------------------
    # File-level entity (created once per file in the parser)
    # ------------------------------------------------------------------

    def _file_entity(self, total_lines: int) -> ParsedEntity:
        return ParsedEntity(
            node_id=self.file_id,
            name=Path(self.rel_path).name,
            entity_type="file",
            file_path=self.rel_path,
            line_number=1,
            lines_of_code=total_lines,
        )

    # ------------------------------------------------------------------
    # Import handling
    # ------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            top = alias.name.split(".")[0]
            mid = _module_id(top)
            # `import a.b` binds the name `a` in the namespace, not the
            # literal string "a.b" — code refers to it as `a.b.func()`.
            # Only an explicit `as` alias binds the dotted string verbatim.
            bound = alias.asname if alias.asname else top
            self._import_map[bound] = top
            self._imported_names[bound] = alias.name  # full module for call resolution
            self._add_rel(self.file_id, mid, "imports", node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""

        if node.level > 0:
            # Relative import — resolve to an absolute module name
            full_module = _resolve_relative_module(
                self.rel_path, module or None, node.level
            )
            top = full_module.split(".")[0] if full_module else ""
        else:
            full_module = module
            top = module.split(".")[0] if module else ""

        if top:
            mid = _module_id(top)
            self._import_map[top] = top
            self._add_rel(self.file_id, mid, "imports", node.lineno)

        for alias in node.names:
            bound = alias.asname or alias.name
            # `from . import utils` — each imported name is itself a submodule
            if node.level > 0 and not module:
                sub_full = f"{full_module}.{alias.name}" if full_module else alias.name
            else:
                sub_full = full_module

            # _import_map: used for module-level edge fallback
            self._import_map[bound] = top or bound
            # _imported_names: full module path, drives call + inheritance resolution
            self._imported_names[bound] = sub_full
            # _submodule_hints: `from pkg import name` is ambiguous — `name`
            # may be a symbol defined in `pkg`, or itself a submodule
            # (`pkg/name.py`). Record the submodule candidate so cross-file
            # resolution can try it first for attribute calls like
            # `name.func()`, instead of always assuming `name` is a symbol
            # inside `full_module` (which looks up the wrong file when
            # `name` is really a submodule).
            self._submodule_hints[bound] = (
                f"{full_module}.{alias.name}" if full_module else alias.name
            )

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Function definitions
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._handle_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._handle_func(node)

    def _handle_func(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        fid = _func_id(self.rel_path, node.name)
        self._defined_funcs[node.name] = fid

        # Count params (exclude self/cls)
        args = node.args
        all_params = (
            args.args + args.posonlyargs + args.kwonlyargs
            + ([args.vararg] if args.vararg else [])
            + ([args.kwarg] if args.kwarg else [])
        )
        non_self = [a for a in all_params if a.arg not in ("self", "cls")]
        num_params = len(non_self)

        entity = ParsedEntity(
            node_id=fid,
            name=node.name,
            entity_type="function",
            file_path=self.rel_path,
            line_number=node.lineno,
            lines_of_code=self._loc(node),
            num_params=num_params,
            has_docstring=self._has_docstring(node),
            complexity=_cyclomatic_complexity(node),
        )
        self.entities.append(entity)

        # file --contains--> function
        self._add_rel(self.file_id, fid, "contains", node.lineno)

        # Walk calls that belong to this function only — not to any nested
        # function, so a call inside a nested `def` is attributed solely to
        # that nested function, not double-counted against the outer one too.
        for child in _own_calls(node):
            self._handle_call(fid, child)

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Class definitions
    # ------------------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        cid = _class_id(self.rel_path, node.name)
        self._defined_classes[node.name] = cid

        entity = ParsedEntity(
            node_id=cid,
            name=node.name,
            entity_type="class",
            file_path=self.rel_path,
            line_number=node.lineno,
            lines_of_code=self._loc(node),
            has_docstring=self._has_docstring(node),
        )
        self.entities.append(entity)

        # file --contains--> class
        self._add_rel(self.file_id, cid, "contains", node.lineno)

        # Inheritance edges (best-effort name resolution)
        for base in node.bases:
            base_name = self._extract_name(base)
            if base_name:
                target_id = self._resolve_base(base_name)
                self._add_rel(cid, target_id, "inherits", node.lineno)

        self.generic_visit(node)

    def _resolve_base(self, base_name: str) -> str:
        """
        Resolve a base-class name to a node ID.

        Resolution order:
          1. Class defined in this file       -> class:: node
          2. Imported symbol (e.g. Enum)      -> module:: node of its top-level source
          3. Dotted name on an imported alias -> module:: node of the alias
          4. Unknown external base            -> module:: stub (NOT a fake class)
        """
        # 1. Locally defined class
        if base_name in self._defined_classes:
            return self._defined_classes[base_name]
        # 2. Directly imported symbol — _imported_names now stores full module, but
        #    module nodes are keyed by top-level only (module::os, module::enum, ...)
        if base_name in self._imported_names:
            full = self._imported_names[base_name]
            top = full.split(".")[0] if full else base_name
            return _module_id(top)
        # 3. Attribute access on an imported alias (abc.ABC -> module::abc)
        top = base_name.split(".")[0]
        if top in self._import_map:
            return _module_id(self._import_map[top])
        # 4. Unknown external base — treat as a module-level dependency
        return _module_id(top)

    # ------------------------------------------------------------------
    # Call resolution (best-effort)
    # ------------------------------------------------------------------

    def _handle_call(self, caller_id: str, call_node: ast.Call) -> None:
        name = self._extract_call_name(call_node)
        if not name:
            return
        lineno = getattr(call_node, "lineno", 0)

        # 1. Locally defined function — same-file edge, immediate resolution
        if name in self._defined_funcs:
            self._add_rel(caller_id, self._defined_funcs[name], "calls", lineno)
            return

        # 2. Directly imported symbol: `from requests.utils import func; func()`
        if name in self._imported_names:
            full_mod = self._imported_names[name]
            if full_mod:
                self._pending_calls.append((caller_id, name, full_mod, lineno, ""))
            return

        # 3. Attribute call on an imported alias/module: `utils.func()`,
        #    `mod.Class()`, or `a.b.func()` after `import a.b`.
        if "." in name:
            prefix, _, func_name = name.rpartition(".")
            root = prefix.split(".", 1)[0]

            full_mod: Optional[str] = None
            hint = ""
            if prefix in self._imported_names:
                # Single-level attribute call, e.g. `utils.func()`.
                full_mod = self._imported_names[prefix]
                hint = self._submodule_hints.get(prefix, "")
            elif root in self._imported_names:
                base = self._imported_names[root]
                # `import a.b` (no alias) binds the name `a`, but the call
                # may spell out the full imported path (`a.b.func()`).
                # Prefer the literal dotted prefix when it matches the
                # imported target so the call resolves against the right
                # submodule file instead of the (possibly wrong) package.
                full_mod = prefix if prefix == base else base
                hint = prefix if prefix == base else self._submodule_hints.get(root, "")

            if full_mod:
                self._pending_calls.append((caller_id, func_name, full_mod, lineno, hint))
            elif root in self._import_map:
                # Fall back to a module-level edge when cross-file resolution isn't possible
                mid = _module_id(self._import_map[root])
                self._add_rel(caller_id, mid, "calls", 0)
            return

        # 4. Bare name not yet defined in this file — may be a forward
        # reference to a function defined later in the same file (Python
        # doesn't require functions be defined before use). Defer until the
        # whole file has been walked so definition order doesn't matter;
        # names that still don't match a local function are genuinely
        # unresolvable (builtins, closures over local variables, star
        # imports, etc.) and stay dropped, same as before.
        self._pending_same_file_calls.append((caller_id, name, lineno))

    def _resolve_pending_same_file_calls(self) -> None:
        """Resolve same-file forward-reference calls now that every function
        in the file has been registered in ``_defined_funcs``."""
        for caller_id, name, lineno in self._pending_same_file_calls:
            target = self._defined_funcs.get(name)
            if target:
                self._add_rel(caller_id, target, "calls", lineno)

    def _extract_call_name(self, call: ast.Call) -> Optional[str]:
        return self._extract_name(call.func)

    def _extract_name(self, node: ast.expr) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._extract_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return None


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class PythonParser:
    """
    Recursively parses a Python repository into entities and relationships.

    Usage
    -----
    >>> parser = PythonParser()
    >>> result = parser.parse("/path/to/repo")
    """

    def __init__(self, skip_dirs: frozenset[str] | None = None) -> None:
        self.skip_dirs = skip_dirs if skip_dirs is not None else SKIP_DIRS

    def parse(self, repo_path: str | Path) -> ParseResult:
        """Parse every .py file under repo_path and return a ParseResult."""
        root = Path(repo_path).resolve()
        result = ParseResult()

        py_files = list(self._walk_py_files(root))
        logger.info(f"Found {len(py_files)} Python files to parse.")

        # Parse each file. This is a genuine two-pass strategy at both levels:
        #   - Within a file, _parse_file's AST walk registers every function
        #     as it's encountered and resolves calls to already-known local
        #     functions immediately, deferring calls to not-yet-seen names
        #     (forward references) until the whole file has been walked
        #     (see _FileVisitor._resolve_pending_same_file_calls).
        #   - Across files, calls to imported symbols are collected as
        #     pending and resolved below in a second full pass, once every
        #     file's entities are known.
        for rel_path, abs_path in py_files:
            self._parse_file(rel_path, abs_path, result)

        self._resolve_cross_file_calls(result)

        logger.info(
            f"Parsed {len(result.entities)} entities, "
            f"{len(result.relationships)} relationships, "
            f"{len(result.errors)} errors."
        )
        return result

    def _resolve_cross_file_calls(self, result: ParseResult) -> None:
        """
        Second pass: resolve cross-file calls collected during AST walking.

        Builds two indices over the fully-parsed entity set:
          module_name -> file_path        (for resolving the import target file)
          (file_path, symbol_name) -> id  (for finding the exact entity)

        Only owned code is resolved — pending calls targeting external libraries
        simply produce no match and are silently dropped.
        """
        if not result.pending_calls:
            return

        # module dotted name -> repo-relative file path
        module_to_path: dict[str, str] = {}
        for entity in result.entities:
            if entity.entity_type == "file":
                mod = _path_to_module(entity.file_path)
                if mod:
                    module_to_path[mod] = entity.file_path

        # (file_path, symbol_name) -> entity_id  (functions + classes)
        sym_index: dict[tuple[str, str], str] = {}
        for entity in result.entities:
            if entity.entity_type in ("function", "class"):
                sym_index[(entity.file_path, entity.name)] = entity.node_id

        resolved = 0
        for caller_id, sym_name, full_module, lineno, submodule_hint in result.pending_calls:
            file_path = None
            # 1. The imported name might itself be a submodule (e.g.
            #    `from pkg import submodule; submodule.func()`) — try that
            #    file first so we don't look up `func` in the wrong file
            #    (pkg/__init__.py instead of pkg/submodule.py).
            if submodule_hint:
                file_path = module_to_path.get(submodule_hint)
            # 2. Fall back to treating the call target as a symbol defined
            #    directly inside `full_module`.
            if not file_path:
                file_path = module_to_path.get(full_module)
            if not file_path:
                continue
            target_id = sym_index.get((file_path, sym_name))
            if target_id and target_id != caller_id:
                result.relationships.append(
                    ParsedRelationship(
                        source_id=caller_id,
                        target_id=target_id,
                        relationship_type="calls",
                        file_path="",
                        line_number=lineno,
                    )
                )
                resolved += 1

        logger.info(
            f"Cross-file calls: {resolved} resolved "
            f"({len(result.pending_calls)} pending, "
            f"{len(result.pending_calls) - resolved} unresolved external/builtin)."
        )

    def _walk_py_files(self, root: Path) -> list[tuple[str, Path]]:
        """Yield (rel_path, abs_path) for every .py file not under a skipped dir."""
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skipped directories in-place so os.walk doesn't recurse
            dirnames[:] = [
                d for d in dirnames if d not in self.skip_dirs and not d.startswith(".")
            ]
            for fname in filenames:
                if fname.endswith(".py"):
                    abs_path = Path(dirpath) / fname
                    rel_path = str(abs_path.relative_to(root)).replace("\\", "/")
                    files.append((rel_path, abs_path))
        return files

    def _parse_file(
        self, rel_path: str, abs_path: Path, result: ParseResult
    ) -> None:
        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            result.errors.append(f"Cannot read {rel_path}: {exc}")
            return

        try:
            tree = ast.parse(source, filename=str(abs_path))
        except SyntaxError as exc:
            result.errors.append(f"SyntaxError in {rel_path}: {exc}")
            return

        source_lines = source.splitlines()
        visitor = _FileVisitor(rel_path, source_lines)

        # Register the file entity
        file_entity = visitor._file_entity(len(source_lines))
        result.entities.append(file_entity)

        # Walk the AST, then resolve any same-file forward-reference calls
        # now that every function in the file is known (see
        # _resolve_pending_same_file_calls).
        visitor.visit(tree)
        visitor._resolve_pending_same_file_calls()

        result.entities.extend(visitor.entities)
        result.relationships.extend(visitor.relationships)
        result.pending_calls.extend(visitor._pending_calls)
