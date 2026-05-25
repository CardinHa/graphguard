"""
Tests for the graph builder.

Verifies that:
  - nodes are created for each parsed entity
  - edges are created with correct relationship types
  - stub nodes are created for external modules
  - graph serialization succeeds (graphml, json, csv)
  - graph summary reports correct counts
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import networkx as nx
import pytest

from graphguard.graph.graph_builder import GraphBuilder
from graphguard.parser.python_parser import ParseResult, ParsedEntity, ParsedRelationship


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_result() -> ParseResult:
    """A hand-crafted ParseResult for deterministic testing."""
    entities = [
        ParsedEntity(
            node_id="file::a.py",
            name="a.py",
            entity_type="file",
            file_path="a.py",
            line_number=1,
        ),
        ParsedEntity(
            node_id="func::a.py::foo",
            name="foo",
            entity_type="function",
            file_path="a.py",
            line_number=3,
            num_params=2,
            has_docstring=True,
            complexity=2,
        ),
        ParsedEntity(
            node_id="class::a.py::Bar",
            name="Bar",
            entity_type="class",
            file_path="a.py",
            line_number=8,
        ),
    ]
    relationships = [
        ParsedRelationship(
            source_id="file::a.py",
            target_id="func::a.py::foo",
            relationship_type="contains",
            file_path="a.py",
        ),
        ParsedRelationship(
            source_id="file::a.py",
            target_id="class::a.py::Bar",
            relationship_type="contains",
            file_path="a.py",
        ),
        ParsedRelationship(
            source_id="file::a.py",
            target_id="module::os",
            relationship_type="imports",
            file_path="a.py",
        ),
    ]
    return ParseResult(entities=entities, relationships=relationships)


@pytest.fixture()
def builder() -> GraphBuilder:
    return GraphBuilder()


# ---------------------------------------------------------------------------
# Node construction
# ---------------------------------------------------------------------------

class TestNodeCreation:
    def test_node_count(self, builder: GraphBuilder, simple_result: ParseResult) -> None:
        G = builder.build(simple_result)
        # 3 entities + 1 stub module node = 4
        assert G.number_of_nodes() == 4

    def test_node_attributes_preserved(
        self, builder: GraphBuilder, simple_result: ParseResult
    ) -> None:
        G = builder.build(simple_result)
        attrs = G.nodes["func::a.py::foo"]
        assert attrs["name"] == "foo"
        assert attrs["entity_type"] == "function"
        assert attrs["num_params"] == 2
        assert attrs["has_docstring"] == 1

    def test_external_module_stub_created(
        self, builder: GraphBuilder, simple_result: ParseResult
    ) -> None:
        G = builder.build(simple_result)
        assert "module::os" in G.nodes
        assert G.nodes["module::os"]["entity_type"] == "module"


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------

class TestEdgeCreation:
    def test_contains_edges(self, builder: GraphBuilder, simple_result: ParseResult) -> None:
        G = builder.build(simple_result)
        contains_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d["relationship_type"] == "contains"
        ]
        assert len(contains_edges) == 2

    def test_imports_edge(self, builder: GraphBuilder, simple_result: ParseResult) -> None:
        G = builder.build(simple_result)
        assert G.has_edge("file::a.py", "module::os")

    def test_duplicate_edges_deduplicated(self, builder: GraphBuilder) -> None:
        entities = [
            ParsedEntity("file::a.py", "a.py", "file", "a.py", 1),
        ]
        rels = [
            ParsedRelationship("file::a.py", "module::os", "imports", "a.py"),
            ParsedRelationship("file::a.py", "module::os", "imports", "a.py"),  # duplicate
        ]
        result = ParseResult(entities=entities, relationships=rels)
        G = builder.build(result)
        assert G.number_of_edges() == 1

    def test_directed_graph(self, builder: GraphBuilder, simple_result: ParseResult) -> None:
        G = builder.build(simple_result)
        assert isinstance(G, nx.DiGraph)


# ---------------------------------------------------------------------------
# Integration with real parser
# ---------------------------------------------------------------------------

class TestBuilderWithParser:
    def test_full_parse_and_build(self, tmp_path: Path, builder: GraphBuilder) -> None:
        from graphguard.parser.python_parser import PythonParser

        code = textwrap.dedent("""\
            import os

            class Animal:
                def speak(self):
                    pass

            class Dog(Animal):
                def speak(self):
                    return "Woof"
        """)
        (tmp_path / "animals.py").write_text(code)

        result = PythonParser().parse(tmp_path)
        G = builder.build(result)

        # Should have at least: file, Animal, Dog, speak (x2), module::os
        assert G.number_of_nodes() >= 5
        # Inheritance edge
        inherits = [
            (u, v) for u, v, d in G.edges(data=True)
            if d["relationship_type"] == "inherits"
        ]
        assert len(inherits) == 1


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_save_all_creates_files(
        self, builder: GraphBuilder, simple_result: ParseResult, tmp_path: Path
    ) -> None:
        G = builder.build(simple_result)
        builder.save_all(G, tmp_path)

        assert (tmp_path / "graph.graphml").exists()
        assert (tmp_path / "graph.json").exists()
        assert (tmp_path / "edges.csv").exists()
        assert (tmp_path / "nodes.csv").exists()

    def test_edge_csv_has_correct_columns(
        self, builder: GraphBuilder, simple_result: ParseResult, tmp_path: Path
    ) -> None:
        import pandas as pd

        G = builder.build(simple_result)
        builder.save_edge_csv(G, tmp_path / "edges.csv")
        df = pd.read_csv(tmp_path / "edges.csv")
        assert "source" in df.columns
        assert "target" in df.columns
        assert "relationship_type" in df.columns
