"""
Tests for PyG dataset construction, focusing on the scorable-node rule:
external module nodes must never be labeled, trained on, or reported as
risk targets — only files, functions, and classes are scorable.
"""

from __future__ import annotations

import networkx as nx
import pytest

from graphguard.data.dataset import CodeGraphDataset, _SCORABLE_TYPES
from graphguard.graph.features import FeatureExtractor
from graphguard.graph.graph_builder import GraphBuilder
from graphguard.parser.python_parser import (
    ParseResult,
    ParsedEntity,
    ParsedRelationship,
    PythonParser,
)


def _toy_graph_with_module() -> nx.DiGraph:
    """A file + function + a high-fan-in external module."""
    entities = [
        ParsedEntity("file::a.py", "a.py", "file", "a.py", 1, lines_of_code=20),
        ParsedEntity("func::a.py::run", "run", "function", "a.py", 3,
                     num_params=2, complexity=5),
        ParsedEntity("class::a.py::Thing", "Thing", "class", "a.py", 8),
    ]
    rels = [
        ParsedRelationship("file::a.py", "func::a.py::run", "contains", "a.py"),
        ParsedRelationship("file::a.py", "class::a.py::Thing", "contains", "a.py"),
        # External module imported by everything -> very high fan-in
        ParsedRelationship("file::a.py", "module::os", "imports", "a.py"),
        ParsedRelationship("func::a.py::run", "module::os", "calls", "a.py"),
        ParsedRelationship("class::a.py::Thing", "module::os", "inherits", "a.py"),
    ]
    return GraphBuilder().build(ParseResult(entities=entities, relationships=rels))


@pytest.fixture()
def builder() -> CodeGraphDataset:
    return CodeGraphDataset()


class TestScorableNodes:
    def test_module_nodes_never_labeled_risky(self, builder: CodeGraphDataset) -> None:
        G = _toy_graph_with_module()
        df = FeatureExtractor().extract(G)
        data, _ = builder.build(G, df)

        node_ids = data.node_ids
        y = data.y.tolist()
        for nid, label in zip(node_ids, y):
            if nid.startswith("module::"):
                assert label == 0, f"Module {nid} was labeled risky"

    def test_masks_exclude_modules(self, builder: CodeGraphDataset) -> None:
        G = _toy_graph_with_module()
        df = FeatureExtractor().extract(G)
        data, _ = builder.build(G, df)

        combined = data.train_mask | data.val_mask | data.test_mask
        for i, nid in enumerate(data.node_ids):
            if nid.startswith("module::"):
                assert not bool(combined[i]), f"Module {nid} appeared in a split mask"

    def test_scorable_mask_matches_entity_type(self, builder: CodeGraphDataset) -> None:
        G = _toy_graph_with_module()
        df = FeatureExtractor().extract(G)
        data, _ = builder.build(G, df)

        for i, nid in enumerate(data.node_ids):
            etype = G.nodes[nid].get("entity_type")
            expected = etype in _SCORABLE_TYPES
            assert bool(data.scorable_mask[i]) == expected

    def test_predictions_csv_excludes_modules(
        self, builder: CodeGraphDataset, tmp_path
    ) -> None:
        import pandas as pd

        G = _toy_graph_with_module()
        df = FeatureExtractor().extract(G)
        data, _ = builder.build(G, df)

        nodes = data.node_ids
        probs = [0.5] * len(nodes)
        labels = data.y.tolist()
        out = tmp_path / "predictions.csv"
        builder.save_predictions(nodes, probs, labels, df, out)

        preds = pd.read_csv(out)
        assert not preds["entity_type"].isin(["module", "unknown"]).any()


class TestDataShape:
    def test_edge_index_shape(self, builder: CodeGraphDataset) -> None:
        G = _toy_graph_with_module()
        df = FeatureExtractor().extract(G)
        data, meta = builder.build(G, df)
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == meta.num_edges

    def test_undirected_doubles_edges(self, builder: CodeGraphDataset) -> None:
        G = _toy_graph_with_module()
        df = FeatureExtractor().extract(G)
        directed, _ = builder.build(G, df, undirected=False)
        undirected, _ = builder.build(G, df, undirected=True)
        assert undirected.edge_index.shape[1] == 2 * directed.edge_index.shape[1]
