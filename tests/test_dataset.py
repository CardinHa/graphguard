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


def _toy_graph_many_files(num_files: int = 20, funcs_per_file: int = 2) -> nx.DiGraph:
    """Many small files, each with a file node + a few functions, so the
    file-grouped split has enough groups to exercise ratio preservation."""
    entities: list[ParsedEntity] = []
    rels: list[ParsedRelationship] = []
    for i in range(num_files):
        fp = f"pkg/file_{i}.py"
        file_id = f"file::{fp}"
        entities.append(ParsedEntity(file_id, f"file_{i}.py", "file", fp, 1, lines_of_code=10))
        for j in range(funcs_per_file):
            fid = f"func::{fp}::fn_{j}"
            entities.append(
                ParsedEntity(fid, f"fn_{j}", "function", fp, 2 + j, num_params=1, complexity=2)
            )
            rels.append(ParsedRelationship(file_id, fid, "contains", fp))
    return GraphBuilder().build(ParseResult(entities=entities, relationships=rels))


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


class TestFileGroupedSplit:
    """Regression tests for the group-leakage fix: same-file siblings must
    never land in two different splits (see dataset.py _split_masks)."""

    def test_no_file_spans_two_splits(self, builder: CodeGraphDataset) -> None:
        G = _toy_graph_many_files()
        df = FeatureExtractor().extract(G)
        data, _ = builder.build(G, df)

        splits_by_file: dict[str, set[str]] = {}
        for i, nid in enumerate(data.node_ids):
            fp = df.loc[nid, "file_path"] if nid in df.index else ""
            if not fp:
                continue
            if bool(data.train_mask[i]):
                splits_by_file.setdefault(fp, set()).add("train")
            elif bool(data.val_mask[i]):
                splits_by_file.setdefault(fp, set()).add("val")
            elif bool(data.test_mask[i]):
                splits_by_file.setdefault(fp, set()).add("test")

        offenders = {fp: s for fp, s in splits_by_file.items() if len(s) > 1}
        assert offenders == {}, f"Files spanning multiple splits: {offenders}"

    def test_split_is_deterministic(self, builder: CodeGraphDataset) -> None:
        G = _toy_graph_many_files()
        df = FeatureExtractor().extract(G)
        data1, _ = builder.build(G, df)
        data2, _ = builder.build(G, df)
        assert data1.train_mask.tolist() == data2.train_mask.tolist()
        assert data1.val_mask.tolist() == data2.val_mask.tolist()
        assert data1.test_mask.tolist() == data2.test_mask.tolist()

    def test_split_ratios_approximately_preserved(self, builder: CodeGraphDataset) -> None:
        G = _toy_graph_many_files(num_files=30, funcs_per_file=2)
        df = FeatureExtractor().extract(G)
        data, meta = builder.build(G, df)

        n_scorable = meta.train_size + meta.val_size + meta.test_size
        assert n_scorable > 0
        assert meta.train_size / n_scorable == pytest.approx(0.70, abs=0.15)
        assert meta.val_size / n_scorable == pytest.approx(0.15, abs=0.1)

    def test_small_repo_all_splits_nonempty(self, builder: CodeGraphDataset) -> None:
        """Regression test for the empty-test-split crash: a repo with only
        a handful of unevenly sized files (like examples/sample_project)
        must still populate train, val, AND test — greedy group packing
        alone could exhaust every group in train+val, and empty test masks
        crashed sklearn metrics downstream."""
        # 5 uneven files mirroring the sample project's shape
        # (group sizes 20, 14, 8, 8, 7 including the file node itself)
        sizes = [19, 13, 7, 7, 6]
        entities = []
        rels = []
        for i, n_funcs in enumerate(sizes):
            fp = f"pkg/file_{i}.py"
            file_id = f"file::{fp}"
            entities.append(ParsedEntity(file_id, f"file_{i}.py", "file", fp, 1))
            for j in range(n_funcs):
                fid = f"func::{fp}::fn_{j}"
                entities.append(ParsedEntity(fid, f"fn_{j}", "function", fp, 2 + j))
                rels.append(ParsedRelationship(file_id, fid, "contains", fp))
        G = GraphBuilder().build(ParseResult(entities=entities, relationships=rels))
        df = FeatureExtractor().extract(G)
        data, meta = builder.build(G, df)

        assert meta.train_size > 0
        assert meta.val_size > 0
        assert meta.test_size > 0

    def test_two_file_repo_gets_train_and_test(self, builder: CodeGraphDataset) -> None:
        """With only 2 file groups, train and test each get one; val may be
        empty (documented degradation) but evaluation must stay possible."""
        G = _toy_graph_many_files(num_files=2, funcs_per_file=3)
        df = FeatureExtractor().extract(G)
        data, meta = builder.build(G, df)

        assert meta.train_size > 0
        assert meta.test_size > 0

    def test_single_file_repo_does_not_crash(self, builder: CodeGraphDataset) -> None:
        """A single-file repo has no leakage-safe way to build an eval
        split; everything goes to train (with a warning) and nothing
        crashes."""
        G = _toy_graph_many_files(num_files=1, funcs_per_file=4)
        df = FeatureExtractor().extract(G)
        data, meta = builder.build(G, df)

        assert meta.train_size > 0
        assert meta.val_size == 0
        assert meta.test_size == 0


class TestEmptySplitMetrics:
    def test_compute_metrics_tolerates_empty_split(self) -> None:
        """compute_metrics on an empty split warns and returns zeros
        instead of crashing with sklearn's empty-input ValueError."""
        import numpy as np

        from graphguard.models.evaluate import compute_metrics

        m = compute_metrics("TEST", np.array([]), np.array([]), np.array([]))
        assert m.accuracy == 0.0
        assert m.roc_auc == 0.0


class TestModelConfigPersistence:
    """dataset_meta.json must carry the model hyperparameters used for a
    run, so `graphguard explain` / POST /explain can reconstruct an
    architecture matching the saved state_dict instead of crashing on a
    shape mismatch after non-default training."""

    def test_metadata_persists_model_hyperparameters(self) -> None:
        from graphguard.utils.config import Config, ModelConfig

        G = _toy_graph_with_module()
        df = FeatureExtractor().extract(G)
        config = Config(
            model=ModelConfig(model_type="gcn", hidden_dim=32, num_layers=3, dropout=0.5)
        )
        builder = CodeGraphDataset(config)
        _, meta = builder.build(G, df)

        assert meta.model_type == "gcn"
        assert meta.hidden_dim == 32
        assert meta.num_layers == 3
        assert meta.dropout == 0.5

        as_dict = meta.to_dict()
        assert as_dict["model_type"] == "gcn"
        assert as_dict["hidden_dim"] == 32
        assert as_dict["num_layers"] == 3
        assert as_dict["dropout"] == 0.5


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
