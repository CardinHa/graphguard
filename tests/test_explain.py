"""Tests for the GNNExplainer module."""

from __future__ import annotations

import pytest
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

from graphguard.models.explain import resolve_node, explain_node
from graphguard.models.gnn import build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_data(n_nodes: int = 8, n_features: int = 5) -> tuple[Data, list[str], pd.DataFrame]:
    """Build a tiny synthetic Data object for testing."""
    node_ids = [f"func::file.py::fn_{i}" for i in range(n_nodes)]
    x = torch.rand(n_nodes, n_features)
    # Simple chain edges: 0->1->2->...
    src = list(range(n_nodes - 1))
    dst = list(range(1, n_nodes))
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)
    data.node_ids = node_ids

    names = [f"fn_{i}" for i in range(n_nodes)]
    features_df = pd.DataFrame({"name": names}, index=node_ids)
    return data, node_ids, features_df


def _minimal_model(n_features: int = 5) -> torch.nn.Module:
    model = build_model("sage", in_channels=n_features, hidden_dim=16, num_layers=2, dropout=0.0)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# resolve_node tests
# ---------------------------------------------------------------------------

class TestResolveNode:
    def test_exact_node_id(self):
        data, node_ids, df = _minimal_data()
        idx, nid = resolve_node("func::file.py::fn_3", node_ids, df)
        assert idx == 3
        assert nid == "func::file.py::fn_3"

    def test_exact_name_match(self):
        data, node_ids, df = _minimal_data()
        idx, nid = resolve_node("fn_2", node_ids, df)
        assert idx == 2
        assert nid == "func::file.py::fn_2"

    def test_substring_match(self):
        data, node_ids, df = _minimal_data()
        idx, nid = resolve_node("fn_4", node_ids, df)
        assert idx == 4

    def test_missing_node_raises(self):
        data, node_ids, df = _minimal_data()
        with pytest.raises(ValueError, match="not found"):
            resolve_node("nonexistent_xyz", node_ids, df)

    def test_ambiguous_raises(self):
        # Give two nodes the same name to trigger ambiguity
        _, node_ids, df = _minimal_data()
        df.loc["func::file.py::fn_1", "name"] = "shared_name"
        df.loc["func::file.py::fn_2", "name"] = "shared_name"
        with pytest.raises(ValueError, match="Ambiguous"):
            resolve_node("shared_name", node_ids, df)


# ---------------------------------------------------------------------------
# explain_node tests
# ---------------------------------------------------------------------------

class TestExplainNode:
    def test_explain_returns_required_keys(self):
        data, node_ids, df = _minimal_data()
        model = _minimal_model()
        result = explain_node(
            query="fn_0",
            data=data,
            model=model,
            features_df=df,
            feature_names=[f"feat_{i}" for i in range(5)],
            top_k=3,
            explainer_epochs=20,
        )
        assert "node_id" in result
        assert "risk_score" in result
        assert "feature_importance" in result
        assert "influential_neighbors" in result

    def test_feature_importance_length(self):
        data, node_ids, df = _minimal_data()
        model = _minimal_model()
        result = explain_node(
            query="fn_3",
            data=data,
            model=model,
            features_df=df,
            feature_names=[f"feat_{i}" for i in range(5)],
            top_k=3,
            explainer_epochs=20,
        )
        assert len(result["feature_importance"]) == 3

    def test_risk_score_in_range(self):
        data, node_ids, df = _minimal_data()
        model = _minimal_model()
        result = explain_node(
            query="func::file.py::fn_0",
            data=data,
            model=model,
            features_df=df,
            feature_names=[f"feat_{i}" for i in range(5)],
            top_k=3,
            explainer_epochs=20,
        )
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_feature_weights_non_negative(self):
        data, node_ids, df = _minimal_data()
        model = _minimal_model()
        result = explain_node(
            query="fn_2",
            data=data,
            model=model,
            features_df=df,
            feature_names=[f"feat_{i}" for i in range(5)],
            top_k=5,
            explainer_epochs=20,
        )
        for item in result["feature_importance"]:
            assert item["weight"] >= 0.0, "Feature attribution weights must be non-negative"

    def test_isolated_node_has_no_neighbors(self):
        """A node with no edges should have an empty influential_neighbors list."""
        n_nodes, n_features = 4, 5
        node_ids = [f"func::file.py::fn_{i}" for i in range(n_nodes)]
        x = torch.rand(n_nodes, n_features)
        # No edges at all
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)
        data.node_ids = node_ids
        df = pd.DataFrame({"name": [f"fn_{i}" for i in range(n_nodes)]}, index=node_ids)

        model = _minimal_model(n_features)
        result = explain_node(
            query="fn_0",
            data=data,
            model=model,
            features_df=df,
            feature_names=[f"feat_{i}" for i in range(n_features)],
            top_k=3,
            explainer_epochs=20,
        )
        assert result["influential_neighbors"] == []
