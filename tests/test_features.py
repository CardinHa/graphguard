"""
Tests for graph feature extraction.

Verifies that:
  - feature DataFrame has the expected columns
  - degree features are correct
  - centrality values are in valid ranges
  - fan-in / fan-out are computed correctly
  - normalization preserves relative ordering
  - numeric_feature_columns() returns only columns present in output
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from graphguard.graph.features import FeatureExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def star_graph() -> nx.DiGraph:
    """Hub node with 4 spokes — easy to verify degree/fan-in."""
    G = nx.DiGraph()
    G.add_node("hub", name="hub", entity_type="file", file_path="hub.py",
               line_number=1, lines_of_code=10, num_params=0, has_docstring=1, complexity=1)
    for i in range(4):
        nid = f"spoke_{i}"
        G.add_node(nid, name=nid, entity_type="function", file_path=f"spoke_{i}.py",
                   line_number=1, lines_of_code=5, num_params=1, has_docstring=0, complexity=1)
        G.add_edge(nid, "hub", relationship_type="imports")
    return G


@pytest.fixture()
def chain_graph() -> nx.DiGraph:
    """Linear chain A → B → C."""
    G = nx.DiGraph()
    for nid in ["A", "B", "C"]:
        G.add_node(nid, name=nid, entity_type="function", file_path=f"{nid}.py",
                   line_number=1, lines_of_code=3, num_params=0, has_docstring=0, complexity=1)
    G.add_edge("A", "B", relationship_type="calls")
    G.add_edge("B", "C", relationship_type="calls")
    return G


@pytest.fixture()
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


# ---------------------------------------------------------------------------
# Output shape and columns
# ---------------------------------------------------------------------------

class TestFeatureShape:
    def test_output_is_dataframe(self, extractor: FeatureExtractor, star_graph: nx.DiGraph) -> None:
        df = extractor.extract(star_graph)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_equals_node_count(
        self, extractor: FeatureExtractor, star_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(star_graph)
        assert len(df) == star_graph.number_of_nodes()

    def test_expected_columns_present(
        self, extractor: FeatureExtractor, star_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(star_graph)
        expected = ["in_degree", "out_degree", "pagerank", "betweenness", "fan_in", "fan_out"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_numeric_feature_columns_are_in_output(
        self, extractor: FeatureExtractor, star_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(star_graph)
        for col in FeatureExtractor.numeric_feature_columns():
            assert col in df.columns, f"Numeric feature column missing: {col}"


# ---------------------------------------------------------------------------
# Degree correctness
# ---------------------------------------------------------------------------

class TestDegreeFeatures:
    def test_hub_in_degree_equals_spoke_count(
        self, extractor: FeatureExtractor, star_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(star_graph)
        assert df.loc["hub", "in_degree"] == 4

    def test_hub_out_degree_is_zero(
        self, extractor: FeatureExtractor, star_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(star_graph)
        assert df.loc["hub", "out_degree"] == 0

    def test_spoke_in_degree_is_zero(
        self, extractor: FeatureExtractor, star_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(star_graph)
        for i in range(4):
            assert df.loc[f"spoke_{i}", "in_degree"] == 0

    def test_chain_middle_node_degree(
        self, extractor: FeatureExtractor, chain_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(chain_graph)
        assert df.loc["B", "in_degree"] == 1
        assert df.loc["B", "out_degree"] == 1

    def test_total_degree_is_sum(
        self, extractor: FeatureExtractor, star_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(star_graph)
        for nid in star_graph.nodes():
            assert df.loc[nid, "total_degree"] == df.loc[nid, "in_degree"] + df.loc[nid, "out_degree"]


# ---------------------------------------------------------------------------
# Centrality ranges
# ---------------------------------------------------------------------------

class TestCentralityRanges:
    def test_pagerank_sums_to_one(
        self, extractor: FeatureExtractor, star_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(star_graph)
        assert abs(df["pagerank"].sum() - 1.0) < 1e-4

    def test_betweenness_in_zero_one(
        self, extractor: FeatureExtractor, star_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(star_graph)
        assert df["betweenness"].between(0.0, 1.0).all()

    def test_hub_has_highest_pagerank(
        self, extractor: FeatureExtractor, star_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(star_graph)
        assert df.loc["hub", "pagerank"] == df["pagerank"].max()

    def test_chain_middle_has_highest_betweenness(
        self, extractor: FeatureExtractor, chain_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(chain_graph)
        # B is on all A→C paths
        assert df.loc["B", "betweenness"] >= df.loc["A", "betweenness"]
        assert df.loc["B", "betweenness"] >= df.loc["C", "betweenness"]


# ---------------------------------------------------------------------------
# Fan-in / fan-out
# ---------------------------------------------------------------------------

class TestFanInFanOut:
    def test_hub_fan_in_equals_spoke_count(
        self, extractor: FeatureExtractor, star_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(star_graph)
        # 4 spokes import hub
        assert df.loc["hub", "fan_in"] == 4

    def test_spoke_fan_out_equals_one(
        self, extractor: FeatureExtractor, star_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(star_graph)
        for i in range(4):
            assert df.loc[f"spoke_{i}", "fan_out"] == 1


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_normalized_values_in_zero_one(
        self, extractor: FeatureExtractor, star_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(star_graph)
        cols = ["in_degree", "out_degree", "pagerank"]
        norm_df = FeatureExtractor.normalize(df, cols)
        for col in cols:
            assert norm_df[col].between(0.0, 1.0 + 1e-9).all(), f"{col} out of range"

    def test_normalization_preserves_order(
        self, extractor: FeatureExtractor, star_graph: nx.DiGraph
    ) -> None:
        df = extractor.extract(star_graph)
        norm_df = FeatureExtractor.normalize(df, ["in_degree"])
        # hub should still have the highest in_degree after normalization
        assert norm_df.loc["hub", "in_degree"] == norm_df["in_degree"].max()
