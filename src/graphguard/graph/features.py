"""
Graph-theoretic and code-structure feature extraction.

For each node in the dependency graph we compute a feature vector that
combines discrete-math graph properties with static code metrics.

Graph theory connection
-----------------------
The dependency graph G = (V, E) has adjacency matrix A where A[i,j] = 1
iff node i has a directed edge to node j.

  in_degree(v)   = column sum of A for column v  → how many things depend on v
  out_degree(v)  = row sum of A for row v         → how many things v depends on
  PageRank       = dominant eigenvector of normalized A^T (power iteration)
  Betweenness    = fraction of shortest paths that pass through v
  Closeness      = inverse of average shortest-path distance from v
  Clustering     = local transitivity (triangles / possible triangles)

High betweenness + high fan-in = structural bottleneck → often correlated with bugs.
"""

from __future__ import annotations

import warnings
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

from graphguard.utils.logging import get_logger

logger = get_logger(__name__)

# One-hot encoding index for entity types
_ENTITY_TYPE_IDX: dict[str, int] = {
    "file": 0,
    "function": 1,
    "class": 2,
    "module": 3,
    "unknown": 4,
}
_NUM_TYPES = len(_ENTITY_TYPE_IDX)


class FeatureExtractor:
    """
    Computes a feature matrix (DataFrame) from a NetworkX DiGraph.

    Each row corresponds to one node; each column is one feature.
    The DataFrame index is the node_id.
    """

    def extract(self, G: nx.DiGraph) -> pd.DataFrame:
        """Return a DataFrame of features for every node in G."""
        logger.info("Computing graph-theoretic features...")

        nodes = list(G.nodes())
        n = len(nodes)

        # ---- Graph centrality metrics ----
        # PageRank: stationary distribution of a random walk on A^T
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=200)

        # Betweenness centrality: fraction of (s,t) shortest paths through v
        # Normalized by (n-1)(n-2) for directed graphs
        betweenness = nx.betweenness_centrality(G, normalized=True)

        # Closeness centrality: inverse avg shortest path length (per component)
        closeness = nx.closeness_centrality(G)

        # Local clustering coefficient on undirected view
        G_undirected = G.to_undirected()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clustering = nx.clustering(G_undirected)

        logger.info("Centrality metrics computed.")

        # ---- Build rows ----
        rows = []
        for node in nodes:
            attrs = G.nodes[node]
            row = self._node_row(G, node, attrs, pagerank, betweenness, closeness, clustering)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("node_id")
        logger.info(f"Feature matrix: {df.shape[0]} nodes × {df.shape[1]} features.")
        return df

    # ------------------------------------------------------------------
    # Per-node feature vector
    # ------------------------------------------------------------------

    def _node_row(
        self,
        G: nx.DiGraph,
        node: str,
        attrs: dict,
        pagerank: dict,
        betweenness: dict,
        closeness: dict,
        clustering: dict,
    ) -> dict:
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)

        # fan-in = number of nodes that import/call/inherit THIS node
        fan_in = sum(
            1 for _, _, d in G.in_edges(node, data=True)
            if d.get("relationship_type") in ("imports", "calls", "inherits")
        )
        # fan-out = number of nodes THIS node imports/calls/inherits
        fan_out = sum(
            1 for _, _, d in G.out_edges(node, data=True)
            if d.get("relationship_type") in ("imports", "calls", "inherits")
        )

        # One-hot entity type
        etype = attrs.get("entity_type", "unknown")
        type_idx = _ENTITY_TYPE_IDX.get(etype, _ENTITY_TYPE_IDX["unknown"])
        type_onehot = [0] * _NUM_TYPES
        type_onehot[type_idx] = 1

        row = {
            "node_id": node,
            # identity
            "name": attrs.get("name", ""),
            "entity_type": etype,
            "file_path": attrs.get("file_path", ""),
            # degree
            "in_degree": in_deg,
            "out_degree": out_deg,
            "total_degree": in_deg + out_deg,
            # centrality
            "pagerank": pagerank.get(node, 0.0),
            "betweenness": betweenness.get(node, 0.0),
            "closeness": closeness.get(node, 0.0),
            "clustering": clustering.get(node, 0.0),
            # dependency counts
            "fan_in": fan_in,
            "fan_out": fan_out,
            "num_dependents": in_deg,
            "num_dependencies": out_deg,
            # code metrics
            "lines_of_code": attrs.get("lines_of_code", 0),
            "num_params": attrs.get("num_params", 0),
            "has_docstring": attrs.get("has_docstring", 0),
            "complexity": attrs.get("complexity", 1),
        }

        # Append one-hot type columns
        for i, tname in enumerate(_ENTITY_TYPE_IDX.keys()):
            row[f"type_{tname}"] = type_onehot[i]

        return row

    # ------------------------------------------------------------------
    # Normalisation helpers (used before feeding into ML models)
    # ------------------------------------------------------------------

    @staticmethod
    def numeric_feature_columns() -> list[str]:
        """Return column names that form the actual feature vector for ML."""
        return [
            "in_degree", "out_degree", "total_degree",
            "pagerank", "betweenness", "closeness", "clustering",
            "fan_in", "fan_out", "num_dependents", "num_dependencies",
            "lines_of_code", "num_params", "has_docstring", "complexity",
            "type_file", "type_function", "type_class", "type_module", "type_unknown",
        ]

    @staticmethod
    def normalize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Min-max normalize a subset of columns (returns a copy)."""
        out = df.copy()
        for col in cols:
            if col not in out.columns:
                continue
            mn, mx = out[col].min(), out[col].max()
            if mx > mn:
                out[col] = (out[col] - mn) / (mx - mn)
            else:
                out[col] = 0.0
        return out
