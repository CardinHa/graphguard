"""
PyTorch Geometric dataset construction.

Converts the NetworkX dependency graph + feature DataFrame + labels into a
PyG Data object ready for GNN training.

Graph → tensor pipeline
------------------------
1. Build integer node index  (node_id → int)
2. Stack feature columns → float32 tensor  x  of shape [N, F]
3. Convert edge list → edge_index tensor   of shape [2, E]
4. Attach binary labels                    y  of shape [N]
5. Create boolean train / val / test masks

Handling directed vs undirected
--------------------------------
GraphSAGE works on directed graphs natively.
GCN expects a symmetric adjacency — pass undirected=True to symmetrize.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from graphguard.graph.features import FeatureExtractor
from graphguard.utils.config import Config, DEFAULT_CONFIG
from graphguard.utils.logging import get_logger

logger = get_logger(__name__)

# Entity types that represent code we own and can act on. External modules
# (numpy, os, enum, ...) are kept in the graph for their structural signal
# but are NOT prediction targets — you cannot fix a bug in the stdlib.
_SCORABLE_TYPES: frozenset[str] = frozenset({"file", "function", "class"})

# ⚠️  SYNTHETIC LABEL WARNING — displayed everywhere labels are created
_SYNTHETIC_WARNING = (
    "WARNING (SYNTHETIC LABELS): Risk labels are heuristic estimates based on graph "
    "centrality and code complexity. They are NOT derived from real bug data. "
    "For real predictions, run on a repository with git history (--label-mode git)."
)


@dataclass
class DatasetMetadata:
    """Human-readable stats about the constructed dataset."""
    num_nodes: int
    num_edges: int
    num_features: int
    num_positive: int
    num_negative: int
    label_mode: str
    train_size: int
    val_size: int
    test_size: int

    def to_dict(self) -> dict:
        return self.__dict__


class CodeGraphDataset:
    """
    Wraps a dependency graph as a PyTorch Geometric Data object.

    Parameters
    ----------
    config : Config
        Project configuration (split fractions, label mode, etc.)
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self.config = config
        self.feature_cols: list[str] = FeatureExtractor.numeric_feature_columns()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(
        self,
        G: nx.DiGraph,
        features_df: pd.DataFrame,
        labels: Optional[dict[str, int]] = None,
        undirected: bool = False,
    ) -> tuple[Data, DatasetMetadata]:
        """
        Build a PyG Data object from graph + features + (optional) labels.

        Parameters
        ----------
        G            : NetworkX DiGraph
        features_df  : DataFrame indexed by node_id, from FeatureExtractor
        labels       : dict mapping node_id → 0/1.  If None, synthetic labels
                       are generated automatically.
        undirected   : symmetrize edges (for GCN compatibility)

        Returns
        -------
        data     : PyG Data object
        metadata : DatasetMetadata
        """
        nodes = list(G.nodes())
        node_idx: dict[str, int] = {n: i for i, n in enumerate(nodes)}
        N = len(nodes)

        # ---- Scorable nodes (code we own) ----
        # External modules stay in the graph for message passing but are never
        # labeled, trained on, evaluated, or reported as risk targets.
        scorable = self._scorable_flags(features_df, nodes)
        scorable_idx = [i for i, s in enumerate(scorable) if s]

        # ---- Feature matrix ----
        x = self._build_feature_matrix(features_df, nodes)

        # ---- Edge index ----
        edge_index = self._build_edge_index(G, node_idx, undirected=undirected)

        # ---- Labels ----
        label_mode = self.config.label_mode
        if labels is None:
            logger.warning(_SYNTHETIC_WARNING)
            labels = self._synthetic_labels(features_df, nodes, scorable)
            label_mode = "synthetic"

        y = torch.zeros(N, dtype=torch.long)
        for i, nid in enumerate(nodes):
            # Force non-scorable (module) nodes to label 0 regardless of source
            y[i] = labels.get(nid, 0) if scorable[i] else 0

        # ---- Masks (only over scorable nodes) ----
        train_mask, val_mask, test_mask = self._split_masks(N, scorable_idx)
        scorable_mask = torch.zeros(N, dtype=torch.bool)
        for i in scorable_idx:
            scorable_mask[i] = True

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=N,
        )
        # Store node IDs and scorable mask for downstream reporting
        data.node_ids = nodes  # type: ignore[attr-defined]
        data.scorable_mask = scorable_mask  # type: ignore[attr-defined]

        n_scorable = len(scorable_idx)
        meta = DatasetMetadata(
            num_nodes=N,
            num_edges=edge_index.shape[1],
            num_features=x.shape[1],
            num_positive=int(y.sum().item()),
            num_negative=int(n_scorable - y.sum().item()),
            label_mode=label_mode,
            train_size=int(train_mask.sum().item()),
            val_size=int(val_mask.sum().item()),
            test_size=int(test_mask.sum().item()),
        )
        pct = (meta.num_positive / n_scorable * 100) if n_scorable else 0.0
        logger.info(
            f"Dataset: {N} nodes ({n_scorable} scorable), {meta.num_features} features, "
            f"{meta.num_positive} positive ({pct:.1f}% of scorable)."
        )
        return data, meta

    @staticmethod
    def _scorable_flags(df: pd.DataFrame, nodes: list[str]) -> list[bool]:
        """Return a per-node boolean: True if the node is owned code."""
        if "entity_type" not in df.columns:
            return [True] * len(nodes)
        types = df.reindex(nodes)["entity_type"].fillna("unknown").tolist()
        return [t in _SCORABLE_TYPES for t in types]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_feature_matrix(
        self, df: pd.DataFrame, nodes: list[str]
    ) -> torch.Tensor:
        """Stack numeric feature columns into [N, F] float32 tensor."""
        available = [c for c in self.feature_cols if c in df.columns]
        sub = df.reindex(nodes)[available].fillna(0.0)

        # Min-max normalize each column
        for col in sub.columns:
            mn, mx = sub[col].min(), sub[col].max()
            if mx > mn:
                sub[col] = (sub[col] - mn) / (mx - mn)
            else:
                sub[col] = 0.0

        return torch.tensor(sub.values, dtype=torch.float32)

    def _build_edge_index(
        self,
        G: nx.DiGraph,
        node_idx: dict[str, int],
        undirected: bool,
    ) -> torch.Tensor:
        """Convert edge list to [2, E] int64 tensor."""
        src_list, dst_list = [], []
        for u, v in G.edges():
            if u in node_idx and v in node_idx:
                src_list.append(node_idx[u])
                dst_list.append(node_idx[v])
                if undirected:
                    src_list.append(node_idx[v])
                    dst_list.append(node_idx[u])

        if not src_list:
            return torch.zeros((2, 0), dtype=torch.long)

        return torch.tensor([src_list, dst_list], dtype=torch.long)

    def _synthetic_labels(
        self, df: pd.DataFrame, nodes: list[str], scorable: list[bool]
    ) -> dict[str, int]:
        """
        ⚠️  SYNTHETIC LABELS ONLY — for demo purposes.

        Risk score = 0.4 * norm(fan_in) + 0.3 * norm(betweenness) + 0.3 * norm(complexity)
        Only scorable (owned-code) nodes are eligible; the threshold is computed
        over the scorable population so external modules never become targets.
        """
        sub = df.reindex(nodes).fillna(0.0)
        scorable_series = pd.Series(scorable, index=sub.index)

        def _norm(col: str) -> pd.Series:
            s = sub[col] if col in sub.columns else pd.Series(0.0, index=sub.index)
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.0, index=s.index)

        score = (
            0.4 * _norm("fan_in")
            + 0.3 * _norm("betweenness")
            + 0.3 * _norm("complexity")
        )
        scorable_scores = score[scorable_series]
        if scorable_scores.empty:
            return {n: 0 for n in nodes}
        threshold = scorable_scores.quantile(self.config.synthetic_risk_percentile)
        risky = ((score >= threshold) & scorable_series).astype(int)
        return risky.to_dict()

    def _split_masks(
        self, N: int, scorable_idx: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create reproducible train/val/test masks over scorable nodes only."""
        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(np.asarray(scorable_idx, dtype=np.int64))
        n_s = len(perm)

        train_end = int(n_s * self.config.train_frac)
        val_end = train_end + int(n_s * self.config.val_frac)

        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)

        train_mask[perm[:train_end].tolist()] = True
        val_mask[perm[train_end:val_end].tolist()] = True
        test_mask[perm[val_end:].tolist()] = True

        return train_mask, val_mask, test_mask

    def save_labels(self, labels: dict[str, int], path: Path) -> None:
        path.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    def save_predictions(
        self,
        nodes: list[str],
        probs: list[float],
        labels: list[int],
        features_df: pd.DataFrame,
        path: Path,
    ) -> None:
        """Save per-node predictions as a CSV for reporting."""
        rows = []
        name_map = (
            features_df["name"].to_dict() if "name" in features_df.columns else {}
        )
        type_map = (
            features_df["entity_type"].to_dict()
            if "entity_type" in features_df.columns
            else {}
        )
        file_map = (
            features_df["file_path"].to_dict()
            if "file_path" in features_df.columns
            else {}
        )
        for nid, prob, lbl in zip(nodes, probs, labels):
            # Only report owned code — skip external module nodes
            if type_map.get(nid, "") not in _SCORABLE_TYPES:
                continue
            rows.append({
                "node_id": nid,
                "name": name_map.get(nid, ""),
                "entity_type": type_map.get(nid, ""),
                "file_path": file_map.get(nid, ""),
                "risk_score": round(prob, 4),
                "predicted_risky": int(prob >= 0.5),
                "true_label": lbl,
            })
        df = pd.DataFrame(rows).sort_values("risk_score", ascending=False)
        df.to_csv(path, index=False)
        logger.info(f"Predictions saved -> {path}")
