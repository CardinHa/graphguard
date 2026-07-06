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
    """Human-readable stats about the constructed dataset.

    Also carries the model hyperparameters used for this run
    (model_type/hidden_dim/num_layers/dropout). ``graphguard explain`` and
    POST /explain reconstruct the model from saved weights without
    retraining, so they need these values to build an architecture that
    matches the checkpoint — otherwise state_dict shapes mismatch whenever
    training used non-default hyperparameters.
    """
    num_nodes: int
    num_edges: int
    num_features: int
    num_positive: int
    num_negative: int
    label_mode: str
    train_size: int
    val_size: int
    test_size: int
    model_type: str = "sage"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3

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
        file_keys = self._file_group_keys(features_df, nodes)
        train_mask, val_mask, test_mask = self._split_masks(N, scorable_idx, file_keys)
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
            model_type=self.config.model.model_type,
            hidden_dim=self.config.model.hidden_dim,
            num_layers=self.config.model.num_layers,
            dropout=self.config.model.dropout,
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

    @staticmethod
    def _file_group_keys(df: pd.DataFrame, nodes: list[str]) -> list[str]:
        """
        Return a per-node "file group" key used to keep same-file entities
        together during the train/val/test split (see ``_split_masks``).

        Every node carries the repo-relative path of the file it belongs to
        (the file node's own path, or the containing file's path for a
        function/class — see ``ParsedEntity.file_path``). Nodes with no
        usable file path (e.g. external ``module::`` stubs, which are never
        scorable anyway) fall back to a unique per-node key so they never
        get accidentally grouped with unrelated nodes.
        """
        if "file_path" not in df.columns:
            return [f"__node_{i}__" for i in range(len(nodes))]
        raw = df.reindex(nodes)["file_path"].fillna("").astype(str).tolist()
        return [fp if fp else f"__node_{i}__" for i, fp in enumerate(raw)]

    def _split_masks(
        self, N: int, scorable_idx: list[int], group_keys: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create reproducible train/val/test masks over scorable nodes only,
        grouped by containing file.

        Git-mode labels are assigned per-file and then propagated to every
        function/class in that file (see
        ``GitMiner.file_labels_to_node_labels``). A naive random node-level
        split would therefore routinely place same-file siblings — which
        share an identical label — on both sides of train/test, leaking
        label information and inflating held-out metrics. Splitting by
        file group instead guarantees every node belonging to the same file
        lands in exactly one of train/val/test.

        Split fractions (``train_frac`` / ``val_frac``) are preserved
        approximately: whole file-groups are assigned to a split in
        seed-shuffled order until that split's target size is reached.

        Small-repo guarantees
        ---------------------
        Because whole groups move together, a repo with few (or unevenly
        sized) files can exhaust every group in train+val, leaving the test
        split empty — which breaks evaluation downstream. So:

        - With >= 3 groups, every split is guaranteed at least one group
          (rebalanced after the greedy pass, test first since an empty test
          split is the hardest failure).
        - With 2 groups, train and test each get one (val stays empty —
          early stopping degrades to "keep last epoch" but training and
          evaluation still work).
        - With 1 group, everything lands in train and a warning is logged;
          there is no leakage-safe way to evaluate a single-file repo.
        """
        rng = np.random.default_rng(seed=42)

        # Group scorable node indices by their containing file.
        groups: dict[str, list[int]] = {}
        for i in scorable_idx:
            groups.setdefault(group_keys[i], []).append(i)

        # Deterministic base order, then seeded shuffle of the groups
        # themselves (not the individual nodes) so a whole file always
        # moves together.
        group_order = sorted(groups.keys())
        perm = rng.permutation(len(group_order))
        shuffled_keys = [group_order[p] for p in perm]

        n_s = len(scorable_idx)
        train_target = int(n_s * self.config.train_frac)
        val_target = int(n_s * self.config.val_frac)

        # Greedy pass: pack whole groups into train, then val, then test.
        assigned: dict[str, list[list[int]]] = {"train": [], "val": [], "test": []}
        counts = {"train": 0, "val": 0, "test": 0}
        for key in shuffled_keys:
            members = groups[key]
            if counts["train"] < train_target:
                split = "train"
            elif counts["val"] < val_target:
                split = "val"
            else:
                split = "test"
            assigned[split].append(members)
            counts[split] += len(members)

        # Rebalance pass: guarantee val/test each hold at least one group
        # whenever enough groups exist, donating from whichever other split
        # currently holds the most groups (a donor must keep at least one
        # group). Deterministic: donor choice depends only on group counts,
        # ties broken by fixed split order; the moved group is always the
        # donor's last-assigned one.
        def _donate_to(dst: str) -> None:
            if assigned[dst]:
                return
            donors = [
                s for s in ("train", "val", "test")
                if s != dst and len(assigned[s]) > 1
            ]
            if not donors:
                return  # fewer groups than splits — degrade as documented
            donor = max(donors, key=lambda s: len(assigned[s]))
            moved = assigned[donor].pop()
            assigned[dst].append(moved)
            counts[donor] -= len(moved)
            counts[dst] += len(moved)

        _donate_to("test")
        _donate_to("val")

        # With exactly 2 groups the greedy pass put both in train; ensure
        # test gets one even though the donor rule above requires len > 1.
        if not assigned["test"] and len(assigned["train"]) == 2 and not assigned["val"]:
            moved = assigned["train"].pop()
            assigned["test"].append(moved)
            counts["train"] -= len(moved)
            counts["test"] += len(moved)

        if not assigned["test"]:
            logger.warning(
                "File-grouped split: repo has too few files to populate a "
                "test split (every node shares one file group). Metrics "
                "cannot be computed without leakage; all nodes assigned to "
                "train."
            )
        if not assigned["val"]:
            logger.warning(
                "File-grouped split: no file group available for the "
                "validation split; early stopping will not trigger."
            )

        train_idx = [i for grp in assigned["train"] for i in grp]
        val_idx = [i for grp in assigned["val"] for i in grp]
        test_idx = [i for grp in assigned["test"] for i in grp]

        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)

        if train_idx:
            train_mask[train_idx] = True
        if val_idx:
            val_mask[val_idx] = True
        if test_idx:
            test_mask[test_idx] = True

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
