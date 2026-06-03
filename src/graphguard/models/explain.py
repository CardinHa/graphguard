"""
GNNExplainer — feature and edge attribution for individual node predictions.

Given a trained GraphSAGE model and a target node, GNNExplainer learns a soft
mask over input features and adjacent edges that best explain the model's
prediction for that node via a small optimisation loop.

Reference: Ying et al., "GNNExplainer: Generating Explanations for Graph
Neural Networks", NeurIPS 2019.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

from graphguard.models.gnn import build_model
from graphguard.utils.config import Config, DEFAULT_CONFIG
from graphguard.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(output_dir: Path, data: Data, config: Config = DEFAULT_CONFIG) -> torch.nn.Module:
    """Load saved GNN weights into a freshly-constructed model."""
    weights_path = output_dir / config.model_weights_file
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No trained model at {weights_path}. Run `graphguard train` first."
        )
    mc = config.model
    model = build_model(
        model_type=mc.model_type,
        in_channels=data.num_node_features,
        hidden_dim=mc.hidden_dim,
        num_layers=mc.num_layers,
        dropout=mc.dropout,
    )
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Node resolution
# ---------------------------------------------------------------------------

def resolve_node(
    query: str,
    node_ids: list[str],
    features_df: pd.DataFrame,
) -> tuple[int, str]:
    """
    Map a user-supplied name or id to (node_idx, node_id).

    Resolution order:
      1. Exact node_id match
      2. Exact ``name`` column match in features_df
      3. Substring match on node_id
    Raises ValueError on ambiguous or missing queries.
    """
    # 1. Exact node_id
    if query in node_ids:
        return node_ids.index(query), query

    # 2. Exact name column match
    if "name" in features_df.columns:
        hits = features_df[features_df["name"] == query]
        if len(hits) == 1:
            nid = hits.index[0]
            if nid in node_ids:
                return node_ids.index(nid), nid
        if len(hits) > 1:
            raise ValueError(
                f"Ambiguous name '{query}' matches multiple nodes:\n"
                + "\n".join(f"  {n}" for n in hits.index.tolist())
                + "\nUse the full node_id to disambiguate."
            )

    # 3. Substring match on node_id
    matches = [nid for nid in node_ids if query in nid]
    if len(matches) == 1:
        return node_ids.index(matches[0]), matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous query '{query}' matches multiple nodes:\n"
            + "\n".join(f"  {m}" for m in matches)
        )

    raise ValueError(
        f"Node '{query}' not found. "
        "Run `graphguard report` to see available nodes."
    )


# ---------------------------------------------------------------------------
# Core explainer
# ---------------------------------------------------------------------------

def explain_node(
    query: str,
    data: Data,
    model: torch.nn.Module,
    features_df: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 5,
    explainer_epochs: int = 200,
) -> dict:
    """
    Run GNNExplainer for a single node.

    GNNExplainer optimises a continuous mask M_f over input features and a
    mask M_e over edges to maximise the mutual information between the masked
    subgraph and the model's prediction. The masks are the explanation.

    Parameters
    ----------
    query            : node_id, function/class name, or substring of a node_id
    data             : PyG Data object (must have data.node_ids)
    model            : trained GNN in eval mode
    features_df      : DataFrame used to build the feature matrix (for name lookup)
    feature_names    : ordered list of feature column names
    top_k            : number of top features / neighbors to surface
    explainer_epochs : GNNExplainer optimisation steps (more = more stable masks)

    Returns
    -------
    dict with keys:
      node_id, name, risk_score,
      feature_importance (list of {feature, weight}),
      influential_neighbors (list of {node, weight})
    """
    node_idx, node_id = resolve_node(query, data.node_ids, features_df)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=explainer_epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    explanation = explainer(data.x, data.edge_index, index=node_idx)

    # Feature importance: node_mask shape is [N, F]; take the target node's row
    node_mask = explanation.node_mask
    if node_mask.dim() == 2:
        row_idx = node_idx if node_mask.shape[0] > 1 else 0
        feat_weights = node_mask[row_idx].cpu().detach().numpy()
    else:
        feat_weights = node_mask.cpu().detach().numpy()

    feat_importance = sorted(
        zip(feature_names, feat_weights.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    # Edge importance: collect edges incident to node_idx, deduplicate by neighbor
    edge_mask = explanation.edge_mask.cpu().detach().numpy()
    ei = data.edge_index.cpu().numpy()
    incident_idx = np.where((ei[0] == node_idx) | (ei[1] == node_idx))[0]

    name_col = features_df["name"] if "name" in features_df.columns else None
    seen: set[str] = set()
    neighbor_importance: list[tuple[str, float]] = []

    for idx in incident_idx:
        src, dst = int(ei[0, idx]), int(ei[1, idx])
        nbr_idx = dst if src == node_idx else src
        nbr_id = data.node_ids[nbr_idx]
        if nbr_id in seen:
            continue
        seen.add(nbr_id)
        nbr_name = (
            name_col.get(nbr_id, nbr_id) if name_col is not None else nbr_id
        )
        neighbor_importance.append((str(nbr_name), float(edge_mask[idx])))

    neighbor_importance.sort(key=lambda x: x[1], reverse=True)

    # Risk score from model forward pass
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        risk_score = float(torch.sigmoid(logits[node_idx]).item())

    return {
        "node_id": node_id,
        "name": str(
            name_col.get(node_id, node_id) if name_col is not None else node_id
        ),
        "risk_score": round(risk_score, 4),
        "feature_importance": [
            {"feature": f, "weight": round(float(w), 4)} for f, w in feat_importance
        ],
        "influential_neighbors": [
            {"node": n, "weight": round(w, 4)}
            for n, w in neighbor_importance[:top_k]
        ],
    }
