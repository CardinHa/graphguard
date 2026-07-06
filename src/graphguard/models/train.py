"""
End-to-end training pipeline: parse → graph → features → labels → train → evaluate → save.

This module orchestrates the full workflow:
  1. Parse repository with PythonParser
  2. Build dependency graph with GraphBuilder
  3. Extract node features with FeatureExtractor
  4. Label nodes (git history or synthetic heuristic)
  5. Build PyG Data object with CodeGraphDataset
  6. Train GraphSAGE (or GCN) with early stopping
  7. Train baseline classifiers (LogReg, RandomForest)
  8. Evaluate all models and print comparison table
  9. Save weights, predictions, metrics, and graph files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from graphguard.data.dataset import CodeGraphDataset
from graphguard.data.git_mining import GitLabelPathMismatchError, GitMiner
from graphguard.graph.features import FeatureExtractor
from graphguard.graph.graph_builder import GraphBuilder
from graphguard.models.baselines import BaselineModels
from graphguard.models.evaluate import ModelMetrics, compute_metrics, print_metrics_table, save_metrics
from graphguard.models.gnn import build_model
from graphguard.parser.python_parser import PythonParser
from graphguard.utils.config import Config, DEFAULT_CONFIG
from graphguard.utils.logging import get_logger, console

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = criterion(logits[data.train_mask], data.y[data.train_mask].float())
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def _evaluate_split(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns (loss, y_true, y_prob) for the given node mask."""
    model.eval()
    logits = model(data.x, data.edge_index)
    loss = criterion(logits[mask], data.y[mask].float()).item()
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    y_true = data.y[mask].cpu().numpy()
    return float(loss), y_true, probs


def train_gnn(
    data: Data,
    config: Config = DEFAULT_CONFIG,
    device: Optional[torch.device] = None,
) -> tuple[nn.Module, list[ModelMetrics]]:
    """
    Train the GNN and return the best model + evaluation metrics.

    Parameters
    ----------
    data   : PyG Data object from CodeGraphDataset.build()
    config : Config with model hyperparameters
    device : torch device (cpu or cuda)

    Returns
    -------
    model   : trained GNN (best checkpoint by val loss)
    metrics : list with one ModelMetrics entry
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = data.to(device)

    # Class imbalance: weight positive class inversely proportional to its frequency
    n_pos = float(data.y[data.train_mask].sum().item())
    n_neg = float((data.y[data.train_mask] == 0).sum().item())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    mc = config.model
    model = build_model(
        model_type=mc.model_type,
        in_channels=data.num_node_features,
        hidden_dim=mc.hidden_dim,
        num_layers=mc.num_layers,
        dropout=mc.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=mc.lr, weight_decay=mc.weight_decay
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    # Very small repos may leave the validation split empty (see
    # CodeGraphDataset._split_masks). Track training loss for early
    # stopping/checkpointing in that case so training still completes
    # instead of comparing against a NaN loss forever.
    has_val = bool(data.val_mask.sum().item() > 0)
    if not has_val:
        logger.warning(
            "Validation split is empty — early stopping/checkpointing will "
            "track training loss instead of validation loss."
        )

    console.print(f"[bold cyan]Training {mc.model_type.upper()} on {device}...[/]")

    for epoch in range(1, mc.epochs + 1):
        train_loss = _train_epoch(model, data, optimizer, criterion)
        if has_val:
            val_loss, _, _ = _evaluate_split(model, data, data.val_mask, criterion)
        else:
            val_loss = train_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:>4}/{mc.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

        if patience_counter >= mc.patience:
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on test set
    _, y_true, y_prob = _evaluate_split(model, data, data.test_mask, criterion)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = [compute_metrics(mc.model_type.upper(), y_true, y_pred, y_prob)]

    return model, metrics


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    repo_path: str | Path,
    config: Config = DEFAULT_CONFIG,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run the complete parse → train → evaluate pipeline.

    Returns a summary dict with paths to all saved outputs.
    """
    repo_path = Path(repo_path)
    out = output_dir or config.resolve_output_dir(repo_path)

    console.rule("[bold cyan]GraphGuard Training Pipeline[/]")

    # Step 1: Parse
    console.print("[cyan]Step 1/7: Parsing repository...[/]")
    parser = PythonParser()
    parse_result = parser.parse(repo_path)

    # Step 2: Build graph
    console.print("[cyan]Step 2/7: Building dependency graph...[/]")
    builder = GraphBuilder()
    G = builder.build(parse_result)
    builder.save_all(G, out)

    # Step 3: Extract features
    console.print("[cyan]Step 3/7: Extracting node features...[/]")
    extractor = FeatureExtractor()
    features_df = extractor.extract(G)
    features_df.to_csv(out / "features.csv")

    # Step 4: Labels
    console.print("[cyan]Step 4/7: Generating labels...[/]")
    labels: Optional[dict[str, int]] = None
    if config.label_mode == "git":
        miner = GitMiner(repo_path)
        file_counts = miner.mine_bug_fix_labels()
        if file_counts:
            try:
                labels = miner.file_labels_to_node_labels(file_counts, list(G.nodes()))
                console.print(
                    f"[green]Git labels: {sum(labels.values())} risky / "
                    f"{len(labels)} total nodes[/]"
                )
            except GitLabelPathMismatchError as exc:
                console.print(f"[bold red]{exc}[/]")
                console.print(
                    "[yellow]Falling back to synthetic labels due to the path "
                    "mismatch above.[/]"
                )
                labels = None
        else:
            console.print("[yellow]No git history found — falling back to synthetic labels.[/]")

    # Step 5: Build dataset
    console.print("[cyan]Step 5/7: Building PyG dataset...[/]")
    dataset_builder = CodeGraphDataset(config)
    undirected = config.model.model_type == "gcn"
    data, meta = dataset_builder.build(G, features_df, labels=labels, undirected=undirected)

    # Save dataset metadata
    (out / "dataset_meta.json").write_text(
        json.dumps(meta.to_dict(), indent=2), encoding="utf-8"
    )

    # Step 6: Train GNN
    console.print("[cyan]Step 6/7: Training GNN...[/]")
    gnn_model, gnn_metrics = train_gnn(data, config)
    torch.save(gnn_model.state_dict(), out / config.model_weights_file)

    # GNN predictions on all nodes
    gnn_model.eval()
    with torch.no_grad():
        all_probs = torch.sigmoid(
            gnn_model(data.x, data.edge_index)
        ).cpu().numpy().tolist()

    # Step 7: Train baselines
    console.print("[cyan]Step 7/7: Training baseline models...[/]")
    feat_cols = FeatureExtractor.numeric_feature_columns()
    available_cols = [c for c in feat_cols if c in features_df.columns]
    X = features_df.reindex(data.node_ids)[available_cols].fillna(0.0).values

    y_all = data.y.cpu().numpy()
    X_train = X[data.train_mask.cpu().numpy()]
    y_train = y_all[data.train_mask.cpu().numpy()]
    X_test = X[data.test_mask.cpu().numpy()]
    y_test = y_all[data.test_mask.cpu().numpy()]

    baselines = BaselineModels()
    baseline_results = baselines.fit_predict(X_train, y_train, X_test, y_test)
    baseline_metrics = [
        compute_metrics(br.model_name, br.y_true, br.y_pred, br.y_prob)
        for br in baseline_results
    ]

    # Combine and display
    all_metrics = gnn_metrics + baseline_metrics
    print_metrics_table(all_metrics)
    save_metrics(all_metrics, out / config.metrics_file)

    # Save RandomForest feature importances — shows which structural signals
    # drive risk (a great talking point and a sanity check on the features).
    importances = baselines.feature_importances(feature_names=available_cols)
    if importances is not None:
        importances.to_csv(out / "feature_importances.csv", index=False)

    # Save predictions
    all_labels = data.y.cpu().numpy().tolist()
    dataset_builder.save_predictions(
        nodes=data.node_ids,
        probs=all_probs,
        labels=all_labels,
        features_df=features_df,
        path=out / config.predictions_file,
    )

    summary = {
        "output_dir": str(out),
        "graph_ml": str(out / "graph.graphml"),
        "features_csv": str(out / "features.csv"),
        "predictions_csv": str(out / config.predictions_file),
        "metrics_json": str(out / config.metrics_file),
        "model_weights": str(out / config.model_weights_file),
        "num_nodes": meta.num_nodes,
        "num_edges": meta.num_edges,
        "label_mode": meta.label_mode,
    }

    console.rule("[bold green]Pipeline complete[/]")
    return summary
