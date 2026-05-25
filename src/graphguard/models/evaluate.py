"""
Evaluation metrics and comparison table for GNN vs baselines.

Metrics computed
----------------
  Accuracy    — overall correctness
  Precision   — TP / (TP + FP)  — how many predicted-risky are truly risky
  Recall      — TP / (TP + FN)  — how many truly risky are caught
  F1 Score    — harmonic mean of precision and recall
  ROC-AUC     — area under receiver-operating-characteristic curve
  PR-AUC      — area under precision-recall curve
              (preferred when classes are imbalanced)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from graphguard.utils.logging import get_logger, console

logger = get_logger(__name__)


@dataclass
class ModelMetrics:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> ModelMetrics:
    """Compute all classification metrics for a single model."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC metrics require probability scores
    roc_auc = pr_auc = 0.0
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        except Exception:
            pass

    return ModelMetrics(
        model_name=model_name,
        accuracy=round(float(acc), 4),
        precision=round(float(prec), 4),
        recall=round(float(rec), 4),
        f1=round(float(f1), 4),
        roc_auc=round(float(roc_auc), 4),
        pr_auc=round(float(pr_auc), 4),
    )


def print_metrics_table(metrics_list: list[ModelMetrics]) -> None:
    """Render a rich comparison table to the console."""
    from rich.table import Table

    table = Table(title="Model Comparison", style="cyan")
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    for c in cols:
        table.add_column(c, justify="center")

    for m in metrics_list:
        table.add_row(
            m.model_name,
            f"{m.accuracy:.4f}",
            f"{m.precision:.4f}",
            f"{m.recall:.4f}",
            f"{m.f1:.4f}",
            f"{m.roc_auc:.4f}",
            f"{m.pr_auc:.4f}",
        )

    console.print(table)


def save_metrics(metrics_list: list[ModelMetrics], path: Path) -> None:
    data = [m.to_dict() for m in metrics_list]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Metrics saved -> {path}")
