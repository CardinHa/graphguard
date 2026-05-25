"""Central configuration for GraphGuard."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Folders the parser should always skip
SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", "venv", ".venv", "env", ".env",
    "node_modules", "dist", "build", "outputs", ".tox",
    ".mypy_cache", ".pytest_cache", "site-packages",
})

# Keywords that signal a bug-fix commit in git history
BUG_KEYWORDS: tuple[str, ...] = (
    "fix", "bug", "issue", "error", "crash",
    "regression", "patch", "hotfix", "defect", "fault",
)


@dataclass
class ModelConfig:
    """Hyperparameters for the GNN."""
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 20          # early stopping patience
    model_type: str = "sage"    # "sage" | "gcn"


@dataclass
class Config:
    """Top-level project configuration."""
    # Paths
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    graph_file: str = "graph.graphml"
    features_file: str = "features.csv"
    predictions_file: str = "predictions.csv"
    metrics_file: str = "metrics.json"
    model_weights_file: str = "gnn_weights.pt"

    # Graph construction
    add_undirected_option: bool = False   # convert to undirected for GCN if needed

    # Labeling
    label_mode: str = "synthetic"         # "git" | "synthetic"
    synthetic_risk_percentile: float = 0.70   # top 30% are "risky"

    # Train/val/test split fractions
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac is inferred as 1 - train - val

    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    def resolve_output_dir(self, base: Path | None = None) -> Path:
        """Return output_dir relative to base (or cwd if None)."""
        root = base or Path.cwd()
        out = root / self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        return out


# Singleton used across the project
DEFAULT_CONFIG = Config()
