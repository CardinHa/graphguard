"""Models subpackage: GNN classifiers and sklearn baselines.

Submodules are exposed lazily (PEP 562) because ``gnn`` imports torch, an
optional 'gnn' extra — an eager import here would make ``BaselineModels``
(sklearn-only, part of the core install) unimportable without torch.
"""

from typing import Any

__all__ = ["GraphSAGEClassifier", "GCNClassifier", "BaselineModels"]


def __getattr__(name: str) -> Any:
    if name in ("GraphSAGEClassifier", "GCNClassifier"):
        from graphguard.models import gnn
        return getattr(gnn, name)
    if name == "BaselineModels":
        from graphguard.models.baselines import BaselineModels
        return BaselineModels
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
