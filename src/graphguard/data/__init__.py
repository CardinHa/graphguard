"""Data subpackage: PyG dataset construction and git-history labeling.

Submodules are exposed lazily (PEP 562) because ``dataset`` imports torch,
an optional 'gnn' extra — an eager import here would make ``GitMiner``
(which only needs the optional 'git' extra) unimportable without torch.
"""

from typing import Any

__all__ = ["CodeGraphDataset", "GitMiner"]


def __getattr__(name: str) -> Any:
    if name == "CodeGraphDataset":
        from graphguard.data.dataset import CodeGraphDataset
        return CodeGraphDataset
    if name == "GitMiner":
        from graphguard.data.git_mining import GitMiner
        return GitMiner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
