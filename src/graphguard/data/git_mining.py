"""
Git-history-based risk labeling.

Mines commit messages for bug-fix keywords and maps affected files to risk
labels. This is the real-label mode; the synthetic fallback lives in
dataset.py.

Signal quality note
-------------------
Bug-fix commit frequency is a noisy but well-studied proxy for code risk.
Files touched in many bug-fix commits tend to have higher coupling and
weaker encapsulation — exactly the structural properties GNNs can learn.
See:  Nagappan & Ball (2005), Zimmermann et al. (2007).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from graphguard.utils.config import BUG_KEYWORDS
from graphguard.utils.logging import get_logger

logger = get_logger(__name__)

_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in BUG_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


class GitMiner:
    """
    Extracts per-file risk signals from git commit history.

    Requires the `gitpython` library and a valid git repository.
    Falls back gracefully if git is not available.
    """

    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path)
        self._repo: Optional[object] = None

    def _load_repo(self) -> bool:
        """Attempt to load the git repository. Returns True on success."""
        try:
            import git  # gitpython
            self._repo = git.Repo(self.repo_path, search_parent_directories=True)
            return True
        except Exception as exc:
            logger.warning(f"Git mining unavailable: {exc}")
            return False

    def mine_bug_fix_labels(self) -> dict[str, int]:
        """
        Return a dict mapping repo-relative file paths to bug-fix commit counts.

        Files with count > 0 are candidates for the "risky" label.
        Only Python files are included.
        """
        if not self._load_repo():
            return {}

        import git  # noqa: F401 — already confirmed available above

        repo = self._repo  # type: ignore[assignment]
        file_bug_counts: dict[str, int] = {}

        try:
            commits = list(repo.iter_commits())  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning(f"Could not iterate commits: {exc}")
            return {}

        logger.info(f"Scanning {len(commits)} commits for bug-fix keywords...")

        for commit in commits:
            message = commit.message or ""
            if not _KEYWORD_PATTERN.search(message):
                continue

            # Find files changed in this commit
            try:
                if commit.parents:
                    diffs = commit.diff(commit.parents[0])
                    for diff in diffs:
                        fpath = diff.a_path or diff.b_path or ""
                        if fpath.endswith(".py"):
                            norm = fpath.replace("\\", "/")
                            file_bug_counts[norm] = file_bug_counts.get(norm, 0) + 1
                else:
                    # Initial commit — list all files
                    for item in commit.tree.traverse():
                        if hasattr(item, "path") and item.path.endswith(".py"):
                            norm = item.path.replace("\\", "/")
                            file_bug_counts[norm] = file_bug_counts.get(norm, 0) + 1
            except Exception:
                continue  # Skip malformed commits

        logger.info(
            f"Bug-fix mining complete. {len(file_bug_counts)} files flagged."
        )
        return file_bug_counts

    def file_labels_to_node_labels(
        self, file_counts: dict[str, int], node_ids: list[str]
    ) -> dict[str, int]:
        """
        Map file-level bug counts to node-level binary labels.

        A node is labeled 1 (risky) if:
          - it IS a file node AND that file has bug_count > 0, OR
          - it is a function/class node AND its containing file has bug_count > 0
        """
        labels: dict[str, int] = {}
        for nid in node_ids:
            label = 0
            parts = nid.split("::")
            if parts[0] == "file" and len(parts) >= 2:
                rel_path = parts[1]
                label = int(file_counts.get(rel_path, 0) > 0)
            elif parts[0] in ("func", "class") and len(parts) >= 3:
                rel_path = parts[1]
                label = int(file_counts.get(rel_path, 0) > 0)
            labels[nid] = label
        return labels
