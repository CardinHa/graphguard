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
from graphguard.utils.optional_deps import missing_dependency_message

logger = get_logger(__name__)

_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in BUG_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


class GitLabelPathMismatchError(RuntimeError):
    """
    Raised when none of the mined git file paths overlap with any parsed
    file path.

    This is almost always a path-frame bug — e.g. ``search_parent_directories``
    discovered a ``.git`` in an ancestor directory, so GitPython's diff paths
    are relative to that ancestor while the parser's node paths are relative
    to the analyzed subdirectory — rather than a genuine absence of risky
    files. Proceeding would silently label every node non-risky, which looks
    like "a very safe codebase" instead of "a path bug".
    """


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
        except ImportError:
            logger.warning(
                f"Git mining unavailable: {missing_dependency_message('git', 'Git-based labeling')}"
            )
            return False
        try:
            self._repo = git.Repo(self.repo_path, search_parent_directories=True)
            return True
        except Exception as exc:
            logger.warning(f"Git mining unavailable: {exc}")
            return False

    def _path_offset(self) -> str:
        """
        Repo-relative offset between the discovered git root and the
        analyzed directory.

        ``search_parent_directories=True`` means the git root GitPython
        finds can be an ancestor of ``self.repo_path`` (e.g. analyzing a
        package nested inside a larger monorepo). GitPython's diff paths
        are always relative to that git root, but node paths built by
        PythonParser are relative to ``self.repo_path`` — so if the two
        roots differ, every mined path must be re-relativized or file
        lookups silently miss everything.

        Returns "" when the roots coincide (the common case) or when the
        analyzed path isn't under the git root at all (nothing sensible to
        strip; the zero-overlap check in ``file_labels_to_node_labels``
        will catch a resulting mismatch).
        """
        repo = self._repo
        if repo is None or not getattr(repo, "working_tree_dir", None):
            return ""
        git_root = Path(repo.working_tree_dir).resolve()
        analyzed_root = self.repo_path.resolve()
        try:
            offset = analyzed_root.relative_to(git_root)
        except ValueError:
            return ""
        offset_str = str(offset).replace("\\", "/")
        return "" if offset_str == "." else offset_str

    def _relativize(self, git_relative_path: str, offset: str) -> Optional[str]:
        """Re-relativize a git-root-relative path to the analyzed root.

        Returns None if the path falls outside the analyzed subdirectory
        (irrelevant to this parse) or lies at/above the offset boundary.
        """
        norm = git_relative_path.replace("\\", "/")
        if not offset:
            return norm
        prefix = offset + "/"
        if norm.startswith(prefix):
            return norm[len(prefix):]
        return None

    def mine_bug_fix_labels(self) -> dict[str, int]:
        """
        Return a dict mapping repo-relative file paths to bug-fix commit counts.

        Files with count > 0 are candidates for the "risky" label.
        Only Python files are included. Paths are relative to the analyzed
        directory (``self.repo_path``), not necessarily the git root — see
        ``_path_offset``.
        """
        if not self._load_repo():
            return {}

        import git  # noqa: F401 — already confirmed available above

        repo = self._repo  # type: ignore[assignment]
        offset = self._path_offset()
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
                            norm = self._relativize(fpath, offset)
                            if norm is not None:
                                file_bug_counts[norm] = file_bug_counts.get(norm, 0) + 1
                else:
                    # Initial commit — list all files
                    for item in commit.tree.traverse():
                        if hasattr(item, "path") and item.path.endswith(".py"):
                            norm = self._relativize(item.path, offset)
                            if norm is not None:
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

        Raises
        ------
        GitLabelPathMismatchError
            If ``file_counts`` and the parsed node paths are both non-empty
            but share zero overlap. See ``GitLabelPathMismatchError`` — this
            is almost always a path-frame bug, and proceeding would silently
            produce an all-zero-label dataset instead of failing loudly.
        """
        parsed_paths: set[str] = set()
        for nid in node_ids:
            parts = nid.split("::")
            if parts[0] in ("file", "func", "class") and len(parts) >= 2:
                parsed_paths.add(parts[1])

        if file_counts and parsed_paths and not (set(file_counts) & parsed_paths):
            msg = (
                f"GIT LABEL PATH MISMATCH: none of the {len(file_counts)} mined "
                f"git file path(s) match any of the {len(parsed_paths)} parsed "
                "file path(s). This almost always means the git repository root "
                "and the analyzed directory differ (search_parent_directories "
                "found an ancestor .git). Proceeding would silently label every "
                "node as non-risky. "
                f"Sample mined paths: {sorted(file_counts)[:5]} | "
                f"Sample parsed paths: {sorted(parsed_paths)[:5]}"
            )
            logger.error(msg)
            raise GitLabelPathMismatchError(msg)

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
