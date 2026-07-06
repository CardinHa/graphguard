"""
Tests for GitMiner: label mapping, the git-root/parse-root path-offset fix,
and the loud-error guard against a path-frame mismatch that would otherwise
silently produce an all-zero-label dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphguard.data.git_mining import GitLabelPathMismatchError, GitMiner


class TestFileLabelsToNodeLabels:
    def test_maps_file_and_contained_entities(self) -> None:
        miner = GitMiner("unused")
        file_counts = {"pkg/risky.py": 3, "pkg/safe.py": 0}
        node_ids = [
            "file::pkg/risky.py",
            "func::pkg/risky.py::do_thing",
            "class::pkg/risky.py::Thing",
            "file::pkg/safe.py",
            "func::pkg/safe.py::helper",
            "module::os",
        ]
        labels = miner.file_labels_to_node_labels(file_counts, node_ids)
        assert labels["file::pkg/risky.py"] == 1
        assert labels["func::pkg/risky.py::do_thing"] == 1
        assert labels["class::pkg/risky.py::Thing"] == 1
        assert labels["file::pkg/safe.py"] == 0
        assert labels["func::pkg/safe.py::helper"] == 0
        assert labels["module::os"] == 0

    def test_raises_on_zero_overlap(self) -> None:
        """If none of the mined paths match any parsed path, this is almost
        always a path-frame bug (git root != parse root) — must raise
        loudly rather than silently producing all-zero labels."""
        miner = GitMiner("unused")
        file_counts = {"some/other/repo/file.py": 5}
        node_ids = ["file::pkg/thing.py", "func::pkg/thing.py::run"]
        with pytest.raises(GitLabelPathMismatchError):
            miner.file_labels_to_node_labels(file_counts, node_ids)

    def test_no_raise_when_file_counts_empty(self) -> None:
        """An empty mined-label dict (e.g. no bug-fix commits found) is not
        a mismatch — it's just "nothing flagged"."""
        miner = GitMiner("unused")
        labels = miner.file_labels_to_node_labels({}, ["file::pkg/thing.py"])
        assert labels["file::pkg/thing.py"] == 0


class TestRelativize:
    def test_no_offset_passthrough(self) -> None:
        miner = GitMiner("unused")
        assert miner._relativize("pkg/mod.py", "") == "pkg/mod.py"

    def test_strips_offset_prefix(self) -> None:
        miner = GitMiner("unused")
        assert miner._relativize("sub/pkg/mod.py", "sub") == "pkg/mod.py"

    def test_outside_offset_returns_none(self) -> None:
        miner = GitMiner("unused")
        assert miner._relativize("other/pkg/mod.py", "sub") is None

    def test_normalizes_backslashes(self) -> None:
        miner = GitMiner("unused")
        assert miner._relativize("sub\\pkg\\mod.py", "sub") == "pkg/mod.py"


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """A minimal real git repo with one commit, for path-offset tests."""
    git = pytest.importorskip("git")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    repo = git.Repo.init(repo_root)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test User")
        cw.set_value("user", "email", "test@example.com")
    (repo_root / "sub").mkdir()
    (repo_root / "sub" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    repo.index.add(["sub/mod.py"])
    repo.index.commit("fix: initial commit")
    return repo_root


class TestPathOffset:
    def test_same_root_has_no_offset(self, git_repo: Path) -> None:
        miner = GitMiner(git_repo)
        assert miner._load_repo() is True
        assert miner._path_offset() == ""

    def test_subdirectory_analysis_computes_offset(self, git_repo: Path) -> None:
        """Analyzing `sub/` inside a repo whose root is one level up must
        compute an offset so mined paths get re-relativized to `sub/`."""
        miner = GitMiner(git_repo / "sub")
        assert miner._load_repo() is True
        assert miner._path_offset() == "sub"

    def test_mining_from_subdirectory_strips_offset(self, git_repo: Path) -> None:
        """The bug this fixes: search_parent_directories finds the repo root
        one level above the analyzed directory, so diff paths ("sub/mod.py")
        must be re-relativized to "mod.py" to match parser node paths."""
        miner = GitMiner(git_repo / "sub")
        counts = miner.mine_bug_fix_labels()
        assert counts.get("mod.py", 0) >= 1
        assert not any(p.startswith("sub/") for p in counts)
