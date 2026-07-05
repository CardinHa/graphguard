"""
Tests for API hardening (see graphguard/api/main.py):
  - repo_path / output_dir are constrained to GRAPHGUARD_ROOT
  - AnalyzeRequest / ExplainRequest reject out-of-range or invalid values
  - CORS defaults to localhost-only origins, not "*"
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pydantic = pytest.importorskip("pydantic")

from fastapi import HTTPException  # noqa: E402
from pydantic import ValidationError  # noqa: E402


@pytest.fixture()
def api_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Import graphguard.api.main with GRAPHGUARD_ROOT pinned to a temp dir
    so path-allowlist assertions don't depend on the real cwd, then restore
    the module to its default state afterwards."""
    monkeypatch.setenv("GRAPHGUARD_ROOT", str(tmp_path))
    import graphguard.api.main as main
    importlib.reload(main)  # re-evaluate module-level _ALLOWED_ROOT from env
    try:
        yield main
    finally:
        monkeypatch.delenv("GRAPHGUARD_ROOT", raising=False)
        importlib.reload(main)


class TestPathAllowlist:
    def test_path_inside_root_is_allowed(self, api_main, tmp_path: Path) -> None:
        sub = tmp_path / "myrepo"
        sub.mkdir()
        resolved = api_main._resolve_within_root(str(sub), "repo_path")
        assert resolved == sub.resolve()

    def test_path_outside_root_is_rejected(self, api_main, tmp_path: Path) -> None:
        outside = tmp_path.parent / "elsewhere"
        with pytest.raises(HTTPException) as excinfo:
            api_main._resolve_within_root(str(outside), "repo_path")
        assert excinfo.value.status_code == 400

    def test_traversal_outside_root_is_rejected(self, api_main, tmp_path: Path) -> None:
        traversal = str(tmp_path / ".." / ".." / "etc")
        with pytest.raises(HTTPException) as excinfo:
            api_main._resolve_within_root(traversal, "repo_path")
        assert excinfo.value.status_code == 400

    def test_output_dir_helper_enforces_allowlist(self, api_main, tmp_path: Path) -> None:
        with pytest.raises(HTTPException):
            api_main._output_dir(str(tmp_path.parent / "outside_outputs"))

    def test_output_dir_helper_allows_default(self, api_main) -> None:
        # No override -> the fixed default, never subject to the allowlist
        assert api_main._output_dir(None) == api_main._DEFAULT_OUTPUT_DIR


class TestRequestFieldBounds:
    def test_analyze_epochs_within_bounds_ok(self, api_main) -> None:
        req = api_main.AnalyzeRequest(repo_path=".", epochs=500)
        assert req.epochs == 500

    def test_analyze_epochs_too_high_rejected(self, api_main) -> None:
        with pytest.raises(ValidationError):
            api_main.AnalyzeRequest(repo_path=".", epochs=100_000)

    def test_analyze_epochs_zero_rejected(self, api_main) -> None:
        with pytest.raises(ValidationError):
            api_main.AnalyzeRequest(repo_path=".", epochs=0)

    def test_analyze_label_mode_invalid_rejected(self, api_main) -> None:
        with pytest.raises(ValidationError):
            api_main.AnalyzeRequest(repo_path=".", label_mode="bogus")

    def test_analyze_model_type_invalid_rejected(self, api_main) -> None:
        with pytest.raises(ValidationError):
            api_main.AnalyzeRequest(repo_path=".", model_type="bogus")

    def test_explain_explainer_epochs_too_high_rejected(self, api_main) -> None:
        with pytest.raises(ValidationError):
            api_main.ExplainRequest(repo_path=".", node="foo", explainer_epochs=99_999)

    def test_explain_explainer_epochs_within_bounds_ok(self, api_main) -> None:
        req = api_main.ExplainRequest(repo_path=".", node="foo", explainer_epochs=300)
        assert req.explainer_epochs == 300


class TestCorsDefaults:
    def test_default_cors_origins_are_localhost_only(self, api_main) -> None:
        assert "*" not in api_main._cors_origins
        assert all(
            origin.startswith("http://localhost") or origin.startswith("http://127.0.0.1")
            for origin in api_main._cors_origins
        )
