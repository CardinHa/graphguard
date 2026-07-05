"""
GraphGuard FastAPI backend.

Endpoints
---------
  GET  /health          Liveness check
  POST /analyze         Parse + analyze a repo (accepts repo_path in request body)
  GET  /graph           Return graph stats and edge/node lists from latest run
  GET  /predictions     Return sorted node risk predictions from latest run
  GET  /metrics         Return model comparison metrics from latest run
  POST /explain         GNNExplainer attribution for a single node

Design note: This API is stateless between requests — it reads outputs written
by the training pipeline rather than holding in-memory state, making it easy
to scale horizontally.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graphguard.utils.logging import get_logger
from graphguard.utils.optional_deps import missing_dependency_message

logger = get_logger(__name__)

app = FastAPI(
    title="GraphGuard API",
    description="GNN-Based Code Dependency Risk Analyzer",
    version="0.1.0",
)

# CORS: restricted to localhost dev origins by default (the dashboard and a
# local browser session are the only expected callers). Override with a
# comma-separated GRAPHGUARD_CORS_ORIGINS env var for other deployments.
_DEFAULT_CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:8501",
    "http://127.0.0.1",
    "http://127.0.0.1:8501",
]
_cors_env = os.environ.get("GRAPHGUARD_CORS_ORIGINS")
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else _DEFAULT_CORS_ORIGINS
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default outputs directory — resolved relative to the running directory
_DEFAULT_OUTPUT_DIR = Path("outputs")

# Root directory that user-supplied repo_path/output_dir values must resolve
# within. Without this, /analyze and friends would happily read/write any
# path the caller names (e.g. "../../etc" or an absolute system path).
# Defaults to the server's working directory; override with GRAPHGUARD_ROOT
# for deployments that serve repos from elsewhere.
_ALLOWED_ROOT = Path(os.environ.get("GRAPHGUARD_ROOT", Path.cwd())).resolve()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    repo_path: str
    output_dir: Optional[str] = None
    label_mode: Literal["synthetic", "git"] = "synthetic"
    model_type: Literal["sage", "gcn"] = "sage"
    epochs: int = Field(default=200, ge=1, le=1000)


class ExplainRequest(BaseModel):
    repo_path: str
    node: str
    output_dir: Optional[str] = None
    top_k: int = 5
    explainer_epochs: int = Field(default=200, ge=1, le=500)


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_within_root(raw_path: str, field_name: str) -> Path:
    """Resolve a user-supplied path and reject it (400) if it escapes
    _ALLOWED_ROOT, e.g. via "../.." traversal or an absolute path elsewhere
    on the filesystem."""
    resolved = Path(raw_path).resolve()
    if not resolved.is_relative_to(_ALLOWED_ROOT):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{field_name} must resolve within {_ALLOWED_ROOT} "
                "(set the GRAPHGUARD_ROOT environment variable to allow a "
                "different location)."
            ),
        )
    return resolved


def _output_dir(override: Optional[str] = None) -> Path:
    if override:
        return _resolve_within_root(override, "output_dir")
    return _DEFAULT_OUTPUT_DIR


def _require_file(path: Path) -> None:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {path}. Run /analyze first.",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    """
    Parse a Python repository, build its dependency graph, and train the GNN.

    This is a synchronous endpoint that runs the full pipeline inline.
    For large repositories, consider wrapping in a background task.
    """
    try:
        from graphguard.models.train import run_full_pipeline
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"{missing_dependency_message('gnn', 'POST /analyze')} ({exc})",
        )
    from graphguard.utils.config import Config, ModelConfig

    repo = _resolve_within_root(request.repo_path, "repo_path")
    if not repo.exists():
        raise HTTPException(status_code=400, detail=f"Repository not found: {repo}")

    config = Config(
        label_mode=request.label_mode,
        model=ModelConfig(
            model_type=request.model_type,
            epochs=request.epochs,
        ),
    )
    out = _output_dir(request.output_dir)

    try:
        summary = run_full_pipeline(repo, config=config, output_dir=out)
        return {"status": "ok", **summary}
    except Exception:
        logger.exception(f"analyze failed for repo_path={request.repo_path!r}")
        raise HTTPException(
            status_code=500,
            detail="Internal error while analyzing the repository. See server logs for details.",
        )


@app.get("/graph")
def get_graph(output_dir: Optional[str] = None) -> dict[str, Any]:
    """Return graph statistics and a sample of nodes/edges."""
    out = _output_dir(output_dir)
    nodes_path = out / "nodes.csv"
    edges_path = out / "edges.csv"
    _require_file(nodes_path)
    _require_file(edges_path)

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    type_counts = (
        nodes_df["entity_type"].value_counts().to_dict()
        if "entity_type" in nodes_df.columns
        else {}
    )
    edge_type_counts = (
        edges_df["relationship_type"].value_counts().to_dict()
        if "relationship_type" in edges_df.columns
        else {}
    )

    return {
        "num_nodes": len(nodes_df),
        "num_edges": len(edges_df),
        "node_type_counts": type_counts,
        "edge_type_counts": edge_type_counts,
        "sample_nodes": nodes_df.head(20).to_dict(orient="records"),
        "sample_edges": edges_df.head(20).to_dict(orient="records"),
    }


@app.get("/predictions")
def get_predictions(
    output_dir: Optional[str] = None,
    top_n: int = 50,
    entity_type: Optional[str] = None,
) -> dict[str, Any]:
    """Return top-N risky nodes sorted by risk_score."""
    out = _output_dir(output_dir)
    pred_path = out / "predictions.csv"
    _require_file(pred_path)

    df = pd.read_csv(pred_path)

    if entity_type:
        df = df[df["entity_type"] == entity_type]

    df = df.sort_values("risk_score", ascending=False).head(top_n)

    return {
        "total": len(df),
        "predictions": df.fillna("").to_dict(orient="records"),
    }


@app.get("/metrics")
def get_metrics(output_dir: Optional[str] = None) -> dict[str, Any]:
    """Return model comparison metrics."""
    out = _output_dir(output_dir)
    metrics_path = out / "metrics.json"
    _require_file(metrics_path)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {"models": metrics}


@app.post("/explain")
def explain(request: ExplainRequest) -> dict[str, Any]:
    """
    Run GNNExplainer for a single node and return feature + edge attribution.

    Reconstructs the dependency graph from the repository (fast — no training),
    loads the saved model weights, and returns which input features and graph
    neighbors most influenced the risk prediction for the requested node.

    The `node` field accepts:
      - Exact node_id (e.g. "func::src/requests/utils.py::resolve_proxies")
      - Function or class name (e.g. "resolve_proxies")
      - Substring of a node_id (e.g. "utils.py")
    """
    try:
        from graphguard.data.dataset import CodeGraphDataset
        from graphguard.models.explain import explain_node, load_model
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"{missing_dependency_message('gnn', 'POST /explain')} ({exc})",
        )
    from graphguard.data.git_mining import GitLabelPathMismatchError, GitMiner
    from graphguard.graph.features import FeatureExtractor
    from graphguard.graph.graph_builder import GraphBuilder
    from graphguard.parser.python_parser import PythonParser
    from graphguard.utils.config import Config, ModelConfig

    repo = _resolve_within_root(request.repo_path, "repo_path")
    if not repo.exists():
        raise HTTPException(status_code=400, detail=f"Repository not found: {repo}")

    out = _output_dir(request.output_dir)
    meta_path = out / "dataset_meta.json"
    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No training outputs at {out}. Run /analyze first.",
        )

    meta_info = json.loads(meta_path.read_text(encoding="utf-8"))
    label_mode = meta_info.get("label_mode", "synthetic")
    # Reuse the model hyperparameters persisted at train time so the
    # reconstructed architecture matches the saved state_dict.
    default_mc = ModelConfig()
    config = Config(
        label_mode=label_mode,
        model=ModelConfig(
            model_type=meta_info.get("model_type", default_mc.model_type),
            hidden_dim=meta_info.get("hidden_dim", default_mc.hidden_dim),
            num_layers=meta_info.get("num_layers", default_mc.num_layers),
            dropout=meta_info.get("dropout", default_mc.dropout),
        ),
    )

    try:
        # Rebuild graph + dataset (no training — typically < 3s for small repos)
        parser = PythonParser()
        parse_result = parser.parse(repo)

        builder = GraphBuilder()
        G = builder.build(parse_result)

        extractor = FeatureExtractor()
        features_df = extractor.extract(G)

        labels = None
        if label_mode == "git":
            miner = GitMiner(repo)
            file_counts = miner.mine_bug_fix_labels()
            if file_counts:
                try:
                    labels = miner.file_labels_to_node_labels(file_counts, list(G.nodes()))
                except GitLabelPathMismatchError as exc:
                    logger.error(str(exc))
                    labels = None

        dataset_builder = CodeGraphDataset(config)
        data, _ = dataset_builder.build(G, features_df, labels=labels, undirected=False)

        model = load_model(out, data, config)

        feat_names = FeatureExtractor.numeric_feature_columns()
        available = [c for c in feat_names if c in features_df.columns]

        result = explain_node(
            query=request.node,
            data=data,
            model=model,
            features_df=features_df,
            feature_names=available,
            top_k=request.top_k,
            explainer_epochs=request.explainer_epochs,
        )
        return result

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        logger.exception(
            f"explain failed for repo_path={request.repo_path!r}, node={request.node!r}"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal error while explaining the node. See server logs for details.",
        )
