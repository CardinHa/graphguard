"""
GraphGuard CLI — professional command-line interface.

Commands
--------
  graphguard analyze   <repo>   Parse repo and build dependency graph
  graphguard train     <repo>   Full train pipeline (parse + features + GNN + baselines)
  graphguard report    <repo>   Print risk report from existing outputs
  graphguard dashboard [repo]   Launch Streamlit dashboard
  graphguard api                Launch FastAPI server

Module-level entrypoint (for dev without installation):
  python -m graphguard.cli analyze examples/sample_project
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from graphguard.utils.config import Config, ModelConfig
from graphguard.utils.logging import console, get_logger

logger = get_logger(__name__)

app = typer.Typer(
    name="graphguard",
    help="GNN-Based Code Dependency Risk Analyzer",
    add_completion=False,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

@app.command()
def analyze(
    repo_path: str = typer.Argument(..., help="Path to Python repository"),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Output directory (default: <repo>/outputs)"
    ),
    model_type: str = typer.Option(
        "sage", "--model", "-m", help="GNN type: sage | gcn"
    ),
) -> None:
    """[cyan]Parse a Python repository and build its dependency graph.[/]"""
    from graphguard.graph.features import FeatureExtractor
    from graphguard.graph.graph_builder import GraphBuilder, print_graph_summary
    from graphguard.parser.python_parser import PythonParser

    repo = Path(repo_path)
    if not repo.exists():
        console.print(f"[red]Error:[/] Repository not found: {repo}")
        raise typer.Exit(1)

    out = Path(output_dir) if output_dir else repo / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]GraphGuard · Analyze[/]")

    # Parse
    console.print("[cyan]Parsing...[/]")
    parser = PythonParser()
    result = parser.parse(repo)

    if result.errors:
        console.print(f"[yellow]Parsing warnings ({len(result.errors)}):[/]")
        for e in result.errors[:5]:
            console.print(f"  [yellow]{e}[/]")

    # Graph
    console.print("[cyan]Building graph...[/]")
    builder = GraphBuilder()
    G = builder.build(result)
    builder.save_all(G, out)

    print_graph_summary(G)

    # Features
    console.print("[cyan]Extracting features...[/]")
    extractor = FeatureExtractor()
    df = extractor.extract(G)
    df.to_csv(out / "features.csv")

    console.print(
        f"[green]Done.[/] Graph saved to [bold]{out}[/] "
        f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)."
    )


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@app.command()
def train(
    repo_path: str = typer.Argument(..., help="Path to Python repository"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o"),
    label_mode: str = typer.Option(
        "synthetic", "--label-mode", "-l",
        help="Labeling strategy: git | synthetic"
    ),
    model_type: str = typer.Option("sage", "--model", "-m", help="sage | gcn"),
    epochs: int = typer.Option(200, "--epochs", "-e"),
    hidden_dim: int = typer.Option(64, "--hidden-dim"),
    layers: int = typer.Option(2, "--layers"),
    dropout: float = typer.Option(0.3, "--dropout"),
) -> None:
    """[cyan]Full training pipeline: parse -> graph -> GNN -> baselines -> report.[/]"""
    from graphguard.models.train import run_full_pipeline

    repo = Path(repo_path)
    if not repo.exists():
        console.print(f"[red]Error:[/] Repository not found: {repo}")
        raise typer.Exit(1)

    out = Path(output_dir) if output_dir else repo / "outputs"

    config = Config(
        label_mode=label_mode,
        model=ModelConfig(
            model_type=model_type,
            epochs=epochs,
            hidden_dim=hidden_dim,
            num_layers=layers,
            dropout=dropout,
        ),
    )

    try:
        summary = run_full_pipeline(repo, config=config, output_dir=out)
        console.print("\n[bold green]Training complete. Outputs:[/]")
        for k, v in summary.items():
            console.print(f"  {k}: [dim]{v}[/]")
    except Exception as exc:
        console.print(f"[red]Training failed:[/] {exc}")
        logger.exception("Training error")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@app.command()
def report(
    repo_path: str = typer.Argument(..., help="Path to Python repository"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o"),
    top_n: int = typer.Option(20, "--top-n", "-n", help="Top N risky nodes to display"),
) -> None:
    """[cyan]Print a risk report from existing outputs (run train first).[/]"""
    import pandas as pd

    repo = Path(repo_path)
    out = Path(output_dir) if output_dir else repo / "outputs"

    pred_path = out / "predictions.csv"
    metrics_path = out / "metrics.json"

    if not pred_path.exists():
        console.print(
            f"[red]No predictions found at {pred_path}.[/] Run [bold]graphguard train[/] first."
        )
        raise typer.Exit(1)

    console.rule("[bold cyan]GraphGuard · Risk Report[/]")

    # Metrics
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        table = Table(title="Model Performance", style="cyan")
        cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
        for c in cols:
            table.add_column(c, justify="center")
        for m in metrics:
            table.add_row(
                m.get("model_name", ""),
                f"{m.get('accuracy', 0):.4f}",
                f"{m.get('precision', 0):.4f}",
                f"{m.get('recall', 0):.4f}",
                f"{m.get('f1', 0):.4f}",
                f"{m.get('roc_auc', 0):.4f}",
                f"{m.get('pr_auc', 0):.4f}",
            )
        console.print(table)

    # Top risky nodes
    preds = pd.read_csv(pred_path)
    preds = preds.sort_values("risk_score", ascending=False).head(top_n)

    risk_table = Table(title=f"Top {top_n} Risky Nodes", style="red")
    risk_table.add_column("Name", style="bold")
    risk_table.add_column("Type")
    risk_table.add_column("File")
    risk_table.add_column("Risk Score", justify="right")
    risk_table.add_column("Predicted Risky", justify="center")

    for _, row in preds.iterrows():
        risk_table.add_row(
            str(row.get("name", "")),
            str(row.get("entity_type", "")),
            str(row.get("file_path", "")),
            f"{row.get('risk_score', 0):.4f}",
            "[red]YES[/]" if row.get("predicted_risky", 0) else "[green]NO[/]",
        )

    console.print(risk_table)


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

@app.command()
def dashboard(
    repo_path: Optional[str] = typer.Argument(None, help="Path to analyzed repository"),
    port: int = typer.Option(8501, "--port", "-p"),
) -> None:
    """[cyan]Launch the Streamlit dashboard.[/]"""
    import subprocess

    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"

    env_args = []
    if repo_path:
        env_args = ["--", f"--repo={repo_path}"]

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(port),
        "--server.headless", "true",
    ] + env_args

    console.print(f"[cyan]Launching dashboard at http://localhost:{port}[/]")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("[yellow]Dashboard stopped.[/]")
    except Exception as exc:
        console.print(f"[red]Failed to launch dashboard:[/] {exc}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# api
# ---------------------------------------------------------------------------

@app.command()
def api(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", help="Hot reload (dev only)"),
) -> None:
    """[cyan]Launch the FastAPI REST server.[/]"""
    import uvicorn

    console.print(f"[cyan]Starting GraphGuard API at http://{host}:{port}[/]")
    uvicorn.run(
        "graphguard.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


# ---------------------------------------------------------------------------
# Entrypoint for `python -m graphguard.cli`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
