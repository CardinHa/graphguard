"""
GraphGuard Streamlit Dashboard.

Layout
------
  Sidebar : repo path input + Run Analysis button
  Tab 1   : Graph Overview (stats, pyvis visualization)
  Tab 2   : Risk Table (sortable predictions)
  Tab 3   : Model Metrics (GNN vs baselines)
  Tab 4   : Node Inspector (click a node to explain its risk score)

Run
---
  streamlit run src/graphguard/dashboard/app.py
  or via CLI: graphguard dashboard path/to/repo
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GraphGuard · Risk Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_predictions(out: Path) -> Optional[pd.DataFrame]:
    p = out / "predictions.csv"
    return pd.read_csv(p) if p.exists() else None


def _load_metrics(out: Path) -> Optional[list[dict]]:
    p = out / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def _load_edges(out: Path) -> Optional[pd.DataFrame]:
    p = out / "edges.csv"
    return pd.read_csv(p) if p.exists() else None


def _load_features(out: Path) -> Optional[pd.DataFrame]:
    p = out / "features.csv"
    return pd.read_csv(p) if p.exists() else None


def _build_pyvis_html(edges_df: pd.DataFrame, predictions_df: Optional[pd.DataFrame]) -> str:
    """Build a pyvis HTML graph and return it as a string."""
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p>pyvis not installed. Run: pip install pyvis</p>"

    risk_map: dict[str, float] = {}
    type_map: dict[str, str] = {}
    name_map: dict[str, str] = {}

    if predictions_df is not None:
        for _, row in predictions_df.iterrows():
            nid = str(row.get("node_id", ""))
            risk_map[nid] = float(row.get("risk_score", 0.0))
            type_map[nid] = str(row.get("entity_type", "unknown"))
            name_map[nid] = str(row.get("name", nid))

    _TYPE_COLORS = {
        "file": "#4e9af1",
        "function": "#f1c40f",
        "class": "#2ecc71",
        "module": "#95a5a6",
        "unknown": "#bdc3c7",
    }

    net = Network(height="600px", width="100%", directed=True, bgcolor="#1a1a2e")
    net.set_options("""
    {
      "physics": { "stabilization": { "iterations": 100 } },
      "edges": { "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } }, "color": "#555" },
      "nodes": { "font": { "color": "#ffffff", "size": 12 } }
    }
    """)

    # Limit to first 200 edges for performance
    sample = edges_df.head(200)

    all_nodes: set[str] = set()
    for _, row in sample.iterrows():
        all_nodes.add(str(row["source"]))
        all_nodes.add(str(row["target"]))

    for nid in all_nodes:
        risk = risk_map.get(nid, 0.0)
        etype = type_map.get(nid, "unknown")
        label = name_map.get(nid, nid.split("::")[-1])[:25]
        color = "#e74c3c" if risk >= 0.5 else _TYPE_COLORS.get(etype, "#bdc3c7")
        size = 15 + risk * 20
        net.add_node(nid, label=label, color=color, size=size, title=f"{nid}\nRisk: {risk:.3f}")

    for _, row in sample.iterrows():
        net.add_edge(str(row["source"]), str(row["target"]))

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        return Path(f.name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🔍 GraphGuard")
st.sidebar.markdown("**GNN-Based Code Dependency Risk Analyzer**")
st.sidebar.divider()

repo_path_input = st.sidebar.text_input(
    "Repository path",
    value="examples/sample_project",
    help="Absolute or relative path to the Python repository to analyze.",
)

output_dir_input = st.sidebar.text_input(
    "Output directory",
    value="",
    help="Leave blank to use <repo>/outputs",
)

label_mode = st.sidebar.selectbox("Label mode", ["synthetic", "git"])
model_type = st.sidebar.selectbox("GNN model", ["sage", "gcn"])
epochs = st.sidebar.slider("Training epochs", 50, 500, 200, step=50)

run_btn = st.sidebar.button("▶ Run Full Pipeline", type="primary", use_container_width=True)

if run_btn:
    from graphguard.models.train import run_full_pipeline
    from graphguard.utils.config import Config, ModelConfig

    repo = Path(repo_path_input)
    out = Path(output_dir_input) if output_dir_input else repo / "outputs"

    config = Config(
        label_mode=label_mode,
        model=ModelConfig(model_type=model_type, epochs=epochs),
    )

    with st.spinner("Running analysis and training pipeline..."):
        try:
            summary = run_full_pipeline(repo, config=config, output_dir=out)
            st.sidebar.success("Pipeline complete!")
            st.session_state["output_dir"] = str(out)
        except Exception as exc:
            st.sidebar.error(f"Error: {exc}")

st.sidebar.divider()
st.sidebar.caption("Outputs auto-loaded from repo/outputs when available.")

# ---------------------------------------------------------------------------
# Resolve output dir
# ---------------------------------------------------------------------------

out_dir_str = st.session_state.get(
    "output_dir",
    (Path(repo_path_input) / "outputs").as_posix() if repo_path_input else "outputs",
)
out_dir = Path(out_dir_str)

predictions = _load_predictions(out_dir)
metrics = _load_metrics(out_dir)
edges_df = _load_edges(out_dir)
features_df = _load_features(out_dir)

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("GraphGuard · Code Dependency Risk Analyzer")
st.markdown(
    "Static analysis + Graph Neural Network predicting structurally risky code components."
)

# Top KPIs
if predictions is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Nodes", len(predictions))
    c2.metric("Predicted Risky", int(predictions["predicted_risky"].sum()))
    c3.metric(
        "Risk Rate",
        f"{predictions['predicted_risky'].mean()*100:.1f}%",
    )
    top_risk = predictions.sort_values("risk_score", ascending=False).iloc[0]
    c4.metric("Top Risk Score", f"{top_risk['risk_score']:.3f}", delta=top_risk.get("name", ""))
else:
    st.info("No outputs found. Enter a repo path and click **Run Full Pipeline**.")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Dependency Graph", "⚠️ Risk Table", "📊 Model Metrics", "🔬 Node Inspector"]
)

# ---------------------------------------------------------------------------
# Tab 1: Graph visualization
# ---------------------------------------------------------------------------

with tab1:
    st.subheader("Dependency Graph")
    if edges_df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Edges", len(edges_df))
        if predictions is not None:
            col2.metric("Nodes", len(predictions))

        edge_counts = edges_df["relationship_type"].value_counts() if "relationship_type" in edges_df.columns else {}
        if not edge_counts.empty:
            col3.metric("Edge Types", len(edge_counts))
            st.bar_chart(edge_counts)

        st.markdown("**Interactive graph** (red = high risk, node size ∝ risk score)")
        html = _build_pyvis_html(edges_df, predictions)
        st.components.v1.html(html, height=620, scrolling=False)
    else:
        st.info("No graph data found. Run the pipeline first.")

# ---------------------------------------------------------------------------
# Tab 2: Risk table
# ---------------------------------------------------------------------------

with tab2:
    st.subheader("Node Risk Predictions")
    if predictions is not None:
        # Filters
        col1, col2 = st.columns(2)
        etype_filter = col1.multiselect(
            "Filter by type",
            options=sorted(predictions["entity_type"].unique().tolist()),
            default=[],
        )
        risky_only = col2.checkbox("Show risky only", value=False)

        df = predictions.copy()
        if etype_filter:
            df = df[df["entity_type"].isin(etype_filter)]
        if risky_only:
            df = df[df["predicted_risky"] == 1]

        df = df.sort_values("risk_score", ascending=False)

        # Colour-code risk score
        def _highlight(val: float) -> str:
            if val >= 0.75:
                return "background-color: #8B0000; color: white"
            if val >= 0.5:
                return "background-color: #c0392b; color: white"
            if val >= 0.25:
                return "background-color: #e67e22; color: white"
            return ""

        display_cols = [c for c in ["name", "entity_type", "file_path", "risk_score", "predicted_risky"] if c in df.columns]
        st.dataframe(
            df[display_cols].style.applymap(_highlight, subset=["risk_score"]),
            use_container_width=True,
            height=450,
        )
        st.download_button(
            "⬇️ Download predictions CSV",
            data=df.to_csv(index=False),
            file_name="graphguard_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("No predictions found.")

# ---------------------------------------------------------------------------
# Tab 3: Model metrics
# ---------------------------------------------------------------------------

with tab3:
    st.subheader("GNN vs Baseline Performance")
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.set_index("model_name")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="#2ecc71"), use_container_width=True)

        # Bar charts per metric
        plot_cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"] if c in metrics_df.columns]
        if plot_cols:
            st.bar_chart(metrics_df[plot_cols])

        st.markdown("""
**Interpreting the metrics**
- **PR-AUC** is the primary signal when classes are imbalanced (more "safe" nodes than "risky").
- **ROC-AUC** measures ranking quality across all thresholds.
- A GNN that outperforms the random forest on PR-AUC shows that **neighborhood embeddings capture risk signals** beyond local node features.
        """)
    else:
        st.info("No metrics found. Run the training pipeline first.")

# ---------------------------------------------------------------------------
# Tab 4: Node inspector
# ---------------------------------------------------------------------------

with tab4:
    st.subheader("Node Inspector")
    if predictions is not None and features_df is not None:
        node_names = predictions.sort_values("risk_score", ascending=False)["name"].tolist()
        selected_name = st.selectbox("Select a node to inspect:", node_names)

        row = predictions[predictions["name"] == selected_name].iloc[0]
        nid = row["node_id"]

        col1, col2 = st.columns(2)
        col1.metric("Risk Score", f"{row['risk_score']:.4f}")
        col2.metric("Entity Type", row.get("entity_type", ""))
        st.metric("File", row.get("file_path", ""))

        st.markdown("**Why might this node be risky?**")

        if nid in features_df.index:
            feat = features_df.loc[nid]
            explanations = []
            if float(feat.get("betweenness", 0)) > 0.05:
                explanations.append("🔴 **High betweenness centrality**: many shortest paths pass through this node — it's a structural bottleneck.")
            if float(feat.get("fan_in", 0)) > 3:
                explanations.append(f"🔴 **High fan-in** ({int(feat.get('fan_in', 0))}): many modules depend on this node — changes here cascade widely.")
            if float(feat.get("complexity", 1)) > 5:
                explanations.append(f"🔴 **High complexity** ({int(feat.get('complexity', 1))}): estimated cyclomatic complexity suggests many branches.")
            if not int(feat.get("has_docstring", 0)):
                explanations.append("🟡 **No docstring**: undocumented code is harder to maintain correctly.")
            if float(feat.get("lines_of_code", 0)) > 50:
                explanations.append(f"🟡 **Large function/file** ({int(feat.get('lines_of_code', 0))} lines): larger units tend to accumulate more responsibilities.")

            if explanations:
                for exp in explanations:
                    st.markdown(f"- {exp}")
            else:
                st.markdown("- No strong individual risk signals. The GNN may have detected a risk pattern in the neighborhood structure.")

            # Show raw features
            with st.expander("Raw feature values"):
                numeric_cols = [c for c in features_df.columns if features_df[c].dtype in ["float64", "int64", "float32", "int32"]]
                if numeric_cols:
                    st.dataframe(feat[numeric_cols].to_frame(name="value").T)
    else:
        st.info("Run the pipeline first to enable node inspection.")
