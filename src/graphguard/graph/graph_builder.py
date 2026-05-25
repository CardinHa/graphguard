"""
Builds a NetworkX directed graph from a ParseResult.

Graph semantics
---------------
Nodes  : code entities (file, function, class, module)
Edges  : directed dependency relationships (imports, calls, inherits, contains)

Each node carries its ParsedEntity metadata as attributes.
Each edge carries its relationship_type as an attribute.

The graph can be exported to:
  - GraphML  (.graphml)
  - JSON     (.json)
  - Edge-list CSV
  - Node-feature CSV (populated by FeatureExtractor)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd

from graphguard.parser.python_parser import ParseResult, ParsedEntity
from graphguard.utils.logging import get_logger

logger = get_logger(__name__)


class GraphBuilder:
    """
    Converts a ParseResult into a NetworkX DiGraph.

    The directed graph models software as a *dependency matrix*:
      A[i,j] = 1  iff  node i depends on (or relates to) node j.
    PageRank, betweenness, and reachability are all computed on this
    adjacency structure, connecting graph theory to linear algebra.
    """

    def build(self, parse_result: ParseResult) -> nx.DiGraph:
        """Build and return the dependency graph."""
        G = nx.DiGraph()

        # Add nodes
        for entity in parse_result.entities:
            G.add_node(entity.node_id, **self._node_attrs(entity))

        # Add edges — deduplicate multi-edges by keeping the first occurrence
        seen: set[tuple[str, str, str]] = set()
        for rel in parse_result.relationships:
            key = (rel.source_id, rel.target_id, rel.relationship_type)
            if key in seen:
                continue
            seen.add(key)

            # Auto-create stub nodes for targets not found in the parse
            # (e.g. external modules like numpy, os)
            if rel.source_id not in G:
                G.add_node(rel.source_id, entity_type="unknown", name=rel.source_id)
            if rel.target_id not in G:
                top_name = rel.target_id.replace("module::", "")
                G.add_node(
                    rel.target_id,
                    entity_type="module",
                    name=top_name,
                    file_path="",
                    line_number=0,
                )

            G.add_edge(
                rel.source_id,
                rel.target_id,
                relationship_type=rel.relationship_type,
                file_path=rel.file_path,
                line_number=rel.line_number,
            )

        logger.info(
            f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
        )
        return G

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def save_graphml(self, G: nx.DiGraph, path: Path) -> None:
        """Save graph as GraphML (lossless, all attributes preserved)."""
        # GraphML requires string attribute values
        G_str = nx.DiGraph()
        for n, attrs in G.nodes(data=True):
            G_str.add_node(n, **{k: str(v) for k, v in attrs.items()})
        for u, v, attrs in G.edges(data=True):
            G_str.add_edge(u, v, **{k: str(v2) for k, v2 in attrs.items()})
        nx.write_graphml(G_str, str(path))
        logger.info(f"GraphML saved -> {path}")

    def save_json(self, G: nx.DiGraph, path: Path) -> None:
        """Save graph as node-link JSON."""
        data = nx.node_link_data(G)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info(f"JSON saved -> {path}")

    def save_edge_csv(self, G: nx.DiGraph, path: Path) -> None:
        """Save edge list as CSV with columns: source, target, relationship_type."""
        rows = [
            {
                "source": u,
                "target": v,
                "relationship_type": data.get("relationship_type", ""),
            }
            for u, v, data in G.edges(data=True)
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info(f"Edge CSV saved -> {path}")

    def save_node_csv(self, G: nx.DiGraph, path: Path) -> None:
        """Save node attributes as CSV."""
        rows = [{"node_id": n, **attrs} for n, attrs in G.nodes(data=True)]
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info(f"Node CSV saved -> {path}")

    def save_all(self, G: nx.DiGraph, output_dir: Path) -> None:
        """Save graph in all supported formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.save_graphml(G, output_dir / "graph.graphml")
        self.save_json(G, output_dir / "graph.json")
        self.save_edge_csv(G, output_dir / "edges.csv")
        self.save_node_csv(G, output_dir / "nodes.csv")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _node_attrs(entity: ParsedEntity) -> dict:
        return {
            "name": entity.name,
            "entity_type": entity.entity_type,
            "file_path": entity.file_path,
            "line_number": entity.line_number,
            "lines_of_code": entity.lines_of_code,
            "num_params": entity.num_params,
            "has_docstring": int(entity.has_docstring),
            "complexity": entity.complexity,
        }


def print_graph_summary(G: nx.DiGraph) -> None:
    """Print a human-readable graph summary to the console."""
    from graphguard.utils.logging import console
    from rich.table import Table

    counts: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        t = attrs.get("entity_type", "unknown")
        counts[t] = counts.get(t, 0) + 1

    edge_counts: dict[str, int] = {}
    for _, _, attrs in G.edges(data=True):
        rt = attrs.get("relationship_type", "unknown")
        edge_counts[rt] = edge_counts.get(rt, 0) + 1

    table = Table(title="Dependency Graph Summary", style="cyan")
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right")

    table.add_row("Total nodes", str(G.number_of_nodes()))
    table.add_row("Total edges", str(G.number_of_edges()))
    table.add_row("", "")
    for t, c in sorted(counts.items()):
        table.add_row(f"  {t} nodes", str(c))
    table.add_row("", "")
    for t, c in sorted(edge_counts.items()):
        table.add_row(f"  {t} edges", str(c))

    # Connectivity
    weakly = nx.number_weakly_connected_components(G)
    table.add_row("Weakly connected components", str(weakly))

    console.print(table)
