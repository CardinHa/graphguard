# GraphGuard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-quality static analysis tool that converts Python codebases into dependency graphs and trains a GNN to predict structurally risky code components.

**Architecture:** Python AST parser → NetworkX DiGraph → node feature matrix → PyTorch Geometric Data → GraphSAGE node classifier. Baselines via sklearn on same features. Outputs served via Typer CLI, FastAPI, and Streamlit dashboard.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Geometric, NetworkX, pandas, scikit-learn, Typer, FastAPI, Streamlit, pyvis, GitPython, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/graphguard/utils/config.py` | Dataclass config, paths, hyperparams |
| `src/graphguard/utils/logging.py` | Rich-based console logging |
| `src/graphguard/parser/python_parser.py` | AST walker → ParsedEntity + ParsedRelationship |
| `src/graphguard/graph/graph_builder.py` | ParseResult → NetworkX DiGraph with metadata |
| `src/graphguard/graph/features.py` | DiGraph → per-node feature DataFrame |
| `src/graphguard/data/git_mining.py` | Git log mining → file risk labels |
| `src/graphguard/data/dataset.py` | DataFrame + graph → PyG Data object |
| `src/graphguard/models/gnn.py` | GraphSAGE + GCN PyTorch models |
| `src/graphguard/models/baselines.py` | LogReg + RandomForest wrappers |
| `src/graphguard/models/train.py` | Full training pipeline, saves weights+metrics |
| `src/graphguard/models/evaluate.py` | Metrics computation and comparison |
| `src/graphguard/cli.py` | Typer CLI: analyze/train/report/dashboard/api |
| `src/graphguard/api/main.py` | FastAPI with /health /analyze /graph /predictions /metrics |
| `src/graphguard/dashboard/app.py` | Streamlit dashboard with pyvis graph viz |
| `tests/test_parser.py` | Parser behavior tests |
| `tests/test_graph_builder.py` | Graph construction tests |
| `tests/test_features.py` | Feature extraction tests |
| `examples/sample_project/*.py` | 5-file demo codebase with intentional complexity |

---

## Task 1: Project scaffold

- [ ] Create pyproject.toml, requirements.txt, .gitignore, Dockerfile, outputs/.gitkeep, all __init__.py stubs

## Task 2: Utils

- [ ] Write config.py (dataclass-based config)
- [ ] Write logging.py (rich console logger)

## Task 3: Parser

- [ ] Write python_parser.py (AST walker, ParsedEntity, ParsedRelationship, ParseResult)
- [ ] Write test_parser.py with tests for entity extraction and relationship detection

## Task 4: Graph

- [ ] Write graph_builder.py (NetworkX DiGraph from ParseResult, save to GraphML/JSON/CSV)
- [ ] Write features.py (in/out degree, PageRank, betweenness, closeness, clustering, fan-in/out)
- [ ] Write test_graph_builder.py and test_features.py

## Task 5: Data pipeline

- [ ] Write git_mining.py (GitPython log mining, bug keyword detection, file-level labels)
- [ ] Write dataset.py (features + labels → PyG Data, train/val/test masks)

## Task 6: Models

- [ ] Write gnn.py (GraphSAGE + optional GCN, node binary classification)
- [ ] Write baselines.py (LogReg + RF wrappers, same interface)
- [ ] Write evaluate.py (precision, recall, F1, ROC-AUC, PR-AUC, comparison table)
- [ ] Write train.py (full pipeline: load → features → labels → train GNN → train baselines → save)

## Task 7: CLI

- [ ] Write cli.py (Typer: analyze, train, report, dashboard, api commands)

## Task 8: API + Dashboard

- [ ] Write api/main.py (FastAPI endpoints)
- [ ] Write dashboard/app.py (Streamlit with risk table and graph viz)

## Task 9: Sample project + README

- [ ] Write examples/sample_project/ (5 files with realistic dependency graph)
- [ ] Write README.md (full production README with all required sections)
