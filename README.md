# GraphGuard: GNN-Based Code Dependency Risk Analyzer

> **Represent any Python codebase as a directed dependency graph, then train a Graph Neural Network to predict which files, functions, and classes are structurally risky or bug-prone — using only the shape of the code.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3+-orange.svg)](https://pyg.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-79%20passing-brightgreen.svg)](tests/)
[![CI](https://github.com/CardinHa/graphguard/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

---

## Why This Project Matters

Software bugs are not randomly distributed. Empirical studies (Nagappan & Ball 2005, Zimmermann et al. 2007) show that files with high coupling, high fan-in, and structural centrality in a dependency graph are disproportionately bug-prone. Most static analysis tools catch *what* the code says. GraphGuard reasons about *how code relates to other code*.

This connects directly to a class of problems Meta solves at scale:
- **Social graphs** — ranking entities by structural importance (PageRank on the social graph)
- **Code intelligence** — understanding code structure to guide refactoring and testing priorities
- **Recommendation systems** — predicting future risk from graph topology, much like predicting link formation
- **Infrastructure reliability** — identifying bottleneck services in dependency graphs before they cause outages

---

## Technical Architecture

```
Python Repo
    │
    ▼
┌─────────────────────────────────┐
│  PythonParser (ast module)      │  → ParsedEntity[], ParsedRelationship[]
│  Files, functions, classes,     │
│  imports, calls, inheritance    │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  GraphBuilder (NetworkX)        │  → nx.DiGraph
│  Directed dependency graph      │
│  Nodes: entities, Edges: rels   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  FeatureExtractor               │  → DataFrame [N × F]
│  PageRank, betweenness,         │
│  fan-in, complexity, LOC, etc.  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Labeling                       │  → y ∈ {0,1}^N
│  Mode A: git commit mining      │
│  Mode B: synthetic heuristic    │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  CodeGraphDataset (PyG)         │  → Data(x, edge_index, y, masks)
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  GraphSAGE / GCN                │  → risk logits ∈ R^N
│  Node-level binary classifier   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Baseline: LogReg + RandomForest│  → metrics comparison
└─────────────────────────────────┘
    │
    ▼
 CLI · FastAPI · Streamlit Dashboard
```

---

## How the Parser Works

GraphGuard uses Python's built-in `ast` module to walk the syntax tree of every `.py` file in a repository. For each file it records:

| Entity | What is extracted |
|--------|------------------|
| **file** | path, total lines |
| **function** | name, line number, LOC, parameter count, docstring presence, cyclomatic complexity |
| **class** | name, line number, LOC, docstring presence |
| **module** | top-level name of every import |

**Cyclomatic complexity** is estimated as 1 + the number of branching nodes (`if`, `for`, `while`, `try`, `with`, `assert`, boolean operators). This approximates McCabe complexity without full control-flow analysis.

**Call resolution** is best-effort: when `foo()` is called inside a function, the parser checks if `foo` is defined in the same file and creates a `calls` edge if so.

---

## How the Dependency Graph Is Built

The parser output is converted into a **directed graph** G = (V, E):

- **Nodes** V = {files, functions, classes, external modules}
- **Edges** E carry one of four `relationship_type` values:

| Edge type | Meaning |
|-----------|---------|
| `contains` | file → function/class (structural containment) |
| `imports`  | file → module (import dependency) |
| `calls`    | function → function (call-graph edge) |
| `inherits` | class → base class (inheritance hierarchy) |

This is exactly the adjacency matrix representation used in graph theory:

```
A[i, j] = 1  iff  entity i has a directed edge to entity j
```

The graph is saved as GraphML, JSON, and edge-list CSV for downstream analysis.

---

## How Node Features Are Created

For each node we compute a **feature vector** in R^F (F ≈ 20) combining:

### Graph-theoretic features (connected to discrete math)

| Feature | Graph theory meaning | Why it predicts risk |
|---------|---------------------|---------------------|
| **in-degree** | Column sum of A | Many dependents → high blast radius |
| **out-degree** | Row sum of A | Many dependencies → high coupling |
| **PageRank** | Dominant eigenvector of D⁻¹A^T (power iteration) | Structurally influential nodes |
| **Betweenness centrality** | Fraction of shortest paths through v | Bottleneck — changes here propagate widely |
| **Closeness centrality** | Inverse avg shortest-path length | How quickly changes propagate |
| **Fan-in / fan-out** | Directed degree filtered by edge type | Coupling metric |
| **Clustering coefficient** | Local triangle density | Measures tight grouping |

> **Linear algebra connection**: PageRank solves the eigenvector equation r = αA^T D⁻¹ r + (1-α)e/N via power iteration — it is matrix–vector multiplication repeated until convergence, exactly the same operation as one GNN layer.

### Code-structure features

| Feature | Source |
|---------|--------|
| Lines of code | AST end_lineno - lineno |
| Parameter count | len(func.args) |
| Docstring present | isinstance(body[0], ast.Expr) |
| Cyclomatic complexity | Branch-node count |
| Entity type (one-hot) | file/function/class/module |

---

## How the GNN Works

### Message passing = matrix multiplication

A single GraphSAGE layer computes:

```
h_v^(k) = σ( W · CONCAT( h_v^(k-1),  MEAN_{u ∈ N(v)} h_u^(k-1) ) )
```

In matrix form this is:

```
H^(k) = σ( [I | D⁻¹A] H^(k-1) W^T )
```

where D⁻¹A is the row-normalised adjacency matrix. Each layer performs one step of **neighborhood aggregation** — essentially a graph convolution. A 2-layer GraphSAGE aggregates information from 2-hop neighborhoods.

### Architecture

```
Input  x ∈ R^{N×F}
  │
SAGEConv(F → 64) + BatchNorm + ReLU + Dropout
  │
SAGEConv(64 → 64) + BatchNorm + ReLU + Dropout
  │
Linear(64 → 1)
  │
Sigmoid → risk probability per node
```

### Class imbalance

Most code nodes are "safe." We weight the positive class inversely to its frequency using PyTorch's `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`.

### Why GNNs could outperform baselines here — and what we actually measured

The theoretical case: a random forest on node features knows that *this function* is complex and central. A GNN also knows that *this function's callers* are complex and central. Structural risk propagates through neighborhoods — GNNs can capture this in principle; tabular models cannot.

In practice, on the `psf/requests` benchmark below (re-run under the fixed, leakage-free
file-grouped split), this advantage did not show up: GraphSAGE's mean ROC-AUC across 3
seeds was below chance and below both tabular baselines, with RandomForest the strongest
performer. See "Model Results" for the honest numbers and the caveats around repo size and
label skew that likely contributed.

---

## Labeling Strategy

### Mode A: Git history (`--label-mode git`)

```python
# A commit is a "bug-fix" commit if its message matches:
r"\b(fix|bug|issue|error|crash|regression|patch|hotfix)\b"

# Files touched in bug-fix commits are labeled 1 (risky)
```

This uses the intuition from defect prediction literature: files with frequent bug fixes have embedded fragility that static structure tends to reflect.

### Mode B: Synthetic heuristic (`--label-mode synthetic`)

```python
risk_score = 0.4 * norm(fan_in) + 0.3 * norm(betweenness) + 0.3 * norm(complexity)
# Nodes in the top 30% by risk_score → label 1
```

> ⚠️ **Synthetic labels are a demo tool only.** They are not real bug predictions. They are generated to make the pipeline runnable on any repository, including ones with no git history. All results using synthetic labels should be interpreted as "the model learned to reproduce a structural heuristic," not "the model found real bugs."

---

## Installation

Core install (`pip install -e .`) only pulls in networkx/pandas/numpy/
scikit-learn/typer/rich/pydantic/scipy — enough to run `graphguard analyze`
(parse a repo, build the dependency graph, extract features). Everything
else is an optional extra, since not every use of GraphGuard needs a GNN
runtime, a REST server, a dashboard, or git history mining:

| Extra    | Adds                        | Needed for |
|----------|------------------------------|------------|
| `gnn`    | torch, torch-geometric       | `graphguard train`, `graphguard explain`, POST /analyze, POST /explain |
| `serve`  | fastapi, uvicorn             | `graphguard api` |
| `dash`   | streamlit, plotly, pyvis     | `graphguard dashboard` |
| `git`    | gitpython                    | `--label-mode git` |

Commands that need a missing extra print a `pip install graphguard[...]`
hint instead of a raw import traceback.

### Option 1: pip (development install)

```bash
git clone https://github.com/CardinHa/graphguard
cd graphguard

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Core only (graphguard analyze):
pip install -e .

# Everything (dev convenience — matches requirements.txt):
pip install -r requirements.txt
pip install -e ".[dev,gnn,serve,dash,git]"

# Or pick only the extras you need, e.g. training + the API:
pip install -e ".[gnn,serve]"
```

### Option 2: Docker

```bash
docker build -t graphguard .
docker run --rm graphguard --help
```

The Docker image installs every extra (see `requirements.txt` in the
image), so all commands work out of the box inside the container.

### PyTorch Geometric note

PyG sometimes requires matching PyTorch + CUDA versions. If the `pip install` above fails for PyG, follow the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to get the correct wheel URL for your system.

---

## Usage

### Analyze a repository (parse + build graph + extract features)

```bash
graphguard analyze examples/sample_project
# or:
python -m graphguard.cli analyze examples/sample_project
```

### Full training pipeline

```bash
# Synthetic labels (works on any repo with no git history)
graphguard train examples/sample_project --label-mode synthetic

# Git history labels (requires a git repository)
graphguard train /path/to/your/repo --label-mode git

# Custom model settings
graphguard train examples/sample_project \
  --model sage \
  --epochs 300 \
  --hidden-dim 128 \
  --layers 3
```

### Generate a risk report

```bash
graphguard report examples/sample_project --top-n 20
```

### Explain a specific node (GNNExplainer)

```bash
# By function name
graphguard explain examples/requests_live resolve_proxies

# By substring of node_id
graphguard explain examples/requests_live utils.py --top-k 8

# Save explanation to outputs/explanations.json
graphguard explain examples/requests_live resolve_proxies --save
```

Example output:

```
Node: func::src/requests/utils.py::resolve_proxies
Risk Score: 0.9964

Top Contributing Features
┌───────────────┬────────────────────┐
│ Feature       │ Attribution Weight │
├───────────────┼────────────────────┤
│ complexity    │             0.7773 │
│ clustering    │             0.6565 │
│ fan_out       │             0.6138 │
│ type_function │             0.2887 │
│ has_docstring │             0.2392 │
└───────────────┴────────────────────┘

Most Influential Neighbors
┌───────────────────────┬──────────────────┐
│ Neighbor              │ Edge Attribution │
├───────────────────────┼──────────────────┤
│ utils.py              │           0.8788 │
│ get_environ_proxies   │           0.0000 │
│ should_bypass_proxies │           0.0000 │
└───────────────────────┴──────────────────┘
```

GNNExplainer optimises a soft mask over input features and adjacent edges to
maximise mutual information with the model's prediction. Higher attribution
weight = that feature or neighbor contributed more to the risk score.

### Launch the dashboard

```bash
graphguard dashboard examples/sample_project
# Opens at http://localhost:8501
```

### Launch the API server

```bash
graphguard api
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Module-level invocation (no install needed)

```bash
python -m graphguard.cli analyze examples/sample_project
python -m graphguard.cli train examples/sample_project
python -m graphguard.cli report examples/sample_project
```

---

## API Reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe |
| POST | `/analyze` | Run full pipeline on a repo path |
| GET | `/graph` | Graph stats + sample nodes/edges |
| GET | `/predictions?top_n=50` | Top-N risky nodes sorted by score |
| GET | `/metrics` | GNN vs baseline comparison |
| POST | `/explain` | GNNExplainer attribution for a single node |

**Example:**

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "0.1.0"}

curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "examples/sample_project", "label_mode": "synthetic"}'

curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "examples/sample_project", "node": "buggy_function", "top_k": 5}'
# {"node_id": "func::...", "risk_score": 0.97, "feature_importance": [...], "influential_neighbors": [...]}
```

---

## Dashboard

```bash
graphguard dashboard
```

The Streamlit dashboard provides:
- **Graph tab**: Interactive pyvis visualization (red nodes = high risk)
- **Risk table**: Sortable predictions with filtering by entity type
- **Metrics tab**: GNN vs baseline comparison with highlighted best scores
- **Node inspector**: Select any node to see why it was flagged (betweenness, fan-in, complexity explanations)

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=graphguard --cov-report=term-missing

# Single test file
pytest tests/test_parser.py -v
```

---

## Model Results

### Real-world validation: `psf/requests` (`--label-mode git`)

> **Previously reported numbers on this benchmark were inflated by a split-leakage bug**
> (`CodeGraphDataset._split_masks` assigned nodes to train/val/test independently at
> random, so same-file siblings sharing an identical git-mined label could land on both
> sides of the split — most visibly inflating precision to 1.0000 and ROC-AUC to 0.9894).
> That bug was fixed in PR #1 (the split now groups every node by its containing file, so
> a file lands entirely in one of train/val/test). The table below is the **re-run under
> the fixed, file-grouped split** — this is the honest number, not the historical one.

**Setup.** Cloned [`psf/requests`](https://github.com/psf/requests) at commit
[`4c800e9`](https://github.com/psf/requests/commit/4c800e9aea2059660b8306b0fc8f9e9a4232cb3e)
(`main`, 2026-07-06). Ran `graphguard train <repo> --label-mode git --seed {42,43,44}` —
the repo's own CLI, default hyperparameters (SAGE, 200 epochs / patience 20, hidden_dim 64,
2 layers, dropout 0.3). `--seed` is a new CLI flag added for this benchmark rerun (see the
separate reproducibility-fix commit); it seeds the file-grouped split shuffle, the
LogReg/RandomForest `random_state`, and torch/numpy global RNG before GNN init/training,
so each of the 3 runs below is an independent draw, not 3x the same split.

**Deviation from the old methodology note:** the README previously described the label
miner as using "500 commits of git history." The current `GitMiner.mine_bug_fix_labels`
has no commit-window parameter — it always scans full history via `repo.iter_commits()`.
There is no code path to reproduce a "500 commit" limit, so full history was mined as-is
(6,482 commits scanned this run, 149 files flagged via bug-fix keyword matching — up from
the previously reported 124, since more history means more files eventually touched by a
fix commit). One consequence worth flagging: this makes the label highly prevalent —
661 of 757 scorable nodes (87.3%) are labeled risky — because on a mature, actively
maintained repo, nearly every live file eventually gets touched by *some* fix commit over
its full lifetime. That skew, combined with `requests` having only 37 Python files (so the
file-grouped splitter has few "groups" to allocate), is the main driver of the run-to-run
variance below.

**Graph:** 822 nodes · 1,297 edges · 757 scorable (files/functions/classes)

| Model | Seed | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Test size |
|-------|------|----------|-----------|--------|----|---------|--------|-----------|
| GraphSAGE | 42 | 0.7632 | 0.7632 | 1.0000 | 0.8657 | 0.1111 | 0.6174 | 38 |
| GraphSAGE | 43 | 0.2162 | 1.0000 | 0.0440 | 0.0842 | 0.5154 | 0.8583 | 111 |
| GraphSAGE | 44 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000† | 0.0000† | 37 |
| **GraphSAGE (mean)** | — | **0.3265** | **0.5877** | **0.3480** | **0.3166** | **0.2088** | **0.4919** | — |
| LogisticRegression | 42 | 0.2632 | 1.0000 | 0.0345 | 0.0667 | 0.8123 | 0.9365 | 38 |
| LogisticRegression | 43 | 0.3694 | 0.7059 | 0.3956 | 0.5070 | 0.4423 | 0.8327 | 111 |
| LogisticRegression | 44 | 0.5676 | 1.0000 | 0.5676 | 0.7241 | 0.0000† | 0.0000† | 37 |
| **LogisticRegression (mean)** | — | **0.4001** | **0.9020** | **0.3326** | **0.4326** | **0.4182** | **0.5897** | — |
| RandomForest | 42 | 0.7632 | 0.7778 | 0.9655 | 0.8615 | 0.8621 | 0.9559 | 38 |
| RandomForest | 43 | 0.4144 | 0.7167 | 0.4725 | 0.5695 | 0.3679 | 0.8100 | 111 |
| RandomForest | 44 | 0.6757 | 1.0000 | 0.6757 | 0.8065 | 0.0000† | 0.0000† | 37 |
| **RandomForest (mean)** | — | **0.6178** | **0.8315** | **0.7046** | **0.7458** | **0.4100** | **0.5886** | — |

† Seed 44's test split (37 nodes) happened to contain only the "risky" class (single-class
test set), so ROC-AUC/PR-AUC are undefined and reported as 0.0 by `evaluate.py`'s
convention rather than reflecting genuinely poor ranking. This is a real artifact of the
file-grouped split on a 37-file repo, not a scoring bug — and it's exactly the "small test
splits make point metrics noisy" failure mode already called out in Limitations below.

**Honest finding: GraphSAGE does not beat the tabular baselines here.** Averaged over 3
seeds, GraphSAGE's ROC-AUC (0.2088) is well below chance (0.5) and below both
LogisticRegression (0.4182) and RandomForest (0.4100) — RandomForest is the strongest and
most stable of the three across seeds. This directly contradicts the "GNNs outperform
baselines here" narrative in the pre-fix table, and we are not spinning it: under a
leakage-free, file-grouped split, on this repo, with this label source, structural
message-passing did not add measurable value over handcrafted per-node features. Plausible
contributing factors (not excuses): a 37-file repo gives the splitter very few groups to
work with, producing high-variance, occasionally single-class test sets; the label is
90%-skewed positive after mining full git history, which starves the GNN's minority class
in an already-small graph; and no hyperparameter search was done for the GNN specifically
(baselines and GNN both used their out-of-the-box defaults). Whether a larger repo, a
class-balanced label source, or GNN-specific tuning would change this conclusion is open
follow-up work — we deliberately did not tune until the GNN won.

---

### Synthetic-label benchmark: `examples/sample_project`

For repos without git history, the pipeline falls back to a heuristic label. These numbers
are included for reproducibility but should not be read as real risk predictions.

**Graph:** 75 nodes · 57 scorable · synthetic labels (top-30% by fan-in + betweenness + complexity)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| **GraphSAGE** | 0.70 | 0.63 | 1.00 | 0.77 | **0.92** | **0.94** |
| LogisticRegression | 0.70 | 0.67 | 0.80 | 0.73 | 0.76 | 0.84 |
| RandomForest | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**RF = 1.00 here is target leakage.** The synthetic label is computed *from*
`complexity`, `fan_in`, and `betweenness` — the exact features RF splits on. The
`feature_importances.csv` output confirms it: `complexity` alone accounts for 49% of
importance. This is not a win; it is a sanity check that the pipeline ran correctly.
The `--label-mode git` table above is the meaningful benchmark.

---

## Project Structure

```
graphguard/
├── src/graphguard/
│   ├── parser/          # AST-based Python code parser
│   ├── graph/           # NetworkX graph builder + feature extractor
│   ├── data/            # PyG dataset builder + git mining
│   ├── models/          # GNN, baselines, training loop, evaluation
│   ├── api/             # FastAPI REST endpoints
│   ├── dashboard/       # Streamlit interactive dashboard
│   ├── utils/           # Config, logging
│   └── cli.py           # Typer CLI (analyze, train, report, dashboard, api)
├── tests/               # pytest test suite
├── examples/
│   └── sample_project/  # Demo codebase with intentional complexity
├── outputs/             # Generated graphs, features, predictions, metrics
├── Dockerfile
└── pyproject.toml
```

---

## Limitations

1. **Call resolution is best-effort.** GraphGuard resolves cross-file calls through import tracking (absolute and relative imports), adding hundreds of real inter-module edges. Dynamic dispatch, star imports, and calls through variables remain unresolvable without running the code.
2. **Synthetic labels reproduce the heuristic, not real bugs.** Because the synthetic label is derived from node features, tabular baselines can over-fit it (see Model Results). To get meaningful predictions, run on a repository with ≥ 6 months of git history and >10 bug-fix commits via `--label-mode git`.
3. **Small graphs struggle.** GNNs need sufficient graph structure to learn meaningful embeddings. Repos with fewer than ~50 nodes may not show a benefit over baselines, and small test splits make point metrics noisy.
4. **Python only.** Tree-sitter could extend this to other languages; the architecture is language-agnostic.

### Design note: external modules are kept but never scored

Imported stdlib/third-party modules (`os`, `numpy`, `enum`, ...) stay in the graph because they carry real structural signal — a file's fan-out and a function's coupling depend on them. But they are **excluded from labeling, training, evaluation, and the prediction report**, because you cannot act on a bug in the standard library. Only files, functions, and classes you own are scored. This keeps the risk report focused on code a developer can actually fix.

---

## Future Improvements

- [ ] Tree-sitter parser for multi-language support (JavaScript, Java, C++)
- [ ] Inter-file call resolution using import tracking + name binding
- [ ] Temporal graph features (how the graph changes over time)
- [ ] Heterogeneous GNN (different convolutions for different edge types)
- [ ] Integration with GitHub Actions for CI-time risk scoring
- [ ] Graph-level community detection to identify risky modules
- [ ] Attention-weighted GNN (GAT) for learned edge importance

---

## Connection to Meta-Scale Systems

| GraphGuard concept | Meta equivalent |
|-------------------|----------------|
| Dependency graph PageRank | Social graph influence ranking |
| GNN message passing | Graph-level recommendation propagation |
| Node risk classification | Content/account quality scoring |
| Fan-in as structural centrality | Social graph degree in network analysis |
| Bug-fix commit labeling | User engagement signal mining |
| Graph serialization → PyG Data | Large-scale graph dataset pipelines |
| FastAPI + async prediction | Production ML serving infrastructure |

The same mathematical primitives — adjacency matrices, graph traversal, eigenvector centrality, message passing — underlie Meta's social graph, News Feed ranking, and Integrity systems.

---

## Resume Bullets

- **Built a Python static-analysis engine** that converts codebases into directed dependency graphs of files, functions, classes, imports, and call relationships, enabling graph-based reasoning over software architecture using NetworkX and Python's AST module.

- **Trained a GraphSAGE Graph Neural Network** with PyTorch Geometric to predict structurally risky code components using node centrality, fan-in/fan-out, dependency depth, cyclomatic complexity, and learned multi-hop neighborhood embeddings.

- **Implemented and benchmarked ML models** — GraphSAGE, GCN, Logistic Regression, and Random Forest — evaluating performance on precision, recall, F1, ROC-AUC, and PR-AUC under a leakage-free, file-grouped train/test split; found that on the `psf/requests` real-world benchmark, relational message passing did **not** outperform handcrafted tabular features, and documented that honest (rather than favorable) result.

- **Added GNNExplainer interpretability** to surface which input features and graph neighbors drive each risk prediction, converting raw scores into actionable developer explanations via per-node mask optimisation.

- **Developed a full production engineering workflow** including Typer CLI tooling, FastAPI REST endpoints, interactive Streamlit dashboard with pyvis graph visualization, automated pytest test suite, and Docker deployment — end-to-end from raw source code to actionable risk predictions.

---

## Interview Talking Points

### How does a codebase become a graph?

"I walk the Python AST of every file with Python's `ast` module and extract entities — files, functions, classes — as nodes, and relationships — imports, calls, inheritance, containment — as directed edges. This gives me a graph where `A[i,j] = 1` means entity i depends on j. It's the same adjacency matrix representation you use for any graph algorithm."

### Why are GNNs the right tool here?

"A random forest on node features knows that *this function* is complex. A GNN also knows that *this function's callers* are complex. Structural risk propagates through dependency chains — if a highly central module has a bug, everything that imports it breaks. GNNs capture this propagation through their neighborhood aggregation; tabular models cannot."

### How does message passing work?

"Each GNN layer computes `H^{(k)} = σ(D⁻¹A · H^{(k-1)} · W)` — it's a matrix multiplication on the normalized adjacency, essentially diffusing features from neighbors. After k layers, each node's embedding encodes information from its k-hop neighborhood. This is the same operation as a graph diffusion or a PageRank iteration, just learned end-to-end."

### How does this connect to graph centrality and linear algebra?

"PageRank is the dominant eigenvector of D⁻¹A^T, computed by power iteration: multiply by the matrix, normalize, repeat. Betweenness centrality counts how many shortest paths pass through a node using BFS from every source. Both are computable from the adjacency matrix alone — the same structure the GNN operates on."

### Why do central nodes tend to be riskier?

"High betweenness means many dependency paths run through a node — a bug there breaks more things. High fan-in means many modules depend on it — a change there forces changes everywhere. These are exactly the structural fragility signals that correlate with bug-fix commit frequency in the literature."

### How could this scale to large repositories?

"For repositories with millions of nodes (think: Meta's monorepo), I'd switch from in-memory NetworkX to a graph database like Neo4j or a distributed graph system. For the GNN, I'd use mini-batch training with neighbor sampling (already built into GraphSAGE) instead of full-graph training. Features could be computed incrementally using graph streaming frameworks."

### Your synthetic benchmark shows RandomForest = 1.00. Isn't that suspicious?

"Yes, deliberately so. The synthetic label is computed *from* each node's complexity, fan-in, and betweenness — the exact features RF splits on — so a tree can reconstruct it trivially. Feature-importance confirms it: complexity alone accounts for 49% of the RF weight. That's a classic target leakage scenario, and I flag it explicitly in the README rather than treating it as a win. Recognizing this kind of leakage matters more than the number itself."

### What happens when you run on a real codebase?

"On `psf/requests`, under the corrected file-grouped split (3 seeds, git-mined labels), GraphSAGE does **not** beat the tabular baselines — its mean ROC-AUC (0.2088) is below chance and below both LogisticRegression (0.4182) and RandomForest (0.4100), which was the most stable performer. That's the honest result, and it directly reverses the pre-fix number (ROC-AUC 0.9894), which turned out to be a split-leakage artifact, not a real GNN advantage. I'm not spinning this: on this repo, with this label source, out-of-the-box hyperparameters, cross-file message passing didn't add measurable value over handcrafted per-node features like PageRank and complexity. A 37-file repo gives the file-grouped splitter very little to work with — test sets ranged from 37 to 111 nodes and one seed produced a single-class test split — so this reads as much as 'the benchmark repo and label source are small and skewed' as it does 'the GNN underperformed.' The honest takeaway is that the earlier 'GNNs win here' claim wasn't supported once the leakage was removed, and I'd want a bigger, more balanced benchmark (and some GNN-specific tuning) before claiming otherwise either way."

### How does GNNExplainer work?

"GNNExplainer optimises two soft masks — one over input features, one over adjacent edges — to maximise the mutual information between the masked subgraph and the model's original prediction. It's a small per-node optimisation loop that runs after training. The result tells you which features (complexity, PageRank, fan-out) and which graph neighbors most influenced the risk score for a specific node. It turns 'this function is flagged' into 'this function is flagged because it has high cyclomatic complexity and is tightly coupled to a hub file.' That interpretability matters in practice: a developer needs a reason, not just a score."

### What are the limitations?

"Call resolution without running the code is best-effort — we miss dynamic dispatch and cross-file calls through variables. Synthetic labels are heuristic-derived and shouldn't be reported as real risk predictions. And GNNs need enough graph structure to learn from; very small repositories won't benefit much over a simple random forest."
