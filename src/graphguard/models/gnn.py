"""
Graph Neural Network models for node-level risk classification.

Architecture overview
---------------------
Both models perform *node-level binary classification*:
  input  : x ∈ R^{N×F}  (feature matrix) and edge_index ∈ Z^{2×E}
  output : logits ∈ R^{N}  (one per node, sigmoid → risk probability)

GraphSAGE (Hamilton et al., 2017)
  Each layer aggregates neighbor features via MEAN pooling, then
  concatenates with the node's own embedding:
    h_v^{(k)} = σ( W · CONCAT(h_v^{(k-1)}, MEAN_{u∈N(v)} h_u^{(k-1)}) )

  This is equivalent to one step of matrix multiplication on a row-normalized
  adjacency:  H^{(k)} = σ( [I | D^{-1}A] H^{(k-1)} W^T )

GCN (Kipf & Welling, 2017)
  Symmetric normalization:  H^{(k+1)} = σ( D̃^{-1/2} Ã D̃^{-1/2} H^{(k)} W )
  where Ã = A + I (self-loops).  Each layer is one step of graph diffusion.

Both are used as feature encoders; the final layer maps to a single logit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GraphSAGEClassifier(nn.Module):
    """
    GraphSAGE node classifier with configurable depth.

    Parameters
    ----------
    in_channels  : number of input features (F)
    hidden_dim   : width of hidden layers
    num_layers   : number of SAGEConv layers (≥ 1)
    dropout      : dropout rate applied after each hidden layer
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Classification head
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : [N, F] node feature matrix
        edge_index : [2, E] edge index tensor

        Returns
        -------
        logits : [N] raw logits (apply sigmoid for probabilities)
        """
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h).squeeze(-1)  # [N]

    def predict_proba(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Return sigmoid probabilities [N]."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x, edge_index))


class GCNClassifier(nn.Module):
    """
    GCN node classifier (Kipf & Welling 2017).

    Note: GCN expects an undirected (symmetric) edge_index.
    Use CodeGraphDataset.build(..., undirected=True) when training GCN.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h).squeeze(-1)

    def predict_proba(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x, edge_index))


def build_model(
    model_type: str,
    in_channels: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> nn.Module:
    """Factory function — returns the requested model."""
    if model_type == "sage":
        return GraphSAGEClassifier(in_channels, hidden_dim, num_layers, dropout)
    elif model_type == "gcn":
        return GCNClassifier(in_channels, hidden_dim, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Choose 'sage' or 'gcn'.")
