"""
Graph classifiers on a SHARED gene co-expression topology.

All models consume the same batched representation for strict fairness:
    forward(x) where x is (B, N) per-cell expression over N gene-nodes,
    returning logits (B, n_classes).

Node = gene, edges = Spearman co-expression graph (fixed per fold). This is
Ravasio's per-cell graph-classification setup, vectorised over the shared
topology so it runs fast on CPU and guarantees identical data handling.

GCN/GAT use PyTorch-Geometric's optimised convolutions (exactly Ravasio's
building blocks) by assembling a block-diagonal big-graph per minibatch.
The Sheaf model uses the verified VectorizedSheafDiffusion operator with a
single shared connection Laplacian per batch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

from sheaf_tp53._sheaf import (
    NormalizedVectorizedSheafDiffusion,
    VectorizedSheafDiffusion,
)


def _batch_graph(x: torch.Tensor, edge_index: torch.Tensor):
    """(B, N) features + shared (2, E) edges -> block-diagonal big graph.

    Returns big_x (B*N, 1), big_edge (2, B*E), batch (B*N,).
    """
    B, N = x.shape
    E = edge_index.shape[1]
    big_x = x.reshape(B * N, 1)
    offs = (torch.arange(B, device=x.device) * N).repeat_interleave(E)  # (B*E,)
    big_edge = edge_index.repeat(1, B) + offs.unsqueeze(0)              # (2, B*E)
    batch = torch.arange(B, device=x.device).repeat_interleave(N)       # (B*N,)
    return big_x, big_edge, batch


class GCNClassifier(nn.Module):
    """Ravasio GCN baseline (PyG GCNConv x2 + global mean pool)."""

    def __init__(self, num_nodes, edge_index, hidden=64, n_layers=2,
                 n_classes=2, dropout=0.3):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_dim = 1
        for _ in range(n_layers):
            self.convs.append(GCNConv(in_dim, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
            in_dim = hidden
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, x):
        big_x, big_edge, batch = _batch_graph(x, self.edge_index)
        h = big_x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, big_edge)
            h = F.relu(bn(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch)
        return self.head(g)


class GATClassifier(nn.Module):
    """Ravasio's best baseline (PyG GATConv multi-head x2 + global mean pool)."""

    def __init__(self, num_nodes, edge_index, hidden=64, heads=4, n_layers=2,
                 n_classes=2, dropout=0.3):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        per_head = max(1, hidden // heads)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_dim = 1
        for _ in range(n_layers):
            self.convs.append(GATConv(in_dim, per_head, heads=heads, concat=True,
                                      dropout=dropout))
            self.bns.append(nn.BatchNorm1d(per_head * heads))
            in_dim = per_head * heads
        self.head = nn.Linear(per_head * heads, n_classes)
        self.dropout = dropout

    def forward(self, x):
        big_x, big_edge, batch = _batch_graph(x, self.edge_index)
        h = big_x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, big_edge)
            h = F.elu(bn(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch)
        return self.head(g)


class SheafClassifier(nn.Module):
    """
    Static parametric Neural Sheaf Diffusion classifier.

    Per gene-node a d-dim stalk; per edge a learned d x d restriction map.
    The sheaf connection Laplacian L_F (shared across the batch) drives
    diffusion. Sign-inverting restriction maps encode anti-correlated
    repressor edges natively (the homophily failure mode of GCN/GAT is
    avoided by construction).
    """

    def __init__(self, num_nodes, edge_index, stalk_dim=4, hidden=64,
                 n_layers=2, n_classes=2, dropout=0.3, normalized=False,
                 diffusion_steps=1, per_gene_lift=True, residual=True,
                 pool="meanmax", flat_skip=False):
        super().__init__()
        self.n = num_nodes
        self.d = stalk_dim
        self.n_layers = n_layers
        self.steps = diffusion_steps
        self.per_gene_lift = per_gene_lift
        self.residual = residual
        self.pool = pool
        self.flat_skip = flat_skip
        op = NormalizedVectorizedSheafDiffusion if normalized else VectorizedSheafDiffusion
        self.sheaf = op(num_nodes, edge_index.shape[1], stalk_dim, edge_index)
        if per_gene_lift:
            # each gene gets its own learned d-direction, modulated by expression
            self.node_scale = nn.Parameter(torch.randn(num_nodes, stalk_dim) * 0.1)
            self.node_bias = nn.Parameter(torch.zeros(num_nodes, stalk_dim))
        else:
            self.lift = nn.Linear(1, stalk_dim)
        self.Wl = nn.ParameterList(nn.Parameter(torch.empty(stalk_dim, stalk_dim))
                                   for _ in range(n_layers))
        self.alpha = nn.Parameter(torch.full((n_layers,), 0.5))
        pool_mult = 2 if pool == "meanmax" else 1
        head_in = stalk_dim * pool_mult
        if flat_skip:
            # recover per-gene signal that graph pooling discards: a learned
            # projection of the raw expression vector, concatenated before the head
            self.flat_proj = nn.Sequential(
                nn.Linear(num_nodes, hidden), nn.ReLU(), nn.Dropout(dropout))
            head_in += hidden
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes),
        )
        for w in self.Wl:
            nn.init.xavier_uniform_(w)

    def forward(self, x):
        B, N = x.shape
        L_F = self.sheaf.compute_connection_laplacian_vectorized()   # (Nd, Nd)
        if self.per_gene_lift:
            h = x.unsqueeze(-1) * self.node_scale + self.node_bias   # (B, N, d)
        else:
            h = self.lift(x.unsqueeze(-1))                           # (B, N, d)
        for li in range(self.n_layers):
            for _ in range(self.steps):
                z = h @ self.Wl[li]                                 # (B, N, d)
                z_flat = z.reshape(B, N * self.d).T                 # (Nd, B)
                diff = (L_F @ z_flat).T.reshape(B, N, self.d)       # (B, N, d)
                h = (h - self.alpha[li] * diff) if self.residual else (-self.alpha[li] * diff)
            h = F.relu(h)
        if self.pool == "meanmax":
            g = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)   # (B, 2d)
        else:
            g = h.mean(dim=1)                                       # (B, d)
        if self.flat_skip:
            g = torch.cat([g, self.flat_proj(x)], dim=-1)          # + raw-expr signal
        return self.head(g)


MODEL_REGISTRY = {
    "gcn": GCNClassifier,
    "gat": GATClassifier,
    "sheaf": SheafClassifier,
}
