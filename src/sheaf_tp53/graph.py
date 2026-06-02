"""
Gene co-expression graph construction (Ravasio-faithful, train-only).

Spearman correlation across TRAIN cells -> threshold |rho| >= tau -> undirected
edge set (unweighted, exactly like Ravasio's `dense_to_sparse` on the thresholded
matrix). The graph is built from TRAIN cells only, every fold, to avoid leaking
test-cell covariance into the topology.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def spearman_matrix(X: np.ndarray) -> np.ndarray:
    """Spearman correlation among the N gene columns of X (cells x N)."""
    # rank each gene (column) — pandas rank is vectorised + handles ties,
    # then Pearson on ranks == Spearman.
    ranks = pd.DataFrame(X).rank(axis=0).to_numpy(dtype=np.float64)
    ranks -= ranks.mean(axis=0, keepdims=True)
    std = ranks.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    ranks /= std
    n = ranks.shape[0]
    corr = (ranks.T @ ranks) / n
    np.fill_diagonal(corr, 0.0)
    return corr


def build_coexpression_graph(
    X_train: np.ndarray, threshold: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Returns:
        edge_undirected: (2, E) unique undirected edges (i<j) — for the sheaf op.
        edge_bidirectional: (2, 2E) both directions — for GCN/GAT message passing.
        info: dict with n_edges, density, mean|rho|, frac_negative.
    """
    corr = spearman_matrix(X_train)
    N = corr.shape[0]
    mask = np.abs(corr) >= threshold
    iu = np.triu_indices(N, k=1)
    sel = mask[iu]
    src = iu[0][sel]
    tgt = iu[1][sel]
    rho = corr[src, tgt]

    edge_und = torch.tensor(np.stack([src, tgt]), dtype=torch.long)
    edge_bi = torch.tensor(
        np.stack([np.concatenate([src, tgt]), np.concatenate([tgt, src])]),
        dtype=torch.long,
    )
    info = {
        "n_nodes": N,
        "n_edges_undirected": int(edge_und.shape[1]),
        "density": float(2 * edge_und.shape[1] / (N * (N - 1))) if N > 1 else 0.0,
        "mean_abs_rho": float(np.abs(rho).mean()) if rho.size else 0.0,
        "frac_negative": float((rho < 0).mean()) if rho.size else 0.0,
    }
    return edge_und, edge_bi, info
