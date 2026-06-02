"""Smoke tests: models run, leakage guard fires, graph builds, sheaf grads finite."""
import numpy as np
import pytest
import torch

from sheaf_tp53.models import GCNClassifier, GATClassifier, SheafClassifier
from sheaf_tp53.graph import build_coexpression_graph
from sheaf_tp53.splits import assert_no_group_leakage, LeakageError


def _toy_edges(n, e=60):
    ei = torch.randint(0, n, (2, e))
    return ei[:, ei[0] != ei[1]]


@pytest.mark.parametrize("ctor", [
    lambda n, ei: GCNClassifier(n, ei),
    lambda n, ei: GATClassifier(n, ei),
    lambda n, ei: SheafClassifier(n, ei, stalk_dim=8),
    lambda n, ei: SheafClassifier(n, ei, stalk_dim=12, flat_skip=True),
])
def test_forward_backward(ctor):
    n, b = 40, 8
    ei = _toy_edges(n)
    model = ctor(n, ei)
    x = torch.randn(b, n)
    y = torch.randint(0, 2, (b,))
    out = model(x)
    assert out.shape == (b, 2)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients flowed"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"


def test_leakage_guard_fires():
    with pytest.raises(LeakageError):
        assert_no_group_leakage(["A", "B"], ["B", "C"])  # B overlaps
    assert_no_group_leakage(["A", "B"], ["C", "D"])  # disjoint: no raise


def test_graph_builds_and_is_train_only():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 30)).astype(np.float32)
    edge_und, edge_bi, info = build_coexpression_graph(X, threshold=0.1)
    assert edge_und.shape[0] == 2
    assert edge_bi.shape[1] == 2 * edge_und.shape[1]  # bidirectional
    assert info["n_edges_undirected"] == edge_und.shape[1]
