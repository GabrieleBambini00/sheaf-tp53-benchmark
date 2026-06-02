"""
Leakage-free cross-validation harness for the sheaf-vs-Ravasio benchmark.

Protocol
--------
- Outer: StratifiedGroupKFold by cell line (groups intact => no cell-line leak).
- Inner: group-disjoint val split from train lines (early stopping).
- Standardisation: z-score using TRAIN gene stats only.
- Graph: Spearman co-expression on TRAIN cells only (per fold).
- Class imbalance (~81% MUT): class-weighted cross-entropy for ALL models.
- Metrics: AUROC + balanced accuracy (leakage-robust) + F1/acc/prec/rec
  (F1 == Ravasio's headline metric).

`leaky=True` switches the outer split to random cell-level KFold to reproduce
Ravasio's inflated numbers (demonstration of the failure mode).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sheaf_tp53.splits import assert_no_group_leakage, group_split
from sheaf_tp53.seeds import set_global_seed
from sheaf_tp53.graph import build_coexpression_graph
from sheaf_tp53.models import MODEL_REGISTRY


def compute_metrics(y_true, prob, pred) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }
    out["auroc"] = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else float("nan")
    return out


def _standardize(Xtr, *others):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return tuple((A - mu) / sd for A in (Xtr, *others))


def train_torch(model, Xtr, ytr, Xva, yva, *, edge_kind_und, max_epochs=60,
                patience=8, lr=1e-3, wd=1e-5, batch=256, device="cpu", seed=0):
    set_global_seed(seed)
    model = model.to(device)
    # class weights (inverse frequency)
    cls_count = np.bincount(ytr, minlength=2).astype(np.float64)
    w = cls_count.sum() / (2 * np.clip(cls_count, 1, None))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)

    n = Xtr_t.shape[0]
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logits = model(Xtr_t[idx])
            loss = crit(logits, ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        # val AUROC
        model.eval()
        with torch.no_grad():
            prob_va = torch.softmax(model(Xva_t), dim=1)[:, 1].cpu().numpy()
        auc = roc_auc_score(yva, prob_va) if len(np.unique(yva)) > 1 else 0.0
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_torch(model, X, device="cpu", batch=1024):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    probs = []
    for i in range(0, Xt.shape[0], batch):
        probs.append(torch.softmax(model(Xt[i:i + batch]), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def _subsample_per_group(groups, y, max_per_group, seed):
    """Deterministically cap cells per cell line (within-line cells are highly
    redundant; capping speeds training without changing the task)."""
    if not max_per_group:
        return np.arange(len(groups))
    rng = np.random.default_rng(seed)
    keep = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) > max_per_group:
            idx = rng.choice(idx, size=max_per_group, replace=False)
        keep.append(idx)
    out = np.sort(np.concatenate(keep))
    return out


def prepare_folds(npz_path, *, n_splits=5, threshold=0.3, seed=42, leaky=False,
                  max_cells_per_group=400, verbose=True):
    """Build (and cache) the per-fold standardised arrays + co-expression graph.

    Cached to results/folds/<key>.pkl so sheaf redesigns reuse IDENTICAL folds
    (fair) without recomputing graphs/standardisation (fast).
    """
    cache_dir = Path(npz_path).resolve().parent.parent / "results" / "folds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = (f"{Path(npz_path).stem}_n{n_splits}_thr{threshold}_s{seed}"
           f"_cap{max_cells_per_group}_{'leaky' if leaky else 'grouped'}.pkl")
    cache = cache_dir / key
    if cache.exists():
        if verbose:
            print(f"[folds] loading cache {cache.name}")
        return pickle.loads(cache.read_bytes())

    data = np.load(npz_path, allow_pickle=True)
    X, y, groups = data["X"].astype(np.float32), data["y"].astype(int), data["groups"].astype(str)
    set_global_seed(seed)
    sub = _subsample_per_group(groups, y, max_cells_per_group, seed)
    X, y, groups = X[sub], y[sub], groups[sub]

    if leaky:
        split_iter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y)
    else:
        split_iter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                           random_state=seed).split(X, y, groups)

    folds = []
    for fold, (tr, te) in enumerate(split_iter):
        if not leaky:
            assert_no_group_leakage(groups[tr], groups[te])
        if leaky:
            tr2_rel, va_rel = next(StratifiedKFold(n_splits=5, shuffle=True,
                                                   random_state=seed).split(X[tr], y[tr]))
        else:
            tr2_rel, va_rel = group_split(groups[tr], y[tr], test_size=0.2, random_seed=seed)
        tr2, va = tr[tr2_rel], tr[va_rel]
        Xtr, Xva, Xte = _standardize(X[tr2], X[va], X[te])
        ytr, yva, yte = y[tr2], y[va], y[te]
        edge_und, edge_bi, ginfo = build_coexpression_graph(Xtr, threshold=threshold)
        folds.append(dict(
            fold=fold, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte,
            edge_und=edge_und, edge_bi=edge_bi, n_nodes=X.shape[1],
            test_lines=sorted(set(groups[te])), test_pos_rate=float(yte.mean()),
            info=ginfo))
        if verbose:
            print(f"[fold {fold}] train={len(tr2)} val={len(va)} test={len(te)} "
                  f"| edges={ginfo['n_edges_undirected']} negfrac={ginfo['frac_negative']:.2f} "
                  f"| test_pos={yte.mean():.2f}")
    cache.write_bytes(pickle.dumps(folds))
    if verbose:
        print(f"[folds] cached {cache.name}")
    return folds


def run_cv(npz_path, model_specs, *, n_splits=5, threshold=0.3, seed=42,
           leaky=False, max_epochs=60, max_cells_per_group=400, verbose=True):
    """model_specs: list of (name, model_key, kwargs). Returns results dict."""
    folds = prepare_folds(npz_path, n_splits=n_splits, threshold=threshold, seed=seed,
                          leaky=leaky, max_cells_per_group=max_cells_per_group, verbose=verbose)
    results = {name: [] for name, _, _ in model_specs}
    fold_info = []

    for fd in folds:
        fold = fd["fold"]
        Xtr, ytr, Xva, yva, Xte, yte = (fd["Xtr"], fd["ytr"], fd["Xva"], fd["yva"],
                                        fd["Xte"], fd["yte"])
        edge_und, edge_bi, N = fd["edge_und"], fd["edge_bi"], fd["n_nodes"]
        fold_info.append({"fold": fold, "test_lines": fd["test_lines"],
                          "test_pos_rate": fd["test_pos_rate"], **fd["info"]})

        for name, key, kw in model_specs:
            if key in ("hgb", "logreg"):
                if key == "hgb":
                    clf = HistGradientBoostingClassifier(
                        class_weight="balanced", random_state=seed, **kw)
                else:
                    clf = LogisticRegression(
                        class_weight="balanced", max_iter=2000, **kw)
                clf.fit(np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva]))
                prob = clf.predict_proba(Xte)[:, 1]
            else:
                edge = edge_und if key == "sheaf" else edge_bi
                model = MODEL_REGISTRY[key](num_nodes=N, edge_index=edge, **kw)
                model = train_torch(model, Xtr, ytr, Xva, yva, edge_kind_und=(key == "sheaf"),
                                    max_epochs=max_epochs, seed=seed + fold)
                prob = predict_torch(model, Xte)
            pred = (prob >= 0.5).astype(int)
            m = compute_metrics(yte, prob, pred)
            results[name].append(m)
            if verbose:
                print(f"    [f{fold}] {name:18s} AUROC={m['auroc']:.3f} bAcc={m['balanced_accuracy']:.3f} "
                      f"F1={m['f1']:.3f} acc={m['accuracy']:.3f}")

    # aggregate
    summary = {}
    for name in results:
        keys = results[name][0].keys()
        summary[name] = {k: {"mean": float(np.nanmean([r[k] for r in results[name]])),
                             "std": float(np.nanstd([r[k] for r in results[name]]))}
                         for k in keys}
    return {"summary": summary, "folds": results, "fold_info": fold_info,
            "config": {"n_splits": n_splits, "threshold": threshold, "seed": seed,
                       "leaky": leaky, "max_cells_per_group": max_cells_per_group,
                       "npz": str(npz_path)}}
