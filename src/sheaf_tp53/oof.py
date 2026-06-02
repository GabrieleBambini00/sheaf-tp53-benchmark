"""
Cell-line-level out-of-fold (OOF) evaluation — the statistically honest protocol.

Cells within a cell line are pseudo-replicates (one TP53 label per line), so
cell-level AUROC is pseudoreplicated and unstable (per-fold AUROC swings 0.13-1.0
with only ~6 test lines/fold). The correct unit of evaluation is the CELL LINE.

Protocol
--------
- Repeated StratifiedGroupKFold by cell line: every line is held out in test
  exactly once per repeat (n_repeats repeats -> n_repeats OOF predictions/line).
- For each held-out line, the predicted score = mean cell probability.
- Aggregate to ONE score per line (mean across repeats) -> AUROC / AP over the
  32 lines. This is a stable, leakage-free single number per model.

Everything else (train-only standardisation, train-only Spearman graph,
class-weighted training, group-disjoint inner val) matches cv.py.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from sheaf_tp53.cv import (
    _standardize,
    _subsample_per_group,
    predict_torch,
    train_torch,
)
from sheaf_tp53.graph import build_coexpression_graph
from sheaf_tp53.models import MODEL_REGISTRY
from sheaf_tp53.splits import assert_no_group_leakage, group_split
from sheaf_tp53.seeds import set_global_seed


def _fit_predict(key, kw, Xtr, ytr, Xva, yva, Xte, edge_und, edge_bi, N,
                 seed, max_epochs):
    if key in ("hgb", "logreg"):
        Xall = np.concatenate([Xtr, Xva])
        yall = np.concatenate([ytr, yva])
        if key == "hgb":
            clf = HistGradientBoostingClassifier(class_weight="balanced",
                                                 random_state=seed, **kw)
        else:
            clf = LogisticRegression(class_weight="balanced", max_iter=2000, **kw)
        clf.fit(Xall, yall)
        return clf.predict_proba(Xte)[:, 1]
    edge = edge_und if key == "sheaf" else edge_bi
    set_global_seed(seed)  # seed BEFORE construction so init is order-independent
    model = MODEL_REGISTRY[key](num_nodes=N, edge_index=edge, **kw)
    model = train_torch(model, Xtr, ytr, Xva, yva, edge_kind_und=(key == "sheaf"),
                        max_epochs=max_epochs, seed=seed)
    return predict_torch(model, Xte)


def prepare_oof_folds(npz_path, *, n_splits=5, n_repeats=3, threshold=0.2,
                      seed=42, cap=300, verbose=True):
    """Build (and cache) the per-(repeat,fold) standardised arrays + graph so
    sheaf redesigns reuse IDENTICAL folds (fair) without recomputing (fast)."""
    cache_dir = Path(npz_path).resolve().parent.parent / "results" / "oof_folds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = (f"{Path(npz_path).stem}_n{n_splits}_r{n_repeats}_thr{threshold}"
           f"_s{seed}_cap{cap}.pkl")
    cache = cache_dir / key
    if cache.exists():
        if verbose:
            print(f"[oof-folds] loading cache {cache.name}", flush=True)
        return pickle.loads(cache.read_bytes())

    data = np.load(npz_path, allow_pickle=True)
    X, y, groups = (data["X"].astype(np.float32), data["y"].astype(int),
                    data["groups"].astype(str))
    set_global_seed(seed)
    sub = _subsample_per_group(groups, y, cap, seed)
    X, y, groups = X[sub], y[sub], groups[sub]
    lines = sorted(set(groups))
    line_status = {g: int(y[groups == g][0]) for g in lines}

    prepped = []
    for rep in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for fold, (tr, te) in enumerate(skf.split(X, y, groups)):
            assert_no_group_leakage(groups[tr], groups[te])
            tr2_rel, va_rel = group_split(groups[tr], y[tr], test_size=0.2,
                                          random_seed=seed + rep)
            tr2, va = tr[tr2_rel], tr[va_rel]
            Xtr, Xva, Xte = _standardize(X[tr2], X[va], X[te])
            edge_und, edge_bi, info = build_coexpression_graph(Xtr, threshold=threshold)
            prepped.append(dict(rep=rep, fold=fold, Xtr=Xtr, ytr=y[tr2], Xva=Xva,
                                yva=y[va], Xte=Xte, gte=groups[te],
                                edge_und=edge_und, edge_bi=edge_bi, info=info))
            if verbose:
                print(f"[oof-folds] rep{rep} fold{fold} train={len(tr2)} test={len(te)} "
                      f"edges={info['n_edges_undirected']} negfrac={info['frac_negative']:.2f}",
                      flush=True)
    bundle = {"prepped": prepped, "lines": lines, "line_status": line_status,
              "n_nodes": int(X.shape[1])}
    cache.write_bytes(pickle.dumps(bundle))
    return bundle


def run_oof(npz_path, model_specs, *, n_splits=5, n_repeats=3, threshold=0.2,
            seed=42, cap=300, max_epochs=50, verbose=True):
    bundle = prepare_oof_folds(npz_path, n_splits=n_splits, n_repeats=n_repeats,
                               threshold=threshold, seed=seed, cap=cap, verbose=verbose)
    prepped, lines, line_status, N = (bundle["prepped"], bundle["lines"],
                                      bundle["line_status"], bundle["n_nodes"])
    scores = {name: {g: [] for g in lines} for name, _, _ in model_specs}

    for fd in prepped:
        gte = fd["gte"]
        if verbose:
            print(f"[rep {fd['rep']} fold {fd['fold']}] edges={fd['info']['n_edges_undirected']} "
                  f"negfrac={fd['info']['frac_negative']:.2f}", flush=True)
        for name, key, kw in model_specs:
            prob = _fit_predict(key, kw, fd["Xtr"], fd["ytr"], fd["Xva"], fd["yva"],
                                fd["Xte"], fd["edge_und"], fd["edge_bi"], N,
                                seed + fd["rep"], max_epochs)
            for g in np.unique(gte):
                scores[name][g].append(float(prob[gte == g].mean()))
            if verbose:
                print(f"    {name:18s} done", flush=True)

    yL = np.array([line_status[g] for g in lines])
    results = {}
    for name in scores:
        sL = np.array([np.mean(scores[name][g]) for g in lines])
        results[name] = {
            "line_auroc": float(roc_auc_score(yL, sL)),
            "line_ap": float(average_precision_score(yL, sL)),
            "line_bacc": float(balanced_accuracy_score(yL, (sL >= 0.5).astype(int))),
            "line_scores": {g: float(np.mean(scores[name][g])) for g in lines},
        }
    return {"results": results, "lines": list(lines), "y_line": yL.tolist(),
            "config": {"n_splits": n_splits, "n_repeats": n_repeats,
                       "threshold": threshold, "seed": seed, "cap": cap,
                       "npz": str(npz_path)}}
