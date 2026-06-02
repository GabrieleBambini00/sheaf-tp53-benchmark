#!/usr/bin/env python3
"""
Statistical comparison of OOF per-line scores with the paired bootstrap.

With only 7 WT lines, independent AUROC SE ~0.19, so absolute-AUROC gaps are
noisy. The PAIRED bootstrap (resample the 32 lines once per replicate, evaluate
all models on the SAME resample) cancels line-difficulty and is the correct way
to ask "does model A rank cell lines better than model B".

Usage: python scripts/oof_stats.py results/oof_iter1_*.json results/oof_v2_*.json
       (merges all models found across the given JSONs; lines/labels must match)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--ref", default="Sheaf", help="substring of the model to test as 'ours'")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    lines, yL = None, None
    model_scores = {}  # name -> dict line->score
    for jp in args.jsons:
        d = json.loads(Path(jp).read_text())
        ll = d["lines"]
        yy = dict(zip(ll, d["y_line"]))
        if lines is None:
            lines, yL = ll, yy
        for name, r in d["results"].items():
            model_scores[name] = r["line_scores"]

    lines = list(lines)
    y = np.array([yL[l] for l in lines])
    names = list(model_scores)
    S = {n: np.array([model_scores[n][l] for l in lines]) for n in names}

    print("=" * 70)
    print(f"Per-model line AUROC + 95% bootstrap CI  ({len(lines)} lines, "
          f"{int(y.sum())} MUT / {int((1-y).sum())} WT)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    idx_boot = [rng.integers(0, len(lines), len(lines)) for _ in range(args.n_boot)]
    point = {}
    for n in sorted(names, key=lambda n: -auroc(y, S[n])):
        a = auroc(y, S[n])
        point[n] = a
        bs = [auroc(y[ix], S[n][ix]) for ix in idx_boot]
        bs = np.array([b for b in bs if not np.isnan(b)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{n:22s} AUROC={a:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]")

    ref = next((n for n in names if args.ref.lower() in n.lower()), None)
    if ref is None:
        print(f"\n(no model matching --ref '{args.ref}')")
        return
    print("\n" + "=" * 70)
    print(f"PAIRED bootstrap: AUROC({ref}) - AUROC(baseline)   [P = P(ours > baseline)]")
    print("=" * 70)
    for n in names:
        if n == ref:
            continue
        diffs = []
        for ix in idx_boot:
            yb = y[ix]
            if len(np.unique(yb)) < 2:
                continue
            diffs.append(auroc(yb, S[ref][ix]) - auroc(yb, S[n][ix]))
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_win = float((diffs > 0).mean())
        star = "  *" if (lo > 0 or hi < 0) else ""
        print(f"  vs {n:22s} diff={point[ref]-point[n]:+.3f}  95%CI=[{lo:+.3f}, {hi:+.3f}]  "
              f"P(ours>base)={p_win:.2f}{star}")


if __name__ == "__main__":
    sys.exit(main())
