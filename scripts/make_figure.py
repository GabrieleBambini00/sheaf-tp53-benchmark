#!/usr/bin/env python3
"""
Deliverable figure: per-model honest cell-line AUROC with bootstrap 95% CIs,
for one or more OOF result JSONs (e.g. Gambardella scRNA + CCLE bulk).

Usage:
  python scripts/make_figure.py --out results/figure_comparison.png \
      "Gambardella scRNA (32 lines)=results/oof_r5_*.json" \
      "CCLE bulk (1385 lines)=results/oof_ccle_*.json"
"""
import argparse
import glob
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ORDER = ["LogReg (flat)", "GradBoost (flat)", "GCN (Ravasio)",
         "GAT (Ravasio best)", "Sheaf d8", "Sheaf d12", "Sheaf (ours)",
         "Sheaf+flat (hybrid)"]


def _color(name: str) -> str:
    n = name.lower()
    if "sheaf" in n:
        return "#d1495b"   # red — ours
    if "ravasio" in n or n.startswith(("gcn", "gat")):
        return "#3a7ca5"   # blue — Ravasio GNNs
    return "#8d99ae"       # gray — flat baselines


def boot_ci(y, s, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    a = roc_auc_score(y, s)
    bs = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) > 1:
            bs.append(roc_auc_score(y[ix], s[ix]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return a, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panels", nargs="+", help='"Title=path-glob.json"')
    ap.add_argument("--out", default="results/figure_comparison.png")
    args = ap.parse_args()

    n = len(args.panels)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    for ax, panel in zip(axes[0], args.panels):
        title, gpath = panel.split("=", 1)
        path = sorted(glob.glob(gpath))[-1]
        d = json.loads(open(path).read())
        y = np.array(d["y_line"])
        lines = d["lines"]
        rows = []
        for name, r in d["results"].items():
            s = np.array([r["line_scores"][l] for l in lines])
            a, lo, hi = boot_ci(y, s)
            rows.append((name, a, lo, hi))
        rows.sort(key=lambda t: ORDER.index(t[0]) if t[0] in ORDER else 99)
        names = [r[0] for r in rows]
        ys = np.arange(len(rows))
        for i, (name, a, lo, hi) in enumerate(rows):
            ax.barh(i, a, color=_color(name),
                    xerr=[[a - lo], [hi - a]], capsize=4, alpha=0.9)
            ax.text(a + 0.01, i, f"{a:.3f}", va="center", fontsize=9)
        ax.axvline(0.5, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_yticks(ys)
        ax.set_yticklabels(names)
        ax.set_xlim(0.3, 1.0)
        ax.set_xlabel("cell-line AUROC (honest, leakage-free)")
        ax.set_title(title)
        ax.invert_yaxis()
    fig.suptitle("TP53-status prediction: Sheaf vs Ravasio (leakage-free, "
                 "per-cell-line AUROC, 95% bootstrap CI)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
