#!/usr/bin/env python3
"""Cell-line-level OOF benchmark: AUROC over the 32 lines (stable, honest)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sheaf_tp53.oof import run_oof  # noqa: E402

CATALOG = {
    "logreg": ("LogReg (flat)", "logreg", {}),
    "hgb": ("GradBoost (flat)", "hgb", {}),
    "gcn": ("GCN (Ravasio)", "gcn", dict(hidden=64, n_layers=2, dropout=0.3)),
    "gat": ("GAT (Ravasio best)", "gat", dict(hidden=64, heads=4, n_layers=2, dropout=0.3)),
    "sheaf": ("Sheaf (ours)", "sheaf", dict(stalk_dim=4, hidden=64, n_layers=2,
                                            dropout=0.3, diffusion_steps=1)),
    "sheaf_d8": ("Sheaf d8", "sheaf", dict(stalk_dim=8, hidden=64, n_layers=2,
                                           dropout=0.3, diffusion_steps=1)),
    "sheaf_d12": ("Sheaf d12", "sheaf", dict(stalk_dim=12, hidden=96, n_layers=2,
                                             dropout=0.3, diffusion_steps=1)),
    "sheaf_d16": ("Sheaf d16", "sheaf", dict(stalk_dim=16, hidden=128, n_layers=2,
                                             dropout=0.3, diffusion_steps=1)),
    "sheaf_hybrid": ("Sheaf+flat (hybrid)", "sheaf", dict(stalk_dim=12, hidden=128,
                     n_layers=2, dropout=0.3, diffusion_steps=1, flat_skip=True)),
    "sheaf_deep": ("Sheaf d8 deep", "sheaf", dict(stalk_dim=8, hidden=128, n_layers=3,
                                                  dropout=0.3, diffusion_steps=2)),
    "sheaf_norm": ("Sheaf d8 norm", "sheaf", dict(stalk_dim=8, hidden=64, n_layers=2,
                                                  dropout=0.3, diffusion_steps=1,
                                                  normalized=True)),
}


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--npz", default=str(repo / "data" / "preprocessed" / "gambardella_hvg300.npz"))
    ap.add_argument("--models", nargs="+", default=["logreg", "hgb", "gcn", "gat", "sheaf"])
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--cap", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    specs = [CATALOG[k] for k in args.models]
    t0 = time.time()
    res = run_oof(args.npz, specs, n_splits=args.n_splits, n_repeats=args.n_repeats,
                  threshold=args.threshold, seed=args.seed, cap=args.cap,
                  max_epochs=args.epochs)
    res["runtime_sec"] = time.time() - t0

    R = res["results"]
    print("\n" + "=" * 70)
    print(f"CELL-LINE OOF RESULTS  (AUROC over {len(res['lines'])} lines, "
          f"{args.n_repeats}x{args.n_splits} grouped folds, thr={args.threshold})")
    print("=" * 70)
    print(f"{'model':22s} {'line_AUROC':>11s} {'line_AP':>9s} {'line_bAcc':>10s}")
    print("-" * 70)
    order = sorted(R, key=lambda n: -R[n]["line_auroc"])
    for name in order:
        print(f"{name:22s} {R[name]['line_auroc']:>11.3f} {R[name]['line_ap']:>9.3f} "
              f"{R[name]['line_bacc']:>10.3f}")
    print("-" * 70)
    print(f"runtime {res['runtime_sec']:.0f}s")

    out_dir = repo / "results"
    out_dir.mkdir(exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    out = out_dir / f"oof_{tag}{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
