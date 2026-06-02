#!/usr/bin/env python3
"""
Prepare the CCLE bulk RNA-seq powered benchmark (DepMap 21Q4).

Each cell line is ONE sample (node=gene, feature=line's log2(TPM+1) expression).
Label = damaging TP53 mutation (CCLE_mutations_bool_damaging). Because each line
is a single sample, leave-lines-out CV is leakage-free by construction and there
are HUNDREDS of TP53-WT lines -> proper statistical power (unlike the 7 WT lines
in the Gambardella single-cell atlas).

Output: data/preprocessed/ccle_tp53_hvg{N}[_<lineage>].npz with
    X (lines x N, float32 lognorm), y (int8 damaging-TP53), groups (DepMap_ID,
    unique), gene_ids (N), lineage (lines).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _symbol(col: str) -> str:
    # "TP53 (7157)" -> "TP53"
    return col.split(" (")[0].strip()


def _tp53_col(path: Path) -> pd.Series:
    """Load only the TP53 column (boolean) from a DepMap bool mutation matrix."""
    head = pd.read_csv(path, index_col=0, nrows=1)
    tp_cols = [c for c in head.columns if _symbol(c) == "TP53"]
    if not tp_cols:
        raise SystemExit(f"TP53 column not found in {path.name}")
    s = pd.read_csv(path, index_col=0, usecols=["Unnamed: 0", tp_cols[0]]).iloc[:, 0]
    return s.astype(bool)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(repo / "data" / "raw" / "ccle_depmap"))
    ap.add_argument("--out_dir", default=str(repo / "data" / "preprocessed"))
    ap.add_argument("--n_hvg", type=int, default=300)
    ap.add_argument("--lineage", default="", help="optional lineage filter, e.g. breast")
    args = ap.parse_args()

    raw = Path(args.raw)
    print("[1] loading expression (422MB, ~1-2 min) ...")
    exp = pd.read_csv(raw / "CCLE_expression.csv", index_col=0)
    exp.columns = [_symbol(c) for c in exp.columns]
    exp = exp.loc[:, ~exp.columns.duplicated()]
    exp = exp.astype(np.float32)
    print(f"    expression: {exp.shape[0]} lines x {exp.shape[1]} genes")

    print("[2] loading TP53 status = damaging OR hotspot ...")
    dmg = _tp53_col(raw / "CCLE_mutations_bool_damaging.csv")
    hot = _tp53_col(raw / "CCLE_mutations_bool_hotspot.csv")
    idx = dmg.index.union(hot.index)
    tp53 = (dmg.reindex(idx).fillna(False) | hot.reindex(idx).fillna(False))
    print(f"    TP53 MUT={int(tp53.sum())}/{len(tp53)} ({100*tp53.mean():.1f}%) "
          f"[damaging={int(dmg.sum())}, hotspot={int(hot.sum())}]")

    info = pd.read_csv(raw / "sample_info.csv", index_col=0)
    lin = info["lineage"] if "lineage" in info.columns else pd.Series("NA", index=info.index)

    lines = exp.index.intersection(tp53.index)
    if args.lineage:
        keep = lin.reindex(lines).fillna("NA").str.lower() == args.lineage.lower()
        lines = lines[keep.values]
    print(f"[3] lines with expression+mutation: {len(lines)}"
          + (f" (lineage={args.lineage})" if args.lineage else " (all lineages)"))

    X = exp.loc[lines]
    y = tp53.reindex(lines).fillna(False).astype(bool).astype(np.int8).values
    lineage = lin.reindex(lines).fillna("NA").astype(str).values

    # HVG: top-N by variance across lines
    var = X.var(axis=0)
    top = var.sort_values(ascending=False).index[: args.n_hvg]
    top = [g for g in X.columns if g in set(top)]  # preserve order, unique
    Xh = X[top].to_numpy(dtype=np.float32)
    gene_ids = np.array(top)

    n_mut = int(y.sum())
    print(f"[4] X={Xh.shape}  MUT={n_mut} WT={len(y)-n_mut} ({100*n_mut/len(y):.1f}% MUT)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.lineage.lower()}" if args.lineage else ""
    out = out_dir / f"ccle_tp53_hvg{args.n_hvg}{tag}.npz"
    np.savez_compressed(out, X=Xh, y=y.astype(np.int8),
                        groups=np.array([str(s) for s in lines]),
                        gene_ids=gene_ids.astype(str), lineage=lineage.astype(str))
    print(f"[5] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
