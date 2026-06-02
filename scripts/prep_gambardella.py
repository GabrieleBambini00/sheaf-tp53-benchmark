#!/usr/bin/env python3
"""
Preprocess the Gambardella 2022 BC cell-line atlas into compact arrays for the
sheaf-vs-Ravasio benchmark.

Pipeline (Ravasio-faithful, leakage-aware):
  1. Read 10x MatrixMarket (genes x cells) -> AnnData (cells x genes).
  2. cell_line = barcode prefix before '_'.
  3. Attach TP53 binary label from data/labels/gambardella_tp53_status.csv.
  4. QC: drop cells with <min_counts UMIs, drop genes in <min_cells cells.
  5. Normalize total (1e4) + log1p.
  6. Select top-N highly variable genes (seurat flavor).
  7. (NO z-scoring here — done per-fold on TRAIN stats to avoid leakage.)
  8. Save data/preprocessed/gambardella_hvg{N}.npz with:
        X (float32, cells x N lognorm), y (int8), groups (cell_line str),
        gene_ids (N str).

Output is the single cached input for the cross-validation harness. Z-scoring
and graph construction happen per-fold (train-only) downstream.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, "rt") as f:
        return [ln.split("\t")[0].strip() for ln in f if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--raw_dir", default=str(repo / "data" / "raw" / "breast_gambardella"))
    ap.add_argument("--labels", default=str(repo / "data" / "labels" / "gambardella_tp53_status.csv"))
    ap.add_argument("--out_dir", default=str(repo / "data" / "preprocessed"))
    ap.add_argument("--n_hvg", type=int, default=300)
    ap.add_argument("--min_counts", type=int, default=500)
    ap.add_argument("--min_cells", type=int, default=20)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    print(f"[1] Reading MatrixMarket from {raw} ...")
    mat = scipy.io.mmread(str(raw / "matrix.mtx.gz"))  # genes x cells (COO)
    mat = sp.csr_matrix(mat)
    genes = _read_lines_gz(raw / "features.tsv.gz")
    barcodes = _read_lines_gz(raw / "barcodes.tsv.gz")
    print(f"    raw matrix shape (genes x cells) = {mat.shape}; genes={len(genes)} cells={len(barcodes)}")
    assert mat.shape == (len(genes), len(barcodes)), "matrix/feature/barcode mismatch"

    # cells x genes
    X = sp.csr_matrix(mat.T)
    cell_line = np.array([bc.split("_")[0] for bc in barcodes])

    # labels
    lab = pd.read_csv(args.labels)
    lab = lab[lab["tp53_status"].isin(["MUT", "WT"])]
    status_map = dict(zip(lab["stripped_name"], lab["tp53_status"]))
    y_status = np.array([status_map.get(cl, "NA") for cl in cell_line])
    keep = y_status != "NA"
    print(f"[2] Label coverage: {keep.sum()}/{len(keep)} cells mapped "
          f"({(~keep).sum()} dropped as unlabeled)")
    X = X[keep]
    cell_line = cell_line[keep]
    y = (y_status[keep] == "MUT").astype(np.int8)

    # QC
    counts = np.asarray(X.sum(axis=1)).ravel()
    cell_ok = counts >= args.min_counts
    X = X[cell_ok]
    cell_line = cell_line[cell_ok]
    y = y[cell_ok]
    print(f"[3] QC cells: kept {cell_ok.sum()}/{len(cell_ok)} (min_counts={args.min_counts})")

    genes = np.array(genes)
    gene_ncells = np.asarray((X > 0).sum(axis=0)).ravel()
    gene_ok = gene_ncells >= args.min_cells
    X = X[:, gene_ok]
    genes = genes[gene_ok]
    print(f"[4] QC genes: kept {gene_ok.sum()}/{len(gene_ok)} (min_cells={args.min_cells})")

    # Normalize total + log1p (scanpy-free to avoid heavy import / OMP issues)
    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    Xn = X.multiply((1e4 / lib)[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data)

    # HVG via dispersion (seurat-like): mean & variance per gene on lognorm
    Xd = Xn.toarray().astype(np.float32)  # cells x genes (dense; fine after gene QC)
    mean = Xd.mean(axis=0)
    var = Xd.var(axis=0)
    disp = np.divide(var, mean, out=np.zeros_like(var), where=mean > 1e-8)
    order = np.argsort(disp)[::-1]
    top = np.sort(order[: args.n_hvg])
    Xhvg = Xd[:, top]
    gene_ids = genes[top]
    print(f"[5] Selected top {args.n_hvg} HVGs -> X shape {Xhvg.shape}")

    n_mut = int(y.sum())
    n_lines = len(np.unique(cell_line))
    print(f"    cells={Xhvg.shape[0]} genes={Xhvg.shape[1]} lines={n_lines} "
          f"MUT_cells={n_mut} WT_cells={len(y)-n_mut} "
          f"({100*n_mut/len(y):.1f}% MUT)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gambardella_hvg{args.n_hvg}.npz"
    np.savez_compressed(
        out,
        X=Xhvg.astype(np.float32),
        y=y.astype(np.int8),
        groups=cell_line.astype(str),
        gene_ids=gene_ids.astype(str),
    )
    print(f"[6] Wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
