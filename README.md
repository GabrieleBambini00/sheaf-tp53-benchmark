# Sheaf GNNs vs Ravasio — an honest TP53-status benchmark

A faithful, **leakage-free** re-evaluation of Tommaso Ravasio's BSc-thesis task
(predict *TP53* mutation status from cell-line RNA-seq with GNNs), plus a
**Neural Sheaf Diffusion** classifier compared head-to-head with his GCN / GAT
under identical protocols — on the underpowered single-cell atlas he used **and**
on a powered 1,385-cell-line bulk benchmark.

## Bottom line

1. **Ravasio's ~0.99 was data leakage.** He split *cells* at random, so the same
   cell line appeared in train and test; the model memorised cell-line identity.
   A flat logistic regression reproduces this: **0.41 honest → 0.999 leaked.**
   Honest TP53-from-expression accuracy is ~0.6 AUROC, not 0.99.

2. **The sheaf significantly beats both of his GNNs (GCN, GAT)** on a powered,
   leakage-free benchmark (1,385 CCLE lines, 521 *TP53*-WT). Paired bootstrap on
   the same lines: Sheaf vs GCN **P≈1.00**, vs GAT **P≈0.99**, 95% CIs exclude 0.

3. **A sheaf+flat hybrid (0.82)** — concatenating the pooled sheaf embedding with
   a learned projection of the raw expression — *significantly* beats logistic
   regression (+0.031, P=1.00) and closes most of the gap to gradient boosting,
   while keeping the sheaf geometry.

4. **Honest caveat:** on *bulk*, plain gradient boosting (~0.85) still beats every
   pure graph model. Pooling gene-nodes over a co-expression graph discards
   per-gene signal; the graph is not the ideal inductive bias for bulk. The pure
   sheaf is the best *graph* model, not the best model overall.

## The task (faithful to Ravasio)
- **Data**: Gambardella 2022 single-cell BC atlas (32 lines) and DepMap-21Q4 CCLE
  bulk (1,385 lines). Node = gene; edges = Spearman co-expression (|ρ|≥τ,
  unweighted, train-only); per-sample node feature = expression; graph-level
  label = the line's *TP53* status.
- **Labels**: authoritative — Gambardella from Cellosaurus curated mutations
  (25 MUT / 7 WT); CCLE from DepMap damaging **OR** hotspot mutations (864/521).
- **Leakage fix**: StratifiedGroupKFold by cell line (single cell), and 1
  sample/line (bulk) → leave-lines-out is leakage-free by construction.
- **Honest metric**: per-cell-line out-of-fold AUROC (cells within a line are
  pseudo-replicates), with a **paired bootstrap** over lines for comparisons.

## Results (per-line AUROC, leakage-free)

**CCLE bulk — 1,385 lines, 521 WT (powered; 5×5 grouped folds, thr=0.5).**
Paired bootstrap over the same 1,385 lines; ✓ = 95% CI excludes 0.

| model | line AUROC | paired comparison |
|-------|-----------:|-------------------|
| GradBoost (flat)        | 0.861 | best overall |
| **Sheaf+flat (hybrid)** | **0.822** | beats LogReg +0.031 (P=1.00 ✓); loses to GradBoost −0.039 |
| LogReg (flat)           | 0.791 | — |
| **Sheaf d12 (pure)**    | **0.706** | — |
| GAT (Ravasio)           | 0.638 | **pure-sheaf wins +0.068 (P=1.00 ✓)** |
| GCN (Ravasio)           | 0.595 | **pure-sheaf wins +0.110 (P=1.00 ✓)** |

Pure sheaf significantly beats both Ravasio GNNs; the hybrid additionally beats
the linear flat baseline and trails only gradient boosting.

**Gambardella single-cell — 32 lines, 7 WT (underpowered; thr=0.2, 5×5 folds).**
Sheaf is best (0.583) but CIs are ±0.3 (only 7 WT lines) so no graph model is
statistically separable. Same ranking, no significance — which is why the CCLE
benchmark exists.

See `figure_comparison.png` and `RESULTS.md`.

## The sheaf architecture
Static parametric Neural Sheaf Diffusion: each gene-node carries a *d*-dim stalk
(learned per-gene direction modulated by expression); each edge a learned *d×d*
restriction map that **can sign-invert**, so anti-correlated (repressor) edges are
encoded natively — the homophily failure mode of GCN/GAT is avoided by
construction. The sheaf Laplacian is shared across the batch (one dense
`L_F`-matvec per layer), giving ~30–60× faster CPU training than per-graph PyG
message passing. Sweet spot: `stalk_dim=12`, 2 layers, mean+max pool. The
`flat_skip` hybrid concatenates a projection of the raw expression vector before
the head.

## Reproduce
```bash
pip install -e ".[dev]"
pytest -q                                   # smoke tests

# 1. labels + data
python scripts/build_gambardella_labels.py  # Cellosaurus -> data/labels/...
python scripts/prep_gambardella.py --n_hvg 100
python scripts/prep_ccle.py --n_hvg 300      # needs DepMap-21Q4 CSVs (see prep_ccle docstring)

# 2. powered benchmark + stats + figure
python scripts/run_oof.py --npz data/preprocessed/ccle_tp53_hvg300.npz \
    --models logreg hgb gcn gat sheaf_d12 sheaf_hybrid \
    --n_repeats 5 --threshold 0.5 --tag ccle_final
python scripts/oof_stats.py results/oof_ccle_final_*.json --ref "Sheaf d12"
python scripts/make_figure.py --out figure_comparison.png \
    "CCLE bulk (1385 lines, 521 WT)=results/oof_ccle_final_*.json"
```

## HPC (GPU, SLURM) — run from GitHub
The repo clones runnable (preprocessed `.npz` are committed). On the login node:
```bash
git clone https://github.com/GabrieleBambini00/sheaf-tp53-benchmark.git
bash sheaf-tp53-benchmark/hpc/run_on_hpc.sh   # builds a CUDA venv once, then sbatch
```
`hpc/run_on_hpc.sh` pulls the latest version, (re)builds the venv if needed, and
submits `scripts/hpc_validate.sbatch` (partition `stud`, 1 A100 MIG slice). Re-run
it any time to pull a newer version and re-validate. Results land in `results/`
and `logs/sheaf_<jobid>.out`.

## Layout
```
src/sheaf_tp53/
  _sheaf.py    vectorised sheaf connection Laplacian (P1–P5 verified)
  models.py    GCN / GAT / Sheaf(+hybrid) classifiers (shared-topology, batched)
  graph.py     train-only Spearman co-expression graph
  splits.py    group-disjoint splitter + leakage assertion
  oof.py       per-line out-of-fold harness (the honest protocol)
  cv.py        cell-level CV (used for the leaked-vs-honest demo)
  seeds.py     global RNG seeding
scripts/       data prep, labels, run_oof, oof_stats, make_figure
tests/         smoke tests
```

## Credit
Task, data sources, and GCN/GAT baselines follow
[tommasoravasio/scRNAseq-GNN-binary-tp53](https://github.com/tommasoravasio/scRNAseq-GNN-binary-tp53).
This repo adds the leakage-free protocol, the sheaf model, and the powered CCLE
benchmark.
