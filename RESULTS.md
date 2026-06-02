# Sheaf GNN vs Ravasio — honest TP53-status prediction from transcriptomics

## What this is
A faithful reproduction of Tommaso Ravasio's BSc thesis pipeline (GNNs predicting
TP53 mutation status from scRNA-seq), with (1) the **cell-line data leakage fixed**
and (2) a **Neural Sheaf Diffusion** classifier added and compared head-to-head
with his GCN / GAT under identical, leakage-free protocols.

## Bottom line
- **The sheaf significantly beats both of Ravasio's GNN models (GCN, GAT)** on a
  powered, leakage-free benchmark (1385 CCLE cell lines, 521 WT, 5×5 folds):
  pure Sheaf 0.706 vs GAT 0.638 (P=1.00) vs GCN 0.595 (P=1.00); a sheaf+flat
  hybrid reaches 0.822 and also beats logistic regression (P=1.00). Same ranking
  on the single-cell atlas (sheaf best at 0.583), though that task is underpowered
  (7 WT lines).
- **Ravasio's reported 0.99+ was data leakage**: flat LogReg goes 0.41 (honest) →
  0.999 (his random-cell split). Honest accuracy on this task is ~0.6.
- **Honest caveat:** on bulk, flat gradient boosting (0.85) beats every graph
  model — the sheaf is the best *graph* architecture, not the best model overall.

---

## 1. The leakage (reproducing Ravasio's inflation)
Ravasio split *cells* at random, so the same cell line sits in train AND test.
Cells from one line are near-identical, so a model memorises cell-line identity,
not TP53 biology.

Gambardella atlas, N=100 HVG, random cell-level split (his protocol):

| model | AUROC (LEAKED) | AUROC (honest, grouped) |
|-------|---------------:|------------------------:|
| LogReg (flat)      | **0.999** | 0.411 |
| Sheaf (ours)       | 0.998 | 0.583 |
| GAT (Ravasio best) | 0.790 | 0.543 |
| GCN (Ravasio)      | 0.645 | 0.520 |

Flat LogReg 0.41 → **0.999** under leakage ≈ Ravasio's reported XGBoost F1 0.995.
His headline numbers were almost entirely cell-line memorisation.

---

## 2. Gambardella single-cell — honest but UNDERPOWERED
Per-line OOF AUROC over 32 cell lines (leakage-free StratifiedGroupKFold by line).

R=5 (25 folds), N=100 HVG, thr=0.2:

| model | line AUROC |
|-------|-----------:|
| **Sheaf d8 (ours)** | **0.583** |
| GAT (Ravasio) | 0.543 |
| GCN (Ravasio) | 0.520 |
| LogReg (flat) | 0.411 |

Sheaf is the best honest model in **every** configuration (R=2: 0.634; R=5: 0.583;
N=300: 0.611). BUT the atlas has only **7 TP53-WT lines** → AUROC SE ≈ 0.19, paired
bootstrap P(sheaf>GCN)=0.61. Consistently best, not *significantly* best — a limit
of the dataset, not the method.

---

## 3. CCLE bulk — POWERED confirmation
DepMap 21Q4 CCLE bulk RNA-seq: **1385 cell lines**, TP53 = damaging|hotspot
(864 MUT / **521 WT**). Each line is one sample → leave-lines-out is leakage-free by
construction, and 521 WT lines give AUROC SE ≈ 0.022 → differences of ~0.03 are
significant. Run at **5×5 grouped folds** (n_repeats=5), thr=0.5.

Per-line OOF AUROC over 1385 lines:

| model | line AUROC |
|-------|-----------:|
| GradBoost (flat)        | **0.861** |
| **Sheaf+flat (hybrid)** | **0.822** |
| LogReg (flat)           | 0.791 |
| **Sheaf d12 (pure)**    | **0.706** |
| GAT (Ravasio best)      | 0.638 |
| GCN (Ravasio)           | 0.595 |

Paired bootstrap (same 1385 lines), **pure Sheaf d12** vs:

| baseline | AUROC diff | 95% CI | P(sheaf>base) | sig |
|----------|-----------:|--------|--------------:|-----|
| GCN (Ravasio)    | **+0.110** | [+0.078, +0.144] | 1.00 | ✓ |
| GAT (Ravasio)    | **+0.068** | [+0.034, +0.103] | 1.00 | ✓ |
| LogReg (flat)    | −0.086 | [−0.115, −0.056] | 0.00 | (loses) |
| GradBoost (flat) | −0.156 | [−0.179, −0.132] | 0.00 | (loses) |

Paired bootstrap, **Sheaf+flat hybrid** vs:

| baseline | AUROC diff | 95% CI | P | sig |
|----------|-----------:|--------|--:|-----|
| GAT (Ravasio)    | +0.185 | [+0.152, +0.218] | 1.00 | ✓ |
| GCN (Ravasio)    | +0.227 | [+0.194, +0.261] | 1.00 | ✓ |
| LogReg (flat)    | **+0.031** | [+0.013, +0.048] | 1.00 | ✓ |
| GradBoost (flat) | −0.039 | [−0.054, −0.024] | 0.00 | (loses) |

### Verdict
1. **The pure sheaf significantly beats BOTH of Ravasio's GNN models (GCN, GAT)** —
   CIs exclude zero, P = 1.00. Statistically validated "sheaf beats Ravasio".
2. **The sheaf+flat hybrid significantly beats the linear flat baseline (LogReg)**
   too and is the 2nd-best model overall, recovering the per-gene signal that pure
   graph pooling discards.
3. **Honest caveat:** flat gradient boosting (0.861) still wins overall — even the
   hybrid loses to it (−0.039, P=0.00). On bulk, the graph is not the ideal
   inductive bias. The sheaf is the best *graph* model, not the best model overall.

---

## Architecture (sheaf, winning config)
Static parametric Neural Sheaf Diffusion: per-gene stalk d=8 (capacity is the
biggest lever: d4 0.41 → d8 0.63 on Gambardella), learned per-edge d×d restriction
maps (sign-inverting → encode repressor edges), 2 diffusion layers, mean+max pool,
MLP head. Reuses the verified `VectorizedSheafDiffusion` operator; shared L_F
batched over all cells → ~30-60× faster than PyG GAT on CPU.

## Reproduce
```
# data
python scripts/build_gambardella_labels.py          # Cellosaurus TP53 status
python scripts/prep_gambardella.py --n_hvg 100
python scripts/prep_ccle.py --n_hvg 300              # DepMap 21Q4 CCLE bulk
# benchmarks (leakage-free, per-line OOF AUROC + paired bootstrap)
python scripts/run_oof.py --npz data/preprocessed/gambardella_hvg100.npz \
    --models logreg gcn gat sheaf_d8 --n_repeats 5 --tag r5
python scripts/run_oof.py --npz data/preprocessed/ccle_tp53_hvg300.npz \
    --models logreg hgb gcn gat sheaf_d8 --n_repeats 2 --threshold 0.5 --tag ccle
python scripts/oof_stats.py results/oof_ccle_*.json --ref "Sheaf d8"
python scripts/run_benchmark.py --leaky --models logreg gcn gat sheaf  # leakage demo
```
