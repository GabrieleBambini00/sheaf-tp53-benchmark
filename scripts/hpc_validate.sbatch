#!/bin/bash
#SBATCH --job-name=sheaf_p53
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sheaf_%j.out
#SBATCH --error=logs/sheaf_%j.out
set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
module load cuda/12.4 2>/dev/null || module load cuda/12.1 2>/dev/null || true
source .venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=src

echo "=== node: $(hostname) ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"

echo "=== powered CCLE benchmark (5x5 folds) on GPU ==="
python -u scripts/run_oof.py --npz data/preprocessed/ccle_tp53_hvg300.npz \
    --models logreg hgb gcn gat sheaf_d12 sheaf_hybrid \
    --n_repeats 5 --n_splits 5 --threshold 0.5 --cap 5 --epochs 40 --tag hpc_ccle

echo "=== paired bootstrap: pure sheaf vs Ravasio GNNs ==="
python scripts/oof_stats.py results/oof_hpc_ccle_*.json --ref "Sheaf d12"
echo "=== paired bootstrap: hybrid ==="
python scripts/oof_stats.py results/oof_hpc_ccle_*.json --ref "hybrid"
echo "HPC_VALIDATION_DONE"
