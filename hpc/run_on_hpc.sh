#!/bin/bash
# One-command HPC runner: clone/update from GitHub, build the env once, submit
# the GPU validation job. Re-run any time to pull the latest version and rerun.
#
# Usage (on the HPC login node):
#   bash <(curl -sL https://raw.githubusercontent.com/GabrieleBambini00/sheaf-tp53-benchmark/main/hpc/run_on_hpc.sh)
# or, if already cloned:
#   bash ~/sheaf-tp53-benchmark/hpc/run_on_hpc.sh
set -e
REPO_URL="https://github.com/GabrieleBambini00/sheaf-tp53-benchmark.git"
DIR="$HOME/sheaf-tp53-benchmark"

if [ -d "$DIR/.git" ]; then
  echo "[1/4] updating repo (git pull) ..."
  cd "$DIR" && git fetch origin && git reset --hard origin/main
else
  echo "[1/4] cloning repo ..."
  # Remove stale non-git dir if present
  [ -e "$DIR" ] && rm -rf "$DIR"
  git clone "$REPO_URL" "$DIR"
fi
cd "$DIR"

echo "[2/4] python env ..."
module load miniconda3 2>/dev/null || true
if [ ! -x "$DIR/.venv/bin/python" ]; then
  python3 -m venv "$DIR/.venv" || python -m venv "$DIR/.venv"
  source "$DIR/.venv/bin/activate"
  python -m pip install -q --upgrade pip
  pip install -q torch --index-url https://download.pytorch.org/whl/cu121
  pip install -q torch_geometric
  pip install -q pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
      -f https://data.pyg.org/whl/torch-2.5.0+cu121.html 2>/dev/null || true
  pip install -q numpy scipy scikit-learn pandas matplotlib
else
  source "$DIR/.venv/bin/activate"
fi
python -c "import torch; print('    torch', torch.__version__, '| cuda_build', torch.version.cuda)"

echo "[3/4] submitting GPU job ..."
mkdir -p logs results
JOBID=$(sbatch --parsable scripts/hpc_validate.sbatch)
echo "    SUBMITTED_JOBID=$JOBID"

echo "[4/4] queue:"
squeue -u "$USER"
echo ""
echo "watch log:  tail -f $DIR/logs/sheaf_${JOBID}.out"
