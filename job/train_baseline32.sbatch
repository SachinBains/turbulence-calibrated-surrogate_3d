#!/bin/bash
#SBATCH --job-name=baseline32
#SBATCH --partition=gpuV
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_full/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_full/logs/slurm/%x-%j.err

set -euo pipefail

# --- activate Python venv (no conda needed) ---
source ~/venvs/turbml/bin/activate

# --- go to repo root ---
cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_full
mkdir -p logs/slurm

# --- quick env printouts ---
echo "WHOAMI=$(whoami)"
echo "HOST=$(hostname)"
echo "PYTHON=$(which python)"
python -c "import sys, torch; print('Py', sys.version); print('CUDA avail:', torch.cuda.is_available())"

# --- run training then evaluation ---
python -m scripts.run_train --config configs/E1_hit_baseline.yaml
python -m scripts.run_eval  --config configs/E1_hit_baseline.yaml

