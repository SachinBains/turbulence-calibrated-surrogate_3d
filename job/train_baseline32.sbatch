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

module purge
source ~/.bashrc
# Try to activate conda env, print warning if missing
if ! command -v conda &> /dev/null; then
  echo "[ERROR] conda not found in PATH. Please check your environment setup." >&2
  exit 1
fi

if ! conda activate turbml 2>/dev/null; then
  echo "[ERROR] conda env 'turbml' not found or failed to activate." >&2
  exit 1
fi

cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_full

python -m scripts.run_train --config configs/E1_hit_baseline.yaml
python -m scripts.run_eval  --config configs/E1_hit_baseline.yaml
