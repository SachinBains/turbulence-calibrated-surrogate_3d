#!/bin/bash
#SBATCH --job-name=baseline32
#SBATCH --partition=gpuV
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

module purge
source ~/.bashrc
conda activate turbml   # <-- replace with your env name if different

cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_full

python -m scripts.run_train --config configs/E1_hit_baseline.yaml
python -m scripts.run_eval  --config configs/E1_hit_baseline.yaml
