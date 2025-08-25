#!/bin/bash
#SBATCH --job-name=e3_ab_baseline
#SBATCH --partition=gpuV
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_full/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_full/logs/slurm/%x-%j.err

set -euo pipefail

source ~/venvs/turbml/bin/activate
cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_full
mkdir -p logs/slurm

python -m scripts.run_train --config configs/E3_hit_ab_baseline.yaml
python -m scripts.predict_mc --config configs/E3_hit_ab_baseline.yaml --T 32 --split test
