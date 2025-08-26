#!/bin/bash
#SBATCH --job-name=analysis_summary
#SBATCH --partition=serial
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_full/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_full/logs/slurm/%x-%j.err

set -euo pipefail
source ~/venvs/turbml/bin/activate
cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_full

# ensure JSON metrics for baselines
python -m scripts.run_eval --config configs/E1_hit_baseline.yaml
python -m scripts.run_eval --config configs/E3_hit_ab_baseline.yaml

# ensure sigma-error + comparisons for MC
CUDA_VISIBLE_DEVICES="" python -m scripts.plot_sigma_error --config configs/E4_hit_ab_dropout.yaml --split test --T 32 --max_points 50000
CUDA_VISIBLE_DEVICES="" python -m scripts.compare_uq \
  --id_base configs/E1_hit_baseline.yaml \
  --id_mc   configs/E2_hit_bayes.yaml \
  --ood_base configs/E3_hit_ab_baseline.yaml \
  --ood_mc   configs/E4_hit_ab_dropout.yaml \
  --T 32
