#!/bin/bash
#SBATCH --job-name=analysis_summary
#SBATCH --partition=serial
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_full/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_full/logs/slurm/%x-%j.err

set -euo pipefail
source ~/venvs/turbml/bin/activate
export CUDA_VISIBLE_DEVICES=""   # force CPU
cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_full

# Baseline eval JSONs
python -m scripts.run_eval --config configs/E1_hit_baseline.yaml
python -m scripts.run_eval --config configs/E3_hit_ab_baseline.yaml

# MC σ–|err| + summary table (CPU)
python -m scripts.plot_sigma_error --config configs/E2_hit_bayes.yaml --split test --T 32 --max_points 50000
python -m scripts.plot_sigma_error --config configs/E4_hit_ab_dropout.yaml --split test --T 32 --max_points 50000

python -m scripts.compare_uq \
  --id_base  configs/E1_hit_baseline.yaml \
  --id_mc    configs/E2_hit_bayes.yaml \
  --ood_base configs/E3_hit_ab_baseline.yaml \
  --ood_mc   configs/E4_hit_ab_dropout.yaml \
  --T 32
