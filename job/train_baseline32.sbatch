#!/bin/bash
#SBATCH --job-name=baseline32
#SBATCH --output=logs/slurm/baseline32_%j.out
#SBATCH --error=logs/slurm/baseline32_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=06:00:00

set -e

module load cuda/11.7 || true  # Adjust as needed for your cluster

# Activate your environment if needed
# source ~/envs/yourenv/bin/activate

export OMP_NUM_THREADS=6

srun python -m scripts.run_train --config configs/E1_hit_baseline.yaml
srun python -m scripts.run_eval --config configs/E1_hit_baseline.yaml
