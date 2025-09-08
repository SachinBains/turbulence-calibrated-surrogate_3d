#!/bin/bash
#SBATCH --job-name=streamlined_stage2
#SBATCH --partition=gpuV
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/streamlined_stage2_%j.out
#SBATCH --error=logs/slurm/streamlined_stage2_%j.err

# Load modules
module load python/3.13.1
module load cuda/12.6.2
module load gcc/13.3.0

# Activate environment
source /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/venvs/turbml/bin/activate

# Set environment variables
export ARTIFACTS_ROOT="/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Run Stage 2: Method-specific predictions
echo "Starting Stage 2: Method-specific Predictions"
python scripts/run_streamlined_evaluation.py --stage 2

echo "Stage 2 completed at $(date)"
