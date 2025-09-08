#!/bin/bash
#SBATCH --job-name=streamlined_stage1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/streamlined_stage1_%j.out
#SBATCH --error=logs/slurm/streamlined_stage1_%j.err

# Load modules
module load python/3.9.7
module load cuda/11.8
module load gcc/9.3.0

# Activate environment
source $HOME/venvs/tcs3d/bin/activate

# Set environment variables
export ARTIFACTS_ROOT="/mnt/iusers01/support/mbaxkb7/scratch/artifacts_3d"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Run Stage 1: Core evaluation
echo "Starting Stage 1: Core Evaluation"
python scripts/run_streamlined_evaluation.py --stage 1

echo "Stage 1 completed at $(date)"
