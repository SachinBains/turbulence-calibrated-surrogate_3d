#!/bin/bash
#SBATCH --job-name=C3D4_cranfield
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/C3D4_cranfield_%j.out
#SBATCH --error=logs/C3D4_cranfield_%j.err

# Load modules (UPDATE FOR CRANFIELD HPC)
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Activate virtual environment (UPDATE PATH)
source /home/USERNAME/venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/USERNAME/turbulence-calibrated-surrogate_3d:$PYTHONPATH

# Create necessary directories
mkdir -p /home/USERNAME/artifacts_3d/{results,checkpoints,logs,cache}
mkdir -p logs

# Run training
python scripts/run_train.py --config cranfield/C3D4_cranfield.yaml --cuda
