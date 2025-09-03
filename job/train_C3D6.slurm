#!/bin/bash
#SBATCH --job-name=C3D6_physics
#SBATCH --partition=gpuV
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=30:00:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/training/C3D6_%j.out
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/training/C3D6_%j.err

# Load required modules
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Activate virtual environment
source /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/venvs/turbml/bin/activate

# Set working directory and Python path
cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d
export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d:$PYTHONPATH

# Run physics-informed training
python scripts/run_train.py configs/3d/C3D6_channel_physics_informed_128.yaml 42 --cuda
