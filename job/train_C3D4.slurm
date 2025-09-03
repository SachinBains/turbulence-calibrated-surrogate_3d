#!/bin/bash
#SBATCH --job-name=C3D4_variational
#SBATCH --partition=gpuV
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/training/C3D4_%j.out
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/training/C3D4_%j.err

# Load required modules
module load python/3.13.1
module load cuda/12.6.2
module load gcc/13.3.0

# Activate virtual environment
source /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/venvs/turbml/bin/activate

# Set working directory and Python path
cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d
export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d:$PYTHONPATH

# Run variational training
python scripts/run_train.py --config configs/3d/C3D4_channel_variational_128.yaml --seed 42 --cuda
