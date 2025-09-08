#!/bin/bash
#SBATCH --job-name=C3D_secondary_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=../logs/slurm/secondary_evaluation_%j.out
#SBATCH --error=../logs/slurm/secondary_evaluation_%j.err
#SBATCH --exclude=gpu[01-04]

# Load modules
module purge
module load python/3.13.1
module load cuda/12.6.2
module load gcc/13.3.0

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"

# Change to project directory
cd ${SLURM_SUBMIT_DIR}

# Print job info
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Python Version: $(python --version)"
echo "CUDA Version: $(nvcc --version | grep release)"
echo "GPU Info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
echo "=================================="

# Run secondary evaluation for C3D1 baseline
echo "Starting secondary evaluation for C3D1 baseline..."
python scripts/run_secondary_evaluation.py \
    --config configs/3d_secondary/C3D1_secondary_5200.yaml

echo "Secondary evaluation completed at: $(date)"
