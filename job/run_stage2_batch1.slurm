#!/bin/bash
#SBATCH --job-name=stage2_batch1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/slurm/stage2_batch1_%j.out
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/slurm/stage2_batch1_%j.err

# STAGE 2 BATCH 1: C3D1 (Baseline) + C3D2 (MC Dropout)
echo "Starting STAGE 2 BATCH 1: C3D1 + C3D2"
echo "Job ID: $SLURM_JOB_ID"

# Environment setup
module load apps/anaconda3/2022.10
source activate tcs3d

export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d:$PYTHONPATH
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d

cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d

# Fix ensemble structure
python scripts/fix_ensemble_structure.py --artifacts_dir $ARTIFACTS_ROOT

# C3D1 - Baseline (~15 minutes)
echo "=== Running C3D1 Baseline ==="
if find $ARTIFACTS_ROOT/results/C3D1_channel_baseline_128 -name "*.pth" | head -1 | grep -q .; then
    echo "✓ C3D1: Checkpoint found"
    python scripts/run_secondary_evaluation.py --config configs/3d_secondary/C3D1_secondary_5200.yaml --cuda
    python scripts/validate_physics.py --config configs/3d_secondary/C3D1_secondary_5200.yaml --cuda
    echo "✓ C3D1 completed"
else
    echo "✗ C3D1: No checkpoint found"
fi

# C3D2 - MC Dropout (~25 minutes)
echo "=== Running C3D2 MC Dropout ==="
if find $ARTIFACTS_ROOT/results/C3D2_channel_mc_dropout_128 -name "*.pth" | head -1 | grep -q .; then
    echo "✓ C3D2: Checkpoint found"
    python scripts/run_secondary_evaluation.py --config configs/3d_secondary/C3D2_secondary_5200.yaml --cuda
    python scripts/validate_physics.py --config configs/3d_secondary/C3D2_secondary_5200.yaml --cuda
    echo "✓ C3D2 completed"
else
    echo "✗ C3D2: No checkpoint found"
fi

echo "BATCH 1 completed at: $(date)"
