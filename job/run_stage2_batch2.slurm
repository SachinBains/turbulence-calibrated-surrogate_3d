#!/bin/bash
#SBATCH --job-name=stage2_batch2
#SBATCH --partition=gpuV
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:50:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/slurm/stage2_batch2_%j.out
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/logs/slurm/stage2_batch2_%j.err

# STAGE 2 BATCH 2: C3D3 (Ensemble) + C3D6 (Physics)
echo "Starting STAGE 2 BATCH 2: C3D3 + C3D6"
echo "Job ID: $SLURM_JOB_ID"

# Environment setup
module load apps/anaconda3/2022.10
source activate tcs3d

export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d:$PYTHONPATH
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d

cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d

# Fix ensemble structure
python scripts/fix_ensemble_structure.py --artifacts_dir $ARTIFACTS_ROOT

# C3D3 - Ensemble (~45 minutes)
echo "=== Running C3D3 Ensemble ==="
if find $ARTIFACTS_ROOT/results/C3D3_channel_ensemble_128/members -name "*.pth" | head -1 | grep -q .; then
    echo "✓ C3D3: Ensemble members found"
    python scripts/run_secondary_evaluation.py --config configs/3d_secondary/C3D3_secondary_5200.yaml --cuda
    python scripts/validate_physics.py --config configs/3d_secondary/C3D3_secondary_5200.yaml --results_dir $ARTIFACTS_ROOT/results/C3D3_secondary_5200 --method ens
    echo "✓ C3D3 completed"
else
    echo "✗ C3D3: No ensemble members found"
fi

# C3D6 - Physics Informed (~15 minutes)
echo "=== Running C3D6 Physics Informed ==="
if find $ARTIFACTS_ROOT/results/C3D6_channel_physics_informed_128 -name "*.pth" | head -1 | grep -q .; then
    echo "✓ C3D6: Checkpoint found"
    python scripts/run_secondary_evaluation.py --config configs/3d_secondary/C3D6_secondary_5200.yaml --cuda
    python scripts/validate_physics.py --config configs/3d_secondary/C3D6_secondary_5200.yaml --results_dir $ARTIFACTS_ROOT/results/C3D6_secondary_5200 --method mc
    echo "✓ C3D6 completed"
else
    echo "✗ C3D6: No checkpoint found"
fi

echo "BATCH 2 completed at: $(date)"
